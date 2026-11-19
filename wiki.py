# wiki_rag.py - LLM Wiki style Q&A + Ingest over Obsidian vault
import os
import re
from collections import defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile

# Workaround for unstructured (you already had this)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from chaincrafter import Chain, Prompt

from model_provider import model_names
from prompt_builder import PromptBuilder
from retrieval_augmented_generator import GeneratedAnswer, RetrievalAugmentedGenerator
from retriever import ChunkerKind


STRUCTURAL_WIKI_FILES = {"index.md", "log.md", "SCHEMA.md"}


def wiki_documents(vault_path: Path) -> list[Path]:
    return sorted(
        path
        for path in vault_path.rglob("*.md")
        if not (path.parent == vault_path and path.name in STRUCTURAL_WIKI_FILES)
        and not any(part.startswith(".") for part in path.relative_to(vault_path).parts)
    )


class WikiPromptBuilder(PromptBuilder):
    def ingest_prompt(self, new_source_text: str, relevant_chunks: list[str], existing_pages: list[str], schema: str):
        # We use a template string here to let Chain handle formatting correctly
        # and we use placeholders {context} and {new_source_text} that we pass in input_vars
        return Prompt(
            """You are the maintainer of a personal LLM Wiki stored as an Obsidian vault.

WIKI CONVENTIONS & SCHEMA:
{schema}

Current relevant wiki context (snippets):
{context}

Full content of relevant existing pages:
{relevant_pages_content}

All existing pages in the vault:
{existing_pages}

=== NEW SOURCE TO INGEST ===
{new_source_text}
=== END OF NEW SOURCE ===

Your job is to process the new source according to the WIKI CONVENTIONS & SCHEMA provided above.
Do not explain your reasoning. Just output the markdown.
"""
        )

    def lint_prompt(self, index_content: str, recent_log: str, schema: str):
        return Prompt(
            """You are the maintainer of a personal LLM Wiki. Review the current index and recent activity log to identify health issues.
            
            WIKI CONVENTIONS & SCHEMA:
            {schema}
            
            WIKI INDEX:
            {index_content}
            
            RECENT LOG:
            {recent_log}
            
            Use the focus areas defined in the schema to identify health issues.
            IMPORTANT: Specifically identify information gaps and suggest new topics or pages that should be created to enrich the wiki.
            
            Output your findings as a concise markdown report. Include a section for "Suggested New Topics".
            """
        )

    def select_pages_prompt(self, query: str, index_content: str, schema: str):
        return Prompt(
            """You are a personal wiki assistant. Given a user query and the wiki index, identify the most relevant pages that should be read in full to answer the query accurately.
            
            WIKI CONVENTIONS & SCHEMA:
            {schema}
            
            WIKI INDEX:
            {index_content}
            
            QUERY: {query}
            
            Output a comma-separated list of the page titles (exactly as they appear in the index, without the brackets [[ ]]). If no pages are relevant, output "None".
            """
        )

    def wiki_answer_prompt(self, query: str, context: str, full_pages_content: str, schema: str):
        return Prompt(
            """You are a personal wiki assistant. Answer the user's query using the provided snippets and the full content of relevant pages.
            
            WIKI CONVENTIONS & SCHEMA:
            {schema}
            
            SNIPPETS (from vector search):
            {context}
            
            FULL PAGE CONTENT (relevant pages):
            {full_pages_content}
            
            QUERY: {query}
            
            Answer the query according to the WIKI CONVENTIONS & SCHEMA provided above.
            """
        )


class WikiRetrievalAugmentedGenerator(RetrievalAugmentedGenerator):
    def __init__(self, *args, vault_path: str | Path, **kwargs):
        self.vault_path = Path(vault_path)
        super().__init__(*args, **kwargs)

    def _relative_page_name(self, page: Path) -> str:
        return page.relative_to(self.vault_path).with_suffix("").as_posix()

    @staticmethod
    def _page_title(content: str) -> str | None:
        yaml_match = re.search(r"^---\s*\n(.*?)\n---", content, re.DOTALL | re.MULTILINE)
        if yaml_match:
            title_match = re.search(
                r"^\s*title:\s*(.+)$",
                yaml_match.group(1),
                re.MULTILINE | re.IGNORECASE,
            )
            if title_match:
                return title_match.group(1).strip()
        h1_match = re.search(r"^\s*#\s+(.+)$", content, re.MULTILINE)
        return h1_match.group(1).strip() if h1_match else None

    def _target_for_title(self, title: str) -> Path:
        filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "-", title).strip(" .")
        if not filename:
            raise ValueError("wiki page title does not produce a valid filename")
        if f"{filename}.md".casefold() in {name.casefold() for name in STRUCTURAL_WIKI_FILES}:
            raise ValueError(f"wiki page title is reserved: {title}")

        target = self.vault_path / f"{filename}.md"
        if target.exists():
            existing_title = self._page_title(target.read_text(encoding="utf-8"))
            if existing_title != title:
                raise ValueError(
                    f"wiki page title {title!r} collides with existing file {target.name!r}"
                )
        return target

    def _write_pages_atomically(self, pages: list[dict[str, str]]) -> list[Path]:
        plans = []
        normalized_targets = set()
        for page in pages:
            target = self._target_for_title(page["title"])
            normalized_target = target.name.casefold()
            if normalized_target in normalized_targets:
                raise ValueError(f"multiple generated pages target {target.name!r}")
            normalized_targets.add(normalized_target)
            plans.append((target, page["content"]))

        self.vault_path.mkdir(parents=True, exist_ok=True)
        temporary_files: list[tuple[Path, Path]] = []
        try:
            for target, content in plans:
                with NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=self.vault_path,
                    prefix=f".{target.stem}-",
                    suffix=".tmp",
                    delete=False,
                ) as handle:
                    handle.write(content)
                    temporary_files.append((Path(handle.name), target))
            for temporary, target in temporary_files:
                temporary.replace(target)
        finally:
            for temporary, _ in temporary_files:
                temporary.unlink(missing_ok=True)
        return [target for _, target in temporary_files]

    @staticmethod
    def _read_source_text(source_path: Path) -> str:
        if source_path.suffix.lower() in {".md", ".txt"}:
            return source_path.read_text(encoding="utf-8")

        from unstructured.partition.auto import partition as unstructured_partition

        elements = unstructured_partition(str(source_path))
        return "\n\n".join(str(element) for element in elements)

    def _update_index(self, vault_path: Path):
        """Maintains a catalog of all pages in the wiki (index.md)."""
        index_file = vault_path / "index.md"
        categories = defaultdict(list)
        yaml_pattern = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL | re.MULTILINE)

        for page in wiki_documents(vault_path):
            content = page.read_text(encoding="utf-8")
            match = yaml_pattern.search(content)
            category = "General"
            summary = "No summary provided."
            if match:
                yaml_block = match.group(1)
                category_match = re.search(
                    r"^\s*category:\s*(.+)$",
                    yaml_block,
                    re.MULTILINE | re.IGNORECASE,
                )
                summary_match = re.search(
                    r"^\s*summary:\s*(.+)$",
                    yaml_block,
                    re.MULTILINE | re.IGNORECASE,
                )
                if category_match:
                    category = category_match.group(1).strip()
                if summary_match:
                    summary = summary_match.group(1).strip()
            page_name = self._relative_page_name(page)
            categories[category].append(f"- [[{page_name}]]: {summary}")

        lines = ["# Wiki Index", "", "## Summary", "A categorized list of all wiki pages.", ""]
        for category in sorted(categories):
            lines.append(f"### {category}")
            lines.extend(sorted(categories[category]))
            lines.append("")
        index_file.write_text("\n".join(lines), encoding="utf-8")
        print(f"→ Structural Maintenance: {index_file} updated.")

    def _update_log(self, vault_path: Path, action: str, details: str):
        """Maintains a chronological record of activity (log.md)."""
        log_file = vault_path / "log.md"
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        entry = f"## [{timestamp}] {action} | {details}"
        
        if log_file.exists():
            content = log_file.read_text(encoding="utf-8")
            if not content.startswith("# Wiki Log"):
                content = "# Wiki Log\n\n" + content
            new_content = content.strip() + "\n\n" + entry + "\n"
        else:
            new_content = f"# Wiki Log\n\n{entry}\n"
            
        log_file.write_text(new_content, encoding="utf-8")
        print(f"→ Structural Maintenance: {log_file} updated.")

    def _load_schema(self) -> str:
        """Loads the wiki schema from SCHEMA.md in the vault."""
        schema_file = self.vault_path / "SCHEMA.md"
        if schema_file.exists():
            return schema_file.read_text(encoding="utf-8")
        return "No external schema found. Follow standard wiki conventions."

    def lint(self):
        """Performs a health check on the wiki."""
        vault_path = self.vault_path
        all_pages = wiki_documents(vault_path)
        page_names = {self._relative_page_name(page) for page in all_pages}
        
        print(f"\n--- Running Wiki Health Check on {len(all_pages)} pages ---")
        
        # 1. Programmatic Linting
        inbound_links = {name: [] for name in page_names}
        broken_links = []
        missing_cross_refs = []
        
        link_pattern = re.compile(r"\[\[(.*?)\]\]")
        
        for p in all_pages:
            content = p.read_text(encoding="utf-8")
            links = link_pattern.findall(content)
            
            linked_targets = set()
            for link in links:
                target = link.split("|")[0].strip()
                linked_targets.add(target)
                if target in page_names:
                    inbound_links[target].append(p.stem)
                elif target not in ("index", "log"):
                    broken_links.append((p.stem, target))
            
            # Missing cross-references: check for unlinked mentions of other pages
            for other_name in page_names:
                if other_name != p.stem and other_name not in linked_targets:
                    # Simple whole-word check
                    if re.search(rf"\b{re.escape(other_name)}\b", content, re.IGNORECASE):
                        missing_cross_refs.append((p.stem, other_name))
        
        orphans = [name for name, sources in inbound_links.items() if not sources]
        
        print("\n[Programmatic Results]")
        if orphans:
            print(f"• Orphan pages (no inbound links): {', '.join(f'[[{o}]]' for o in orphans)}")
        if broken_links:
            print("• Broken links (target missing):")
            for source, target in broken_links:
                print(f"  - In [[{source}]]: [[{target}]]")
        
        if missing_cross_refs:
            grouped_missing = defaultdict(list)
            for source, target in missing_cross_refs:
                grouped_missing[source].append(target)
            
            print("• Potential missing cross-references:")
            for source, targets in sorted(list(grouped_missing.items()))[:5]:
                print(f"  - In [[{source}]]: mention of {', '.join(f'[[{t}]]' for t in targets)}")
            if len(grouped_missing) > 5:
                print(f"  - ... and {len(grouped_missing) - 5} more.")
                
        if not orphans and not broken_links and not missing_cross_refs:
            print("• No obvious structural issues found.")

        # 2. LLM-based Health Check
        index_file = vault_path / "index.md"
        log_file = vault_path / "log.md"
        
        index_content = index_file.read_text(encoding="utf-8") if index_file.exists() else "No index found."
        
        recent_log = ""
        if log_file.exists():
            log_lines = log_file.read_text(encoding="utf-8").splitlines()
            recent_log = "\n".join(log_lines[-30:]) # Last 30 lines for context
            
        schema = self._load_schema()
        lint_prompt_obj = self.prompts.lint_prompt(index_content, recent_log, schema)
        chain = Chain(
            self.prompts.system_prompt(),
            (lint_prompt_obj, "lint_output"),
        )
        
        print("\n[LLM Health Analysis]")
        full_analysis = ""
        for message in chain.run(self.chat_model, {
            "index_content": index_content,
            "recent_log": recent_log,
            "schema": schema
        }):
            if message["role"] == "assistant":
                content = message["content"]
                print(content, end="", flush=True)
                full_analysis += content
        print("\n")
        
        self._update_log(vault_path, "lint", "Performed health check.")

    def ingest(self, new_source: str | Path):
        source_path = None
        if isinstance(new_source, Path):
            source_path = new_source
        elif isinstance(new_source, str):
            try:
                candidate = Path(new_source)
                if candidate.exists():
                    source_path = candidate
            except OSError:
                pass

        if source_path is not None:
            print(f"--- Ingesting new source file: {source_path}")
            new_text = self._read_source_text(source_path)
        else:
            print("--- Ingesting new content as wiki page")
            new_text = str(new_source)

        chunks = self.get_retrieved_documents(new_text) if self.retriever.is_ready else []
        context = "\n\n".join(chunk.text for chunk in chunks)

        relevant_pages_content = []
        seen_pages = set()
        for chunk in chunks:
            if chunk.source is None or chunk.source in seen_pages:
                continue
            content = chunk.source.read_text(encoding="utf-8")
            page_name = self._relative_page_name(chunk.source)
            relevant_pages_content.append(f"--- PAGE: {page_name} ---\n{content}")
            seen_pages.add(chunk.source)
        relevant_pages_str = (
            "\n\n".join(relevant_pages_content)
            if relevant_pages_content
            else "No relevant existing pages found."
        )

        existing_page_names = [
            self._relative_page_name(page) for page in self.retriever.documents
        ]
        existing_pages_str = ", ".join(
            f"[[{name}]]" for name in sorted(existing_page_names)
        )

        schema = self._load_schema()
        ingest_prompt_obj = self.prompts.ingest_prompt(
            new_text,
            [chunk.text for chunk in chunks],
            existing_page_names,
            schema,
        )
        chain = Chain(
            self.prompts.system_prompt(),
            (ingest_prompt_obj, "ingest_output"),
        )

        print("LLM is generating wiki updates...\n")
        full_content = ""
        for message in chain.run(
            self.chat_model,
            {
                "context": context,
                "new_source_text": new_text,
                "relevant_pages_content": relevant_pages_str,
                "existing_pages": existing_pages_str,
                "schema": schema,
            },
        ):
            if message["role"] == "assistant":
                content = message["content"]
                print(content)
                full_content += content

        full_content = full_content.strip()
        if full_content.startswith("```"):
            full_content = re.sub(r"^```[a-zA-Z]*\n", "", full_content)
            full_content = re.sub(r"\n```$", "", full_content).strip()

        page_segments = [
            page.strip()
            for page in full_content.split("---PAGE_BREAK---")
            if page.strip()
        ]
        saved_pages = []
        for content in page_segments:
            title = self._page_title(content)
            if title is None:
                raise ValueError("every generated wiki page must have a frontmatter title or H1")
            saved_pages.append({"title": title, "content": content})
        if not saved_pages:
            raise ValueError("LLM did not generate any wiki pages")

        saved_files = self._write_pages_atomically(saved_pages)
        for saved_file in saved_files:
            print(f"Saved: {saved_file}")

        self._update_index(self.vault_path)
        source_name = source_path.name if source_path is not None else "new content"
        saved_titles = [item["title"] for item in saved_pages]
        titles_str = (
            ", ".join(saved_titles)
            if len(saved_titles) <= 5
            else f"{len(saved_titles)} pages"
        )
        self._update_log(
            self.vault_path,
            "ingest",
            f"Source: {source_name} | Saved: {titles_str}",
        )

        self.retriever.documents = wiki_documents(self.vault_path)
        self.prepare()
        print(f"→ {len(saved_files)} page(s) saved to {self.vault_path}.")
        print("→ Index refreshed.")


    def answer(self, query: str) -> GeneratedAnswer:
        """Answers using index-selected pages plus vector retrieval."""
        self._update_log(self.vault_path, "query", f"Q: {query}")
        index_file = self.vault_path / "index.md"
        index_content = (
            index_file.read_text(encoding="utf-8")
            if index_file.exists()
            else "No index found."
        )

        schema = self._load_schema()
        select_pages_prompt_obj = self.prompts.select_pages_prompt(
            query,
            index_content,
            schema,
        )
        select_chain = Chain(
            self.prompts.system_prompt(),
            (select_pages_prompt_obj, "selected_pages"),
        )

        print("Consulting wiki index...")
        selected_pages_str = ""
        for message in select_chain.run(
            self.chat_model,
            {"query": query, "index_content": index_content, "schema": schema},
        ):
            if message["role"] == "assistant":
                selected_pages_str += message["content"]

        relevant_pages_content = []
        if selected_pages_str.strip().lower() != "none":
            page_titles = [
                title.strip().strip("[]").strip()
                for title in selected_pages_str.split(",")
            ]
            for title in page_titles:
                if not title:
                    continue
                relative_page = Path(title)
                if relative_page.is_absolute() or ".." in relative_page.parts:
                    continue
                page_path = self.vault_path / relative_page.with_name(
                    f"{relative_page.name}.md"
                )
                if page_path.exists():
                    content = page_path.read_text(encoding="utf-8")
                    relevant_pages_content.append(f"--- PAGE: {title} ---\n{content}")

        full_pages_content = (
            "\n\n".join(relevant_pages_content)
            if relevant_pages_content
            else "No additional full pages selected."
        )
        chunks = self.get_retrieved_documents(query) if self.retriever.is_ready else []
        context = "\n\n".join(chunk.text for chunk in chunks)

        answer_prompt_obj = self.prompts.wiki_answer_prompt(
            query,
            context,
            full_pages_content,
            schema,
        )
        answer_chain = Chain(
            self.prompts.system_prompt(),
            (answer_prompt_obj, "answer"),
        )
        messages = tuple(
            answer_chain.run(
                self.chat_model,
                {
                    "query": query,
                    "context": context,
                    "full_pages_content": full_pages_content,
                    "schema": schema,
                },
            )
        )
        return GeneratedAnswer(
            query=query,
            messages=messages,
            retrieved_chunks=tuple(chunks),
        )


if __name__ == "__main__":
    vault_path = Path("wiki")
    documents = wiki_documents(vault_path)

    print(f"Documents found in vault: {len(documents)} markdown files")

    prompts = WikiPromptBuilder("You are a personal wiki maintainer.")
    chat_model, embedding_model = model_names("gemma-4-e4b-it")
    rag = WikiRetrievalAugmentedGenerator(
        vault_path=vault_path,
        documents=documents,
        queries=[],
        embedding_model=embedding_model,
        llm_model=chat_model,
        temperature=0.7,
        num_neighbors=5,
        chunker=ChunkerKind.UNSTRUCTURED,
    )
    rag.prompts = prompts
    rag.prepare()

    print("\nLLM Wiki mode ready!")
    print("Just type your question. A new wiki page will be created and ingested automatically.\n")

    try:
        while True:
            user_input = input("> ").strip()
            if not user_input or user_input.lower() == "exit":
                break

            if user_input.lower() == "/lint":
                rag.lint()
                continue

            if user_input.startswith("/ingest "):
                source = user_input[8:].strip()
                rag.ingest(source)
                continue

            query = user_input
            print(f"\n--- Query: {query}")
            
            answer = rag.answer(query)
            for message in answer.messages:
                if message["role"] == "assistant":
                    print(message["content"], end="", flush=True)
            print("\n")

            # Automatically ingest the answer as a new wiki page
            rag.ingest(f"Query: {query}\n\nAnswer: {answer.text}")
    except (KeyboardInterrupt, EOFError):
        print("\nExiting...")
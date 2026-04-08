# wiki_rag.py - LLM Wiki style Q&A + Ingest over Obsidian vault
from pathlib import Path
import os
import re
from collections import defaultdict

# Workaround for unstructured (you already had this)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from prompt_builder import PromptBuilder
from retriever import ChunkerKind
from retrieval_augmented_generator import RetrievalAugmentedGenerator
from chaincrafter import Prompt, Chain
from unstructured.partition.auto import partition as unstructured_partition


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
    @staticmethod
    def _read_source_text(source_path: Path) -> str:
        if source_path.suffix.lower() in {".md", ".txt"}:
            return source_path.read_text(encoding="utf-8")

        elements = unstructured_partition(str(source_path))
        return "\n\n".join(str(element) for element in elements)

    def _update_index(self, vault_path: Path):
        """Maintains a catalog of all pages in the wiki (index.md)."""
        index_file = vault_path / "index.md"
        # Get all md files, excluding index, log and schema itself
        pages = sorted([p for p in vault_path.glob("*.md") if p.name not in ("index.md", "log.md", "SCHEMA.md")])
        
        categories = defaultdict(list)
        yaml_pattern = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL | re.MULTILINE)
        
        for p in pages:
            content = p.read_text(encoding="utf-8")
            match = yaml_pattern.search(content)
            category = "General"
            summary = "No summary provided."
            
            if match:
                yaml_block = match.group(1)
                cat_match = re.search(r"^\s*category:\s*(.+)$", yaml_block, re.MULTILINE | re.IGNORECASE)
                sum_match = re.search(r"^\s*summary:\s*(.+)$", yaml_block, re.MULTILINE | re.IGNORECASE)
                
                if cat_match:
                    category = cat_match.group(1).strip()
                if sum_match:
                    summary = sum_match.group(1).strip()
            
            categories[category].append(f"- [[{p.stem}]]: {summary}")
        
        lines = ["# Wiki Index", "", "## Summary", "A categorized list of all wiki pages.", ""]
        
        for cat in sorted(categories.keys()):
            lines.append(f"### {cat}")
            lines.extend(sorted(categories[cat]))
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
        vault_path = self.retriever.documents[0].parent if self.retriever.documents else Path("wiki")
        schema_file = vault_path / "SCHEMA.md"
        if schema_file.exists():
            return schema_file.read_text(encoding="utf-8")
        return "No external schema found. Follow standard wiki conventions."

    def lint(self):
        """Performs a health check on the wiki."""
        vault_path = self.retriever.documents[0].parent if self.retriever.documents else Path("wiki")
        all_pages = [p for p in vault_path.glob("*.md") if p.name not in ("index.md", "log.md")]
        page_names = {p.stem for p in all_pages}
        
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
            print(f"• Broken links (target missing):")
            for source, target in broken_links:
                print(f"  - In [[{source}]]: [[{target}]]")
        
        if missing_cross_refs:
            grouped_missing = defaultdict(list)
            for source, target in missing_cross_refs:
                grouped_missing[source].append(target)
            
            print(f"• Potential missing cross-references:")
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
        if isinstance(new_source, Path) or (isinstance(new_source, str) and Path(new_source).exists()):
            print(f"--- Ingesting new source file: {new_source}")
            source_path = Path(new_source)
            new_text = self._read_source_text(source_path)
        else:
            print(f"--- Ingesting new content as wiki page")
            new_text = str(new_source)

        chunks = self.get_retrieved_documents(new_text)
        context = "\n\n".join(chunks)
        
        # Identify relevant existing pages from retrieved chunks
        relevant_pages_content = []
        seen_pages = set()
        for doc_path in self.retriever.documents:
            doc_content = doc_path.read_text(encoding="utf-8")
            for chunk in chunks:
                if chunk in doc_content and doc_path not in seen_pages:
                    relevant_pages_content.append(f"--- PAGE: {doc_path.stem} ---\n{doc_content}")
                    seen_pages.add(doc_path)
                    break
        
        relevant_pages_str = "\n\n".join(relevant_pages_content) if relevant_pages_content else "No relevant existing pages found."

        # Get list of existing page names (filenames without .md), excluding structural files
        existing_page_names = [p.stem for p in self.retriever.documents if p.name not in ("index.md", "log.md")]
        existing_pages_str = ", ".join(f"[[{name}]]" for name in sorted(existing_page_names))

        schema = self._load_schema()
        ingest_prompt_obj = self.prompts.ingest_prompt(new_text, chunks, existing_page_names, schema)
        chain = Chain(
            self.prompts.system_prompt(),
            (ingest_prompt_obj, "ingest_output"),
        )

        print("LLM is generating wiki updates...\n")
        full_content = ""
        for message in chain.run(self.chat_model, {
            "context": context, 
            "new_source_text": new_text,
            "relevant_pages_content": relevant_pages_str,
            "existing_pages": existing_pages_str,
            "schema": schema
        }):
            if message["role"] == "assistant":
                content = message["content"]
                print(content)
                full_content += content

        print("\nIngest complete, copying into Obsidian vault.")

        # Strip leading/trailing code blocks if the LLM ignored instructions
        full_content = full_content.strip()
        if full_content.startswith("```"):
            # Handle both ```markdown and ```
            full_content = re.sub(r"^```[a-zA-Z]*\n", "", full_content)
            full_content = re.sub(r"\n```$", "", full_content)
            full_content = full_content.strip()

        pages = [p.strip() for p in full_content.split("---PAGE_BREAK---") if p.strip()]
        
        vault_path = self.retriever.documents[0].parent if self.retriever.documents else Path("wiki")
        
        saved_pages = []
        for page in pages:
            title = None

            # Look for title in YAML frontmatter first
            yaml_match = re.search(r"^---\s*\n(.*?)\n---", page, re.DOTALL | re.MULTILINE)
            if yaml_match:
                yaml_content = yaml_match.group(1)
                title_match = re.search(r"^\s*title:\s*(.+)$", yaml_content, re.MULTILINE | re.IGNORECASE)
                if title_match:
                    title = title_match.group(1).strip()
            
            # If not in YAML, look for H1
            if not title:
                h1_match = re.search(r"^\s*#\s+(.+)$", page, re.MULTILINE)
                if h1_match:
                    title = h1_match.group(1).strip()

            if title is None:
                if saved_pages:
                    # Append to the last page if this segment has no title
                    saved_pages[-1]["content"] += "\n\n---\n\n" + page
                else:
                    # First segment has no title, use default
                    saved_pages.append({"title": "Untitled", "content": page})
            else:
                saved_pages.append({"title": title, "content": page})

        saved_count = 0
        for item in saved_pages:
            title = item["title"]
            content = item["content"]

            filename = "".join(c for c in title if c.isalnum() or c in (" ", "-", "_")).strip()
            if not filename:
                filename = "Untitled"

            target_file = vault_path / f"{filename}.md"
            
            # We now overwrite existing files to fulfill 'Page Updates and Revisions'
            # (No longer appending _1, _2 etc. to filename)

            target_file.write_text(content, encoding="utf-8")
            print(f"Saved: {target_file}")
            saved_count += 1

        self._update_index(vault_path)
        source_name = Path(new_source).name if isinstance(new_source, (str, Path)) and Path(new_source).exists() else "new content"
        saved_titles = [item["title"] for item in saved_pages]
        titles_str = ", ".join(saved_titles) if len(saved_titles) <= 5 else f"{len(saved_titles)} pages"
        self._update_log(vault_path, "ingest", f"Source: {source_name} | Saved: {titles_str}")

        self.retriever.documents = list(vault_path.rglob("*.md"))
        # Optional: exclude hidden folders
        self.retriever.documents = [p for p in self.retriever.documents if not any(part.startswith(".") for part in p.parts)]

        self.prepare()
        print(f"→ {saved_count} page(s) saved to {vault_path}.")
        print("→ Index refreshed.")


    def _answer(self, query):
        """Overrides base _answer with index-first navigation and query logging."""
        vault_path = self.retriever.documents[0].parent if self.retriever.documents else Path("wiki")
        self._update_log(vault_path, "query", f"Q: {query}")
        
        # 1. Consult index.md
        index_file = vault_path / "index.md"
        index_content = index_file.read_text(encoding="utf-8") if index_file.exists() else "No index found."
        
        # 2. Select relevant pages using LLM
        schema = self._load_schema()
        select_pages_prompt_obj = self.prompts.select_pages_prompt(query, index_content, schema)
        select_chain = Chain(
            self.prompts.system_prompt(),
            (select_pages_prompt_obj, "selected_pages")
        )
        
        print("Consulting wiki index...")
        selected_pages_str = ""
        for message in select_chain.run(self.chat_model, {
            "query": query, 
            "index_content": index_content,
            "schema": schema
        }):
            if message["role"] == "assistant":
                selected_pages_str += message["content"]
        
        selected_pages_str = selected_pages_str.strip()
        relevant_pages_content = []
        if selected_pages_str.lower() != "none":
            page_titles = [t.strip().strip("[]").strip() for t in selected_pages_str.split(",")]
            for title in page_titles:
                if not title: continue
                page_path = vault_path / f"{title}.md"
                if not page_path.exists():
                    clean_filename = "".join(c for c in title if c.isalnum() or c in (" ", "-", "_")).strip()
                    page_path = vault_path / f"{clean_filename}.md"
                
                if page_path.exists():
                    content = page_path.read_text(encoding="utf-8")
                    relevant_pages_content.append(f"--- PAGE: {title} ---\n{content}")
        
        full_pages_content_str = "\n\n".join(relevant_pages_content) if relevant_pages_content else "No additional full pages selected."
        
        # 3. Vector search
        chunks = self.get_retrieved_documents(query)
        context = "\n\n".join(chunks)
        
        # 4. Final answer using combined context
        answer_prompt_obj = self.prompts.wiki_answer_prompt(query, context, full_pages_content_str, schema)
        answer_chain = Chain(
            self.prompts.system_prompt(),
            (answer_prompt_obj, "answer")
        )
        
        return answer_chain.run(self.chat_model, {
            "query": query, 
            "context": context, 
            "full_pages_content": full_pages_content_str,
            "schema": schema
        })


if __name__ == "__main__":
    vault_path = Path("wiki")
    documents = list(vault_path.rglob("*.md"))
    documents = [p for p in documents if not any(part.startswith(".") for part in p.parts)]

    print(f"Documents found in vault: {len(documents)} markdown files")

    prompts = WikiPromptBuilder("You are a personal wiki maintainer.")
    rag = WikiRetrievalAugmentedGenerator(
        documents=documents,
        queries=[],
        embedding_model="text-embedding-nomic-embed-text-v1.5",
        llm_model="gemma-4-e4b-it",
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
            
            full_answer = ""
            for message in rag._answer(query):
                if message["role"] == "assistant":
                    content = message["content"]
                    print(content, end="", flush=True)
                    full_answer += content
            print("\n")

            # Automatically ingest the answer as a new wiki page
            rag.ingest(f"Query: {query}\n\nAnswer: {full_answer}")
    except (KeyboardInterrupt, EOFError):
        print("\nExiting...")
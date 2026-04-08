# LLM Wiki Schema & Conventions

## Core Wiki Principles
- The wiki is a persistent, compounding artifact.
- Use Obsidian-style [[Page Name]] wikilinks.
- All pages MUST have YAML frontmatter with a `title`, `summary`, `category`, and `date`.
- The `category` should be one of: `General`, `Project`, `Entity`, `Concept`, or `Source`.
- The `summary` should be a one-line description of the page content.
- The `date` should be in YYYY-MM-DD format.

## Ingestion Workflow
- Extract key takeaways, entities, concepts, and insights.
- Decide what new pages or updates are needed based on existing content.
- **CRITICAL**: If information belongs on an existing page, provide the updated/merged version of that page.
- Do NOT create duplicate pages for the same topic (e.g., don't create "Topic_1" if "Topic" exists).
- Resolve contradictions by favoring newer/specific sources while maintaining historical context.
- Output clean markdown ready to save as .md files.
- Each page must be separated by `---PAGE_BREAK---`.
- For existing pages being updated, use the EXACT same title to ensure overwriting.
- Do NOT wrap the markdown in code blocks.

## Health & Linting Focus
- Identify contradictions between pages.
- Spot stale claims that newer information has superseded.
- Find gaps where important concepts lack dedicated pages.
- Suggest better cross-references between related topics.

## Answering Queries
- Provide comprehensive answers using provided snippets and full page content.
- If the information is not in the context, say so.
- Use [[Page Name]] links to reference wiki pages.

from chaincrafter import Prompt


class PromptBuilder:
    def __init__(self, system_text):
        self.system_text = system_text

    def system_prompt(self):
        return Prompt(self.system_text)

    def user_prompt(self):
        return Prompt(
            "## Context\n{context}\n\n## Query\nAnswer this query: {query}"
        )

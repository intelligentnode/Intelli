from pathlib import Path


class SystemHelper:
    def __init__(self):
        # Setting up the path to the templates directory
        self.systems_path = Path(__file__).parent.parent / "resource" / "templates"

    def get_prompt_path(self, file_type):
        """Returns the file path for the specified prompt type."""
        file_map = {
            "sentiment": "sentiment_prompt.in",
            "summary": "summary_prompt.in",
            "html_page": "html_page_prompt.in",
            "graph_dashboard": "graph_dashboard_prompt.in",
            "instruct_update": "instruct_update.in",
            "prompt_example": "prompt_example.in",
            "augmented_chatbot": "augmented_chatbot.in"
        }

        if file_type in file_map:
            return self.systems_path / file_map[file_type]
        else:
            raise ValueError(f"File type '{file_type}' not supported.")

    def load_prompt(self, file_type):
        """Loads the prompt template from a file."""
        prompt_path = self.get_prompt_path(file_type)
        with open(prompt_path, 'r', encoding='utf-8') as file:
            prompt_template = file.read()

        return prompt_template

    def load_static_prompt(self, file_type):
        static_prompts = {
            "augmented_chatbot": (
                "Using the provided context, craft a cohesive response that addresses the user's query. "
                "If the context lacks relevance, focus on generating accurate answer "
                "based on the user's question alone. Aim for clarity in your reply.\n"
                "Context:\n"
                "${semantic_search}\n"
                "------------------\n"
                "User's Question:\n"
                "${user_query}"
            ),
        }

        if file_type in static_prompts:
            return static_prompts[file_type]
        else:
            raise ValueError(f"Static prompt for file type '{file_type}' not defined.")

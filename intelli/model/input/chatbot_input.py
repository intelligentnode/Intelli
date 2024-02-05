class ChatMessage:
    def __init__(self, content, role, name=None):
        self.content = content
        self.role = role
        self.name = name

class ChatModelInput:
    def __init__(self, system, model, temperature=1, 
                 max_tokens=None, search_k=3, attach_reference=False, 
                 **options):
        self.system = system
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.search_k = search_k
        self.attach_reference = attach_reference
        self.options = options
        self.messages = []

    def add_user_message(self, prompt):
        self.messages.append(ChatMessage(prompt, 'user'))

    def add_assistant_message(self, prompt):
        self.messages.append(ChatMessage(prompt, 'assistant'))

    def add_system_message(self, prompt):
        self.messages.append(ChatMessage(prompt, 'system'))

    def clean_messages(self):
        self.messages = []

    def delete_last_message(self, delete_message):
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i].content == delete_message.content and self.messages[i].role == delete_message.role:
                del self.messages[i]
                return True
        return False

    def get_openai_input(self):
        messages = [{'role': msg.role, 'content': msg.content} for msg in self.messages]
        params = {
            'model': self.model,
            'messages': messages,
            **({'temperature': self.temperature} if self.temperature is not None else {}),
            **({'max_tokens': self.max_tokens} if self.max_tokens is not None else {}),
            **self.options
        }
        return params

    def get_mistral_input(self):
        messages = [{'role': msg.role, 'content': msg.content} for msg in self.messages]
        params = {
            'model': self.model,
            'messages': messages,
            **self.options
        }
        return params

    def get_gemini_input(self):
        """Specific adjustment for Gemini where 'assistant' role is mapped to 'model'"""
        contents = []
        for msg in self.messages:
            # Adjusting role specifically for Gemini
            role = 'model' if msg.role == 'assistant' else msg.role
            contents.append({'role': role, 'parts': [{'text': msg.content}]})
        params = {
            'contents': contents,
            'generationConfig': {
                **({'temperature': self.temperature} if self.temperature is not None else {}),
                **({'maxOutputTokens': self.max_tokens} if self.max_tokens is not None else {})
            },
            **self.options
        }
        return params
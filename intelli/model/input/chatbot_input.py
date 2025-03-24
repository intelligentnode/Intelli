class ChatMessage:
    def __init__(self, content, role, name=None):
        self.content = content
        self.role = role
        self.name = name


class ChatModelInput:
    def __init__(self, system, model=None, temperature=1,
                 max_tokens=None, numberOfOutputs=1, attach_reference=False,
                 filter_options={}, **options):
        self.system = system
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.numberOfOutputs = numberOfOutputs
        self.options = options
        self.messages = []
        self.add_system_message(self.system)
        # augemented search parameters
        self.attach_reference = attach_reference
        self.doc_name = filter_options.get('doc_name', None)
        self.search_k = filter_options.get('search_k', 3) 

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
        system = ""
        for msg in self.messages:
            # Adjusting role specifically for Gemini
            if msg.role == 'system' and msg.content:
                system = msg.content+":"
            else:
                role = 'model' if msg.role == 'assistant' else msg.role
                contents.append({'role': role, 'parts': [{'text': system+msg.content}]})
        params = {
            'contents': contents,
            'generationConfig': {
                **({'temperature': self.temperature} if self.temperature is not None else {}),
                **({'maxOutputTokens': self.max_tokens} if self.max_tokens is not None else {})
            },
            **self.options
        }
        return params

    def get_anthropic_input(self):
        # prepare the messages
        system = ""
        contents = []
        for msg in self.messages:
            if msg.role == 'system':
                system += msg.content + " "
            else:
                contents.append({'role': msg.role, 'content': msg.content})

        # construct params dictionary
        params = {
            'model': self.model,
            'system': system.strip(),  # Use a system prompt
            'messages': contents,
            'max_tokens': self.max_tokens or 2048,
            **({'temperature': self.temperature} if self.temperature is not None else {}),
            **self.options,
        }

        return params

    def get_keras_input(self):
        instructions = ""
        if any(msg.role == 'system' for msg in self.messages):
            instructions += "instructions: " + " ".join([msg.content for msg in self.messages if msg.role == 'system']) + "\n"

        chat_history = []
        for msg in self.messages:
            if msg.role == 'user':
                chat_history.append(f"user: {msg.content}")
            elif msg.role == 'assistant':
                chat_history.append(f"assistant: {msg.content}")

        # at least one user message
        if not any(msg.role == 'user' for msg in self.messages):
            raise "Send at least one user message."

        # end with 'assistant: '
        if not chat_history or not chat_history[-1].startswith("assistant:"):
            chat_history.append("assistant: ")

        prompt = instructions + "\n".join(chat_history)
        params = {
            'prompt': prompt,
            'max_length': self.max_tokens or 180,
            **self.options
        }
        return params

    def get_nvidia_input(self):
        messages = [{'role': msg.role, 'content': msg.content} for msg in self.messages]
        params = {
            'model': self.model,
            'messages': messages,
            **({'temperature': self.temperature} if self.temperature is not None else {}),
            **({'max_tokens': self.max_tokens} if self.max_tokens is not None else {}),
            **self.options
        }
        return params

    def get_llamacpp_input(self):
        """
        Create an input prompt for llama.cpp.

        This method concatenates the conversation messages into a plain text prompt.
        It prefixes system, user, and assistant messages, and ends with an 'Assistant:' prompt.

        Returns:
            A dictionary with keys:
                - prompt: the final text prompt,
                - max_tokens: maximum tokens to generate,
                - temperature: sampling temperature,
                plus any additional options.
        """
        prompt = ""
        for msg in self.messages:
            if msg.role == 'system':
                prompt += f"System: {msg.content}\n"
            elif msg.role == 'user':
                prompt += f"User: {msg.content}\n"
            elif msg.role == 'assistant':
                prompt += f"Assistant: {msg.content}\n"
        if not prompt.endswith("Assistant: "):
            prompt += "Assistant: "
        params = {
            "prompt": prompt,
            "max_tokens": self.max_tokens or 180,
            "temperature": self.temperature,
            **self.options
        }
        return params

    def get_vllm_input(self):
        """
        Format the input for VLLM API.

        Returns:
            dict: Parameters for VLLM API request.
        """
        if len(self.messages) > 1:
            # Check if there are multiple message roles or system messages
            has_system_or_multiple_roles = any(msg.role == "system" for msg in self.messages) or \
                                           len(set(msg.role for msg in self.messages)) > 1

            if has_system_or_multiple_roles:
                # Chat completion format
                messages = [{"role": msg.role, "content": msg.content} for msg in self.messages]
                params = {
                    "model": self.model,
                    "messages": messages,
                }
            else:
                # Text completion format (using the last user message as prompt)
                user_messages = [msg for msg in self.messages if msg.role == "user"]
                if not user_messages:
                    raise ValueError("No user messages found for text completion")

                prompt = user_messages[-1].content
                params = {
                    "model": self.model,
                    "prompt": prompt,
                }
        else:
            # Single message - likely a user prompt for text completion
            prompt = self.messages[0].content if self.messages else ""
            params = {
                "model": self.model,
                "prompt": prompt,
            }

        # Add optional parameters
        if self.temperature is not None:
            params["temperature"] = self.temperature

        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens

        if self.numberOfOutputs is not None and self.numberOfOutputs > 1:
            params["n"] = self.numberOfOutputs

        # Add any additional options
        params.update(self.options)

        return params

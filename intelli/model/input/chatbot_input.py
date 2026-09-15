from intelli.config import config
from intelli.utils.model_helper import (
    is_reasoning_model,
    claude_rejects_sampling_params,
    strip_route_override,
)


class ChatMessage:
    def __init__(self, content, role, name=None):
        self.content = content
        self.role = role
        self.name = name


class ChatModelInput:
    def __init__(self, system, model=None, temperature=1,
                 max_tokens=None, numberOfOutputs=1, attach_reference=False,
                 filter_options={}, tools=None, functions=None, function_call=None,
                 reasoning_effort=None, verbosity=None,
                 tool_choice=None,
                 **options):
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
        # tool/function parameters
        self.tools = tools
        self.functions = functions
        self.function_call = function_call
        self.tool_choice = tool_choice
        # GPT-5+ parameters
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity

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

    def is_reasoning_model(self):
        """Check if the current model is a reasoning model (GPT-5+)"""
        return is_reasoning_model(self.model)

    def get_openai_gpt5_input(self):
        """
        Format input specifically for GPT-5+ API (uses /v1/responses endpoint).
        
        GPT-5+ uses 'input' instead of 'messages' and has different parameters.
        """
        # Combine all messages into a single input string
        input_text = ""
        for msg in self.messages:
            if msg.role == 'system':
                input_text += f"{msg.content}\n\n"
            elif msg.role == 'user':
                input_text += f"{msg.content}\n"
            elif msg.role == 'assistant':
                input_text += f"{msg.content}\n"
        
        params = {
            'model': strip_route_override(self.model),
            'input': input_text.strip(),
        }
        
        # Add reasoning configuration if specified (default to low for GPT-5)
        reasoning_effort = self.reasoning_effort or 'low'
        params['reasoning'] = {'effort': reasoning_effort}
        
        # Add verbosity if specified (GPT-5 uses text.verbosity as string: 'low', 'medium', 'high')
        if self.verbosity:
            # Convert numeric verbosity to string if needed
            if isinstance(self.verbosity, int):
                # Map integers to verbosity levels
                verbosity_map = {1: 'low', 2: 'medium', 3: 'high'}
                verbosity_str = verbosity_map.get(self.verbosity, 'medium')
            else:
                verbosity_str = self.verbosity
            params['text'] = {'verbosity': verbosity_str}

        # Allow tools to be passed to GPT-5+/Responses API when provided.
        if self.tools:
            params['tools'] = self.tools

        # First-class tool_choice for the Responses API (additive).
        if self.tool_choice is not None:
            params['tool_choice'] = self.tool_choice

        # The Responses API uses 'max_output_tokens' (not 'max_tokens').
        # Map the caller's max_tokens through so it is no longer silently dropped.
        if self.max_tokens is not None:
            params['max_output_tokens'] = self.max_tokens

        # GPT-5+ doesn't accept temperature or max_tokens
        # Add any additional options that are compatible (options win on conflict).
        for key, value in self.options.items():
            if key not in ['temperature', 'max_tokens']:
                params[key] = value

        return params

    def get_openai_input(self):
        # Check if this is a reasoning model (GPT-5+) on the RAW model name, so the
        # ':chat'/'#chat'/'|chat' override still routes to chat-completions.
        if self.is_reasoning_model():
            return self.get_openai_gpt5_input()

        # Standard OpenAI format for other models. Strip any route-override suffix
        # so the API receives a clean model id (e.g. 'gpt-5.5:chat' -> 'gpt-5.5').
        messages = [{'role': msg.role, 'content': msg.content} for msg in self.messages]
        params = {
            'model': strip_route_override(self.model),
            'messages': messages,
            **({'temperature': self.temperature} if self.temperature is not None else {}),
            **({'max_tokens': self.max_tokens} if self.max_tokens is not None else {}),
            **self.options
        }

        # Add tools/functions if provided
        if self.tools:
            params['tools'] = self.tools
        if self.functions:
            params['functions'] = self.functions
        if self.function_call is not None:
            params['function_call'] = self.function_call
        # Forward tool_choice (supported by /v1/chat/completions) when set,
        # mirroring the GPT-5 and Anthropic builders.
        if self.tool_choice is not None and 'tool_choice' not in params:
            params['tool_choice'] = self.tool_choice

        return params

    def get_mistral_input(self):
        messages = [{'role': msg.role, 'content': msg.content} for msg in self.messages]
        params = {
            'model': self.model,
            'messages': messages,
            **({'temperature': self.temperature} if self.temperature is not None else {}),
            **({'max_tokens': self.max_tokens} if self.max_tokens is not None else {}),
            **self.options
        }
        # Forward tools for function calling when provided (options win on conflict).
        if self.tools and 'tools' not in params:
            params['tools'] = self.tools
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
            # Emit the per-call model as a sibling key so the function layer can
            # thread it to the wrapper as model_override (the Gemini wrapper takes
            # the model from the URL, not the request body). _chat_gemini pops it
            # before the body is sent. None means "use the configured default".
            'model': self.model,
            'contents': contents,
            'generationConfig': {
                **({'temperature': self.temperature} if self.temperature is not None else {}),
                **({'maxOutputTokens': self.max_tokens} if self.max_tokens is not None else {})
            },
            **self.options
        }
        # Allow tool/function style params for Gemini when provided.
        # (Gemini uses "tools" in generateContent requests for function calling.)
        if self.tools:
            params['tools'] = self.tools
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

        # Resolve a default Anthropic model (centralized in config) when none given,
        # so Anthropic chat works out of the box instead of sending model=None.
        model = self.model or config['url']['anthropic']['models']['chat']

        # construct params dictionary
        params = {
            'model': model,
            'system': system.strip(),  # Use a system prompt
            'messages': contents,
            'max_tokens': self.max_tokens or 2048,
            **self.options,
        }

        # Sampling-parameter policy: Opus 4.7+ and the Fable/Mythos family removed
        # temperature/top_p/top_k (sending them returns HTTP 400). Strip those keys
        # for such models; otherwise emit temperature exactly as before.
        if claude_rejects_sampling_params(model):
            for sampling_key in ('temperature', 'top_p', 'top_k'):
                params.pop(sampling_key, None)
        elif self.temperature is not None:
            params.setdefault('temperature', self.temperature)

        # Add tools if provided for Anthropic
        if self.tools:
            params['tools'] = self.tools
        # First-class tool_choice passthrough (additive).
        if self.tool_choice is not None:
            params.setdefault('tool_choice', self.tool_choice)

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

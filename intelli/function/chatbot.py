import json

from intelli.model.input.chatbot_input import ChatModelInput
from intelli.utils.system_helper import SystemHelper
from intelli.utils.model_helper import is_reasoning_model
from intelli.wrappers.geminiai_wrapper import GeminiAIWrapper
from intelli.wrappers.intellicloud_wrapper import IntellicloudWrapper
from intelli.wrappers.mistralai_wrapper import MistralAIWrapper
from intelli.wrappers.openai_wrapper import OpenAIWrapper
from intelli.wrappers.anthropic_wrapper import AnthropicWrapper
from intelli.wrappers.keras_wrapper import KerasWrapper
from intelli.wrappers.nvidia_wrapper import NvidiaWrapper
from intelli.wrappers.llama_cpp_wrapper import IntelliLlamaCPPWrapper
from intelli.wrappers.vllm_wrapper import VLLMWrapper
from enum import Enum


class ChatProvider(Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    MISTRAL = "mistral"
    ANTHROPIC = "anthropic"
    KERAS = "keras"
    NVIDIA = "nvidia"
    LLAMACPP = "llamacpp"
    VLLM = "vllm"


class Chatbot:

    def __init__(self, api_key=None, provider=None, options=None):
        if options is None:
            options = {}

        self.api_key = api_key
        self.provider = self._get_provider(provider)
        self.options = options
        self.wrapper = self._initialize_provider()
        self.add_rag(options)
        self.system_helper = SystemHelper()

        if self.provider and self.provider == ChatProvider.NVIDIA.value and not api_key:
            print("Please obtain NVIDIA API Key from https://build.nvidia.com/")

    def add_rag(self, options):
        self.extended_search = (
            IntellicloudWrapper(options["one_key"], options.get("api_base", None))
            if "one_key" in options
            else None
        )

    def _get_provider(self, provider):

        if isinstance(provider, str):
            provider = provider.lower()
            if provider not in (p.value for p in ChatProvider):
                raise ValueError(f"Unsupported provider: {provider}")
            return provider
        elif isinstance(provider, ChatProvider):
            return provider.value
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _initialize_provider(self):
        if self.provider == ChatProvider.OPENAI.value:
            proxy_helper = self.options.get("proxy_helper", None)
            return OpenAIWrapper(self.api_key, proxy_helper=proxy_helper)
        elif self.provider == ChatProvider.MISTRAL.value:
            return MistralAIWrapper(self.api_key)
        elif self.provider == ChatProvider.GEMINI.value:
            return GeminiAIWrapper(self.api_key)
        elif self.provider == ChatProvider.ANTHROPIC.value:
            return AnthropicWrapper(self.api_key)
        elif self.provider == ChatProvider.KERAS.value:
            return KerasWrapper(
                self.options["model_name"], self.options.get("model_params", {})
            )
        elif self.provider == ChatProvider.NVIDIA.value:
            nvidia_options = self.options.get("nvidiaOptions", {})
            base_url = self.options.get("baseUrl", {})
            if "baseUrl" in nvidia_options and nvidia_options["baseUrl"]:
                return NvidiaWrapper(self.api_key, base_url=nvidia_options["baseUrl"])
            elif base_url:
                return NvidiaWrapper(self.api_key, base_url=base_url)
            else:
                return NvidiaWrapper(self.api_key)
        elif self.provider == ChatProvider.LLAMACPP.value:
            # assume options has "model_path" and optionally "model_params"
            model_path = self.options.get("model_path")
            model_params = self.options.get("model_params", {"n_ctx": 512})
            return IntelliLlamaCPPWrapper(
                model_path=model_path, model_params=model_params
            )
        elif self.provider == ChatProvider.VLLM.value:
            vllm_base_url = self.options.get("vllmBaseUrl") or self.options.get("baseUrl")
            if not vllm_base_url:
                raise ValueError("VLLM provider requires baseUrl in options")
            return VLLMWrapper(vllm_base_url, self.api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def chat(self, chat_input):
        if not isinstance(chat_input, ChatModelInput):
            raise TypeError("chat_input must be an instance of ChatModelInput")

        references = []
        if self.extended_search:
            references = self._augment_with_semantic_search(chat_input)

        # Set default model to gpt-5.2 for OpenAI provider if not specified
        if self.provider == ChatProvider.OPENAI.value and not chat_input.model:
            chat_input.model = 'gpt-5.2'

        get_input_method = f"get_{self.provider}_input"
        chat_method = getattr(self, f"_chat_{self.provider}", None)
        if not chat_method:
            raise NotImplementedError(
                f"{self.provider.capitalize()} chat is not implemented."
            )

        params = getattr(chat_input, get_input_method)()
        result = chat_method(params)
        return (
            {"result": result, "references": references}
            if chat_input.attach_reference
            else result
        )

    def _chat_vllm(self, params):
        """Process chat requests for VLLM."""
        if "messages" in params:
            # This is a chat completion request
            results = self.wrapper.generate_chat_text(params)
            return self._parse_vllm_chat_responses(results)
        else:
            # This is a regular completion request
            results = self.wrapper.generate_text(params)
            return self._parse_vllm_text_responses(results)

    def _parse_vllm_chat_responses(self, results):
        """Parse VLLM chat completion responses."""
        if "choices" in results and len(results["choices"]) > 0:
            return [choice["message"]["content"] for choice in results["choices"]]
        return [""]

    def _parse_vllm_text_responses(self, results):
        """Parse VLLM text completion responses."""
        if "choices" in results and len(results["choices"]) > 0:
            return [choice["text"] for choice in results["choices"]]
        return [""]

    def _chat_llamacpp(self, params):
        # assume the wrapper returns a dict with key "choices" containing a list of text responses.
        response = self.wrapper.generate_text(params)
        # extract the text.
        return [response["choices"][0]["text"]]

    def _chat_keras(self, params):
        response = self.wrapper.generate(params["prompt"], params["max_length"])
        return [response]

    def _chat_openai(self, params):
        # Check if this is a reasoning model (GPT-5+)
        model_name = params.get('model', '')
        is_gpt5_plus = is_reasoning_model(model_name)
        
        if is_gpt5_plus:
            # GPT-5+ uses different endpoint and response format
            results = self.wrapper.generate_gpt5_response(params)
            return self._parse_gpt5_responses(results)
        else:
            # Extract functions and function_call if present
            functions = params.pop('functions', None)
            function_call = params.pop('function_call', None)
            
            # Pass functions and function_call separately to wrapper
            results = self.wrapper.generate_chat_text(params, functions=functions, function_call=function_call)
            return self._parse_openai_responses(results)

    def _chat_mistral(self, params):
        response = self.wrapper.generate_text(params)
        return [choice["message"]["content"] for choice in response.get("choices", [])]

    def _chat_gemini(self, params):
        response = self.wrapper.generate_content(params)
        output = []
        for candidate in response.get("candidates", []):
            if "content" in candidate:
                output.append(candidate["content"]["parts"][0]["text"])
            else:
                raise Exception("Error when calling gemini: {}".format(response))
        return output

    def _chat_anthropic(self, params):
        response = self.wrapper.generate_text(params)
        
        # Check if this is a tool use response
        if 'stop_reason' in response and response['stop_reason'] == 'tool_use':
            # Return tool use information in a standardized format
            tool_uses = []
            for content in response.get('content', []):
                if content.get('type') == 'tool_use':
                    tool_uses.append({
                        'type': 'tool_response',
                        'tool_calls': [{
                            'id': content.get('id'),
                            'type': 'function',
                            'function': {
                                'name': content.get('name'),
                                'arguments': json.dumps(content.get('input', {}))
                            }
                        }]
                    })
            
            # If we have tool uses, return the first one (can be extended for multiple)
            if tool_uses:
                return [tool_uses[0]]
        
        # Normal text response
        return [message["text"] for message in response["content"]]

    def _chat_nvidia(self, params):
        result = self.wrapper.generate_text(params)
        choices = result.get("choices", [])
        if not choices:
            raise Exception("No choices returned from NVIDIA API")
        return [choices[0]["message"]["content"]]

    def stream(self, chat_input):
        """Streams responses from the selected provider for the given chat input."""

        streaming_method = getattr(self, f"_stream_{self.provider}", None)

        if not streaming_method:
            raise NotImplementedError(
                f"Streaming is not implemented for {self.provider}."
            )

        if self.extended_search:
            _ = self._augment_with_semantic_search(chat_input)

        # Set default model to gpt-5.2 for OpenAI provider if not specified
        if self.provider == ChatProvider.OPENAI.value and not chat_input.model:
            chat_input.model = 'gpt-5.2'

        params = getattr(chat_input, f"get_{self.provider}_input")()

        for content in streaming_method(params):
            yield content

    def _stream_llamacpp(self, params):
        params["stream"] = True
        # Stream text chunks from the llama-cpp wrapper.
        for chunk in self.wrapper.generate_text_stream(params):
            yield chunk

    def _stream_openai(self, params):
        """
        Private helper method to stream text from OpenAI and parse each content chunk.
        Note: GPT-5+ models may not support streaming in the same way as previous models.
        """
        # Check if this is a reasoning model (GPT-5+)
        model_name = params.get('model', '')
        is_gpt5_plus = is_reasoning_model(model_name)
        
        if is_gpt5_plus:
            # GPT-5+ streaming is not yet supported
            raise NotImplementedError(
                "GPT-5+ streaming is not yet supported. Please use the chat() method instead."
            )
        
        params["stream"] = True
        for response in self.wrapper.generate_chat_text(params):
            if (
                response.strip()
                and response.startswith("data: ")
                and response != "data: [DONE]"
            ):
                json_content = response[len("data: ") :].strip()

                try:
                    data_chunk = json.loads(json_content)
                    content = (
                        data_chunk.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content", "")
                    )
                    if content:
                        yield content
                except json.JSONDecodeError as e:
                    print("Error decoding JSON:", e)

    def _stream_anthropic(self, params):
        """Stream text from Anthropic and directly yield text content."""
        params["stream"] = True

        for line in self.wrapper.stream_text(params):
            # process lines starting with 'data:'
            if line.startswith("data:"):
                try:

                    json_payload = line[len("data:") :]
                    line_data = json.loads(json_payload)

                    if (
                        "type" in line_data
                        and line_data["type"] == "content_block_delta"
                        and "text" in line_data["delta"]
                    ):
                        yield line_data["delta"]["text"]
                except json.JSONDecodeError as e:
                    print("Error decoding JSON from stream:", e)

    def _stream_nvidia(self, params):
        params["stream"] = True
        stream = self.wrapper.generate_text_stream(params)
        for line in stream:
            if line.strip() and line.startswith("data: ") and line != "data: [DONE]":
                json_content = line[len("data: ") :].strip()
                try:
                    data_chunk = json.loads(json_content)
                    content = (
                        data_chunk.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content", "")
                    )
                    if content:
                        yield content
                except json.JSONDecodeError as e:
                    print("Error decoding JSON:", e)

    def _stream_vllm(self, params):
        """Stream responses from VLLM."""

        self.wrapper.is_log = self.options.get("debug", False)

        if "messages" in params:
            # This is a chat completion stream
            params["stream"] = True
            for chunk in self.wrapper.generate_chat_text_stream(params):
                if chunk:
                    yield chunk
        else:
            # This is a text completion stream
            params["stream"] = True
            for chunk in self.wrapper.generate_text_stream(params):
                if chunk:
                    yield chunk

    # helpers
    def _parse_gpt5_responses(self, results):
        """
        Parse GPT-5+ API responses.
        
        GPT-5+ response format: 
        {
            "output": [
                {"type": "reasoning", "reasoning": {...}},
                {"type": "message", "content": [...]}
            ]
        }
        """
        responses = []
        
        if 'output' in results and isinstance(results['output'], list):
            # Extract text from message content
            for item in results['output']:
                if item.get('type') == 'message' and 'content' in item:
                    content = item['content']
                    if isinstance(content, list):
                        # Content is an array of parts
                        text_parts = []
                        for part in content:
                            if isinstance(part, dict):
                                text_parts.append(part.get('text', str(part)))
                            else:
                                text_parts.append(str(part))
                        responses.append(''.join(text_parts))
                    elif isinstance(content, str):
                        responses.append(content)
                    else:
                        responses.append(str(content))
        
        # Fallback to choices format if available
        elif 'choices' in results and len(results['choices']) > 0:
            for choice in results['choices']:
                output = choice.get('output') or choice.get('text') or choice.get('message', {}).get('content')
                if output:
                    responses.append(output)
        
        return responses if responses else ['']

    def _parse_openai_responses(self, results):
        responses = []
        for choice in results.get("choices", []):
            # Check for tool/function calls
            message = choice.get("message", {})
            
            # Handle function calls (legacy format)
            if choice.get("finish_reason") == "function_call" and "function_call" in message:
                response = {
                    "type": "function_response",
                    "function_call": message["function_call"]
                }
                responses.append(response)
            # Handle tool calls (new format)
            elif message.get("tool_calls"):
                response = {
                    "type": "tool_response",
                    "tool_calls": message["tool_calls"]
                }
                responses.append(response)
            else:
                # Regular text response
                response = message.get("content", "")
                responses.append(response)
                
        return responses

    def _augment_with_semantic_search(self, chat_input):
        last_user_message = (
            chat_input.messages[-1].content if chat_input.messages else ""
        )
        references = []
        if last_user_message:
            # Perform the semantic search based on the last user message.
            filters = (
                {"document_name": chat_input.doc_name} if chat_input.doc_name else None
            )
            search_results = self.extended_search.semantic_search(
                last_user_message, chat_input.search_k, filters=filters
            )

            # Accumulate document names from the search results for references.
            references = {}
            for doc in search_results:
                doc_name = doc["document_name"]
                if doc_name not in references:
                    references[doc_name] = {"pages": []}
                # Assuming each 'doc' can contain multiple 'pages' or similar structures, adjust as necessary.
                references[doc_name]["pages"].extend(doc.get("pages", []))

            # Generate context data based on the semantic search results.
            context_data = "\n".join(
                data["text"]
                for doc in search_results
                for data in doc["data"]
                if "text" in data
            ).strip()

            # Load the static prompt template for an augmented chatbot response.
            augmented_message_template = self.system_helper.load_static_prompt(
                "augmented_chatbot"
            )
            augmented_message = augmented_message_template.replace(
                "${semantic_search}", context_data
            ).replace("${user_query}", last_user_message)

            # Replace the content of the last user message with the augmented message in the ChatModelInput.
            chat_input.messages[-1].content = augmented_message

        return references

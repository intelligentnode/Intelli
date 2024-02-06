import json
from intelli.wrappers.openai_wrapper import OpenAIWrapper
from intelli.wrappers.mistralai_wrapper import MistralAIWrapper
from intelli.wrappers.geminiai_wrapper import GeminiAIWrapper
from intelli.wrappers.intellicloud_wrapper import IntellicloudWrapper
from intelli.model.input.chatbot_input import ChatModelInput
from intelli.utils.system_helper import SystemHelper

class Chatbot:
   
    def __init__(self, api_key, provider, options=None):
        if options is None:
            options = {}
        self.api_key = api_key
        self.provider = provider.lower()
        self.options = options
        self.wrapper = self._initialize_provider()
        self.extended_search = IntellicloudWrapper(options['one_key'], options.get('api_base',None)) if 'one_key' in options else None
        self.system_helper = SystemHelper()
    
    def _initialize_provider(self):
        if self.provider == 'openai':
            proxy_helper = self.options.get('proxy_helper', None)
            return OpenAIWrapper(self.api_key, proxy_helper=proxy_helper)
        elif self.provider == 'mistral':
            return MistralAIWrapper(self.api_key)
        elif self.provider == 'gemini':
            return GeminiAIWrapper(self.api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def chat(self, chat_input):
        if not isinstance(chat_input, ChatModelInput):
            raise TypeError("chat_input must be an instance of ChatModelInput")
        
        references = []
        if self.extended_search:
            references = self._augment_with_semantic_search(chat_input)
        
        get_input_method = f"get_{self.provider}_input"
        chat_method = getattr(self, f"_chat_{self.provider}", None)
        if not chat_method:
            raise NotImplementedError(f"{self.provider.capitalize()} chat is not implemented.")
        
        params = getattr(chat_input, get_input_method)()
        result = chat_method(params)
        return {'result': result, 'references': references} if chat_input.attach_reference else result

    def _chat_openai(self, params):
        results = self.wrapper.generate_chat_text(params)
        return self._parse_openai_responses(results)
    
    def _chat_mistral(self, params):
        response = self.wrapper.generate_text(params)
        return [choice['message']['content'] for choice in response.get('choices', [])]
    
    def _chat_gemini(self, params):
        response = self.wrapper.generate_content(params)
        return [candidate["content"]["parts"][0]["text"] for candidate in response["candidates"]]

    def stream(self, chat_input):
        """
        Streams responses from OpenAI for the given chat input.
        
        Each yielded content is the text content alone, extracted from the streamed response.
        """
        if self.provider != 'openai':
            raise NotImplementedError("Streaming is only supported for OpenAI.")
        
        if self.extended_search:
            _ = self._augment_with_semantic_search(chat_input)
        
        params = chat_input.get_openai_input()
        
        for content in self._stream_openai(params):
            yield content

    def _stream_openai(self, params):
        """
        Private helper method to stream text from OpenAI and parse each content chunk.
        """
        params['stream']=True
        for response in self.wrapper.generate_chat_text(params):
            if response.strip() and response.startswith('data: ') and response != 'data: [DONE]':
                json_content = response[len('data: '):].strip()
                
                try:
                    data_chunk = json.loads(json_content)
                    content = data_chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if content:
                        yield content
                except json.JSONDecodeError as e:
                    print("Error decoding JSON:", e)
    
    # helpers
    def _parse_openai_responses(self, results):
        responses = []
        for choice in results.get('choices', []):
            response = choice.get('message', {}).get('content', '')
            if choice.get('finish_reason') == 'function_call' and 'function_call' in choice.get('message', {}):
                response['function_call'] = choice['message']['function_call']
            responses.append(response)
        return responses
    
    def _augment_with_semantic_search(self, chat_input):
        last_user_message = chat_input.messages[-1].content if chat_input.messages else ""
        references = []
        if last_user_message:
            # Perform the semantic search based on the last user message.
            search_results = self.extended_search.semantic_search(last_user_message, chat_input.search_k)
            
            # Accumulate document names from the search results for references.
            references = {}
            for doc in search_results:
                doc_name = doc['document_name']
                if doc_name not in references:
                    references[doc_name] = {'pages': []}
                # Assuming each 'doc' can contain multiple 'pages' or similar structures, adjust as necessary.
                references[doc_name]['pages'].extend(doc.get('pages', []))

            # Generate context data based on the semantic search results.
            context_data = '\n'.join(
                doc['data']['text'] for doc in search_results if 'data' in doc and 'text' in doc['data']
            ).strip()

            # Load the static prompt template for an augmented chatbot response.
            augmented_message_template = self.system_helper.load_prompt("augmented_chatbot")
            augmented_message = augmented_message_template.replace("${semantic_search}", context_data).replace("${user_query}", last_user_message)
            
            # Replace the content of the last user message with the augmented message in the ChatModelInput.
            chat_input.messages[-1].content = augmented_message

        return references

    
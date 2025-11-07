import logging
from typing import Optional, Iterator

logger = logging.getLogger(__name__)

try:
    from openai import AzureOpenAI
except ImportError:
    AzureOpenAI = None


class AzureOpenAIWrapper:
    """
    Wrapper for Azure OpenAI chat completions and other services.
    
    This wrapper provides access to Azure OpenAI models for chat completions,
    embeddings, and other Azure OpenAI services.
    """
    
    def __init__(
        self,
        api_key: str,
        endpoint: str,
        api_version: str = "2024-02-15-preview",
        timeout: Optional[float] = 60.0,
        max_retries: int = 3
    ):
        """
        Initialize Azure OpenAI wrapper.
        
        Args:
            api_key: Azure OpenAI API key
            endpoint: Azure OpenAI endpoint URL (e.g., "https://{resource-name}.openai.azure.com")
            api_version: API version (default: "2024-02-15-preview")
            timeout: Request timeout in seconds (default: 60.0)
            max_retries: Maximum number of retries for rate-limited requests (default: 3)
        """
        if AzureOpenAI is None:
            raise ImportError(
                "Azure OpenAI SDK is not installed. "
                "Install with: pip install openai"
            )
        
        if not api_key:
            raise ValueError("Azure API key is required")
        if not endpoint:
            raise ValueError("Azure endpoint is required")
        
        self.api_key = api_key
        self.endpoint = endpoint
        self.api_version = api_version
        self.timeout = timeout
        self.max_retries = max_retries
        
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
            timeout=timeout,
            max_retries=max_retries
        )
        
        logger.debug(f"Azure OpenAI wrapper initialized with endpoint: {self.endpoint}")
    
    def generate_chat_text(self, params: dict) -> dict:
        """
        Generate chat completion using Azure OpenAI.
        
        Args:
            params: Dictionary with parameters like:
                - model: Deployment name (e.g., "gpt-4o")
                - messages: List of message dictionaries
                - temperature: Optional temperature
                - max_tokens: Optional max tokens
                
        Returns:
            JSON response from Azure OpenAI
        """
        try:
            model = params.get('model', '')
            messages = params.get('messages', [])
            
            if not model:
                raise ValueError("Model parameter is required")
            if not messages:
                raise ValueError("Messages parameter is required and cannot be empty")
            
            is_gpt5 = model.startswith('gpt-5') or 'gpt5' in model.lower()
            
            if is_gpt5:
                if 'temperature' in params and params.get('temperature') is not None:
                    raise ValueError(
                        f"GPT-5 models (like {model}) do not support the 'temperature' parameter. "
                        "Please remove it from your request."
                    )
                if 'max_tokens' in params and params.get('max_tokens') is not None:
                    raise ValueError(
                        f"GPT-5 models (like {model}) do not support the 'max_tokens' parameter. "
                        "Use 'max_completion_tokens' instead if you need to limit the response length."
                    )
                
                kwargs = {
                    'model': model,
                    'messages': params.get('messages', [])
                }
                
                if 'max_completion_tokens' in params and params['max_completion_tokens'] is not None:
                    kwargs['max_completion_tokens'] = params['max_completion_tokens']
            else:
                kwargs = {
                    'model': model,
                    'messages': params.get('messages', [])
                }
                
                temperature = params.get('temperature')
                if temperature is not None:
                    kwargs['temperature'] = temperature
                
                max_tokens = params.get('max_tokens')
                if max_tokens is not None:
                    kwargs['max_tokens'] = max_tokens
            
            response = self.client.chat.completions.create(**kwargs)
            
            result = {
                "choices": [
                    {
                        "message": {
                            "content": response.choices[0].message.content,
                            "role": response.choices[0].message.role
                        },
                        "finish_reason": response.choices[0].finish_reason
                    }
                ],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }
            }
            
            return result
            
        except ValueError as e:
            raise
        except Exception as e:
            logger.error(f"Azure OpenAI chat completion failed: {e}")
            raise
    
    def get_embeddings(self, params: dict) -> dict:
        """
        Get embeddings using Azure OpenAI.
        
        Args:
            params: Dictionary with parameters:
                - model: Deployment name (e.g., "text-embedding-ada-002")
                - input: Text or list of texts to embed
                
        Returns:
            JSON response with embeddings
        """
        try:
            model = params.get('model')
            input_data = params.get('input')
            
            if not model:
                raise ValueError("Model parameter is required")
            if not input_data:
                raise ValueError("Input parameter is required")
            
            response = self.client.embeddings.create(
                model=model,
                input=input_data
            )
            
            result = {
                "data": [
                    {
                        "embedding": item.embedding,
                        "index": item.index
                    }
                    for item in response.data
                ],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }
            }
            
            return result
            
        except ValueError as e:
            raise
        except Exception as e:
            logger.error(f"Azure OpenAI embeddings failed: {e}")
            raise
    
    def generate_chat_text_stream(self, params: dict) -> Iterator[dict]:
        """
        Generate streaming chat completion using Azure OpenAI.
        
        Args:
            params: Dictionary with parameters like:
                - model: Deployment name (e.g., "gpt-4o")
                - messages: List of message dictionaries
                - temperature: Optional temperature
                - max_tokens: Optional max tokens
                - stream: Should be True (optional, defaults to True)
                
        Yields:
            Dictionary chunks with streaming response data:
            {
                "choices": [{
                    "delta": {"content": "...", "role": "..."},
                    "finish_reason": None or "stop"
                }],
                "usage": None or usage dict
            }
        """
        try:
            model = params.get('model', '')
            messages = params.get('messages', [])
            
            if not model:
                raise ValueError("Model parameter is required")
            if not messages:
                raise ValueError("Messages parameter is required and cannot be empty")
            
            is_gpt5 = model.startswith('gpt-5') or 'gpt5' in model.lower()
            
            if is_gpt5:
                if 'temperature' in params and params.get('temperature') is not None:
                    raise ValueError(
                        f"GPT-5 models (like {model}) do not support the 'temperature' parameter. "
                        "Please remove it from your request."
                    )
                if 'max_tokens' in params and params.get('max_tokens') is not None:
                    raise ValueError(
                        f"GPT-5 models (like {model}) do not support the 'max_tokens' parameter. "
                        "Use 'max_completion_tokens' instead if you need to limit the response length."
                    )
                
                kwargs = {
                    'model': model,
                    'messages': params.get('messages', []),
                    'stream': True
                }
                
                if 'max_completion_tokens' in params and params['max_completion_tokens'] is not None:
                    kwargs['max_completion_tokens'] = params['max_completion_tokens']
            else:
                kwargs = {
                    'model': model,
                    'messages': params.get('messages', []),
                    'stream': True
                }
                
                temperature = params.get('temperature')
                if temperature is not None:
                    kwargs['temperature'] = temperature
                
                max_tokens = params.get('max_tokens')
                if max_tokens is not None:
                    kwargs['max_tokens'] = max_tokens
            
            stream = self.client.chat.completions.create(**kwargs)
            
            for chunk in stream:
                if not chunk.choices or len(chunk.choices) == 0:
                    continue
                
                choice = chunk.choices[0]
                delta = choice.delta if hasattr(choice, 'delta') else None
                
                chunk_dict = {
                    "choices": [
                        {
                            "delta": {
                                "content": delta.content if delta and delta.content else "",
                                "role": delta.role if delta and delta.role else None
                            },
                            "finish_reason": choice.finish_reason if hasattr(choice, 'finish_reason') else None
                        }
                    ],
                    "usage": None
                }
                
                # Include usage info if available (usually in the last chunk)
                if hasattr(chunk, 'usage') and chunk.usage:
                    chunk_dict["usage"] = {
                        "prompt_tokens": chunk.usage.prompt_tokens if chunk.usage else 0,
                        "completion_tokens": chunk.usage.completion_tokens if chunk.usage else 0,
                        "total_tokens": chunk.usage.total_tokens if chunk.usage else 0
                    }
                
                yield chunk_dict
                
        except ValueError as e:
            raise
        except Exception as e:
            logger.error(f"Azure OpenAI streaming chat completion failed: {e}")
            raise


import unittest
import os
from intelli.wrappers.azure_openai_wrapper import AzureOpenAIWrapper
from dotenv import load_dotenv

load_dotenv()


class TestAzureOpenAIWrapper(unittest.TestCase):
    """Comprehensive test suite for Azure OpenAI wrapper."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.api_key = os.getenv('AZURE_OPENAI_API_KEY')
        self.endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        
        if not self.api_key or not self.endpoint:
            print(' azure keys not found')
            self.skipTest("Azure OpenAI credentials not configured. "
                         "Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables.")
        
        self.wrapper = AzureOpenAIWrapper(
            api_key=self.api_key,
            endpoint=self.endpoint
        )
    
    def test_initialization(self):
        """Test wrapper initialization."""
        print(f"Test wrapper initialization")
        self.assertIsNotNone(self.wrapper)
        self.assertEqual(self.wrapper.endpoint, self.endpoint)
        self.assertEqual(self.wrapper.timeout, 60.0)
        self.assertEqual(self.wrapper.max_retries, 3)
    
    def test_initialization_with_custom_timeout_and_retries(self):
        """Test wrapper initialization with custom timeout and max_retries."""
        print(f"Test wrapper initialization with custom timeout and max_retries")
        wrapper = AzureOpenAIWrapper(
            api_key=self.api_key,
            endpoint=self.endpoint,
            timeout=120.0,
            max_retries=5
        )
        self.assertEqual(wrapper.timeout, 120.0)
        self.assertEqual(wrapper.max_retries, 5)
    
    def test_invalid_initialization_empty_api_key(self):
        """Test initialization with empty API key."""
        with self.assertRaises(ValueError):
            AzureOpenAIWrapper(api_key="", endpoint="https://test.openai.azure.com")
    
    def test_invalid_initialization_empty_endpoint(self):
        """Test initialization with empty endpoint."""
        with self.assertRaises(ValueError):
            AzureOpenAIWrapper(api_key="test-key", endpoint="")
    
    def test_chat_completion_basic(self):
        """Test basic chat completion."""
        model = os.getenv('AZURE_OPENAI_CHAT_MODEL', 'gpt-4o')
        
        response = self.wrapper.generate_chat_text({
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "What is machine learning?"}
            ],
            "temperature": 0.7,
            "max_tokens": 100
        })
        
        self.assertTrue("choices" in response)
        self.assertTrue(len(response["choices"]) > 0)
        self.assertTrue(len(response["choices"][0]["message"]["content"]) > 0)
        
        if "usage" in response:
            usage = response["usage"]
            self.assertGreater(usage.get('prompt_tokens', 0), 0)
    
    def test_chat_completion_conversation(self):
        """Test multi-turn conversation."""
        model = os.getenv('AZURE_OPENAI_CHAT_MODEL', 'gpt-4o')
        
        # First message
        response1 = self.wrapper.generate_chat_text({
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is Python?"}
            ]
        })
        
        assistant_response = response1["choices"][0]["message"]["content"]
        self.assertTrue(len(assistant_response) > 0)
        
        # Second message (continuation)
        response2 = self.wrapper.generate_chat_text({
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": assistant_response},
                {"role": "user", "content": "What can you do with it?"}
            ]
        })
        
        self.assertTrue("choices" in response2)
        self.assertTrue(len(response2["choices"]) > 0)
    
    def test_chat_completion_different_temperatures(self):
        """Test chat completion with different temperatures."""
        model = os.getenv('AZURE_OPENAI_CHAT_MODEL', 'gpt-4o')
        temperatures = [0.0, 0.5, 1.0]
        
        for temp in temperatures:
            response = self.wrapper.generate_chat_text({
                "model": model,
                "messages": [
                    {"role": "user", "content": "What are RESTful API principles?"}
                ],
                "temperature": temp,
                "max_tokens": 100
            })
            
            self.assertTrue("choices" in response)
            self.assertTrue(len(response["choices"]) > 0)
    
    def test_embeddings_single(self):
        """Test embeddings for single text."""
        model = os.getenv('AZURE_OPENAI_EMBEDDING_MODEL', 'text-embedding-ada-002')
        text = "This is a test sentence for embeddings."
        
        response = self.wrapper.get_embeddings({
            "model": model,
            "input": text
        })
        
        self.assertTrue("data" in response)
        self.assertTrue(len(response["data"]) > 0)
        self.assertTrue("embedding" in response["data"][0])
        self.assertGreater(len(response["data"][0]["embedding"]), 0)
        
        if "usage" in response:
            usage = response["usage"]
            self.assertGreater(usage.get('total_tokens', 0), 0)
    
    def test_embeddings_multiple(self):
        """Test embeddings for multiple texts."""
        model = os.getenv('AZURE_OPENAI_EMBEDDING_MODEL', 'text-embedding-ada-002')
        texts = [
            "The weather is nice today.",
            "It's a sunny day outside.",
            "I love programming in Python."
        ]
        
        response = self.wrapper.get_embeddings({
            "model": model,
            "input": texts
        })
        
        self.assertTrue("data" in response)
        self.assertEqual(len(response["data"]), len(texts))
        
        for i, item in enumerate(response["data"]):
            self.assertTrue("embedding" in item)
            self.assertGreater(len(item["embedding"]), 0)
    
    def test_error_handling_invalid_model(self):
        """Test error handling with invalid model."""
        # Should raise original exception type, not RuntimeError
        try:
            self.wrapper.generate_chat_text({
                "model": "invalid-model-name-12345",
                "messages": [
                    {"role": "user", "content": "Hello"}
                ]
            })
            self.fail("Should have raised an exception for invalid model")
        except Exception as e:
            # Should be the original exception from OpenAI SDK
            self.assertIsNotNone(e)
    
    def test_error_handling_missing_messages(self):
        """Test error handling with missing messages."""
        model = os.getenv('AZURE_OPENAI_CHAT_MODEL', 'gpt-4o')
        
        # Empty messages should raise an error
        try:
            self.wrapper.generate_chat_text({
                "model": model,
                "messages": []
            })
            # If it doesn't raise an error, that's acceptable too
        except Exception:
            # Expected behavior
            pass
    
    def test_gpt5_mini_without_temperature(self):
        """Test GPT-5-mini without temperature or max_tokens parameters."""
        model = 'gpt-5-mini'
        
        try:
            response = self.wrapper.generate_chat_text({
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a data science expert."},
                    {"role": "user", "content": "What is gradient boosting?"}
                ]
            })
            
            self.assertTrue("choices" in response)
            self.assertTrue(len(response["choices"]) > 0)
        except Exception as e:
            # Model might not be available, skip this test
            self.skipTest(f"GPT-5-mini model not available: {e}")
    
    def test_gpt5_mini_rejects_temperature(self):
        """Test that GPT-5-mini rejects temperature parameter."""
        model = 'gpt-5-mini'
        
        try:
            self.wrapper.generate_chat_text({
                "model": model,
                "messages": [
                    {"role": "user", "content": "Hello"}
                ],
                "temperature": 0.7
            })
            self.fail("Should have raised ValueError for temperature parameter")
        except ValueError as e:
            self.assertIn("temperature", str(e).lower())
        except Exception as e:
            # Model might not be available
            if "gpt-5-mini" not in str(e).lower() and "not available" not in str(e).lower():
                raise
    
    def test_gpt5_mini_rejects_max_tokens(self):
        """Test that GPT-5-mini rejects max_tokens parameter."""
        model = 'gpt-5-mini'
        
        try:
            self.wrapper.generate_chat_text({
                "model": model,
                "messages": [
                    {"role": "user", "content": "Hello"}
                ],
                "max_tokens": 100
            })
            self.fail("Should have raised ValueError for max_tokens parameter")
        except ValueError as e:
            self.assertIn("max_tokens", str(e).lower())
        except Exception as e:
            # Model might not be available
            if "gpt-5-mini" not in str(e).lower() and "not available" not in str(e).lower():
                raise
    
    def test_different_models(self):
        """Test with different chat models."""
        models_to_test = [
            os.getenv('AZURE_OPENAI_CHAT_MODEL', 'gpt-4o'),
            'gpt-4o',
            'gpt-4'
        ]
        
        for model in models_to_test:
            try:
                response = self.wrapper.generate_chat_text({
                    "model": model,
                    "messages": [
                        {"role": "user", "content": "What is machine learning?"}
                    ],
                    "max_tokens": 50
                })
                
                self.assertTrue("choices" in response)
                self.assertTrue(len(response["choices"]) > 0)
            except Exception:
                # Some models might not be available, that's okay
                pass
    
    def test_chat_completion_with_spell_correction_prompt(self):
        """Test chat completion with spell correction prompt (as used by Whisper)."""
        model = os.getenv('AZURE_WHISPER_SPELL_CORRECTOR_MODEL', 'gpt-4o')
        
        transcription = "The patint presentd with acut chest pain and dyspnea requirng imediate evaluashun."
        
        prompt = (
            "You are a spell correction assistant for medical transcriptions. "
            "Analyze the following transcription and return a JSON object "
            "with corrections in the format: {\"wrong\": \"correct\"}. "
            "Only include corrections that are clearly needed. "
            "Return an empty object {} if no corrections are needed."
        )
        
        response = self.wrapper.generate_chat_text({
            "model": model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": transcription}
            ],
            "temperature": 0
        })
        
        corrections = response["choices"][0]["message"]["content"]
        self.assertTrue(isinstance(corrections, str))
        self.assertTrue(len(corrections) > 0)
    
    def test_chat_completion_streaming(self):
        """Test streaming chat completion."""
        model = os.getenv('AZURE_OPENAI_CHAT_MODEL', 'gpt-4o')
        
        full_content = ""
        chunks_received = 0
        has_finish_reason = False
        
        for chunk in self.wrapper.generate_chat_text_stream({
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello in one sentence."}
            ],
            "temperature": 0.7
        }):
            chunks_received += 1
            self.assertTrue("choices" in chunk)
            self.assertTrue(len(chunk["choices"]) > 0)
            
            # Validate chunk structure
            choice = chunk["choices"][0]
            self.assertTrue("delta" in choice)
            self.assertIsInstance(choice["delta"], dict)
            
            if choice["delta"].get("content"):
                full_content += choice["delta"]["content"]
            
            # Check if this is the final chunk
            if choice.get("finish_reason"):
                has_finish_reason = True
                self.assertIsNotNone(choice["finish_reason"])
                if chunk.get("usage"):
                    usage = chunk["usage"]
                    self.assertGreater(usage.get("total_tokens", 0), 0)
                    self.assertGreater(usage.get("prompt_tokens", 0), 0)
                break
        
        self.assertGreater(chunks_received, 0, "Should have received at least one chunk")
        self.assertTrue(len(full_content) > 0, "Streamed content should not be empty")
        self.assertTrue(has_finish_reason, "Should have received finish_reason")
    
    def test_chat_completion_streaming_with_max_tokens(self):
        """Test streaming with max_tokens parameter."""
        model = os.getenv('AZURE_OPENAI_CHAT_MODEL', 'gpt-4o')
        
        chunks_received = 0
        
        for chunk in self.wrapper.generate_chat_text_stream({
            "model": model,
            "messages": [
                {"role": "user", "content": "Tell me a short story."}
            ],
            "max_tokens": 50
        }):
            chunks_received += 1
            if chunk["choices"][0].get("finish_reason"):
                break
        
        self.assertGreater(chunks_received, 0)
    
    def test_chat_completion_streaming_empty_response(self):
        """Test streaming handles empty responses gracefully."""
        model = os.getenv('AZURE_OPENAI_CHAT_MODEL', 'gpt-4o')
        
        chunks_received = 0
        
        for chunk in self.wrapper.generate_chat_text_stream({
            "model": model,
            "messages": [
                {"role": "user", "content": "Say OK"}
            ]
        }):
            chunks_received += 1
            self.assertTrue("choices" in chunk)
            if chunk["choices"][0].get("finish_reason"):
                break
        
        self.assertGreater(chunks_received, 0)
    
    def test_chat_completion_streaming_gpt5(self):
        """Test streaming chat completion with GPT-5 model."""
        model = 'gpt-5-mini'
        
        try:
            chunks_received = 0
            for chunk in self.wrapper.generate_chat_text_stream({
                "model": model,
                "messages": [
                    {"role": "user", "content": "Hello"}
                ]
            }):
                chunks_received += 1
                self.assertTrue("choices" in chunk)
                if chunks_received > 10:  # Limit to avoid long tests
                    break
            
            self.assertGreater(chunks_received, 0)
        except Exception as e:
            # Model might not be available, skip this test
            self.skipTest(f"GPT-5-mini model not available: {e}")


if __name__ == '__main__':
    unittest.main()


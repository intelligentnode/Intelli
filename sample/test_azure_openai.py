#!/usr/bin/env python3
"""
Test script for Azure OpenAI Wrapper

This script tests all functionality of the Azure OpenAI wrapper including:
- Chat completions
- Embeddings
- Error handling
- Different models

Usage:
    python sample/test_azure_openai.py

Environment variables required:
    AZURE_OPENAI_API_KEY - Azure OpenAI API key
    AZURE_OPENAI_ENDPOINT - Azure OpenAI endpoint URL
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from intelli.wrappers.azure_openai_wrapper import AzureOpenAIWrapper

load_dotenv()


def test_initialization():
    """Test wrapper initialization."""
    print("\n" + "="*60)
    print("TEST 1: Initialization")
    print("="*60)
    
    api_key = os.getenv('AZURE_OPENAI_API_KEY')
    endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    
    if not api_key or not endpoint:
        print("❌ FAILED: Missing required environment variables:")
        print("   - AZURE_OPENAI_API_KEY")
        print("   - AZURE_OPENAI_ENDPOINT")
        return None
    
    try:
        wrapper = AzureOpenAIWrapper(
            api_key=api_key,
            endpoint=endpoint
        )
        print(f"✅ SUCCESS: Azure OpenAI wrapper initialized")
        print(f"   Endpoint: {endpoint}")
        print(f"   Timeout: {wrapper.timeout}s")
        print(f"   Max retries: {wrapper.max_retries}")
        return wrapper
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return None


def test_invalid_initialization():
    """Test initialization with invalid parameters."""
    print("\n" + "="*60)
    print("TEST 2: Invalid Initialization")
    print("="*60)
    
    # Test with empty API key
    try:
        AzureOpenAIWrapper(api_key="", endpoint="https://test.openai.azure.com")
        print("❌ FAILED: Should have raised ValueError for empty API key")
    except ValueError as e:
        print(f"✅ SUCCESS: Correctly raised ValueError: {e}")
    except Exception as e:
        print(f"❌ FAILED: Raised unexpected exception: {e}")
    
    # Test with empty endpoint
    try:
        AzureOpenAIWrapper(api_key="test-key", endpoint="")
        print("❌ FAILED: Should have raised ValueError for empty endpoint")
    except ValueError as e:
        print(f"✅ SUCCESS: Correctly raised ValueError: {e}")
    except Exception as e:
        print(f"❌ FAILED: Raised unexpected exception: {e}")


def test_chat_completion_basic(wrapper):
    """Test basic chat completion."""
    print("\n" + "="*60)
    print("TEST 3: Basic Chat Completion")
    print("="*60)
    
    model = os.getenv('AZURE_OPENAI_CHAT_MODEL', 'gpt-4o')
    
    try:
        response = wrapper.generate_chat_text({
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "What is machine learning?"}
            ],
            "temperature": 0.7,
            "max_tokens": 100
        })
        
        print(f"✅ SUCCESS: Chat completion completed")
        print(f"   Model: {model}")
        
        if "choices" in response and len(response["choices"]) > 0:
            content = response["choices"][0]["message"]["content"]
            print(f"   Response: {content}")
            print(f"   Response length: {len(content)} characters")
        else:
            print(f"   ⚠️  WARNING: Response structure unexpected: {response}")
        
        if "usage" in response:
            usage = response["usage"]
            print(f"   Usage - Prompt tokens: {usage.get('prompt_tokens', 0)}")
            print(f"   Usage - Completion tokens: {usage.get('completion_tokens', 0)}")
            print(f"   Usage - Total tokens: {usage.get('total_tokens', 0)}")
        
    except Exception as e:
        print(f"❌ FAILED: {e}")


def test_chat_completion_conversation(wrapper):
    """Test multi-turn conversation."""
    print("\n" + "="*60)
    print("TEST 4: Multi-Turn Conversation")
    print("="*60)
    
    model = os.getenv('AZURE_OPENAI_CHAT_MODEL', 'gpt-4o')
    
    try:
        # First message
        response1 = wrapper.generate_chat_text({
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is Python?"}
            ]
        })
        
        assistant_response = response1["choices"][0]["message"]["content"]
        print(f"   User: What is Python?")
        print(f"   Assistant: {assistant_response[:100]}...")
        
        # Second message (continuation)
        response2 = wrapper.generate_chat_text({
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": assistant_response},
                {"role": "user", "content": "What can you do with it?"}
            ]
        })
        
        print(f"   User: What can you do with it?")
        print(f"   Assistant: {response2['choices'][0]['message']['content'][:100]}...")
        
        print(f"✅ SUCCESS: Multi-turn conversation completed")
        
    except Exception as e:
        print(f"❌ FAILED: {e}")


def test_chat_completion_different_temperatures(wrapper):
    """Test chat completion with different temperatures."""
    print("\n" + "="*60)
    print("TEST 5: Different Temperature Settings")
    print("="*60)
    
    model = os.getenv('AZURE_OPENAI_CHAT_MODEL', 'gpt-4o')
    temperatures = [0.0, 0.5, 1.0]
    
    for temp in temperatures:
        try:
            response = wrapper.generate_chat_text({
                "model": model,
            "messages": [
                {"role": "user", "content": "What are RESTful API principles?"}
            ],
            "temperature": temp,
            "max_tokens": 100
            })
            
            content = response["choices"][0]["message"]["content"]
            print(f"   Temperature {temp}: {content[:80]}...")
            
        except Exception as e:
            print(f"   ❌ FAILED for temperature {temp}: {e}")
    
    print(f"✅ SUCCESS: Tested different temperature settings")


def test_embeddings_single(wrapper):
    """Test embeddings for single text."""
    print("\n" + "="*60)
    print("TEST 6: Single Text Embedding")
    print("="*60)
    
    model = os.getenv('AZURE_OPENAI_EMBEDDING_MODEL', 'text-embedding-ada-002')
    text = "This is a test sentence for embeddings."
    
    try:
        response = wrapper.get_embeddings({
            "model": model,
            "input": text
        })
        
        print(f"✅ SUCCESS: Embedding generated")
        print(f"   Model: {model}")
        print(f"   Input text: {text}")
        
        if "data" in response and len(response["data"]) > 0:
            embedding = response["data"][0]["embedding"]
            print(f"   Embedding dimensions: {len(embedding)}")
            print(f"   Embedding sample (first 5): {embedding[:5]}")
        else:
            print(f"   ⚠️  WARNING: Response structure unexpected: {response}")
        
        if "usage" in response:
            usage = response["usage"]
            print(f"   Usage - Prompt tokens: {usage.get('prompt_tokens', 0)}")
            print(f"   Usage - Total tokens: {usage.get('total_tokens', 0)}")
        
    except Exception as e:
        print(f"❌ FAILED: {e}")


def test_embeddings_multiple(wrapper):
    """Test embeddings for multiple texts."""
    print("\n" + "="*60)
    print("TEST 7: Multiple Texts Embedding")
    print("="*60)
    
    model = os.getenv('AZURE_OPENAI_EMBEDDING_MODEL', 'text-embedding-ada-002')
    texts = [
        "The weather is nice today.",
        "It's a sunny day outside.",
        "I love programming in Python."
    ]
    
    try:
        response = wrapper.get_embeddings({
            "model": model,
            "input": texts
        })
        
        print(f"✅ SUCCESS: Embeddings generated for {len(texts)} texts")
        print(f"   Model: {model}")
        
        if "data" in response:
            print(f"   Number of embeddings: {len(response['data'])}")
            for i, item in enumerate(response["data"]):
                embedding = item["embedding"]
                print(f"   Text {i+1} ('{texts[i][:30]}...'): {len(embedding)} dimensions")
        
        if "usage" in response:
            usage = response["usage"]
            print(f"   Usage - Total tokens: {usage.get('total_tokens', 0)}")
        
    except Exception as e:
        print(f"❌ FAILED: {e}")


def test_error_handling_invalid_model(wrapper):
    """Test error handling with invalid model."""
    print("\n" + "="*60)
    print("TEST 8: Error Handling - Invalid Model")
    print("="*60)
    
    try:
        response = wrapper.generate_chat_text({
            "model": "invalid-model-name-12345",
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        })
        print("❌ FAILED: Should have raised an error for invalid model")
    except RuntimeError as e:
        print(f"✅ SUCCESS: Correctly raised RuntimeError: {e}")
    except Exception as e:
        print(f"⚠️  WARNING: Raised unexpected exception: {e}")


def test_error_handling_missing_messages(wrapper):
    """Test error handling with missing messages."""
    print("\n" + "="*60)
    print("TEST 9: Error Handling - Missing Messages")
    print("="*60)
    
    model = os.getenv('AZURE_OPENAI_CHAT_MODEL', 'gpt-4o')
    
    try:
        response = wrapper.generate_chat_text({
            "model": model,
            "messages": []  # Empty messages
        })
        print("⚠️  WARNING: Should have raised an error for empty messages")
    except Exception as e:
        print(f"✅ SUCCESS: Correctly raised exception: {e}")


def test_gpt5_mini_without_temperature(wrapper):
    """Test GPT-5-mini which doesn't accept temperature or max_tokens parameters."""
    print("\n" + "="*60)
    print("TEST 10: GPT-5-mini Without Temperature or Max Tokens")
    print("="*60)
    
    model = 'gpt-5-mini'
    
    # Test without temperature and max_tokens (should work)
    try:
        print("   Testing GPT-5-mini without temperature or max_tokens parameters...")
        response = wrapper.generate_chat_text({
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a data science expert."},
                {"role": "user", "content": "What is gradient boosting?"}
            ]
            # No temperature or max_tokens - should work fine
        })
        
        content = response["choices"][0]["message"]["content"]
        print(f"   ✅ SUCCESS: GPT-5-mini works without temperature or max_tokens")
        print(f"   Response preview: {content[:100]}...")
        
    except Exception as e:
        error_msg = str(e).lower()
        if "max_tokens" in error_msg and "null" in error_msg:
            print(f"   ⚠️  Note: This error suggests the wrapper may need to handle None values better")
        print(f"   ⚠️  Model might not be available: {e}")
    
    # Test with temperature (should fail or be ignored)
    try:
        print("\n   Testing GPT-5-mini WITH temperature parameter (should fail or be ignored)...")
        response = wrapper.generate_chat_text({
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a data science expert."},
                {"role": "user", "content": "What is gradient boosting?"}
            ],
            "temperature": 0.7  # This should cause an error or be ignored
        })
        
        # If it doesn't fail, check if temperature was actually applied
        print(f"   ⚠️  WARNING: Temperature parameter was accepted, but GPT-5-mini doesn't support it")
        print(f"   This might indicate the parameter is being silently ignored")
        
    except Exception as e:
        error_msg = str(e).lower()
        if "temperature" in error_msg or "parameter" in error_msg or "invalid" in error_msg:
            print(f"   ✅ SUCCESS: Correctly rejected temperature parameter")
            print(f"   Error: {e}")
        else:
            print(f"   ⚠️  Unexpected error: {e}")
    
    # Test with max_tokens (should fail or be ignored)
    try:
        print("\n   Testing GPT-5-mini WITH max_tokens parameter (should fail or be ignored)...")
        response = wrapper.generate_chat_text({
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a data science expert."},
                {"role": "user", "content": "What is gradient boosting?"}
            ],
            "max_tokens": 100  # This should cause an error or be ignored
        })
        
        # If it doesn't fail, check if max_tokens was actually applied
        print(f"   ⚠️  WARNING: max_tokens parameter was accepted, but GPT-5-mini doesn't support it")
        print(f"   This might indicate the parameter is being silently ignored")
        
    except Exception as e:
        error_msg = str(e).lower()
        if "max_tokens" in error_msg or "parameter" in error_msg or "invalid" in error_msg:
            print(f"   ✅ SUCCESS: Correctly rejected max_tokens parameter")
            print(f"   Error: {e}")
        else:
            print(f"   ⚠️  Unexpected error: {e}")


def test_different_models(wrapper):
    """Test with different chat models."""
    print("\n" + "="*60)
    print("TEST 11: Different Chat Models")
    print("="*60)
    
    # Try to get models from environment or use defaults
    models_to_test = [
        os.getenv('AZURE_OPENAI_CHAT_MODEL', 'gpt-4o'),
        'gpt-4o',
        'gpt-4'
    ]
    
    for model in models_to_test:
        try:
            response = wrapper.generate_chat_text({
                "model": model,
                "messages": [
                    {"role": "user", "content": "What is machine learning?"}
                ],
                "max_tokens": 50
            })
            
            content = response["choices"][0]["message"]["content"]
            print(f"   ✅ {model}: {content[:60]}...")
            
        except Exception as e:
            print(f"   ❌ {model}: {str(e)[:60]}...")


def test_initialization_with_custom_timeout_retries():
    """Test initialization with custom timeout and retries."""
    print("\n" + "="*60)
    print("TEST 11: Custom Timeout and Retries Configuration")
    print("="*60)
    
    api_key = os.getenv('AZURE_OPENAI_API_KEY')
    endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    
    if not api_key or not endpoint:
        print("⚠️  SKIPPED: Missing Azure OpenAI credentials")
        return
    
    try:
        wrapper = AzureOpenAIWrapper(
            api_key=api_key,
            endpoint=endpoint,
            timeout=120.0,
            max_retries=5
        )
        print(f"✅ SUCCESS: Wrapper initialized with custom settings")
        print(f"   Timeout: {wrapper.timeout}s")
        print(f"   Max retries: {wrapper.max_retries}")
    except Exception as e:
        print(f"❌ FAILED: {e}")


def test_chat_completion_with_spell_correction_prompt(wrapper):
    """Test chat completion with spell correction prompt (as used by Whisper)."""
    print("\n" + "="*60)
    print("TEST 12: Spell Correction Use Case")
    print("="*60)
    
    model = os.getenv('AZURE_WHISPER_SPELL_CORRECTOR_MODEL', 'gpt-4o')
    
    # Simulate medical transcription with errors
    transcription = "The patint presentd with acut chest pain and dyspnea requirng imediate evaluashun."
    
    prompt = (
        "You are a spell correction assistant for medical transcriptions. "
        "Analyze the following transcription and return a JSON object "
        "with corrections in the format: {\"wrong\": \"correct\"}. "
        "Only include corrections that are clearly needed. "
        "Return an empty object {} if no corrections are needed."
    )
    
    try:
        response = wrapper.generate_chat_text({
            "model": model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": transcription}
            ],
            "temperature": 0
        })
        
        corrections = response["choices"][0]["message"]["content"]
        print(f"✅ SUCCESS: Spell correction completed")
        print(f"   Original: {transcription}")
        print(f"   Corrections: {corrections}")
        
    except Exception as e:
        print(f"❌ FAILED: {e}")


def test_chat_completion_streaming(wrapper):
    """Test streaming chat completion."""
    print("\n" + "="*60)
    print("TEST 13: Streaming Chat Completion")
    print("="*60)
    
    model = os.getenv('AZURE_OPENAI_CHAT_MODEL', 'gpt-4o')
    
    try:
        print("   Streaming response:")
        full_content = ""
        chunks_received = 0
        
        for chunk in wrapper.generate_chat_text_stream({
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Count from 1 to 5, one number per line."}
            ],
            "temperature": 0.7
        }):
            chunks_received += 1
            
            if "choices" in chunk and len(chunk["choices"]) > 0:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                
                if content:
                    print(content, end="", flush=True)
                    full_content += content
            
            # Check if this is the final chunk
            if chunk["choices"][0].get("finish_reason"):
                print()  # New line after streaming
                if chunk.get("usage"):
                    usage = chunk["usage"]
                    print(f"\n   ✅ SUCCESS: Streaming completed")
                    print(f"   Chunks received: {chunks_received}")
                    print(f"   Total tokens: {usage.get('total_tokens', 0)}")
                    print(f"   Full content length: {len(full_content)} characters")
                break
        
        if chunks_received == 0:
            print("   ❌ FAILED: No chunks received")
        elif len(full_content) == 0:
            print("   ⚠️  WARNING: Received chunks but no content")
        
    except Exception as e:
        print(f"❌ FAILED: {e}")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("AZURE OPENAI WRAPPER - COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    # Initialize wrapper
    wrapper = test_initialization()
    
    if wrapper is None:
        print("\n❌ Cannot proceed without initialized wrapper. Please check your configuration.")
        return
    
    # Run all tests
    test_invalid_initialization()
    test_initialization_with_custom_timeout_retries()
    test_chat_completion_basic(wrapper)
    test_chat_completion_conversation(wrapper)
    test_chat_completion_different_temperatures(wrapper)
    test_embeddings_single(wrapper)
    test_embeddings_multiple(wrapper)
    test_error_handling_invalid_model(wrapper)
    test_error_handling_missing_messages(wrapper)
    test_gpt5_mini_without_temperature(wrapper)
    test_different_models(wrapper)
    test_chat_completion_with_spell_correction_prompt(wrapper)
    test_chat_completion_streaming(wrapper)
    
    print("\n" + "="*60)
    print("TEST SUITE COMPLETED")
    print("="*60)


if __name__ == '__main__':
    main()


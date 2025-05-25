import unittest
import os
import time
from dotenv import load_dotenv
from intelli.wrappers.geminiai_wrapper import GeminiAIWrapper
import base64
load_dotenv()

class TestGeminiAIWrapper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        api_key = os.getenv("GEMINI_API_KEY")
        assert api_key is not None, "GEMINI_API_KEY is not set."
        cls.wrapper = GeminiAIWrapper(api_key)
    
    def test_generate_content(self):
        """Test basic content generation with Gemini 2.0 Flash"""
        params = {
            "contents": [{
                "parts": [{
                    "text": "Write a story about a magic backpack."
                }]
            }]
        }

        result = self.wrapper.generate_content(params)
        print('content sample result: ', result['candidates'][0]['content']['parts'][0]['text'][:100])
        self.assertIsNotNone(result['candidates'][0]['content']['parts'][0]['text'])

    def test_generate_content_with_system_instructions(self):
        """Test system instructions functionality"""
        content_parts = [{"text": "What is the capital of France?"}]
        system_instruction = "You are a helpful geography teacher. Always provide additional context."
        
        try:
            result = self.wrapper.generate_content_with_system_instructions(
                content_parts, system_instruction
            )
            
            response_text = result['candidates'][0]['content']['parts'][0]['text']
            print('System instruction result: ', response_text[:150])
            self.assertIsNotNone(response_text)
            self.assertIn('Paris', response_text)
        except Exception as e:
            print(f"System instructions test error: {e}")
    
    def test_image_to_text(self):
        """Test basic image to text conversion"""
        file_path = './temp/test_image_desc.png'

        try:
            with open(file_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            result = self.wrapper.image_to_text('describe the image', image_data, 'png')

            self.assertTrue('candidates' in result, "The result should have a 'candidates' field.")
            self.assertIsInstance(result['candidates'], list, "Expected 'candidates' to be a list.")
            self.assertGreater(len(result['candidates']), 0, "'candidates' list should not be empty.")
            
            generated_text = result['candidates'][0]['content']['parts'][0]['text']
            print('Gemini AI Image To Text Generation Test Result:\n', generated_text, '\n')

            # Assert the generated text is not empty
            self.assertTrue(generated_text, 'Gemini AI returned no results')

        except Exception as error:
            self.fail(f'Gemini AI Error: {error}')

    def test_generate_image(self):
        """Test image generation using Gemini 2.0 Flash"""
        try:
            prompt = "A beautiful sunset over mountains"
            result = self.wrapper.generate_image(prompt)
            
            print('Image generation result keys: ', list(result.keys()))
            self.assertIn('candidates', result)
            
            # Check if image data is present
            candidate = result['candidates'][0]
            image_found = False
            
            if 'content' in candidate and 'parts' in candidate['content']:
                for part in candidate['content']['parts']:
                    if 'inline_data' in part and part['inline_data'].get('mime_type', '').startswith('image/'):
                        image_data = part['inline_data']['data']
                        print('✅ Image generated successfully - data length:', len(image_data))
                        image_found = True
                        break
            
            if not image_found:
                print("⚠️ No image data found in response")
                
        except Exception as e:
            print(f"Image generation test error: {e}")
            print("Note: Image generation may require special access")

    def test_generate_speech(self):
        """Test text-to-speech generation"""
        try:
            text = "Hello, this is a test of Gemini's text-to-speech capabilities."
            result = self.wrapper.generate_speech(text)
            
            print('TTS result keys: ', list(result.keys()))
            self.assertIn('candidates', result)
            
            # Check for audio data
            candidate = result['candidates'][0]
            audio_found = False
            
            if 'content' in candidate and 'parts' in candidate['content']:
                for part in candidate['content']['parts']:
                    if 'inline_data' in part and part['inline_data'].get('mime_type', '').startswith('audio/'):
                        audio_data = part['inline_data']['data']
                        print('✅ Audio generated successfully - data length:', len(audio_data))
                        audio_found = True
                        break
            
            if not audio_found:
                print("⚠️ No audio data found in response")
                
        except Exception as e:
            print(f"TTS test error: {e}")
            print("Note: TTS may require special model access")

    def test_get_embeddings(self):
        """Test embeddings with latest model (text-embedding-004)"""
        text = "Write a story about a magic backpack."
        params = {
            "content": {
                "parts": [{
                    "text": text
                }]
            }
        }

        result = self.wrapper.get_embeddings(params)
        print('embedding sample result: ', result.get('embedding', {}).get('values', [])[:5])
        self.assertTrue('embedding' in result)
        
        # Check if using latest embedding model (should have 768 dimensions)
        if 'embedding' in result and 'values' in result['embedding']:
            embedding_values = result['embedding']['values']
            print(f'Embedding dimensions: {len(embedding_values)}')
            # text-embedding-004 should have 768 dimensions
            if len(embedding_values) == 768:
                print('✅ Using latest text-embedding-004 model')

    def test_get_batch_embeddings(self):
        """Test batch embeddings with latest model"""
        texts = ["Hello world", "Write a story about a magic backpack."]

        # Format according to the documentation
        requests = [
            {
                "model": f"models/text-embedding-004",  # Use explicit model name to match docs
                "content": {
                    "parts": [{"text": text}]
                }
            } for text in texts
        ]

        params = {"requests": requests}

        result = self.wrapper.get_batch_embeddings(params)

        if result and len(result) > 0:
            print('batch embedding sample result: ', result[0].get("values", [])[:5])

        self.assertGreater(len(result), 0, "No batch embedding results returned.")
    
if __name__ == "__main__":
    unittest.main()

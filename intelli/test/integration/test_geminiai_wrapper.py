import unittest
import os
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
    
    def test_image_to_text(self):
        file_path = '../temp/test_image_desc.png' 

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
    
    def test_get_embeddings(self):
        text = "Write a story about a magic backpack."
        params = {
            "model": "models/embedding-001",
            "content": {
                "parts": [{
                    "text": text
                }]
            }
        }

        result = self.wrapper.get_embeddings(params)
        print('embedding sample result: ', result['values'][:5])
        self.assertTrue('values' in result)
    
    def test_get_batch_embeddings(self):
        texts = ["Hello world", "Write a story about a magic backpack."]
        requests = [{
            "model": "models/embedding-001",
            "content": {
                "parts": [{"text": text}]
            }
        } for text in texts]

        result = self.wrapper.get_batch_embeddings({"requests": requests})
        print('batch embedding sample result: ', result[0]['values'][:5])
        self.assertGreater(len(result), 0, "No batch embedding results returned.")
    
if __name__ == "__main__":
    unittest.main()

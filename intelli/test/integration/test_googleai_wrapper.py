import unittest
from intelli.wrappers.googleai_wrapper import GoogleAIWrapper
import os 
from dotenv import load_dotenv
load_dotenv()

class TestGoogleAIWrapper(unittest.TestCase):

    def setUp(self):
        self.api_key = os.getenv('GOOGLE_API_KEY')
        self.googleai = GoogleAIWrapper(self.api_key)
   
    def test_generate_speech(self):
        params = {
            "text": 'Welcome to IntelliNode',
            "languageCode": 'en-US',
            "name": 'en-US-Wavenet-A',
            "ssmlGender": 'MALE',
        }

        result = self.googleai.generate_speech(params)
        print('Generate Speech Result:', result)
        self.assertTrue(len(result) > 0)

if __name__ == "__main__":
    unittest.main()

import base64
import os
import unittest
from dotenv import load_dotenv
from intelli.controller.remote_speech_model import RemoteSpeechModel, SupportedSpeechModels
from intelli.model.input.text_speech_input import Text2SpeechInput

load_dotenv()

class TestRemoteSpeechModel(unittest.TestCase):

    def setUp(self):
        self.api_key_google = os.getenv('GOOGLE_API_KEY')
        self.api_key_openai = os.getenv('OPENAI_API_KEY')
        self.remote_speech_model_google = RemoteSpeechModel(self.api_key_google, SupportedSpeechModels['GOOGLE'])
        self.remote_speech_model_openai = RemoteSpeechModel(self.api_key_openai, SupportedSpeechModels['OPENAI'])
        self.temp_dir = '../temp'

        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

    def test_generate_speech_google(self):
        input_params = Text2SpeechInput('Welcome to Intellinode.', 'en-gb')
        audio_content = self.remote_speech_model_google.generate_speech(input_params)
        self.assertTrue(audio_content, "audio_content should not be None")
        
        # eecode the base64
        audio_data = base64.b64decode(audio_content)
        google_file_path = os.path.join(self.temp_dir, 'google_speech.mp3')
        
        # save
        with open(google_file_path, 'wb') as audio_file:
            audio_file.write(audio_data)
        
        self.assertTrue(os.path.exists(google_file_path), "Google TTS MP3 file should be saved")

    def test_generate_speech_openai(self):
        input_params = Text2SpeechInput('Welcome to Intellinode.', 'en-US', 'MALE', 'alloy', 'tts-1', True)
        result = self.remote_speech_model_openai.generate_speech(input_params)
        self.assertTrue(result, "result should not be None")
        
        # write the streaming audio
        openai_file_path = os.path.join(self.temp_dir, 'openai_speech.mp3')
        with open(openai_file_path, 'wb') as audio_file:
            for chunk in result:
                audio_file.write(chunk)
        
        self.assertTrue(os.path.exists(openai_file_path), "OpenAI TTS MP3 file should be saved")

if __name__ == "__main__":
    unittest.main()

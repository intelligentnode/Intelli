import unittest
import os
import tempfile
from intelli.wrappers.elevenlabs_wrapper import ElevenLabsWrapper
from dotenv import load_dotenv

load_dotenv()


class TestElevenLabsWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api_key = os.getenv("ELEVENLABS_API_KEY")
        if not cls.api_key:
            raise unittest.SkipTest("ELEVENLABS_API_KEY environment variable not set")

        cls.wrapper = ElevenLabsWrapper(cls.api_key)

        # For tests that require a voice ID, get the first available voice
        voices = cls.wrapper.list_voices()
        if not voices or 'voices' not in voices or not voices['voices']:
            raise unittest.SkipTest("No voices available for testing")

        cls.voice_id = voices['voices'][0]['voice_id']

    def test_list_voices(self):

        result = self.wrapper.list_voices()
        self.assertIn('voices', result)
        self.assertTrue(len(result['voices']) > 0)
        print(f"Found {len(result['voices'])} voices")

        # Print first voice details
        first_voice = result['voices'][0]
        print(f"First voice: {first_voice['name']} ({first_voice['voice_id']})")

    def test_text_to_speech(self):

        text = "Hello, this is a test of the Eleven Labs text to speech API."

        # Get audio data
        audio_data = self.wrapper.text_to_speech(
            text=text,
            voice_id=self.voice_id
        )

        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_path = temp_file.name

        # Check file exists and has content
        self.assertTrue(os.path.exists(temp_path))
        self.assertTrue(os.path.getsize(temp_path) > 0)

        print(f"Generated audio file at {temp_path}")

        # Clean up
        os.unlink(temp_path)

    def test_stream_text_to_speech(self):

        text = "This is a test of streaming audio from Eleven Labs."

        # Get streaming response
        response = self.wrapper.stream_text_to_speech(
            text=text,
            voice_id=self.voice_id
        )

        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    temp_file.write(chunk)
            temp_path = temp_file.name

        # Check file exists and has content
        self.assertTrue(os.path.exists(temp_path))
        self.assertTrue(os.path.getsize(temp_path) > 0)

        print(f"Generated streaming audio file at {temp_path}")

        # Clean up
        os.unlink(temp_path)

    def test_speech_to_text(self):

        text = "This is a test of the speech to text capability."

        # Get audio data and save to temporary file
        audio_data = self.wrapper.text_to_speech(
            text=text,
            voice_id=self.voice_id
        )

        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            temp_file.write(audio_data)
            audio_path = temp_file.name

        try:

            result = self.wrapper.speech_to_text(audio_path)

            # Verify response
            self.assertIn('text', result)
            print(f"Transcribed text: {result['text']}")

        finally:
            # Clean up
            os.unlink(audio_path)

    def test_speech_to_text_with_bytes(self):

        text = "This is a test of the speech to text capability with bytes input."

        # Get audio data
        audio_data = self.wrapper.text_to_speech(
            text=text,
            voice_id=self.voice_id
        )

        # Use the bytes directly
        result = self.wrapper.speech_to_text(audio_data)

        # Verify response
        self.assertIn('text', result)
        print(f"Transcribed text from bytes: {result['text']}")

    def test_speech_to_speech(self):

        text = "This is a test of the voice transformation capability."

        # Get audio data and save to temporary file
        audio_data = self.wrapper.text_to_speech(
            text=text,
            voice_id=self.voice_id
        )

        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            temp_file.write(audio_data)
            audio_path = temp_file.name

        try:

            transformed_audio = self.wrapper.speech_to_speech(
                audio_file=audio_path,
                voice_id=self.voice_id  # Use same voice for simplicity
            )

            # Save transformed audio to temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as out_file:
                out_file.write(transformed_audio)
                out_path = out_file.name

            # Check file exists and has content
            self.assertTrue(os.path.exists(out_path))
            self.assertTrue(os.path.getsize(out_path) > 0)

            print(f"Generated transformed audio file at {out_path}")

            # Clean up transformed audio
            os.unlink(out_path)

        finally:
            # Clean up original audio
            os.unlink(audio_path)

if __name__ == "__main__":
    unittest.main()
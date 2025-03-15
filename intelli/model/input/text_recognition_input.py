class SpeechRecognitionInput:
    """
    Input parameters for speech recognition API calls.
    Supports both cloud providers (OpenAI) and local models (Keras).
    """

    def __init__(self,
                 audio_file_path=None,
                 audio_data=None,
                 sample_rate=16000,
                 language=None,
                 model="whisper-1",
                 user_prompt=None,
                 condition_on_previous_text=False,
                 max_steps=80,
                 max_chunk_sec=30):
        """
        Initialize speech recognition input parameters.

        Args:
            audio_file_path: Path to audio file
            audio_data: Raw audio data (numpy array)
            sample_rate: Sample rate of audio data
            language: Language code (e.g., 'en' or '<|en|>')
            model: Model name for OpenAI
            user_prompt: Optional text prompt to guide transcription
            condition_on_previous_text: Whether to use previous output as context
            max_steps: Maximum decoding steps
            max_chunk_sec: Maximum chunk size in seconds
        """
        self.audio_file_path = audio_file_path
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.language = language
        self.model = model
        self.user_prompt = user_prompt
        self.condition_on_previous_text = condition_on_previous_text
        self.max_steps = max_steps
        self.max_chunk_sec = max_chunk_sec

        # Validate that either file path or audio data is provided
        if not audio_file_path and audio_data is None:
            raise ValueError("Either audio_file_path or audio_data must be provided")

    def get_openai_input(self):
        """
        Return file path and parameters for OpenAI's speech to text API.
        """
        # Just return the basic information needed for a direct API call
        return {
            'file_path': self.audio_file_path,
            'model': self.model,
            'language': self.language
        }

    def get_keras_input(self):
        """Return parameters for Keras Whisper models"""
        return {
            'audio_data': self.get_audio_data(),
            'sample_rate': self.sample_rate,
            'language': self.language,
            'user_prompt': self.user_prompt,
            'condition_on_previous_text': self.condition_on_previous_text,
            'max_steps': self.max_steps,
            'max_chunk_sec': self.max_chunk_sec
        }

    def get_audio_data(self):
        """Get audio data for processing with local models"""
        if self.audio_data is not None:
            return self.audio_data

        if self.audio_file_path:
            try:
                import numpy as np
                import librosa

                # Load the audio file
                audio_data, sample_rate = librosa.load(self.audio_file_path, sr=self.sample_rate)
                self.sample_rate = sample_rate
                self.audio_data = audio_data
                return self.audio_data

            except ImportError:
                raise ImportError("librosa and numpy are required for loading audio files")

        return None

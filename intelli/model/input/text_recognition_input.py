import os
import tempfile
import numpy as np


class SpeechRecognitionInput:
    """
    Input parameters for speech recognition API calls.
    Supports both cloud providers (OpenAI) and local models (Keras).
    Enhanced to handle audio file paths from speech tasks.
    """

    def __init__(
            self,
            audio_file_path=None,
            audio_data=None,
            sample_rate=16000,
            language=None,
            model="whisper-1",
            user_prompt=None,
            condition_on_previous_text=False,
            max_steps=80,
            max_chunk_sec=30,
    ):
        """
        Initialize speech recognition input parameters.

        Args:
            audio_file_path: Path to audio file
            audio_data: Raw audio data (numpy array or bytes)
            sample_rate: Sample rate of audio data
            language: Language code (e.g., 'en' or '<|en|>')
            model: Model name for OpenAI
            user_prompt: Optional text prompt to guide transcription
            condition_on_previous_text: Whether to use previous output as context
            max_steps: Maximum decoding steps
            max_chunk_sec: Maximum chunk size in seconds
        """
        # Check if audio_data is a dictionary from speech task output
        if isinstance(audio_data, dict) and 'audio_file' in audio_data:
            audio_file_path = audio_data['audio_file']
            audio_data = None
            print(f"Using audio file from speech task: {audio_file_path}")

        # Strip 'file:' prefix if present
        if isinstance(audio_file_path, str) and audio_file_path.startswith("file:"):
            audio_file_path = audio_file_path[5:]
            print(f"Stripped 'file:' prefix, using path: {audio_file_path}")

        self.audio_file_path = audio_file_path
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.language = language
        self.model = model
        self.user_prompt = user_prompt
        self.condition_on_previous_text = condition_on_previous_text

        # Ensure these values are never None to prevent multiplication errors
        self.max_steps = max_steps if max_steps is not None else 80
        self.max_chunk_sec = max_chunk_sec if max_chunk_sec is not None else 30

        self.model_id = None  # For ElevenLabs
        self._temp_file = None

        # Relaxed validation for flow context
        if not audio_file_path and audio_data is None:
            print("Warning: Neither audio_file_path nor audio_data provided")

    def get_openai_input(self):
        """
        Return file path and parameters for OpenAI's speech to text API.
        """
        # Validate that we have a valid file path
        if not self.audio_file_path or not os.path.exists(self.audio_file_path):
            # If no valid file path but we have audio_data bytes, create a temporary file
            if isinstance(self.audio_data, (bytes, bytearray)):
                try:
                    # Create a temporary file with .mp3 extension
                    self._temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
                    self._temp_file.write(self.audio_data)
                    self._temp_file.flush()
                    self.audio_file_path = self._temp_file.name
                    print(f"Created temporary file for OpenAI: {self.audio_file_path}")
                except Exception as e:
                    print(f"Error creating temporary file: {e}")
                    raise ValueError(f"No valid audio file path and failed to create temp file: {e}")
            else:
                raise ValueError(f"Invalid or nonexistent audio file path: {self.audio_file_path}")

        return {
            "file_path": self.audio_file_path,
            "model": self.model,
            "language": self.language,
        }

    def get_keras_input(self):
        """Return parameters for Keras Whisper models"""
        return {
            "audio_data": self.get_audio_data(),
            "sample_rate": self.sample_rate,
            "language": self.language,
            "user_prompt": self.user_prompt,
            "condition_on_previous_text": self.condition_on_previous_text,
            "max_steps": self.max_steps,  # Now guaranteed to not be None
            "max_chunk_sec": self.max_chunk_sec,  # Now guaranteed to not be None
        }

    def get_audio_data(self):
        """
        Get audio data for processing with local models.
        Enhanced to handle bytes data and file paths from flow tasks.
        """
        # If we already have audio data as a numpy array, return it
        if self.audio_data is not None and not isinstance(self.audio_data, (bytes, bytearray)):
            # Assume it's already a numpy array
            return self.audio_data

        # If we have bytes, convert to numpy array
        if isinstance(self.audio_data, (bytes, bytearray)):
            try:
                print(f"Converting audio bytes to numpy array, size: {len(self.audio_data)}")
                # Try using soundfile first which handles various audio formats
                try:
                    import soundfile as sf
                    import io
                    import tempfile
                    import os

                    # Create temporary file from bytes
                    temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
                    temp_file.write(self.audio_data)
                    temp_file.close()

                    try:
                        # Load with soundfile
                        audio_data, sample_rate = sf.read(temp_file.name)
                        self.sample_rate = sample_rate
                        print(f"Converted audio bytes using soundfile, shape: {audio_data.shape}")
                        return audio_data
                    finally:
                        # Clean up temp file
                        os.remove(temp_file.name)
                except Exception as e1:
                    print(f"Soundfile conversion failed: {e1}, trying librosa")
                    # Fall back to librosa
                    import librosa
                    import tempfile
                    import os

                    # Create temporary file from bytes
                    temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
                    temp_file.write(self.audio_data)
                    temp_file.close()

                    try:
                        # Load with librosa
                        audio_data, sample_rate = librosa.load(temp_file.name, sr=self.sample_rate)
                        self.sample_rate = sample_rate
                        print(f"Converted audio bytes using librosa, shape: {audio_data.shape}")
                        return audio_data
                    finally:
                        # Clean up temp file
                        os.remove(temp_file.name)
            except Exception as e:
                print(f"Failed to convert audio bytes: {e}")
                # Return a small dummy array as last resort
                print("Creating dummy audio array")
                return np.zeros(16000)

        # If we have a file path, load it
        if self.audio_file_path and os.path.exists(self.audio_file_path):
            try:
                # Try soundfile first
                try:
                    import soundfile as sf
                    audio_data, sample_rate = sf.read(self.audio_file_path)
                    self.sample_rate = sample_rate
                    print(f"Loaded audio file with soundfile: {self.audio_file_path}")
                    return audio_data
                except Exception:
                    # Fall back to librosa
                    import librosa
                    audio_data, sample_rate = librosa.load(self.audio_file_path, sr=self.sample_rate)
                    self.sample_rate = sample_rate
                    print(f"Loaded audio file with librosa: {self.audio_file_path}")
                    return audio_data
            except ImportError:
                raise ImportError("librosa, soundfile, and numpy are required for loading audio files")
            except Exception as e:
                print(f"Error loading audio file: {e}")
                import traceback
                traceback.print_exc()

        print("Warning: No valid audio data could be obtained. Creating dummy audio array.")
        return np.zeros(16000)  # Return dummy array rather than None

    def get_elevenlabs_input(self):
        """
        Get input parameters for Eleven Labs speech-to-text.
        """
        params = {}

        # Handle file path
        if self.audio_file_path and os.path.exists(self.audio_file_path):
            params['file_path'] = self.audio_file_path

        # Handle audio data
        elif self.audio_data is not None:
            if isinstance(self.audio_data, (bytes, bytearray)):
                try:
                    import tempfile
                    import os

                    # Create a temporary file with .mp3 extension
                    temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
                    temp_file.write(self.audio_data)
                    temp_file.flush()
                    params['file_path'] = temp_file.name
                    self._temp_file = temp_file  # Store for cleanup
                    print(f"Created temporary file for ElevenLabs: {temp_file.name}")
                except Exception as e:
                    print(f"Error creating temporary file: {e}")
                    params['audio_data'] = self.audio_data
            else:
                params['audio_data'] = self.audio_data
        else:
            raise ValueError("Either file_path or audio_data must be provided")

        # Add optional parameters
        if hasattr(self, 'model_id') and self.model_id:
            params['model_id'] = self.model_id

        if self.language:
            params['language'] = self.language

        return params

    def __del__(self):
        """Clean up temporary files on destruction"""
        if hasattr(self, '_temp_file') and self._temp_file:
            try:
                if os.path.exists(self._temp_file.name):
                    os.unlink(self._temp_file.name)
                    print(f"Cleaned up temporary file: {self._temp_file.name}")
            except Exception as e:
                print(f"Error cleaning up temp file: {e}")

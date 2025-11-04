import os
import logging
import io
from typing import Union, Optional

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from openai import AzureOpenAI
except ImportError:
    AzureOpenAI = None


class AzureWhisperWrapper:
    """
    Wrapper for Azure OpenAI Whisper transcription service.
    
    This wrapper provides speech-to-text transcription using Azure OpenAI's Whisper model.
    """
    
    def __init__(
        self,
        api_key: str,
        endpoint: str,
        api_version: str = "2024-02-15-preview",
        deployment_name: str = "whisper",
        timeout: Optional[float] = 60.0,
        max_retries: int = 3
    ):
        """
        Initialize Azure Whisper wrapper.
        
        Args:
            api_key: Azure OpenAI API key
            endpoint: Azure OpenAI endpoint URL (e.g., "https://{resource-name}.openai.azure.com")
            api_version: API version (default: "2024-02-15-preview")
            deployment_name: Deployment name for Whisper model (default: "whisper")
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
        self.deployment_name = deployment_name
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
            timeout=timeout,
            max_retries=max_retries
        )
        
        logger.debug(f"Azure Whisper wrapper initialized with endpoint: {self.endpoint}")
    
    def transcribe(
        self,
        audio_file: Union[str, bytes],
        file_name: Optional[str] = None,
        language: Optional[str] = None,
        prompt_guide: Optional[str] = None,
        temperature: float = 0
    ) -> str:
        """
        Transcribe audio using Azure OpenAI Whisper.
        
        Args:
            audio_file: Path to audio file (str) or audio data (bytes)
            file_name: Name of the audio file (required if audio_file is bytes)
            language: Language code for transcription (e.g., "en", "es", "fr")
            prompt_guide: Prompt guide for transcription (context/vocabulary hints)
            temperature: Sampling temperature (0-1, default: 0 for deterministic)
            
        Returns:
            Transcribed text as string
            
        Raises:
            ValueError: If audio_file is invalid or transcription fails
            RuntimeError: If Azure API call fails
        """
        # Handle both file path and bytes
        if isinstance(audio_file, str):
            if not os.path.exists(audio_file):
                raise ValueError(f"Audio file does not exist: {audio_file}")
            with open(audio_file, 'rb') as f:
                audio_data = f.read()
            file_name = file_name or os.path.basename(audio_file)
        elif isinstance(audio_file, bytes):
            if not audio_file:
                raise ValueError("audio_file bytes cannot be empty")
            audio_data = audio_file
            if not file_name:
                raise ValueError("file_name is required when audio_file is bytes")
        else:
            raise ValueError("audio_file must be a file path (str) or bytes")
        
        language = language or 'en'
        prompt_guide = prompt_guide or ""
        
        logger.debug(f"Starting Azure Whisper transcription for file: {file_name}")
        
        try:
            # Create a file-like object for the API
            audio_file_obj = io.BytesIO(audio_data)
            audio_file_obj.name = file_name
            
            # Call Azure OpenAI Whisper API
            transcription = self.client.audio.transcriptions.create(
                file=(file_name, audio_file_obj, "application/octet-stream"),
                model=self.deployment_name,
                language=language,
                prompt=prompt_guide,
                temperature=temperature,
            ).text
            
            if not transcription or transcription.strip() == "":
                logger.error("Transcription returned empty result")
                raise ValueError(
                    "No speech detected in the audio. "
                    "Please record your voice clearly and try again."
                )
            
            logger.debug(f"Transcription: {transcription}")
            return transcription
            
        except ValueError as e:
            raise
        except Exception as e:
            logger.error(f"Azure Whisper transcription failed: {e}")
            raise

import os
import asyncio
import tempfile
import logging
import json
from typing import Dict, Any, Optional, Union, AsyncGenerator, List
from datetime import timedelta

from intelli.config import config

# Try to import optional dependencies
try:
    import aiohttp
except ImportError:
    aiohttp = None

try:
    import websockets
except ImportError:
    websockets = None

try:
    from speechmatics.batch import AsyncClient, FormatType, JobConfig, JobType, TranscriptionConfig, OperatingPoint
except ImportError:
    AsyncClient = None
    FormatType = None
    JobConfig = None
    JobType = None
    TranscriptionConfig = None
    OperatingPoint = None

logger = logging.getLogger(__name__)


class SpeechmaticsWrapper:
    """Wrapper for Speechmatics API operations."""
    
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        """
        Initialize Speechmatics wrapper.
        
        Args:
            api_key: Speechmatics API key
            base_url: Optional base URL for Speechmatics batch API (defaults to config)
        """
        # Check if required dependencies are installed
        if AsyncClient is None or aiohttp is None or websockets is None:
            raise ImportError(
                "Speechmatics dependencies are not installed. "
                "Install with: pip install intelli[speech]"
            )
        
        self.api_key = api_key
        self.api_url = base_url or config["url"]["speechmatics"]["base"]
        self.realtime_url = os.getenv('SPEECHMATICS_REALTIME_URL', config["url"]["speechmatics"]["realtime"])
        self.default_language = 'en'
        self.default_timeout = 3600
        
        logger.debug(f"Speechmatics API URL: {self.api_url}")
        logger.debug(f"Speechmatics Real-time URL: {self.realtime_url}")
    
    def speech_to_text(self, audio_file: Union[str, bytes], language: str = None, 
                      output_format: str = "text", timeout: int = None) -> str:
        """
        Convert speech to text using Speechmatics API.
        
        Args:
            audio_file: Path to audio file (str) or audio data (bytes)
            language: Language code (defaults to 'en')
            output_format: Output format - 'text', 'speakers', or 'segments'
            timeout: Timeout in seconds (defaults to 3600)
            
        Returns:
            Transcribed text as string
        """
        if not audio_file:
            raise ValueError("audio_file is required")
        
        # Handle both file path and bytes
        if isinstance(audio_file, str):
            if not os.path.exists(audio_file):
                raise ValueError(f"Audio file does not exist: {audio_file}")
            with open(audio_file, 'rb') as f:
                audio_data = f.read()
            file_name = os.path.basename(audio_file)
        else:
            audio_data = audio_file
            file_name = "audio.wav"  # Default filename for bytes input
        
        speechmatics_language = self._map_language(language or self.default_language)
        timeout = timeout or self.default_timeout
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            return loop.run_until_complete(
                self._transcribe_async(file_name, audio_data, speechmatics_language, output_format, timeout)
            )
        except Exception as e:
            logger.error(f"Speechmatics transcription failed: {e}")
            raise RuntimeError(f"Transcription failed: {str(e)}")
    
    async def _transcribe_async(self, file_name: str, audio_file: bytes, language: str, 
                               output_format: str, timeout: int) -> str:
        """Async transcription implementation."""
        temp_file_path = None
        
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_name.split('.')[-1]}") as temp_file:
                temp_file.write(audio_file)
                temp_file_path = temp_file.name
            
            # Get operating point from environment or use default
            operating_point_env = os.getenv('SPEECHMATICS_OPERATING_POINT', 'ENHANCED')
            operating_point = OperatingPoint.ENHANCED if operating_point_env == 'ENHANCED' else OperatingPoint.STANDARD
            
            # Get speaker sensitivity from environment or use default
            speaker_sensitivity = float(os.getenv('SPEECHMATICS_SPEAKER_SENSITIVITY', '0.6'))
            
            # Submit job and wait for completion
            async with AsyncClient(api_key=self.api_key, url=self.api_url) as client:
                config = JobConfig(
                    type=JobType.TRANSCRIPTION,
                    transcription_config=TranscriptionConfig(
                        language=language,
                        operating_point=operating_point,
                        diarization="speaker" if output_format in ["speakers", "segments"] else None,
                        speaker_diarization_config={"speaker_sensitivity": speaker_sensitivity}
                    ),
                )
                
                job = await client.submit_job(temp_file_path, config=config)
                result = await client.wait_for_completion(
                    job.id, 
                    format_type=FormatType.JSON,
                    timeout=timeout
                )
                
                # Clean up job
                await self._cleanup_job(job.id)
                
                return self._format_transcript(result, output_format)
                
        except Exception as e:
            logger.error(f"Speechmatics async transcription failed: {e}")
            raise
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except OSError as e:
                    logger.warning(f"Failed to clean up temp file {temp_file_path}: {e}")
    
    async def _cleanup_job(self, job_id: str) -> bool:
        """Delete the Speechmatics job after transcription."""
        try:
            url = f"{self.api_url.rstrip('/')}/jobs/{job_id}"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            async with aiohttp.ClientSession() as session:
                async with session.delete(url, headers=headers) as response:
                    if response.status in [200, 204]:
                        logger.info(f"Successfully deleted Speechmatics job {job_id}")
                        return True
                    else:
                        logger.warning(f"Failed to delete job {job_id}: HTTP {response.status}")
                        return False
        except Exception as e:
            logger.error(f"Error deleting Speechmatics job {job_id}: {e}")
            return False
    
    def _map_language(self, language: str) -> str:
        """Map language codes to Speechmatics language codes."""
        mapping = {
            'en-IE': 'en',  # Irish English -> en
            'en-US': 'en',
            'en-GB': 'en',
            'en-AU': 'en',
            'en-CA': 'en',
        }
        return mapping.get(language, language)
    
    def _format_transcript(self, result, output_format: str) -> str:
        """Format Speechmatics result based on output format."""
        if not hasattr(result, 'results') or not result.results:
            logger.error("No results found in Speechmatics response")
            return ""
        
        if output_format == "segments":
            return self._format_segments(result.results)
        elif output_format == "speakers":
            return self._format_speakers(result.results)
        else:  # "text" format
            return self._format_text(result.results)
    
    def _format_text(self, results) -> str:
        """Format as plain text without speaker labels."""
        pieces = []
        
        for recognition_result in results:
            if hasattr(recognition_result, 'alternatives') and recognition_result.alternatives:
                alternative = recognition_result.alternatives[0]
                if hasattr(alternative, 'content') and alternative.content:
                    pieces.append(alternative.content)
        
        return " ".join(pieces)
    
    def _format_segments(self, results) -> str:
        """Format as time-stamped segments."""
        pieces = []
        
        for recognition_result in results:
            if hasattr(recognition_result, 'alternatives') and recognition_result.alternatives:
                alternative = recognition_result.alternatives[0]
                if hasattr(alternative, 'content') and alternative.content:
                    start_time = self._format_time(recognition_result.start_time)
                    end_time = self._format_time(recognition_result.end_time)
                    speaker = alternative.speaker if hasattr(alternative, 'speaker') and alternative.speaker else "unknown"
                    pieces.append(f"{speaker} ({start_time}–{end_time}): {alternative.content}")
        
        return "\n\n".join(pieces) + "\n"
    
    def _format_speakers(self, results) -> str:
        """Format as conversation with speaker labels."""
        if not results:
            return ""
        
        pieces = []
        current_speaker = None
        current_text_parts = []
        
        for recognition_result in results:
            if hasattr(recognition_result, 'alternatives') and recognition_result.alternatives:
                alternative = recognition_result.alternatives[0]
                if hasattr(alternative, 'content') and alternative.content:
                    speaker = alternative.speaker if hasattr(alternative, 'speaker') and alternative.speaker else "unknown"
                    content = alternative.content
                    
                    if not content.strip():
                        continue
                        
                    if current_speaker != speaker:
                        if current_speaker and current_text_parts:
                            combined_text = " ".join(current_text_parts)
                            pieces.append(f"{current_speaker.lower()}: {combined_text}")
                        
                        current_speaker = speaker
                        current_text_parts = [content]
                    else:
                        current_text_parts.append(content)
        
        if current_speaker and current_text_parts:
            combined_text = " ".join(current_text_parts)
            pieces.append(f"{current_speaker.lower()}: {combined_text}")
        
        return "\n".join(pieces) + "\n"
    
    def _format_time(self, seconds: float) -> str:
        """Format time in MM:SS.sss format."""
        td = timedelta(seconds=seconds)
        total_seconds = td.total_seconds()
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{int(hours):02}:{int(minutes):02}:{secs:06.3f}"
        return f"{int(minutes):02}:{secs:06.3f}"
    
    # ==================== Streaming Methods ====================
    
    async def start_streaming_session(self, language: str = None, 
                                     sample_rate: int = 16000,
                                     enable_partials: bool = True):
        """
        Start a real-time streaming session with Speechmatics.
        
        Args:
            language: Language code (defaults to 'en')
            sample_rate: Audio sample rate (defaults to 16000)
            enable_partials: Enable partial transcripts (defaults to True)
            
        Returns:
            WebSocket connection for streaming
        """
        speechmatics_language = self._map_language(language or self.default_language)
        
        # Get operating point from environment or use default
        operating_point_env = os.getenv('SPEECHMATICS_OPERATING_POINT', 'ENHANCED')
        operating_point = operating_point_env.lower()
        
        # Get speaker sensitivity
        speaker_sensitivity = float(os.getenv('SPEECHMATICS_SPEAKER_SENSITIVITY', '0.6'))
        
        # Determine if we need diarization
        output_format = os.getenv('SPEECHMATICS_OUTPUT_FORMAT', 'speakers')
        diarization = "speaker" if output_format in ["speakers", "segments"] else None
        
        # Connect to Speechmatics real-time API
        websocket = await websockets.connect(
            self.realtime_url,
            additional_headers={"Authorization": f"Bearer {self.api_key}"},
            max_size=None,  # Allow large messages
            ping_interval=None
        )
        
        # Send configuration
        config = {
            "message": "StartRecognition",
            "transcription_config": {
                "language": speechmatics_language,
                "operating_point": operating_point,
                "enable_partials": enable_partials
            },
            "audio_format": {
                "type": "raw",
                "encoding": "pcm_f32le",
                "sample_rate": sample_rate
            }
        }
        
        # Add diarization if enabled
        if diarization:
            config["transcription_config"]["diarization"] = diarization
            config["transcription_config"]["speaker_diarization_config"] = {
                "speaker_sensitivity": speaker_sensitivity
            }
        
        await websocket.send(json.dumps(config))
        logger.info("Started real-time streaming session")
        
        return websocket
    
    async def stream_audio(self, session, audio_data: bytes) -> None:
        """
        Stream audio data to the real-time session.
        
        Args:
            session: WebSocket session
            audio_data: Audio bytes to send
        """
        try:
            await session.send(audio_data)
        except Exception as e:
            logger.error(f"Failed to stream audio: {str(e)}")
            raise RuntimeError(f"Failed to stream audio: {str(e)}")
    
    async def receive_transcripts(self, session) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Receive transcription results from the real-time session.
        
        Args:
            session: WebSocket session
            
        Yields:
            Dictionary with transcription data. Types:
            - {"type": "partial", "tokens": [...], "confidence": list, "transcript": "..."} - Interim/partial results with per-token confidence scores (0.0-1.0)
            - {"type": "final", "tokens": [...], "speaker": "...", "confidence": list, "transcript": "..."} - Final confirmed results with per-token confidence scores (0.0-1.0)
            - {"type": "error", "message": "..."} - Error messages
            
        Note:
            Tokens and confidence scores are returned directly from Speechmatics API.
            Each token (word or punctuation) has a corresponding confidence score.
            The transcript field is provided for convenience (simple join of tokens).
            
            Example:
                for token, conf in zip(result['tokens'], result['confidence']):
                    print(f"{token} [{conf:.2f}]")
        """
        try:
            async for message in session:
                try:
                    data = json.loads(message)
                    
                    if data.get("message") == "AddTranscript":
                        # Extract all words with their speakers and confidence scores
                        words = []
                        
                        if "results" in data and isinstance(data["results"], list):
                            for result in data["results"]:
                                if isinstance(result, dict) and "alternatives" in result and result["alternatives"]:
                                    alt = result["alternatives"][0]
                                    if isinstance(alt, dict):
                                        content = alt.get("content", "").strip()
                                        speaker = alt.get("speaker", "unknown")
                                        confidence = alt.get("confidence", None)
                                        
                                        if content:
                                            words.append((content, speaker, confidence))
                        
                        if words:
                            current_speaker = None
                            current_tokens = []
                            confidences = []
                            
                            for content, speaker, confidence in words:
                                if speaker != current_speaker and current_tokens:
                                    yield {
                                        "type": "final",
                                        "tokens": current_tokens,
                                        "speaker": current_speaker or "unknown",
                                        "confidence": confidences,
                                        "transcript": " ".join(current_tokens)
                                    }
                                    current_tokens = []
                                    confidences = []
                                
                                current_speaker = speaker
                                current_tokens.append(content)
                                confidences.append(confidence)
                            
                            if current_tokens:
                                yield {
                                    "type": "final",
                                    "tokens": current_tokens,
                                    "speaker": current_speaker or "unknown",
                                    "confidence": confidences,
                                    "transcript": " ".join(current_tokens)
                                }
                    
                    elif data.get("message") == "AddPartialTranscript":
                        # Extract partial transcript with confidence scores
                        partial_tokens = []
                        confidences = []
                        if "results" in data and isinstance(data["results"], list):
                            for result in data["results"]:
                                if isinstance(result, dict) and "alternatives" in result and result["alternatives"]:
                                    alt = result["alternatives"][0]
                                    if isinstance(alt, dict):
                                        content = alt.get("content", "").strip()
                                        confidence = alt.get("confidence", None)
                                        if content:
                                            partial_tokens.append(content)
                                            confidences.append(confidence)
                        
                        if partial_tokens:
                            yield {
                                "type": "partial",
                                "tokens": partial_tokens,
                                "confidence": confidences,
                                "transcript": " ".join(partial_tokens)
                            }
                    elif data.get("message") == "Error":
                        yield {
                            "type": "error",
                            "message": data.get("reason", "Unknown error")
                        }
                        
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logger.warning(f"Error processing message: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            yield {
                "type": "error",
                "message": f"Connection error: {str(e)}"
            }
    
    # ==================== Audio Conversion Methods ====================
    
    @staticmethod
    def detect_audio_format(audio_file: Union[str, bytes]) -> Dict[str, Any]:
        """
        Detect audio format from file or bytes.
        
        Args:
            audio_file: Path to audio file or audio bytes
            
        Returns:
            Dictionary with format information (format, sample_rate, channels, etc.)
        """
        try:
            import librosa
            
            if isinstance(audio_file, str):
                if not os.path.exists(audio_file):
                    raise ValueError(f"Audio file does not exist: {audio_file}")
                
                # Get file info using librosa
                duration = librosa.get_duration(path=audio_file)
                y, sr = librosa.load(audio_file, sr=None, mono=False)
                
                channels = 1 if y.ndim == 1 else y.shape[0]
                
                return {
                    'format': os.path.splitext(audio_file)[1].lstrip('.'),
                    'subtype': os.path.splitext(audio_file)[1].lstrip('.'),
                    'sample_rate': sr,
                    'channels': channels,
                    'duration': duration
                }
            else:
                # For bytes, write to temporary file
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
                    tmp.write(audio_file)
                    tmp_path = tmp.name
                
                try:
                    duration = librosa.get_duration(path=tmp_path)
                    y, sr = librosa.load(tmp_path, sr=None, mono=False)
                    
                    channels = 1 if y.ndim == 1 else y.shape[0]
                    
                    return {
                        'format': os.path.splitext(tmp_path)[1].lstrip('.'),
                        'subtype': os.path.splitext(tmp_path)[1].lstrip('.'),
                        'sample_rate': sr,
                        'channels': channels,
                        'duration': duration
                    }
                finally:
                    os.unlink(tmp_path)
        except ImportError:
            logger.warning("librosa not available, cannot detect audio format")
            return None
        except Exception as e:
            logger.error(f"Error detecting audio format: {e}")
            return None
    
    @staticmethod
    def convert_audio_to_pcm_f32le(audio_file: Union[str, bytes], 
                                   target_sample_rate: int = 16000,
                                   target_channels: int = 1) -> bytes:
        """
        Convert audio to PCM F32LE format required for streaming.
        
        Args:
            audio_file: Path to audio file or audio bytes
            target_sample_rate: Target sample rate (default: 16000)
            target_channels: Target number of channels (default: 1 for mono)
            
        Returns:
            Audio data in PCM F32LE format as bytes
        """
        try:
            import librosa
            import numpy as np
            
            # Read audio - librosa supports many formats including MP3
            if isinstance(audio_file, str):
                if not os.path.exists(audio_file):
                    raise ValueError(f"Audio file does not exist: {audio_file}")
                audio_data, sample_rate = librosa.load(
                    audio_file, 
                    sr=None,  # Load original sample rate
                    mono=False  # Load all channels
                )
            else:
                # For bytes, write to temporary file
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
                    tmp.write(audio_file)
                    tmp_path = tmp.name
                
                try:
                    audio_data, sample_rate = librosa.load(
                        tmp_path,
                        sr=None,
                        mono=False
                    )
                finally:
                    os.unlink(tmp_path)
            
            # Handle multi-channel audio
            if audio_data.ndim > 1:
                if target_channels == 1:
                    # Convert to mono by taking mean
                    audio_data = np.mean(audio_data, axis=0)
                else:
                    # Take first channel
                    audio_data = audio_data[0]
            
            # Resample if needed
            if sample_rate != target_sample_rate:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sample_rate)
            
            # Convert to float32 (librosa already returns float32, but ensure it)
            audio_data = audio_data.astype(np.float32)
            
            # Ensure values are in [-1, 1] range
            audio_data = np.clip(audio_data, -1.0, 1.0)
            
            return audio_data.tobytes()
            
        except ImportError as e:
            missing = str(e).split("'")[1] if "'" in str(e) else str(e)
            if 'librosa' in missing or 'librosa' in str(e):
                raise ImportError("librosa is required for audio conversion. Install with: pip install librosa soundfile")
            else:
                raise ImportError(f"{missing} is required for audio conversion")
        except Exception as e:
            logger.error(f"Error converting audio: {e}")
            raise RuntimeError(f"Failed to convert audio to PCM F32LE: {str(e)}")

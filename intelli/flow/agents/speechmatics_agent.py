from __future__ import annotations
import asyncio
import json
import os
import tempfile
from typing import Any, Callable, Dict, Optional

from intelli.flow.agents.custom_agent import CustomAgent
from intelli.flow.types import AgentTypes
from intelli.wrappers.speechmatics_wrapper import SpeechmaticsWrapper

class StreamingSpeechAgent(CustomAgent):
    """
    StreamingSpeechAgent handles real-time speech-to-text using Speechmatics.
    It yields partial/final transcripts to a listener callback while aggregating
    the final result for the next step in the flow.
    """
    def __init__(self, api_key: str, listener_callback: Optional[Callable[[str], None]] = None, 
                 mission: str = "", provider: str = "speechmatics", 
                 options: Optional[Dict[str, Any]] = None):
        # Set agent type to RECOGNITION as it takes audio input and produces text output
        super().__init__(agent_type=AgentTypes.RECOGNITION.value, provider=provider, mission=mission, options=options)
        self.wrapper = SpeechmaticsWrapper(api_key)
        self.listener = listener_callback

    def execute(self, agent_input: Any, new_params: Optional[Dict[str, Any]] = None) -> str:
        """
        Execute the streaming speech-to-text.
        This is called synchronously by the Task, so we manage the async loop internally.
        """
        # Resolve any combined parameters
        custom_params = dict(self.model_params)
        if new_params:
            custom_params.update(new_params)
            
        # Use a new event loop to ensure a clean environment (Flow runs tasks in a thread).
        # Avoid mutating the thread-global default event loop.
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._stream_and_collect(agent_input, custom_params))
        except Exception as e:
            return f"Error in streaming speech-to-text: {str(e)}"
        finally:
            try:
                loop.close()
            except Exception:
                pass

    async def _stream_and_collect(self, agent_input: Any, params: Dict[str, Any]) -> str:
        full_transcript = []
        last_partial = ""
        any_transcript_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        last_update_ts = loop.time()
        
        # 1. Start the streaming session
        language = params.get("language", "en")
        sample_rate = params.get("sample_rate", 16000)
        max_wait_seconds = float(params.get("max_wait_seconds", 30.0))
        idle_timeout_seconds = float(params.get("idle_timeout_seconds", 2.0))
        chunk_ms = int(params.get("chunk_ms", 50))  # 50ms chunks by default
        
        session = await self.wrapper.start_streaming_session(
            language=language,
            sample_rate=sample_rate
        )
        
        tmp_audio_path: Optional[str] = None
        try:
            # 2. Prepare audio data
            audio_data = None
            if hasattr(agent_input, "audio") and agent_input.audio:
                audio_data = agent_input.audio
            elif isinstance(agent_input, (bytes, bytearray)):
                audio_data = agent_input
            
            if not audio_data:
                raise ValueError("No audio data provided for StreamingSpeechAgent")

            # Prefer passing a file path into convert_audio_to_pcm_f32le so the suffix matches the actual bytes.
            # If the input is MP3 bytes, use a .mp3 temp file; if it's WAV (RIFF), use .wav.
            if isinstance(audio_data, (bytes, bytearray)):
                b = bytes(audio_data)
                is_wav = b[:4] == b"RIFF" and b[8:12] == b"WAVE"
                is_mp3 = b[:3] == b"ID3" or (len(b) > 2 and b[0] == 0xFF and (b[1] & 0xE0) == 0xE0)
                suffix = ".mp3" if is_mp3 and not is_wav else ".wav"

                fd, tmp_audio_path = tempfile.mkstemp(suffix=suffix)
                os.close(fd)
                with open(tmp_audio_path, "wb") as f:
                    f.write(b)
                audio_source: Any = tmp_audio_path
            else:
                audio_source = audio_data

            # Speechmatics requires PCM F32LE format
            pcm_audio = self.wrapper.convert_audio_to_pcm_f32le(
                audio_source,
                target_sample_rate=sample_rate,
                target_channels=1,
            )
            
            # 3. Setup receiver task BEFORE sending audio
            async def receive_and_process():
                nonlocal last_partial
                nonlocal last_update_ts
                async for result in self.wrapper.receive_transcripts(session):
                    if result["type"] == "final":
                        chunk = result["transcript"]
                        if chunk:
                            full_transcript.append(chunk)
                            any_transcript_event.set()
                            last_update_ts = loop.time()
                            if self.listener:
                                if asyncio.iscoroutinefunction(self.listener):
                                    await self.listener(chunk)
                                else:
                                    await loop.run_in_executor(None, self.listener, chunk)
                    elif result["type"] == "partial":
                        last_partial = result.get("transcript", "")
                        if last_partial:
                            any_transcript_event.set()
                            last_update_ts = loop.time()
                    elif result["type"] == "error":
                        raise RuntimeError(f"Speechmatics stream error: {result.get('message')}")

            receiver_task = asyncio.create_task(receive_and_process())
            
            # 4. Stream audio chunks (avoid sending huge buffers in a single websocket frame)
            bytes_per_sample = 4  # float32
            samples_per_chunk = max(1, int(sample_rate * (chunk_ms / 1000.0)))
            chunk_size = samples_per_chunk * bytes_per_sample
            for i in range(0, len(pcm_audio), chunk_size):
                await self.wrapper.stream_audio(session, pcm_audio[i : i + chunk_size])
                # tiny yield to let receiver process network events
                await asyncio.sleep(0)
            
            # 5. Wait for the first transcript (or timeout), then stop after idle window
            try:
                await asyncio.wait_for(any_transcript_event.wait(), timeout=max_wait_seconds)
            except asyncio.TimeoutError:
                # No transcript arrived in time; close and return empty/partial
                pass

            # Idle-wait loop: if no updates for idle_timeout_seconds, stop
            start_ts = loop.time()
            while True:
                now = loop.time()
                if now - start_ts > max_wait_seconds:
                    break
                if any_transcript_event.is_set() and (now - last_update_ts) >= idle_timeout_seconds:
                    break
                await asyncio.sleep(0.1)

            # 6. Close session to stop the receiver loop and await receiver completion
            try:
                await session.close()
            except Exception:
                pass

            try:
                await asyncio.wait_for(receiver_task, timeout=5.0)
            except Exception:
                receiver_task.cancel()
                try:
                    await receiver_task
                except Exception:
                    pass
            
            # If no final transcripts were received, use the last partial
            result_text = " ".join(full_transcript).strip()
            if not result_text and last_partial:
                result_text = last_partial.strip()
                
            return result_text
            
        finally:
            # Final cleanup
            try:
                if not session.closed:
                    await session.close()
            except:
                pass
            if tmp_audio_path:
                try:
                    os.unlink(tmp_audio_path)
                except Exception:
                    pass

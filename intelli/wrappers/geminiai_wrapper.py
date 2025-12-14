import base64
import requests
import json
import time
import os
from typing import List, Dict, Any, Optional, Union

from intelli.config import config


class GeminiAIWrapper:

    def __init__(self, api_key):
        self.API_BASE_URL = config['url']['gemini']['base']
        self.UPLOAD_BASE_URL = config['url']['gemini']['upload_base']
        self.FILES_BASE_URL = config['url']['gemini']['files_base']
        self.VERTEX_BASE_URL = config['url']['gemini']['vertex_base']
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json'
        })
        self.API_KEY = api_key
        self.models = config['url']['gemini']['models']
        self.endpoints = config['url']['gemini']['endpoints']

    # ------------------------------------------------------------------
    # Backwards-compatible request/response normalization
    # Gemini REST APIs commonly use camelCase keys; older Intelli code/tests
    # often use snake_case. We accept either and normalize to avoid breaking
    # existing users.
    # ------------------------------------------------------------------
    _KEY_MAP = {
        # top-level / common
        "system_instruction": "systemInstruction",
        "generation_config": "generationConfig",
        "safety_settings": "safetySettings",
        "tool_config": "toolConfig",
        "response_modalities": "responseModalities",
        "speech_config": "speechConfig",
        "voice_config": "voiceConfig",
        "multi_speaker_voice_config": "multiSpeakerVoiceConfig",
        "speaker_voice_configs": "speakerVoiceConfigs",
        "prebuilt_voice_config": "prebuiltVoiceConfig",
        "voice_name": "voiceName",
        "response_mime_type": "responseMimeType",
        "response_schema": "responseSchema",
        # content parts
        "inline_data": "inlineData",
        "file_data": "fileData",
        "mime_type": "mimeType",
        "file_uri": "fileUri",
    }

    _REVERSE_KEY_MAP = {v: k for k, v in _KEY_MAP.items()}

    def _camelize(self, obj: Any) -> Any:
        """Convert known snake_case keys to camelCase recursively."""
        if isinstance(obj, list):
            return [self._camelize(x) for x in obj]
        if not isinstance(obj, dict):
            return obj
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            new_k = self._KEY_MAP.get(k, k)
            out[new_k] = self._camelize(v)
        return out

    def _snake_alias(self, obj: Any) -> Any:
        """
        Add snake_case aliases for known camelCase keys recursively.
        Does not remove the original camelCase keys.
        """
        if isinstance(obj, list):
            return [self._snake_alias(x) for x in obj]
        if not isinstance(obj, dict):
            return obj
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            aliased_v = self._snake_alias(v)
            out[k] = aliased_v
            snake_k = self._REVERSE_KEY_MAP.get(k)
            if snake_k and snake_k not in out:
                out[snake_k] = aliased_v
        return out

    def generate_content(self, params, vision=False, model_override=None):
        """Generate content using Gemini models"""
        if model_override:
            model = model_override
        else:
            model = self.models['vision'] if vision else self.models['text']
        
        url = f"{self.API_BASE_URL}/{model}{self.endpoints['generateContent']}"
        normalized_params = self._camelize(params)

        try:
            response = self.session.post(url, json=normalized_params, params={'key': self.API_KEY})
            response.raise_for_status()
            return self._snake_alias(response.json())
        except requests.exceptions.RequestException as error:
            if hasattr(error, 'response') and error.response:
                try:
                    error_detail = error.response.json()
                    raise Exception(f"Gemini API error: {error} - Details: {json.dumps(error_detail)}")
                except:
                    pass
            raise Exception(f"Gemini API error: {error}")
        except Exception as error:
            raise Exception(str(error))

    def generate_content_with_system_instructions(self, content_parts, system_instruction=None, model_override=None):
        """Generate content with system instructions support"""
        params = {
            "contents": [{
                "parts": content_parts
            }]
        }
        
        if system_instruction:
            params["system_instruction"] = {
                "parts": [{"text": system_instruction}]
            }
            
        return self.generate_content(params, model_override=model_override)

    def generate_structured_content(
        self,
        content_parts: List[Dict[str, Any]],
        response_schema: Dict[str, Any],
        system_instruction: Optional[str] = None,
        model_override: Optional[str] = None,
        response_mime_type: str = "application/json",
        generation_config: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Generate structured JSON outputs using Gemini structured outputs.
        Backwards compatible: callers can pass snake_case keys; we normalize.
        """
        gen_cfg = generation_config.copy() if generation_config else {}
        gen_cfg.setdefault("response_mime_type", response_mime_type)
        gen_cfg.setdefault("response_schema", response_schema)

        params: Dict[str, Any] = {
            "contents": [{"parts": content_parts}],
            "generation_config": gen_cfg,
        }
        if system_instruction:
            params["system_instruction"] = {"parts": [{"text": system_instruction}]}
        if tools is not None:
            params["tools"] = tools
        if tool_config is not None:
            params["tool_config"] = tool_config

        return self.generate_content(params, model_override=model_override)

    def stream_generate_content(self, params, vision=False, model_override=None):
        """
        Stream content from Gemini (yields decoded lines).
        Backwards compatible: accepts snake_case params and normalizes to camelCase.
        """
        if model_override:
            model = model_override
        else:
            model = self.models['vision'] if vision else self.models['text']

        stream_ep = self.endpoints.get('streamGenerateContent', ':streamGenerateContent')
        url = f"{self.API_BASE_URL}/{model}{stream_ep}"
        normalized_params = self._camelize(params)

        try:
            with self.session.post(url, json=normalized_params, params={'key': self.API_KEY}, stream=True) as response:
                response.raise_for_status()
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        yield line
        except requests.exceptions.RequestException as error:
            if hasattr(error, 'response') and error.response:
                try:
                    error_detail = error.response.json()
                    raise Exception(f"Gemini stream error: {error} - Details: {json.dumps(error_detail)}")
                except:
                    pass
            raise Exception(f"Gemini stream error: {error}")

    def image_to_text(self, user_input, image_data, extension):
        """Convert image to text using vision model"""
        params = {
            "contents": [
                {
                    "parts": [
                        {"text": f"{user_input}"},
                        {
                            "inline_data": {
                                "mime_type": f"image/{extension}",
                                "data": image_data
                            }
                        }
                    ]
                }
            ]
        }

        return self.image_to_text_params(params=params)

    def image_to_text_params(self, params, model_override=None):
        """Process image to text with custom parameters"""
        return self.generate_content(params, True, model_override=model_override)

    def image_to_text_with_file_uri(self, user_input, file_uri, mime_type):
        """Convert image to text using uploaded file URI"""
        params = {
            "contents": [
                {
                    "parts": [
                        {"text": user_input},
                        {
                            "file_data": {
                                "mime_type": mime_type,
                                "file_uri": file_uri
                            }
                        }
                    ]
                }
            ]
        }
        return self.generate_content(params, True)

    def multiple_images_to_text(self, user_input, images_data):
        """Process multiple images with text prompt"""
        parts = [{"text": user_input}]
        
        for img_data in images_data:
            if 'file_uri' in img_data:
                parts.append({
                    "file_data": {
                        "mime_type": img_data['mime_type'],
                        "file_uri": img_data['file_uri']
                    }
                })
            else:
                parts.append({
                    "inline_data": {
                        "mime_type": img_data['mime_type'],
                        "data": img_data['data']
                    }
                })
        
        params = {
            "contents": [{
                "parts": parts
            }]
        }
        
        return self.generate_content(params, True)

    def get_bounding_boxes(self, user_input, image_data, extension):
        """Get bounding box coordinates for objects in image"""
        prompt = f"{user_input}. Return bounding boxes in [ymin, xmin, ymax, xmax] format normalized to 0-1000."
        return self.image_to_text(prompt, image_data, extension)

    def get_image_segmentation(self, user_input, image_data, extension):
        """Get image segmentation masks"""
        prompt = f"""
        {user_input}
        Output a JSON list of segmentation masks where each entry contains the 2D
        bounding box in the key "box_2d", the segmentation mask in key "mask", and
        the text label in the key "label". Use descriptive labels.
        """
        return self.image_to_text(prompt, image_data, extension)

    def generate_image(self, prompt, config_params=None, model_override=None):
        """Generate images using Gemini 2.0 Flash Image Generation"""
        model = model_override or self.models['image_generation']
        url = f"{self.API_BASE_URL}/{model}{self.endpoints['generateContent']}"
        
        default_config = {
            "responseModalities": ["TEXT", "IMAGE"]
        }
        
        if config_params:
            default_config.update(config_params)
        
        params = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": default_config
        }

        try:
            response = self.session.post(url, json=self._camelize(params), params={'key': self.API_KEY})
            response.raise_for_status()
            return self._snake_alias(response.json())
        except requests.exceptions.RequestException as error:
            if hasattr(error, 'response') and error.response:
                try:
                    error_detail = error.response.json()
                    raise Exception(f"Gemini Image Generation error: {error} - Details: {json.dumps(error_detail)}")
                except:
                    pass
            raise Exception(f"Gemini Image Generation error: {error}")

    def generate_video(self, prompt, config_params=None, project_id=None):
        """Generate videos using Veo 2.0"""
        if not project_id:
            raise ValueError("Project ID is required for video generation")
            
        model = self.models['video_generation']
        url = f"{self.VERTEX_BASE_URL}/{project_id}/locations/us-central1/publishers/google/models/{model}{self.endpoints['predictLongRunning']}"
        
        default_params = {
            "aspectRatio": "16:9",
            "personGeneration": "dont_allow"
        }
        
        if config_params:
            default_params.update(config_params)
        
        payload = {
            "instances": [{
                "prompt": prompt
            }],
            "parameters": default_params
        }

        try:
            response = self.session.post(url, json=payload, params={'key': self.API_KEY})
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as error:
            if hasattr(error, 'response') and error.response:
                try:
                    error_detail = error.response.json()
                    raise Exception(f"Veo Video Generation error: {error} - Details: {json.dumps(error_detail)}")
                except:
                    pass
            raise Exception(f"Veo Video Generation error: {error}")

    def check_video_generation_status(self, operation_name, project_id=None):
        """Check the status of video generation operation"""
        if not project_id:
            raise ValueError("Project ID is required to check video generation status")
            
        url = f"{self.VERTEX_BASE_URL}/{operation_name}"
        
        try:
            response = self.session.get(url, params={'key': self.API_KEY})
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as error:
            if hasattr(error, 'response') and error.response:
                try:
                    error_detail = error.response.json()
                    raise Exception(f"Video status check error: {error} - Details: {json.dumps(error_detail)}")
                except:
                    pass
            raise Exception(f"Video status check error: {error}")

    def generate_speech(self, text, voice_config=None, model_override=None):
        """Generate speech using Gemini TTS"""
        model = model_override or self.models['tts']
        url = f"{self.API_BASE_URL}/{model}{self.endpoints['generateContent']}"
        
        default_voice_config = {
            "prebuilt_voice_config": {
                "voice_name": "Kore"
            }
        }
        
        if voice_config:
            default_voice_config.update(voice_config)
        
        params = {
            "contents": [{
                "parts": [{"text": text}]
            }],
            "generation_config": {
                "response_modalities": ["AUDIO"],
                "speech_config": {
                    "voice_config": default_voice_config
                }
            }
        }

        try:
            response = self.session.post(url, json=self._camelize(params), params={'key': self.API_KEY})
            response.raise_for_status()
            return self._snake_alias(response.json())
        except requests.exceptions.RequestException as error:
            if hasattr(error, 'response') and error.response:
                try:
                    error_detail = error.response.json()
                    raise Exception(f"Gemini TTS error: {error} - Details: {json.dumps(error_detail)}")
                except:
                    pass
            raise Exception(f"Gemini TTS error: {error}")

    def generate_multi_speaker_speech(self, text, speaker_configs):
        """Generate multi-speaker speech"""
        model = self.models['tts']
        url = f"{self.API_BASE_URL}/{model}{self.endpoints['generateContent']}"
        
        params = {
            "contents": [{
                "parts": [{"text": text}]
            }],
            "generation_config": {
                "response_modalities": ["AUDIO"],
                "speech_config": {
                    "multi_speaker_voice_config": {
                        "speaker_voice_configs": speaker_configs
                    }
                }
            }
        }

        try:
            response = self.session.post(url, json=self._camelize(params), params={'key': self.API_KEY})
            response.raise_for_status()
            return self._snake_alias(response.json())
        except requests.exceptions.RequestException as error:
            if hasattr(error, 'response') and error.response:
                try:
                    error_detail = error.response.json()
                    raise Exception(f"Gemini Multi-Speaker TTS error: {error} - Details: {json.dumps(error_detail)}")
                except:
                    pass
            raise Exception(f"Gemini Multi-Speaker TTS error: {error}")

    def upload_file(self, file_path, display_name=None):
        """Upload a file using the Files API"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get file info
        file_size = os.path.getsize(file_path)
        mime_type = self._get_mime_type(file_path)
        display_name = display_name or os.path.basename(file_path)
        
        # Start resumable upload
        headers = {
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(file_size),
            "X-Goog-Upload-Header-Content-Type": mime_type,
            "Content-Type": "application/json"
        }
        
        metadata = {
            "file": {
                "display_name": display_name
            }
        }
        
        try:
            # Initial request
            response = self.session.post(
                self.UPLOAD_BASE_URL,
                headers=headers,
                json=metadata,
                params={'key': self.API_KEY}
            )
            response.raise_for_status()
            
            # Get upload URL
            upload_url = response.headers.get('x-goog-upload-url')
            if not upload_url:
                raise Exception("Upload URL not found in response headers")
            
            # Upload file content
            with open(file_path, 'rb') as f:
                file_content = f.read()
            
            upload_headers = {
                "Content-Length": str(file_size),
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize"
            }
            
            upload_response = self.session.post(
                upload_url,
                headers=upload_headers,
                data=file_content
            )
            upload_response.raise_for_status()
            
            return upload_response.json()
            
        except requests.exceptions.RequestException as error:
            if hasattr(error, 'response') and error.response:
                try:
                    error_detail = error.response.json()
                    raise Exception(f"File upload error: {error} - Details: {json.dumps(error_detail)}")
                except:
                    pass
            raise Exception(f"File upload error: {error}")

    def list_files(self):
        """List uploaded files"""
        try:
            response = self.session.get(self.FILES_BASE_URL, params={'key': self.API_KEY})
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as error:
            if hasattr(error, 'response') and error.response:
                try:
                    error_detail = error.response.json()
                    raise Exception(f"List files error: {error} - Details: {json.dumps(error_detail)}")
                except:
                    pass
            raise Exception(f"List files error: {error}")

    def delete_file(self, file_name):
        """Delete an uploaded file"""
        url = f"{self.FILES_BASE_URL}/{file_name}"
        try:
            response = self.session.delete(url, params={'key': self.API_KEY})
            response.raise_for_status()
            return response.json() if response.content else {"status": "deleted"}
        except requests.exceptions.RequestException as error:
            if hasattr(error, 'response') and error.response:
                try:
                    error_detail = error.response.json()
                    raise Exception(f"Delete file error: {error} - Details: {json.dumps(error_detail)}")
                except:
                    pass
            raise Exception(f"Delete file error: {error}")

    def get_embeddings(self, params):
        """Get embeddings for text"""
        model = self.models['embedding']
        url = f"{self.API_BASE_URL}/{model}:embedContent"

        try:
            response = self.session.post(url, json=self._camelize(params), params={'key': self.API_KEY})
            response.raise_for_status()
            return self._snake_alias(response.json())
        except requests.exceptions.RequestException as error:
            if hasattr(error, 'response') and error.response:
                try:
                    error_detail = error.response.json()
                    raise Exception(f"Gemini API error: {error} - Details: {json.dumps(error_detail)}")
                except:
                    pass
            raise Exception(str(error))

    def get_batch_embeddings(self, params):
        """Get batch embeddings for multiple texts"""
        model = self.models['embedding']
        url = f"{self.API_BASE_URL}/{model}:batchEmbedContents"

        # Format according to the documentation
        if "requests" in params:
            batch_params = {
                "requests": [
                    {
                        "model": f"models/{model}",
                        "content": req.get("content", {})
                    } for req in params["requests"]
                ]
            }
        else:
            batch_params = params

        try:
            response = self.session.post(url, json=self._camelize(batch_params), params={'key': self.API_KEY})
            response.raise_for_status()
            data = self._snake_alias(response.json())
            return data.get("embeddings", [])
        except requests.exceptions.RequestException as error:
            if hasattr(error, 'response') and error.response:
                try:
                    error_detail = error.response.json()
                    raise Exception(f"Gemini API error: {error} - Details: {json.dumps(error_detail)}")
                except:
                    pass
            raise Exception(str(error))

    def _get_mime_type(self, file_path):
        """Get MIME type for file"""
        extension = os.path.splitext(file_path)[1].lower()
        mime_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.webp': 'image/webp',
            '.heic': 'image/heic',
            '.heif': 'image/heif',
            '.mp4': 'video/mp4',
            '.mov': 'video/quicktime',
            '.avi': 'video/x-msvideo',
            '.mp3': 'audio/mpeg',
            '.wav': 'audio/wav',
            '.pdf': 'application/pdf',
            '.txt': 'text/plain'
        }
        return mime_types.get(extension, 'application/octet-stream')

    def wait_for_video_completion(self, operation_name, project_id, max_wait_time=300, poll_interval=5):
        """Wait for video generation to complete"""
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            status = self.check_video_generation_status(operation_name, project_id)
            
            if status.get('done', False):
                return status
            
            time.sleep(poll_interval)
        
        raise TimeoutError(f"Video generation did not complete within {max_wait_time} seconds")

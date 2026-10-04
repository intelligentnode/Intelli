"""
Offline tests that lock the legacy GeminiAIWrapper contract after it became a
deprecated facade over GoogleAIWrapper, plus the places that used to build a
GeminiAIWrapper (Chatbot, the remote controllers and the flow agents).

No network: every HTTP call goes to a mocked requests session.
"""
import copy
import json
import os
import tempfile
import unittest
import warnings
from unittest import mock

import requests

from intelli.config import config
from intelli.controller.remote_embed_model import RemoteEmbedModel
from intelli.controller.remote_image_model import RemoteImageModel
from intelli.controller.remote_speech_model import RemoteSpeechModel
from intelli.controller.remote_vision_model import RemoteVisionModel
from intelli.flow.agents.agent import Agent
from intelli.flow.agents.handlers import ImageAgentHandler, SpeechAgentHandler, VisionAgentHandler
from intelli.flow.input.agent_input import ImageAgentInput, TextAgentInput
from intelli.flow.types import AgentTypes
from intelli.function.chatbot import Chatbot
from intelli.model.input.chatbot_input import ChatModelInput
from intelli.model.input.embed_input import EmbedInput
from intelli.model.input.image_input import ImageModelInput
from intelli.model.input.text_speech_input import Text2SpeechInput
from intelli.model.input.vision_input import VisionModelInput
from intelli.wrappers.geminiai_wrapper import GeminiAIWrapper
from intelli.wrappers.googleai_wrapper import GoogleAIError, GoogleAIWrapper

KEY = "AIzaFAKE-compat-key-0123456789"
GEMINI = config["url"]["gemini"]
MODELS = GEMINI["models"]
VERTEX_MODELS = GEMINI["vertex"]["models"]
DEV_MODELS_BASE = GEMINI["base"]  # https://generativelanguage.googleapis.com/v1beta/models
VERTEX_ROOT = "https://aiplatform.googleapis.com/v1beta1"

# Environment variables that can switch GoogleAIWrapper to Vertex AI.
GOOGLE_ENV_VARS = ("GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_GENAI_USE_ENTERPRISE",
                   "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION")

TEXT_RESPONSE = {
    "candidates": [{
        "content": {"role": "model", "parts": [
            {"text": "Hello"},
            {"inlineData": {"mimeType": "image/png", "data": "IMG1"}},
        ]},
        "finishReason": "STOP",
    }],
    "usageMetadata": {"promptTokenCount": 3},
}


class FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, payload=None, status=200, headers=None, lines=None):
        self.payload = payload
        self.status_code = status
        self.headers = headers or {}
        self.lines = lines or []
        self.text = json.dumps(payload) if payload is not None else ""
        self.content = self.text.encode()

    @property
    def ok(self):
        return self.status_code < 400

    def __bool__(self):
        return self.ok

    def json(self):
        if self.payload is None:
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
        return copy.deepcopy(self.payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(
                f"{self.status_code} Client Error: Bad Request for url: https://example.test/x?key={KEY}",
                response=self)

    def iter_lines(self, decode_unicode=False):
        return iter(self.lines)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def api_error_response():
    return FakeResponse({"error": {"code": 400, "message": "bad request", "status": "INVALID_ARGUMENT"}}, status=400)


def sse_lines(*texts):
    lines = []
    for text in texts:
        lines.append("data: " + json.dumps({"candidates": [{"content": {"parts": [{"text": text}]}}]}))
        lines.append("")
    return lines


class CleanGoogleEnv(unittest.TestCase):
    """Removes Vertex-selecting environment variables for the duration of each test."""

    def setUp(self):
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        for var in GOOGLE_ENV_VARS:
            os.environ.pop(var, None)

    @staticmethod
    def facade(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return GeminiAIWrapper(*args, **kwargs)

    @staticmethod
    def mock_post(wrapper, response=None):
        wrapper.session.post = mock.Mock(return_value=response if response is not None else FakeResponse(TEXT_RESPONSE))
        return wrapper.session.post

    @staticmethod
    def sent(sender, index=-1):
        """(url, kwargs) of a recorded session call."""
        call = sender.call_args_list[index]
        return call.args[0], call.kwargs

    @staticmethod
    def deprecation_warnings(caught):
        return [w for w in caught if issubclass(w.category, DeprecationWarning) and "GeminiAIWrapper" in str(w.message)]


# ----------------------------------------------------------------------------
# Deprecation
# ----------------------------------------------------------------------------
class TestGeminiFacadeDeprecation(CleanGoogleEnv):

    def test_construction_warns_deprecated(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            GeminiAIWrapper(KEY)

        found = self.deprecation_warnings(caught)
        self.assertEqual(len(found), 1)
        self.assertIn("GoogleAIWrapper", str(found[0].message))

    def test_warning_points_at_the_caller(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            GeminiAIWrapper(KEY)

        self.assertEqual(os.path.abspath(self.deprecation_warnings(caught)[0].filename), os.path.abspath(__file__))

    def test_docstring_says_deprecated(self):
        self.assertIn("DEPRECATED", GeminiAIWrapper.__doc__)
        self.assertIn("GoogleAIWrapper", GeminiAIWrapper.__doc__)

    def test_chatbot_gemini_does_not_warn(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Chatbot(KEY, "gemini")

        self.assertEqual(self.deprecation_warnings(caught), [])

    def test_controllers_do_not_warn(self):
        builders = {
            "image": lambda: RemoteImageModel(KEY, "gemini"),
            "vision": lambda: RemoteVisionModel(KEY, "gemini"),
            "speech": lambda: RemoteSpeechModel(KEY, "gemini"),
            "embed": lambda: RemoteEmbedModel(KEY, "gemini"),
        }
        for name, build in builders.items():
            with self.subTest(controller=name):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    build()
                self.assertEqual(self.deprecation_warnings(caught), [])


# ----------------------------------------------------------------------------
# Legacy attributes
# ----------------------------------------------------------------------------
class TestGeminiFacadeAttributes(CleanGoogleEnv):

    def test_legacy_url_attributes(self):
        wrapper = self.facade(KEY)

        self.assertEqual(wrapper.API_BASE_URL, GEMINI["base"])
        self.assertEqual(wrapper.UPLOAD_BASE_URL, GEMINI["upload_base"])
        self.assertEqual(wrapper.FILES_BASE_URL, GEMINI["files_base"])
        self.assertEqual(wrapper.VERTEX_BASE_URL, GEMINI["vertex_base"])

    def test_api_key_and_default_timeout(self):
        wrapper = self.facade(KEY)

        self.assertEqual(wrapper.API_KEY, KEY)
        self.assertEqual(wrapper.timeout, 180)
        self.assertEqual(wrapper.google.timeout, 180)

    def test_custom_timeout_is_forwarded(self):
        wrapper = self.facade(KEY, 30)
        post = self.mock_post(wrapper)

        wrapper.generate_content({"contents": [{"parts": [{"text": "hi"}]}]})

        self.assertEqual(wrapper.timeout, 30)
        self.assertEqual(self.sent(post)[1]["timeout"], 30)

    def test_models_is_the_shared_config_dict(self):
        wrapper = self.facade(KEY)

        self.assertIs(wrapper.models, GEMINI["models"])
        self.assertIs(wrapper.endpoints, GEMINI["endpoints"])

    def test_session_is_shared_with_google_wrapper(self):
        wrapper = self.facade(KEY)

        self.assertIsInstance(wrapper.session, requests.Session)
        self.assertEqual(wrapper.session.headers.get("Content-Type"), "application/json")
        self.assertIsInstance(wrapper.google, GoogleAIWrapper)
        self.assertIs(wrapper.google.session, wrapper.session)

    def test_defaults_to_gemini_developer_api(self):
        wrapper = self.facade(KEY)

        self.assertFalse(wrapper.google.vertex)
        self.assertIsNone(wrapper.google.project_id)

    def test_vertex_environment_variable_does_not_switch_the_facade(self):
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"

        wrapper = self.facade(KEY)

        self.assertFalse(wrapper.google.vertex)

    def test_key_maps_match_legacy(self):
        self.assertEqual(GeminiAIWrapper._KEY_MAP, GoogleAIWrapper._KEY_MAP)
        self.assertEqual(GeminiAIWrapper._KEY_MAP["system_instruction"], "systemInstruction")
        self.assertEqual(GeminiAIWrapper._KEY_MAP["inline_data"], "inlineData")
        self.assertEqual(GeminiAIWrapper._REVERSE_KEY_MAP["fileData"], "file_data")

    def test_camelize_converts_known_snake_keys(self):
        wrapper = self.facade(KEY)
        params = {
            "generation_config": {"response_mime_type": "application/json", "response_schema": {"type": "object"}},
            "system_instruction": {"parts": [{"inline_data": {"mime_type": "image/png", "data": "x"}}]},
            "custom_key": 1,
        }

        normalized = wrapper._camelize(params)

        self.assertEqual(normalized["generationConfig"], {"responseMimeType": "application/json",
                                                          "responseSchema": {"type": "object"}})
        self.assertEqual(normalized["systemInstruction"]["parts"][0], {"inlineData": {"mimeType": "image/png", "data": "x"}})
        self.assertEqual(normalized["custom_key"], 1)

    def test_snake_alias_keeps_camel_keys(self):
        wrapper = self.facade(KEY)

        aliased = wrapper._snake_alias({"inlineData": {"mimeType": "audio/wav"}})

        self.assertEqual(aliased["inlineData"]["mimeType"], "audio/wav")
        self.assertEqual(aliased["inline_data"]["mime_type"], "audio/wav")

    def test_get_mime_type_legacy_extensions(self):
        wrapper = self.facade(KEY)
        legacy = {
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
            ".heic": "image/heic", ".heif": "image/heif", ".mp4": "video/mp4", ".mov": "video/quicktime",
            ".avi": "video/x-msvideo", ".mp3": "audio/mpeg", ".wav": "audio/wav", ".pdf": "application/pdf",
            ".txt": "text/plain", ".PNG": "image/png", "": "application/octet-stream",
        }
        for extension, mime in legacy.items():
            with self.subTest(extension=extension):
                self.assertEqual(wrapper._get_mime_type("/tmp/file" + extension), mime)

    def test_api_key_set_after_construction_is_used(self):
        # SUSPECTED REGRESSION: the old GeminiAIWrapper read self.API_KEY on every call, so
        # rotating the key with `wrapper.API_KEY = new_key` worked. The facade copies the key
        # into GoogleAIWrapper once in __init__ (geminiai_wrapper.py:39-44), so the new key is
        # ignored. API_BASE_URL / UPLOAD_BASE_URL / FILES_BASE_URL / timeout set after
        # construction are ignored the same way.
        wrapper = self.facade(KEY)
        post = self.mock_post(wrapper)

        wrapper.API_KEY = "AIzaROTATED-key-987654321"
        wrapper.generate_content({"contents": [{"parts": [{"text": "hi"}]}]})

        self.assertEqual(self.sent(post)[1]["headers"]["x-goog-api-key"], "AIzaROTATED-key-987654321")


# ----------------------------------------------------------------------------
# Legacy requests and return values (Gemini Developer API)
# ----------------------------------------------------------------------------
class TestGeminiFacadeTextAndVision(CleanGoogleEnv):

    def setUp(self):
        super().setUp()
        self.wrapper = self.facade(KEY)
        self.post = self.mock_post(self.wrapper)

    def test_generate_content_url_auth_and_body(self):
        params = {
            "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
            "generation_config": {"temperature": 0.2},
            "system_instruction": {"parts": [{"text": "be brief"}]},
        }

        self.wrapper.generate_content(params)

        url, kwargs = self.sent(self.post)
        self.assertEqual(url, f"{DEV_MODELS_BASE}/{MODELS['text']}:generateContent")
        self.assertEqual(kwargs["json"], {
            "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
            "generationConfig": {"temperature": 0.2},
            "systemInstruction": {"parts": [{"text": "be brief"}]},
        })
        self.assertEqual(kwargs["headers"]["x-goog-api-key"], KEY)
        self.assertNotIn("key", kwargs.get("params") or {})
        self.assertEqual(kwargs["timeout"], 180)

    def test_generate_content_returns_json_with_snake_aliases(self):
        result = self.wrapper.generate_content({"contents": [{"parts": [{"text": "Hi"}]}]})

        part = result["candidates"][0]["content"]["parts"][1]
        self.assertEqual(result["candidates"][0]["content"]["parts"][0]["text"], "Hello")
        self.assertEqual(part["inlineData"]["data"], "IMG1")
        self.assertEqual(part["inline_data"]["mime_type"], "image/png")

    def test_generate_content_vision_uses_vision_model(self):
        self.wrapper.generate_content({"contents": [{"parts": [{"text": "Hi"}]}]}, vision=True)

        self.assertEqual(self.sent(self.post)[0], f"{DEV_MODELS_BASE}/{MODELS['vision']}:generateContent")

    def test_generate_content_model_override(self):
        self.wrapper.generate_content({"contents": [{"parts": [{"text": "Hi"}]}]}, True, "gemini-3.8-flash")

        self.assertEqual(self.sent(self.post)[0], f"{DEV_MODELS_BASE}/gemini-3.8-flash:generateContent")

    def test_generate_content_keeps_body_model_key_on_developer_api(self):
        # Vision inputs carry 'model' in the body; the old wrapper sent it unchanged.
        self.wrapper.generate_content({"contents": [{"parts": [{"text": "Hi"}]}], "model": "gemini-x"}, True, "gemini-x")

        self.assertEqual(self.sent(self.post)[1]["json"]["model"], "gemini-x")

    def test_generate_content_with_system_instructions(self):
        self.wrapper.generate_content_with_system_instructions([{"text": "hi"}], "be kind", model_override="gemini-m")

        url, kwargs = self.sent(self.post)
        self.assertEqual(url, f"{DEV_MODELS_BASE}/gemini-m:generateContent")
        self.assertEqual(kwargs["json"], {"contents": [{"parts": [{"text": "hi"}]}],
                                          "systemInstruction": {"parts": [{"text": "be kind"}]}})

    def test_generate_content_with_system_instructions_without_system(self):
        self.wrapper.generate_content_with_system_instructions([{"text": "hi"}])

        self.assertEqual(self.sent(self.post)[1]["json"], {"contents": [{"parts": [{"text": "hi"}]}]})

    def test_generate_structured_content_body(self):
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}

        self.wrapper.generate_structured_content(
            [{"text": "q"}], schema, system_instruction="strict", generation_config={"temperature": 0},
            tools=[{"googleSearch": {}}], tool_config={"functionCallingConfig": {"mode": "AUTO"}})

        body = self.sent(self.post)[1]["json"]
        self.assertEqual(body["contents"], [{"parts": [{"text": "q"}]}])
        self.assertEqual(body["generationConfig"], {"temperature": 0, "responseMimeType": "application/json",
                                                    "responseSchema": schema})
        self.assertEqual(body["systemInstruction"], {"parts": [{"text": "strict"}]})
        self.assertEqual(body["tools"], [{"googleSearch": {}}])
        self.assertEqual(body["toolConfig"], {"functionCallingConfig": {"mode": "AUTO"}})

    def test_generate_structured_content_custom_mime_type(self):
        self.wrapper.generate_structured_content([{"text": "q"}], {"type": "string"}, None, None, "text/x.enum")

        self.assertEqual(self.sent(self.post)[1]["json"]["generationConfig"]["responseMimeType"], "text/x.enum")

    def test_stream_generate_content_yields_raw_lines(self):
        lines = ['[{"candidates": [{"content": {"parts": [{"text": "Hel"}]}}]}', "", ',{"candidates": []}]']
        self.post.return_value = FakeResponse({}, lines=lines)

        chunks = list(self.wrapper.stream_generate_content({"contents": [{"parts": [{"text": "Hi"}]}]}))

        url, kwargs = self.sent(self.post)
        self.assertEqual(chunks, [lines[0], lines[2]])
        self.assertEqual(url, f"{DEV_MODELS_BASE}/{MODELS['text']}:streamGenerateContent")
        self.assertTrue(kwargs["stream"])
        self.assertIsNone(kwargs.get("params"))

    def test_stream_generate_content_is_lazy(self):
        stream = self.wrapper.stream_generate_content({"contents": []}, True, "gemini-v")

        self.post.assert_not_called()
        self.post.return_value = FakeResponse({}, lines=["x"])
        self.assertEqual(list(stream), ["x"])
        self.assertEqual(self.sent(self.post)[0], f"{DEV_MODELS_BASE}/gemini-v:streamGenerateContent")

    def test_image_to_text(self):
        self.wrapper.image_to_text("describe", "BASE64", "jpeg")

        url, kwargs = self.sent(self.post)
        self.assertEqual(url, f"{DEV_MODELS_BASE}/{MODELS['vision']}:generateContent")
        self.assertEqual(kwargs["json"], {"contents": [{"parts": [
            {"text": "describe"}, {"inlineData": {"mimeType": "image/jpeg", "data": "BASE64"}}]}]})

    def test_image_to_text_params_model_override(self):
        self.wrapper.image_to_text_params({"contents": [{"parts": [{"text": "x"}]}]}, model_override="gemini-v2")

        self.assertEqual(self.sent(self.post)[0], f"{DEV_MODELS_BASE}/gemini-v2:generateContent")

    def test_image_to_text_with_file_uri(self):
        uri = "https://generativelanguage.googleapis.com/v1beta/files/abc"

        self.wrapper.image_to_text_with_file_uri("describe", uri, "image/png")

        self.assertEqual(self.sent(self.post)[1]["json"]["contents"][0]["parts"][1],
                         {"fileData": {"mimeType": "image/png", "fileUri": uri}})

    def test_multiple_images_to_text(self):
        self.wrapper.multiple_images_to_text("compare", [
            {"mime_type": "image/png", "data": "AAA"},
            {"mime_type": "image/jpeg", "file_uri": "gs://bucket/x.jpg"},
        ])

        url, kwargs = self.sent(self.post)
        self.assertEqual(url, f"{DEV_MODELS_BASE}/{MODELS['vision']}:generateContent")
        self.assertEqual(kwargs["json"]["contents"][0]["parts"], [
            {"text": "compare"},
            {"inlineData": {"mimeType": "image/png", "data": "AAA"}},
            {"fileData": {"mimeType": "image/jpeg", "fileUri": "gs://bucket/x.jpg"}},
        ])

    def test_get_bounding_boxes_prompt(self):
        self.wrapper.get_bounding_boxes("find cats", "DATA", "png")

        text = self.sent(self.post)[1]["json"]["contents"][0]["parts"][0]["text"]
        self.assertEqual(text, "find cats. Return bounding boxes in [ymin, xmin, ymax, xmax] format normalized to 0-1000.")

    def test_get_image_segmentation_prompt(self):
        self.wrapper.get_image_segmentation("segment the dogs", "DATA", "png")

        text = self.sent(self.post)[1]["json"]["contents"][0]["parts"][0]["text"]
        self.assertIn("segment the dogs", text)
        self.assertIn('"box_2d"', text)
        self.assertIn('"mask"', text)


class TestGeminiFacadeMedia(CleanGoogleEnv):

    def setUp(self):
        super().setUp()
        self.wrapper = self.facade(KEY)
        self.post = self.mock_post(self.wrapper)

    def test_generate_image_defaults(self):
        self.wrapper.generate_image("a cat")

        url, kwargs = self.sent(self.post)
        self.assertEqual(url, f"{DEV_MODELS_BASE}/{MODELS['image_generation']}:generateContent")
        self.assertEqual(kwargs["json"], {"contents": [{"parts": [{"text": "a cat"}]}],
                                          "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]}})

    def test_generate_image_config_and_model_override(self):
        self.wrapper.generate_image("a cat", {"imageConfig": {"aspectRatio": "16:9"}}, "gemini-img")

        url, kwargs = self.sent(self.post)
        self.assertEqual(url, f"{DEV_MODELS_BASE}/gemini-img:generateContent")
        self.assertEqual(kwargs["json"]["generationConfig"], {"responseModalities": ["TEXT", "IMAGE"],
                                                              "imageConfig": {"aspectRatio": "16:9"}})

    def test_generate_image_returns_aliased_json(self):
        result = self.wrapper.generate_image("a cat")

        self.assertEqual(result["candidates"][0]["content"]["parts"][1]["inline_data"]["data"], "IMG1")

    def test_generate_speech_defaults(self):
        self.wrapper.generate_speech("Hello")

        url, kwargs = self.sent(self.post)
        self.assertEqual(url, f"{DEV_MODELS_BASE}/{MODELS['tts']}:generateContent")
        self.assertEqual(kwargs["json"], {
            "contents": [{"parts": [{"text": "Hello"}]}],
            "generationConfig": {"responseModalities": ["AUDIO"], "speechConfig": {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}}},
        })

    def test_generate_speech_voice_and_model_override(self):
        self.wrapper.generate_speech("Hello", {"prebuilt_voice_config": {"voice_name": "Puck"}}, "gemini-2.5-pro-preview-tts")

        url, kwargs = self.sent(self.post)
        self.assertEqual(url, f"{DEV_MODELS_BASE}/gemini-2.5-pro-preview-tts:generateContent")
        voice = kwargs["json"]["generationConfig"]["speechConfig"]["voiceConfig"]
        self.assertEqual(voice, {"prebuiltVoiceConfig": {"voiceName": "Puck"}})

    def test_generate_multi_speaker_speech(self):
        speakers = [{"speaker": "A", "voice_config": {"prebuilt_voice_config": {"voice_name": "Kore"}}}]

        self.wrapper.generate_multi_speaker_speech("A: hi", speakers)

        url, kwargs = self.sent(self.post)
        self.assertEqual(url, f"{DEV_MODELS_BASE}/{MODELS['tts']}:generateContent")
        self.assertEqual(kwargs["json"]["generationConfig"]["speechConfig"], {"multiSpeakerVoiceConfig": {
            "speakerVoiceConfigs": [{"speaker": "A", "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}}]}})

    def test_generate_video_requires_project_id(self):
        with self.assertRaises(ValueError) as ctx:
            self.wrapper.generate_video("a dog")

        self.assertIn("Project ID is required for video generation", str(ctx.exception))
        self.post.assert_not_called()

    def test_generate_video_request(self):
        self.post.return_value = FakeResponse({"name": "projects/p/locations/us-central1/x/operations/1"})

        result = self.wrapper.generate_video("a dog", {"durationSeconds": 5}, project_id="p")

        url, kwargs = self.sent(self.post)
        self.assertEqual(url, "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/p/locations/us-central1/"
                              f"publishers/google/models/{MODELS['video_generation']}:predictLongRunning")
        self.assertEqual(kwargs["json"], {"instances": [{"prompt": "a dog"}], "parameters": {
            "aspectRatio": "16:9", "personGeneration": "dont_allow", "durationSeconds": 5}})
        self.assertEqual(kwargs["headers"]["x-goog-api-key"], KEY)
        self.assertEqual(result, {"name": "projects/p/locations/us-central1/x/operations/1"})

    def test_check_video_generation_status_requires_project_id(self):
        with self.assertRaises(ValueError) as ctx:
            self.wrapper.check_video_generation_status("projects/p/locations/us-central1/x/operations/1")

        self.assertIn("Project ID is required to check video generation status", str(ctx.exception))

    def test_check_video_generation_status_uses_fetch_predict_operation(self):
        operation = "projects/p/locations/us-central1/publishers/google/models/veo/operations/123"
        self.post.return_value = FakeResponse({"name": operation, "done": True})

        result = self.wrapper.check_video_generation_status(operation, "p")

        url, kwargs = self.sent(self.post)
        self.assertEqual(url, "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/p/locations/us-central1/"
                              "publishers/google/models/veo:fetchPredictOperation")
        self.assertEqual(kwargs["json"], {"operationName": operation})
        self.assertEqual(result, {"name": operation, "done": True})

    def test_wait_for_video_completion_polls_until_done(self):
        self.post.side_effect = [FakeResponse({"done": False}), FakeResponse({"done": True, "response": {}})]

        with mock.patch("time.sleep") as sleep:
            result = self.wrapper.wait_for_video_completion("projects/p/locations/us-central1/m/operations/1", "p",
                                                            max_wait_time=60, poll_interval=2)

        self.assertEqual(result, {"done": True, "response": {}})
        self.assertEqual(self.post.call_count, 2)
        sleep.assert_called_once_with(2)

    def test_wait_for_video_completion_timeout(self):
        with self.assertRaises(TimeoutError):
            self.wrapper.wait_for_video_completion("projects/p/locations/us-central1/m/operations/1", "p", max_wait_time=0)


class TestGeminiFacadeFilesAndEmbeddings(CleanGoogleEnv):

    def setUp(self):
        super().setUp()
        self.wrapper = self.facade(KEY)
        self.post = self.mock_post(self.wrapper)

    def test_upload_file_two_step_resumable_upload(self):
        upload_url = "https://generativelanguage.googleapis.com/upload/v1beta/files?upload_id=XYZ"
        meta = {"file": {"name": "files/abc", "uri": "https://generativelanguage.googleapis.com/v1beta/files/abc"}}
        self.post.side_effect = [FakeResponse({}, headers={"x-goog-upload-url": upload_url}), FakeResponse(meta)]
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "pic.png")
            with open(path, "wb") as file:
                file.write(b"PNGDATA")

            result = self.wrapper.upload_file(path, "My Pic")

        start_url, start = self.sent(self.post, 0)
        finish_url, finish = self.sent(self.post, 1)
        self.assertEqual(start_url, GEMINI["upload_base"])
        self.assertEqual(start["json"], {"file": {"display_name": "My Pic"}})
        self.assertEqual(start["headers"]["X-Goog-Upload-Protocol"], "resumable")
        self.assertEqual(start["headers"]["X-Goog-Upload-Command"], "start")
        self.assertEqual(start["headers"]["X-Goog-Upload-Header-Content-Length"], "7")
        self.assertEqual(start["headers"]["X-Goog-Upload-Header-Content-Type"], "image/png")
        self.assertEqual(finish_url, upload_url)
        self.assertEqual(finish["data"], b"PNGDATA")
        self.assertEqual(finish["headers"]["X-Goog-Upload-Command"], "upload, finalize")
        self.assertEqual(result, meta)

    def test_upload_file_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            self.wrapper.upload_file("/no/such/file.png")

    def test_list_files(self):
        self.wrapper.session.get = mock.Mock(return_value=FakeResponse({"files": [{"name": "files/abc"}]}))

        result = self.wrapper.list_files()

        url, kwargs = self.sent(self.wrapper.session.get)
        self.assertEqual(url, GEMINI["files_base"])
        self.assertEqual(kwargs["headers"]["x-goog-api-key"], KEY)
        self.assertEqual(result, {"files": [{"name": "files/abc"}]})

    def test_delete_file_empty_response(self):
        self.wrapper.session.delete = mock.Mock(return_value=FakeResponse(None))

        result = self.wrapper.delete_file("abc")

        self.assertEqual(self.sent(self.wrapper.session.delete)[0], f"{GEMINI['files_base']}/abc")
        self.assertEqual(result, {"status": "deleted"})

    def test_delete_file_accepts_files_prefix(self):
        self.wrapper.session.delete = mock.Mock(return_value=FakeResponse({}))

        result = self.wrapper.delete_file("files/abc")

        self.assertEqual(self.sent(self.wrapper.session.delete)[0], f"{GEMINI['files_base']}/abc")
        self.assertEqual(result, {})

    def test_get_embeddings_default_model(self):
        self.post.return_value = FakeResponse({"embedding": {"values": [0.1, 0.2]}})

        result = self.wrapper.get_embeddings({"content": {"parts": [{"text": "a"}]}})

        url, kwargs = self.sent(self.post)
        self.assertEqual(url, f"{DEV_MODELS_BASE}/{MODELS['embedding']}:embedContent")
        self.assertEqual(kwargs["json"], {"content": {"parts": [{"text": "a"}]}, "model": f"models/{MODELS['embedding']}"})
        self.assertEqual(result, {"embedding": {"values": [0.1, 0.2]}})

    def test_get_embeddings_explicit_model_with_prefix(self):
        self.wrapper.get_embeddings({"model": "models/text-embedding-004", "content": {"parts": [{"text": "a"}]}})

        url, kwargs = self.sent(self.post)
        self.assertEqual(url, f"{DEV_MODELS_BASE}/text-embedding-004:embedContent")
        self.assertEqual(kwargs["json"]["model"], "models/text-embedding-004")

    def test_get_embeddings_does_not_mutate_input(self):
        params = {"content": {"parts": [{"text": "a"}]}}

        self.wrapper.get_embeddings(params)

        self.assertEqual(params, {"content": {"parts": [{"text": "a"}]}})

    def test_get_batch_embeddings(self):
        self.post.return_value = FakeResponse({"embeddings": [{"values": [1.0]}, {"values": [2.0]}]})

        result = self.wrapper.get_batch_embeddings({"requests": [
            {"content": {"parts": [{"text": "a"}]}, "model": "ignored"},
            {"content": {"parts": [{"text": "b"}]}},
        ]})

        url, kwargs = self.sent(self.post)
        self.assertEqual(url, f"{DEV_MODELS_BASE}/{MODELS['embedding']}:batchEmbedContents")
        self.assertEqual(kwargs["json"], {"requests": [
            {"model": f"models/{MODELS['embedding']}", "content": {"parts": [{"text": "a"}]}},
            {"model": f"models/{MODELS['embedding']}", "content": {"parts": [{"text": "b"}]}},
        ]})
        self.assertEqual(result, [{"values": [1.0]}, {"values": [2.0]}])

    def test_get_batch_embeddings_without_embeddings_returns_empty_list(self):
        self.post.return_value = FakeResponse({})

        self.assertEqual(self.wrapper.get_batch_embeddings({"requests": []}), [])


# ----------------------------------------------------------------------------
# Errors: legacy prefixes, details added, key never leaked
# ----------------------------------------------------------------------------
class TestGeminiFacadeErrors(CleanGoogleEnv):

    def setUp(self):
        super().setUp()
        self.wrapper = self.facade(KEY)
        self.wrapper.session.post = mock.Mock(return_value=api_error_response())
        self.wrapper.session.get = mock.Mock(return_value=api_error_response())
        self.wrapper.session.delete = mock.Mock(return_value=api_error_response())
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.png = os.path.join(self.folder.name, "pic.png")
        with open(self.png, "wb") as file:
            file.write(b"PNG")

    def calls(self):
        params = {"contents": [{"parts": [{"text": "hi"}]}]}
        operation = "projects/p/locations/us-central1/m/operations/1"
        return {
            "Gemini API error": [
                lambda: self.wrapper.generate_content(params),
                lambda: self.wrapper.generate_content_with_system_instructions([{"text": "hi"}], "sys"),
                lambda: self.wrapper.generate_structured_content([{"text": "hi"}], {"type": "string"}),
                lambda: self.wrapper.image_to_text("d", "B64", "png"),
                lambda: self.wrapper.get_embeddings({"content": {"parts": [{"text": "a"}]}}),
                lambda: self.wrapper.get_batch_embeddings({"requests": []}),
            ],
            "Gemini stream error": [lambda: list(self.wrapper.stream_generate_content(params))],
            "Gemini Image Generation error": [lambda: self.wrapper.generate_image("a cat")],
            "Veo Video Generation error": [lambda: self.wrapper.generate_video("a dog", project_id="p")],
            "Video status check error": [lambda: self.wrapper.check_video_generation_status(operation, "p")],
            "Gemini TTS error": [lambda: self.wrapper.generate_speech("hello")],
            "Gemini Multi-Speaker TTS error": [lambda: self.wrapper.generate_multi_speaker_speech("A: hi", [])],
            "File upload error": [lambda: self.wrapper.upload_file(self.png)],
            "List files error": [lambda: self.wrapper.list_files()],
            "Delete file error": [lambda: self.wrapper.delete_file("abc")],
        }

    def test_http_errors_keep_legacy_prefix(self):
        for prefix, calls in self.calls().items():
            for index, call in enumerate(calls):
                with self.subTest(prefix=prefix, call=index):
                    with self.assertRaises(Exception) as ctx:
                        call()
                    self.assertTrue(str(ctx.exception).startswith(prefix + ": 400 Client Error"), str(ctx.exception))

    def test_http_errors_include_details_and_status(self):
        with self.assertRaises(GoogleAIError) as ctx:
            self.wrapper.generate_content({"contents": [{"parts": [{"text": "hi"}]}]})

        self.assertIn("Details:", str(ctx.exception))
        self.assertIn("INVALID_ARGUMENT", str(ctx.exception))
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.details["error"]["code"], 400)

    def test_http_errors_never_contain_the_key(self):
        for prefix, calls in self.calls().items():
            for index, call in enumerate(calls):
                with self.subTest(prefix=prefix, call=index):
                    with self.assertRaises(Exception) as ctx:
                        call()
                    self.assertNotIn(KEY, str(ctx.exception))
                    self.assertIn("key=<redacted>", str(ctx.exception))

    def test_errors_are_plain_exception_subclasses(self):
        self.assertTrue(issubclass(GoogleAIError, Exception))

    def test_connection_error_message(self):
        self.wrapper.session.post = mock.Mock(side_effect=requests.exceptions.ConnectionError("connection refused"))

        with self.assertRaises(Exception) as ctx:
            self.wrapper.generate_content({"contents": [{"parts": [{"text": "hi"}]}]})

        self.assertEqual(str(ctx.exception), "Gemini API error: connection refused")


# ----------------------------------------------------------------------------
# Session sharing and Vertex options
# ----------------------------------------------------------------------------
class TestGeminiFacadeSessionAndVertex(CleanGoogleEnv):

    def test_mocking_facade_session_intercepts_calls(self):
        wrapper = self.facade(KEY)
        wrapper.session.post = mock.Mock(return_value=FakeResponse(TEXT_RESPONSE))
        wrapper.session.get = mock.Mock(return_value=FakeResponse({"files": []}))
        wrapper.session.delete = mock.Mock(return_value=FakeResponse(None))

        wrapper.generate_content({"contents": [{"parts": [{"text": "hi"}]}]})
        wrapper.list_files()
        wrapper.delete_file("abc")

        wrapper.session.post.assert_called_once()
        wrapper.session.get.assert_called_once()
        wrapper.session.delete.assert_called_once()

    def test_vertex_true_uses_vertex_express_defaults(self):
        wrapper = self.facade(KEY, vertex=True)
        post = self.mock_post(wrapper)

        wrapper.generate_content({"contents": [{"parts": [{"text": "hi"}]}]})

        url, kwargs = self.sent(post)
        self.assertTrue(wrapper.google.vertex)
        self.assertEqual(url, f"{VERTEX_ROOT}/publishers/google/models/{VERTEX_MODELS['text']}:generateContent")
        self.assertEqual(kwargs["headers"]["x-goog-api-key"], KEY)
        self.assertEqual(kwargs["json"]["contents"], [{"parts": [{"text": "hi"}], "role": "user"}])

    def test_vertex_model_override_is_kept(self):
        wrapper = self.facade(KEY, vertex=True)
        post = self.mock_post(wrapper)

        wrapper.generate_content({"contents": [{"parts": [{"text": "hi"}]}]}, model_override="gemini-3.5-flash")

        self.assertEqual(self.sent(post)[0], f"{VERTEX_ROOT}/publishers/google/models/gemini-3.5-flash:generateContent")

    def test_vertex_speech_uses_vertex_tts_default(self):
        wrapper = self.facade(KEY, vertex=True)
        post = self.mock_post(wrapper)

        wrapper.generate_speech("Hello")

        self.assertEqual(self.sent(post)[0], f"{VERTEX_ROOT}/publishers/google/models/{VERTEX_MODELS['tts']}:generateContent")

    def test_vertex_project_and_location_select_regional_project_url(self):
        wrapper = self.facade(KEY, vertex=True, project_id="my-proj", location="us-central1")
        post = self.mock_post(wrapper)

        wrapper.generate_content({"contents": [{"parts": [{"text": "hi"}]}]})

        self.assertEqual(self.sent(post)[0],
                         "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/my-proj/locations/us-central1/"
                         f"publishers/google/models/{VERTEX_MODELS['text']}:generateContent")

    def test_vertex_project_uses_global_location_by_default(self):
        wrapper = self.facade(KEY, vertex=True, project_id="my-proj")
        post = self.mock_post(wrapper)

        wrapper.image_to_text("describe", "B64", "png")

        self.assertEqual(self.sent(post)[0],
                         f"{VERTEX_ROOT}/projects/my-proj/locations/global/"
                         f"publishers/google/models/{VERTEX_MODELS['vision']}:generateContent")

    def test_vertex_embeddings_use_predict(self):
        wrapper = self.facade(KEY, vertex=True)
        post = self.mock_post(wrapper, FakeResponse({"predictions": [{"embeddings": {"values": [0.5]}}]}))

        result = wrapper.get_embeddings({"content": {"parts": [{"text": "a"}]}})

        url, kwargs = self.sent(post)
        self.assertEqual(url, f"{VERTEX_ROOT}/publishers/google/models/{VERTEX_MODELS['embedding']}:predict")
        self.assertEqual(kwargs["json"], {"instances": [{"content": "a"}]})
        self.assertEqual(result, {"embedding": {"values": [0.5]}})

    def test_vertex_video_uses_constructor_project(self):
        wrapper = self.facade(KEY, vertex=True, project_id="my-proj")
        post = self.mock_post(wrapper, FakeResponse({"name": "op"}))

        wrapper.generate_video("a dog")

        url, kwargs = self.sent(post)
        self.assertEqual(url, "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/my-proj/locations/"
                              f"us-central1/publishers/google/models/{VERTEX_MODELS['video_generation']}:predictLongRunning")
        self.assertEqual(kwargs["json"]["parameters"], {"aspectRatio": "16:9", "personGeneration": "dont_allow"})

    def test_vertex_files_api_is_not_available(self):
        wrapper = self.facade(KEY, vertex=True)

        with self.assertRaises(NotImplementedError):
            wrapper.list_files()


# ----------------------------------------------------------------------------
# Chatbot
# ----------------------------------------------------------------------------
class TestChatbotGemini(CleanGoogleEnv):

    def chat_input(self, model=None):
        chat_input = ChatModelInput("You are helpful.", model=model)
        chat_input.add_user_message("hi")
        return chat_input

    def test_wrapper_is_google_ai_wrapper_on_developer_api(self):
        chatbot = Chatbot(KEY, "gemini", {"timeout": 42})

        self.assertIsInstance(chatbot.wrapper, GoogleAIWrapper)
        self.assertNotIsInstance(chatbot.wrapper, GeminiAIWrapper)
        self.assertFalse(chatbot.wrapper.vertex)
        self.assertEqual(chatbot.wrapper.timeout, 42)

    def test_options_select_vertex(self):
        chatbot = Chatbot(KEY, "gemini", {"vertex": True, "project_id": "my-proj", "location": "europe-west4"})

        self.assertTrue(chatbot.wrapper.vertex)
        self.assertEqual(chatbot.wrapper.project_id, "my-proj")
        self.assertEqual(chatbot.wrapper.location, "europe-west4")

    def test_vertex_chat_request_url_and_body(self):
        chatbot = Chatbot(KEY, "gemini", {"vertex": True, "project_id": "my-proj", "location": "europe-west4"})
        post = self.mock_post(chatbot.wrapper)

        chatbot.chat(self.chat_input())

        url, kwargs = self.sent(post)
        self.assertEqual(url, "https://europe-west4-aiplatform.googleapis.com/v1beta1/projects/my-proj/locations/"
                              f"europe-west4/publishers/google/models/{VERTEX_MODELS['text']}:generateContent")
        self.assertNotIn("model", kwargs["json"])
        self.assertEqual(kwargs["json"]["contents"][0]["role"], "user")

    def test_vertex_environment_variable_keeps_developer_api(self):
        # SUSPECTED REGRESSION: the old Chatbot always used the Gemini Developer API. Now
        # GoogleAIWrapper.from_options(api_key, {}) honours GOOGLE_GENAI_USE_VERTEXAI
        # (googleai_wrapper.py:646-650), so a user with that variable set (common for the
        # google-genai SDK) silently gets Vertex AI URLs for an AI Studio key. from_options'
        # docstring says "Without these keys the result is ... the Gemini Developer API".
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"

        chatbot = Chatbot(KEY, "gemini")

        self.assertFalse(chatbot.wrapper.vertex)

    def test_chat_returns_text_of_every_candidate(self):
        chatbot = Chatbot(KEY, "gemini")
        response = {"candidates": [
            {"content": {"parts": [{"text": "Hel"}, {"text": "lo"}]}},
            {"content": {"parts": [{"text": "Hi"}, {"inlineData": {"mimeType": "image/png", "data": "x"}}]}},
            {"content": {}, "finishReason": "MAX_TOKENS"},
        ]}
        self.mock_post(chatbot.wrapper, FakeResponse(response))

        self.assertEqual(chatbot.chat(self.chat_input()), ["Hello", "Hi", ""])

    def test_chat_without_candidates_raises(self):
        chatbot = Chatbot(KEY, "gemini")
        self.mock_post(chatbot.wrapper, FakeResponse({"promptFeedback": {"blockReason": "SAFETY"}}))

        with self.assertRaises(Exception) as ctx:
            chatbot.chat(self.chat_input())

        self.assertIn("Error when calling gemini", str(ctx.exception))

    def test_chat_uses_default_model_and_strips_model_from_body(self):
        chatbot = Chatbot(KEY, "gemini")
        post = self.mock_post(chatbot.wrapper)

        chatbot.chat(self.chat_input())

        url, kwargs = self.sent(post)
        self.assertEqual(url, f"{DEV_MODELS_BASE}/{MODELS['text']}:generateContent")
        self.assertNotIn("model", kwargs["json"])
        self.assertEqual(kwargs["json"]["contents"][0]["parts"][0]["text"], "You are helpful.:hi")

    def test_chat_placeholder_model_gemini_uses_default(self):
        chatbot = Chatbot(KEY, "gemini")
        post = self.mock_post(chatbot.wrapper)

        chatbot.chat(self.chat_input(model="gemini"))

        self.assertEqual(self.sent(post)[0], f"{DEV_MODELS_BASE}/{MODELS['text']}:generateContent")

    def test_chat_per_call_model(self):
        chatbot = Chatbot(KEY, "gemini")
        post = self.mock_post(chatbot.wrapper)

        chatbot.chat(self.chat_input(model="gemini-3.8-flash"))

        self.assertEqual(self.sent(post)[0], f"{DEV_MODELS_BASE}/gemini-3.8-flash:generateContent")

    def test_stream_yields_text_from_sse_chunks(self):
        chatbot = Chatbot(KEY, "gemini")
        post = self.mock_post(chatbot.wrapper, FakeResponse({}, lines=sse_lines("Hel", "lo", "")))

        chunks = list(chatbot.stream(self.chat_input(model="gemini-3.8-flash")))

        url, kwargs = self.sent(post)
        self.assertEqual(chunks, ["Hel", "lo"])
        self.assertEqual(url, f"{DEV_MODELS_BASE}/gemini-3.8-flash:streamGenerateContent")
        self.assertEqual(kwargs["params"], {"alt": "sse"})
        self.assertTrue(kwargs["stream"])
        self.assertNotIn("model", kwargs["json"])


# ----------------------------------------------------------------------------
# Controllers
# ----------------------------------------------------------------------------
class TestRemoteImageModelGemini(CleanGoogleEnv):

    def test_builds_google_ai_wrapper(self):
        model = RemoteImageModel(KEY, "gemini", {"timeout": 25})

        self.assertIsInstance(model.provider, GoogleAIWrapper)
        self.assertFalse(model.provider.vertex)
        self.assertEqual(model.provider.timeout, 25)

    def test_options_select_vertex(self):
        model = RemoteImageModel(KEY, "gemini", {"vertex": True, "project_id": "my-proj"})

        self.assertTrue(model.provider.vertex)
        self.assertEqual(model.provider.project_id, "my-proj")

    def test_generate_images_returns_image_parts(self):
        model = RemoteImageModel(KEY, "gemini")
        response = {"candidates": [{"content": {"parts": [
            {"text": "here you go"},
            {"inlineData": {"mimeType": "image/png", "data": "IMG1"}},
            {"inline_data": {"mime_type": "image/jpeg", "data": "IMG2"}},
            {"inlineData": {"mimeType": "audio/wav", "data": "NOT_AN_IMAGE"}},
        ]}}]}
        post = self.mock_post(model.provider, FakeResponse(response))

        images = model.generate_images(ImageModelInput("a cat"))

        url, kwargs = self.sent(post)
        self.assertEqual(images, ["IMG1", "IMG2"])
        self.assertEqual(url, f"{DEV_MODELS_BASE}/{MODELS['image_generation']}:generateContent")
        self.assertEqual(kwargs["json"]["generationConfig"]["responseModalities"], ["TEXT", "IMAGE"])

    def test_generate_images_model_override(self):
        model = RemoteImageModel(KEY, "gemini")
        post = self.mock_post(model.provider)

        model.generate_images(ImageModelInput("a cat", model="gemini-3.1-flash-image"))

        self.assertEqual(self.sent(post)[0], f"{DEV_MODELS_BASE}/gemini-3.1-flash-image:generateContent")


class TestRemoteVisionModelGemini(CleanGoogleEnv):

    def test_builds_google_ai_wrapper_with_options(self):
        model = RemoteVisionModel(KEY, "gemini", {"vertex": True, "project_id": "my-proj", "location": "us"})

        self.assertIsInstance(model.provider_wrapper, GoogleAIWrapper)
        self.assertTrue(model.provider_wrapper.vertex)
        self.assertEqual(model.provider_wrapper.location, "us")

    def test_image_to_text_skips_parts_without_text(self):
        model = RemoteVisionModel(KEY, "gemini")
        response = {"candidates": [{"content": {"parts": [
            {"text": "A cat"}, {"thoughtSignature": "abc"}, {"text": "on a mat"}]}}]}
        post = self.mock_post(model.provider_wrapper, FakeResponse(response))

        text = model.image_to_text(VisionModelInput("describe", image_data="B64", extension="png"))

        url, kwargs = self.sent(post)
        self.assertEqual(text, "A cat on a mat")
        self.assertEqual(url, f"{DEV_MODELS_BASE}/{MODELS['vision']}:generateContent")
        self.assertEqual(kwargs["json"]["contents"][0]["parts"][1], {"inlineData": {"mimeType": "image/png", "data": "B64"}})

    def test_image_to_text_model_override(self):
        model = RemoteVisionModel(KEY, "gemini")
        post = self.mock_post(model.provider_wrapper, FakeResponse({"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}))

        model.image_to_text(VisionModelInput("describe", image_data="B64", model="gemini-3.8-flash"))

        self.assertEqual(self.sent(post)[0], f"{DEV_MODELS_BASE}/gemini-3.8-flash:generateContent")


class TestRemoteSpeechModelGemini(CleanGoogleEnv):

    AUDIO_RESPONSE = {"candidates": [{"content": {"parts": [
        {"inlineData": {"mimeType": "audio/L16;codec=pcm;rate=24000", "data": "PCMDATA"}}]}}]}

    def speech(self, options=None):
        model = RemoteSpeechModel(KEY, "gemini", options)
        post = self.mock_post(model.gemini_wrapper, FakeResponse(self.AUDIO_RESPONSE))
        return model, post

    @staticmethod
    def voice_name(kwargs):
        return kwargs["json"]["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"]

    def test_builds_google_ai_wrapper_with_options(self):
        model = RemoteSpeechModel(KEY, "gemini", {"vertex": True, "project_id": "my-proj"})

        self.assertIsInstance(model.gemini_wrapper, GoogleAIWrapper)
        self.assertTrue(model.gemini_wrapper.vertex)
        self.assertEqual(model.gemini_wrapper.project_id, "my-proj")

    def test_returns_audio_data(self):
        model, _ = self.speech()

        self.assertEqual(model.generate_speech(Text2SpeechInput("Hello")), "PCMDATA")

    def test_openai_defaults_map_to_gemini_defaults(self):
        model, post = self.speech()

        model.generate_speech(Text2SpeechInput("Hello", voice="alloy", model="tts-1"))

        url, kwargs = self.sent(post)
        self.assertEqual(url, f"{DEV_MODELS_BASE}/{MODELS['tts']}:generateContent")
        self.assertEqual(self.voice_name(kwargs), "Kore")

    def test_openai_voice_maps_to_male_gemini_voice(self):
        model, post = self.speech()

        model.generate_speech(Text2SpeechInput("Hello", gender="MALE", voice="onyx"))

        self.assertEqual(self.voice_name(self.sent(post)[1]), "Puck")

    def test_gemini_voice_and_model_are_kept(self):
        model, post = self.speech()

        model.generate_speech(Text2SpeechInput("Hello", voice="Charon", model="gemini-2.5-pro-preview-tts"))

        url, kwargs = self.sent(post)
        self.assertEqual(url, f"{DEV_MODELS_BASE}/gemini-2.5-pro-preview-tts:generateContent")
        self.assertEqual(self.voice_name(kwargs), "Charon")


class TestRemoteEmbedModelGemini(CleanGoogleEnv):

    def test_builds_google_ai_wrapper_with_options(self):
        model = RemoteEmbedModel(KEY, "gemini", {"vertex": True, "project_id": "my-proj", "timeout": 9})

        self.assertIsInstance(model.provider, GoogleAIWrapper)
        self.assertTrue(model.provider.vertex)
        self.assertEqual(model.provider.timeout, 9)

    def test_get_embeddings_request_and_result(self):
        model = RemoteEmbedModel(KEY, "gemini")
        post = self.mock_post(model.provider, FakeResponse({"embedding": {"values": [0.1, 0.2]}}))

        result = model.get_embeddings(EmbedInput(["a", "b"]))

        url, kwargs = self.sent(post)
        self.assertEqual(url, f"{DEV_MODELS_BASE}/{MODELS['embedding']}:embedContent")
        self.assertEqual(kwargs["json"], {"model": f"models/{MODELS['embedding']}",
                                          "content": {"parts": [{"text": "a"}, {"text": "b"}]}})
        self.assertEqual(result, {"embedding": {"values": [0.1, 0.2]}})


# ----------------------------------------------------------------------------
# Flow agents pass options to the image / vision / speech controllers
# ----------------------------------------------------------------------------
class TestFlowAgentsPassOptions(CleanGoogleEnv):

    OPTIONS = {"vertex": True, "project_id": "my-proj"}

    def test_image_handler_passes_options(self):
        with mock.patch("intelli.controller.remote_image_model.RemoteImageModel") as controller:
            controller.return_value.generate_images.return_value = ["IMG"]
            handler = ImageAgentHandler("gemini", "Draw", {"key": KEY}, self.OPTIONS)

            result = handler.execute(TextAgentInput("a cat"), {"key": KEY})

        self.assertEqual(result, "IMG")
        controller.assert_called_once_with(KEY, "gemini", options=self.OPTIONS)

    def test_vision_handler_passes_options(self):
        with mock.patch("intelli.controller.remote_vision_model.RemoteVisionModel") as controller:
            controller.return_value.image_to_text.return_value = "a cat"
            handler = VisionAgentHandler("gemini", "Describe", {"key": KEY, "model": None}, self.OPTIONS)

            result = handler.execute(ImageAgentInput("what is it", "B64"), {"key": KEY, "model": None})

        self.assertEqual(result, "a cat")
        controller.assert_called_once_with(KEY, "gemini", options=self.OPTIONS)

    def test_speech_handler_passes_options(self):
        with mock.patch("intelli.controller.remote_speech_model.RemoteSpeechModel") as controller:
            controller.return_value.generate_speech.return_value = "AUDIO"
            handler = SpeechAgentHandler("gemini", "Say", {"key": KEY}, self.OPTIONS)

            result = handler.execute(TextAgentInput("hello"), {"key": KEY})

        self.assertEqual(result, "AUDIO")
        controller.assert_called_once_with(key_value=KEY, provider="gemini", options=self.OPTIONS)

    def test_agent_legacy_image_path_passes_options(self):
        agent = Agent(AgentTypes.IMAGE.value, "gemini", "Draw", {"key": KEY}, options=self.OPTIONS)
        with mock.patch("intelli.flow.agents.agent.RemoteImageModel") as controller:
            controller.return_value.generate_images.return_value = ["IMG"]

            agent._execute_image_agent(TextAgentInput("a cat"), {"key": KEY})

        controller.assert_called_once_with(KEY, "gemini", options=self.OPTIONS)

    def test_agent_legacy_vision_path_passes_options(self):
        agent = Agent(AgentTypes.VISION.value, "gemini", "Describe", {"key": KEY, "model": None}, options=self.OPTIONS)
        with mock.patch("intelli.flow.agents.agent.RemoteVisionModel") as controller:
            controller.return_value.image_to_text.return_value = "a cat"

            agent._execute_vision_agent(ImageAgentInput("what", "B64"), {"key": KEY, "model": None})

        controller.assert_called_once_with(KEY, "gemini", options=self.OPTIONS)

    def test_agent_legacy_speech_path_passes_options(self):
        agent = Agent(AgentTypes.SPEECH.value, "gemini", "Say", {"key": KEY}, options=self.OPTIONS)
        with mock.patch("intelli.flow.agents.agent.RemoteSpeechModel") as controller, \
                mock.patch("builtins.print"):
            controller.return_value.generate_speech.return_value = "AUDIO"

            agent._execute_speech_agent(TextAgentInput("hello"), {"key": KEY})

        controller.assert_called_once_with(key_value=KEY, provider="gemini", options=self.OPTIONS)

    def test_image_agent_reaches_vertex_project_url(self):
        agent = Agent(AgentTypes.IMAGE.value, "gemini", "Draw", {"key": KEY}, options=self.OPTIONS)
        response = FakeResponse({"candidates": [{"content": {"parts": [
            {"inlineData": {"mimeType": "image/png", "data": "IMG"}}]}}]})
        with mock.patch.object(requests.Session, "post", return_value=response) as post:
            result = agent.execute(TextAgentInput("a cat"))

        self.assertEqual(result, "IMG")
        self.assertEqual(post.call_args.args[0],
                         f"{VERTEX_ROOT}/projects/my-proj/locations/global/publishers/google/models/"
                         f"{VERTEX_MODELS['image_generation']}:generateContent")

    def test_speech_agent_uses_gemini_voice_defaults(self):
        agent = Agent(AgentTypes.SPEECH.value, "gemini", "Say", {"key": KEY})
        response = FakeResponse({"candidates": [{"content": {"parts": [
            {"inlineData": {"mimeType": "audio/L16;rate=24000", "data": "PCM"}}]}}]})
        with mock.patch.object(requests.Session, "post", return_value=response) as post:
            result = agent.execute(TextAgentInput("hello"))

        body = post.call_args.kwargs["json"]
        self.assertEqual(result, "PCM")
        self.assertEqual(post.call_args.args[0], f"{DEV_MODELS_BASE}/{MODELS['tts']}:generateContent")
        self.assertEqual(body["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"], "Kore")


if __name__ == "__main__":
    unittest.main(verbosity=2)

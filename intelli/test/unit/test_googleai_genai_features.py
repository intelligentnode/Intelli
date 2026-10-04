"""
Offline unit tests for the feature methods of GoogleAIWrapper (Gemini Developer API and
Vertex AI / Gemini Enterprise Agent Platform).

Every test runs against a fake requests session (and a fake websockets module for the Live API),
so nothing touches the network. Each test checks the exact URL, HTTP method, headers and JSON
body the wrapper sends for one backend:

- Developer API:    GoogleAIWrapper(key)                                    -> generativelanguage.googleapis.com
- Vertex express:   GoogleAIWrapper(key, vertex=True)                       -> aiplatform.googleapis.com/v1beta1/publishers/...
- Vertex project:   GoogleAIWrapper(key, vertex=True, project_id="my-proj") -> .../projects/my-proj/locations/<loc>/...
"""
import asyncio
import base64
import copy
import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import requests
from requests.structures import CaseInsensitiveDict

from intelli.config import config
from intelli.wrappers.googleai_wrapper import GoogleAIError, GoogleAILiveSession, GoogleAIWrapper

DEV_KEY = "dev-test-key-123"
VERTEX_KEY = "vertex-test-key-456"
PROJECT = "my-proj"

DEV_ROOT = "https://generativelanguage.googleapis.com/v1beta"
GLOBAL_ROOT = "https://aiplatform.googleapis.com/v1beta1"
US_CENTRAL_ROOT = "https://us-central1-aiplatform.googleapis.com/v1beta1"
EUROPE_WEST4_ROOT = "https://europe-west4-aiplatform.googleapis.com/v1beta1"
US_MULTI_ROOT = "https://aiplatform.us.rep.googleapis.com/v1beta1"

PROJECT_GLOBAL = f"projects/{PROJECT}/locations/global"
PROJECT_US_CENTRAL = f"projects/{PROJECT}/locations/us-central1"

DEV_HEADERS = {"Content-Type": "application/json", "x-goog-api-key": DEV_KEY}
VERTEX_HEADERS = {"Content-Type": "application/json", "x-goog-api-key": VERTEX_KEY}

DEFAULT_TIMEOUT = 180

# Environment variables that switch the backend, project or location. Tests clear them.
GENAI_ENV_VARS = ("GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_GENAI_USE_ENTERPRISE",
                  "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION")

_MISSING = object()


def b64(data):
    return base64.b64encode(data).decode("utf-8")


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------
class FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, payload=None, status_code=200, headers=None, lines=None, content=None):
        self._payload = {} if payload is None else payload
        self.status_code = status_code
        self.headers = CaseInsensitiveDict(headers or {})
        self._lines = list(lines or [])
        if content is None:
            content = json.dumps(self._payload).encode("utf-8") if payload is not None else b""
        self.content = content
        self.text = content.decode("utf-8", errors="replace")
        self.closed = False

    def json(self):
        return copy.deepcopy(self._payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error", response=self)

    def iter_lines(self, decode_unicode=False):
        for line in self._lines:
            yield line

    def close(self):
        self.closed = True


class FakeSession:
    """Records every request; returns queued responses in order (an empty JSON object by default)."""

    def __init__(self):
        self.calls = []
        self.responses = []

    def queue(self, *responses):
        self.responses.extend(responses)

    def _send(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        return self.responses.pop(0) if self.responses else FakeResponse({})

    def get(self, url, **kwargs):
        return self._send("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self._send("POST", url, **kwargs)

    def delete(self, url, **kwargs):
        return self._send("DELETE", url, **kwargs)

    def patch(self, url, **kwargs):
        return self._send("PATCH", url, **kwargs)


class FakeClock:
    """Replaces the time module inside googleai_wrapper: sleep() advances time() instantly."""

    def __init__(self, start=1000.0):
        self.now = start
        self.sleeps = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class FakeWebSocket:
    """Records sent messages (as dicts) and serves queued server messages."""

    def __init__(self, incoming=()):
        self.incoming = [m if isinstance(m, str) else json.dumps(m) for m in incoming]
        self.sent = []
        self.closed = False

    async def send(self, message):
        self.sent.append(json.loads(message))

    async def recv(self):
        if not self.incoming:
            raise ConnectionError("connection closed")
        return self.incoming.pop(0)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.incoming:
            raise StopAsyncIteration
        return self.incoming.pop(0)

    async def close(self):
        self.closed = True


def fake_websockets_module(websocket, *, legacy=False, error=None):
    """
    A module that replaces `websockets` in sys.modules. connect() records its arguments.
    legacy=True accepts only extra_headers (websockets < 14), like the old API.
    """
    module = types.ModuleType("websockets")
    module.connect_calls = []

    if legacy:
        async def connect(url, *, extra_headers, max_size, open_timeout):
            module.connect_calls.append({"url": url, "extra_headers": extra_headers,
                                         "max_size": max_size, "open_timeout": open_timeout})
            return websocket
    else:
        async def connect(url, *, additional_headers, max_size, open_timeout):
            module.connect_calls.append({"url": url, "additional_headers": additional_headers,
                                         "max_size": max_size, "open_timeout": open_timeout})
            if error is not None:
                raise error
            return websocket

    module.connect = connect
    return module


# ----------------------------------------------------------------------
# Base test case
# ----------------------------------------------------------------------
class GenAITestCase(unittest.TestCase):

    def setUp(self):
        env_patch = patch.dict(os.environ)
        env_patch.start()
        self.addCleanup(env_patch.stop)
        for name in GENAI_ENV_VARS:
            os.environ.pop(name, None)
        self.session = FakeSession()

    # Wrappers for the three backends, sharing the fake session.
    def dev(self, **kwargs):
        return GoogleAIWrapper(DEV_KEY, session=self.session, **kwargs)

    def express(self, **kwargs):
        return GoogleAIWrapper(VERTEX_KEY, vertex=True, session=self.session, **kwargs)

    def project(self, **kwargs):
        return GoogleAIWrapper(VERTEX_KEY, vertex=True, project_id=PROJECT, session=self.session, **kwargs)

    @property
    def last_call(self):
        self.assertTrue(self.session.calls, "no request was sent")
        return self.session.calls[-1]

    def assertCall(self, method, url, headers, body=_MISSING, params=_MISSING, call=None, stream=False):
        """Exact method, URL and headers; exact JSON body / query params (or their absence)."""
        call = self.last_call if call is None else call
        self.assertEqual(call["method"], method)
        self.assertEqual(call["url"], url)
        self.assertEqual(call["headers"], headers)
        self.assertEqual(call["timeout"], DEFAULT_TIMEOUT)
        if body is _MISSING:
            self.assertNotIn("json", call)
        else:
            self.assertEqual(call["json"], body)
        if params is _MISSING:
            self.assertNotIn("params", call)
        else:
            self.assertEqual(call["params"], params)
        if stream:
            self.assertTrue(call.get("stream"))
        else:
            self.assertNotIn("stream", call)
        return call


# ----------------------------------------------------------------------
# Gemini native image generation and editing
# ----------------------------------------------------------------------
class TestGeminiImageGeneration(GenAITestCase):

    def test_generate_image_developer_api(self):
        self.dev().generate_image("a red fox")

        self.assertCall("POST", f"{DEV_ROOT}/models/gemini-2.5-flash-image:generateContent", DEV_HEADERS, {
            "contents": [{"parts": [{"text": "a red fox"}]}],
            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
        })

    def test_generate_image_vertex_express_uses_vertex_default_model_and_user_role(self):
        self.express().generate_image("a red fox")

        self.assertCall("POST", f"{GLOBAL_ROOT}/publishers/google/models/gemini-3.1-flash-image:generateContent",
                        VERTEX_HEADERS, {
                            "contents": [{"parts": [{"text": "a red fox"}], "role": "user"}],
                            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
                        })

    def test_generate_image_vertex_project_url(self):
        self.project().generate_image("a red fox", model_override="gemini-3-pro-image")

        self.assertEqual(self.last_call["url"],
                         f"{GLOBAL_ROOT}/{PROJECT_GLOBAL}/publishers/google/models/gemini-3-pro-image:generateContent")

    def test_generate_image_model_override(self):
        self.dev().generate_image("a red fox", model_override="gemini-3.1-flash-image")

        self.assertEqual(self.last_call["url"], f"{DEV_ROOT}/models/gemini-3.1-flash-image:generateContent")

    def test_generate_image_config_params_are_merged_into_generation_config(self):
        self.dev().generate_image("a red fox", {"imageConfig": {"aspectRatio": "16:9", "imageSize": "2K"}})

        self.assertEqual(self.last_call["json"]["generationConfig"], {
            "responseModalities": ["TEXT", "IMAGE"],
            "imageConfig": {"aspectRatio": "16:9", "imageSize": "2K"},
        })

    def test_generate_image_config_params_can_override_modalities(self):
        self.dev().generate_image("a red fox", {"responseModalities": ["IMAGE"]})

        self.assertEqual(self.last_call["json"]["generationConfig"], {"responseModalities": ["IMAGE"]})

    def test_generate_image_with_images_adds_media_parts_for_editing(self):
        inline_part = {"inlineData": {"mimeType": "image/webp", "data": "UklGRg=="}}

        self.dev().generate_image("Add a hat", images=[(b"\x89PNG", "image/png"), "gs://bucket/cat.jpg",
                                                       inline_part])

        self.assertEqual(self.last_call["json"]["contents"], [{"parts": [
            {"text": "Add a hat"},
            {"inlineData": {"mimeType": "image/png", "data": b64(b"\x89PNG")}},
            {"fileData": {"mimeType": "image/jpeg", "fileUri": "gs://bucket/cat.jpg"}},
            inline_part,
        ]}])

    def test_generate_image_with_local_image_path(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "photo.jpg")
            with open(path, "wb") as file:
                file.write(b"jpeg-bytes")

            self.dev().generate_image("Make it sunny", images=[path])

        self.assertEqual(self.last_call["json"]["contents"][0]["parts"][1],
                         {"inlineData": {"mimeType": "image/jpeg", "data": b64(b"jpeg-bytes")}})

    def test_generate_image_response_gets_snake_case_aliases(self):
        self.session.queue(FakeResponse({"candidates": [{"content": {"parts": [
            {"text": "Here it is"},
            {"inlineData": {"mimeType": "image/png", "data": "iVBORw0KGgo="}},
        ]}}]}))

        result = self.dev().generate_image("a red fox")

        image_part = result["candidates"][0]["content"]["parts"][1]
        self.assertEqual(image_part["inlineData"]["data"], "iVBORw0KGgo=")
        self.assertEqual(image_part["inline_data"]["mime_type"], "image/png")
        self.assertEqual(GoogleAIWrapper.extract_images(result),
                         [{"mime_type": "image/png", "data": "iVBORw0KGgo="}])

    def test_edit_image_wraps_a_single_image(self):
        self.dev().edit_image("Make it blue", (b"img", "image/png"))

        self.assertCall("POST", f"{DEV_ROOT}/models/gemini-2.5-flash-image:generateContent", DEV_HEADERS, {
            "contents": [{"parts": [{"text": "Make it blue"},
                                    {"inlineData": {"mimeType": "image/png", "data": b64(b"img")}}]}],
            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
        })

    def test_edit_image_with_several_images_config_and_model(self):
        self.express().edit_image("Combine them", ["gs://b/one.png", "gs://b/two.png"],
                                  {"imageConfig": {"aspectRatio": "1:1"}}, "gemini-3-pro-image")

        self.assertCall("POST", f"{GLOBAL_ROOT}/publishers/google/models/gemini-3-pro-image:generateContent",
                        VERTEX_HEADERS, {
                            "contents": [{"role": "user", "parts": [
                                {"text": "Combine them"},
                                {"fileData": {"mimeType": "image/png", "fileUri": "gs://b/one.png"}},
                                {"fileData": {"mimeType": "image/png", "fileUri": "gs://b/two.png"}},
                            ]}],
                            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"],
                                                 "imageConfig": {"aspectRatio": "1:1"}},
                        })

    def test_generate_image_error_prefix(self):
        self.session.queue(FakeResponse({"error": {"message": "blocked"}}, status_code=400))

        with self.assertRaises(GoogleAIError) as caught:
            self.dev().generate_image("a red fox")

        self.assertTrue(str(caught.exception).startswith("Gemini Image Generation error"))
        self.assertEqual(caught.exception.status_code, 400)


# ----------------------------------------------------------------------
# Imagen: generate, edit, upscale
# ----------------------------------------------------------------------
class TestImagen(GenAITestCase):
    # Google retired the Imagen models on 2026-06-30, so config has no Imagen defaults any more.
    # These tests set defaults the way a project that still has Imagen access would.
    IMAGEN_DEFAULTS = {"imagen": "imagen-4.0-generate-001", "imagen_edit": "imagen-3.0-capability-001",
                       "imagen_upscale": "imagen-4.0-upscale-preview"}

    def setUp(self):
        super().setUp()
        for models in (config["url"]["gemini"]["models"], config["url"]["gemini"]["vertex"]["models"]):
            patcher = patch.dict(models, self.IMAGEN_DEFAULTS)
            patcher.start()
            self.addCleanup(patcher.stop)


    def test_imagen_generate_images_express_uses_capability_location(self):
        response = {"predictions": [{"bytesBase64Encoded": "AAA", "mimeType": "image/png"}]}
        self.session.queue(FakeResponse(response))

        result = self.express().imagen_generate_images("a lighthouse", number_of_images=2, aspect_ratio="1:1",
                                                       negative_prompt="blur",
                                                       parameters={"personGeneration": "allow_adult"})

        self.assertCall("POST", f"{US_CENTRAL_ROOT}/publishers/google/models/imagen-4.0-generate-001:predict",
                        VERTEX_HEADERS, {
                            "instances": [{"prompt": "a lighthouse"}],
                            "parameters": {"sampleCount": 2, "aspectRatio": "1:1", "negativePrompt": "blur",
                                           "personGeneration": "allow_adult"},
                        })
        # Predictions are returned as sent by the API (no snake_case aliases).
        self.assertEqual(result, response)
        self.assertEqual(GoogleAIWrapper.extract_images(result), [{"mime_type": "image/png", "data": "AAA"}])

    def test_imagen_generate_images_project_uses_us_central1(self):
        self.project().imagen_generate_images("a lighthouse")

        self.assertCall("POST",
                        f"{US_CENTRAL_ROOT}/{PROJECT_US_CENTRAL}/publishers/google/models/"
                        "imagen-4.0-generate-001:predict",
                        VERTEX_HEADERS, {"instances": [{"prompt": "a lighthouse"}], "parameters": {"sampleCount": 1}})

    def test_imagen_explicit_wrapper_location_wins_over_capability_location(self):
        self.project(location="europe-west4").imagen_generate_images("a lighthouse")

        self.assertEqual(self.last_call["url"],
                         f"{EUROPE_WEST4_ROOT}/projects/{PROJECT}/locations/europe-west4/publishers/google/models/"
                         "imagen-4.0-generate-001:predict")

    def test_imagen_generate_images_model_override(self):
        self.express().imagen_generate_images("a lighthouse", model="imagen-4.0-ultra-generate-001")

        self.assertEqual(self.last_call["url"],
                         f"{US_CENTRAL_ROOT}/publishers/google/models/imagen-4.0-ultra-generate-001:predict")

    def test_imagen_generate_images_developer_api(self):
        self.dev().imagen_generate_images("a lighthouse")

        self.assertCall("POST", f"{DEV_ROOT}/models/imagen-4.0-generate-001:predict", DEV_HEADERS,
                        {"instances": [{"prompt": "a lighthouse"}], "parameters": {"sampleCount": 1}})

    def test_imagen_edit_image_with_mask_image(self):
        self.express().imagen_edit_image("add a boat", b"raw-image", b"mask-image")

        self.assertCall("POST", f"{US_CENTRAL_ROOT}/publishers/google/models/imagen-3.0-capability-001:predict",
                        VERTEX_HEADERS, {
                            "instances": [{"prompt": "add a boat", "referenceImages": [
                                {"referenceType": "REFERENCE_TYPE_RAW", "referenceId": 1,
                                 "referenceImage": {"bytesBase64Encoded": b64(b"raw-image")}},
                                {"referenceType": "REFERENCE_TYPE_MASK", "referenceId": 2,
                                 "referenceImage": {"bytesBase64Encoded": b64(b"mask-image")},
                                 "maskImageConfig": {"maskMode": "MASK_MODE_USER_PROVIDED", "dilation": 0.01}},
                            ]}],
                            "parameters": {"sampleCount": 1, "editMode": "EDIT_MODE_INPAINT_INSERTION"},
                        })

    def test_imagen_edit_image_with_automatic_mask_mode_and_edit_mode(self):
        self.project().imagen_edit_image("a beach behind the car", "gs://bucket/car.png",
                                         mask_mode="MASK_MODE_BACKGROUND", edit_mode="EDIT_MODE_BGSWAP")

        self.assertCall("POST",
                        f"{US_CENTRAL_ROOT}/{PROJECT_US_CENTRAL}/publishers/google/models/"
                        "imagen-3.0-capability-001:predict",
                        VERTEX_HEADERS, {
                            "instances": [{"prompt": "a beach behind the car", "referenceImages": [
                                {"referenceType": "REFERENCE_TYPE_RAW", "referenceId": 1,
                                 "referenceImage": {"gcsUri": "gs://bucket/car.png", "mimeType": "image/png"}},
                                {"referenceType": "REFERENCE_TYPE_MASK", "referenceId": 2,
                                 "maskImageConfig": {"maskMode": "MASK_MODE_BACKGROUND", "dilation": 0.01}},
                            ]}],
                            "parameters": {"sampleCount": 1, "editMode": "EDIT_MODE_BGSWAP"},
                        })

    def test_imagen_edit_image_mask_mode_defaults_to_inpaint_insertion(self):
        self.express().imagen_edit_image("remove the person", b"raw", mask_mode="MASK_MODE_FOREGROUND")

        self.assertEqual(self.last_call["json"]["parameters"]["editMode"], "EDIT_MODE_INPAINT_INSERTION")

    def test_imagen_edit_image_without_mask_uses_default_edit_mode(self):
        self.express().imagen_edit_image("make it a watercolor", "QUJD", parameters={"sampleCount": 3})

        body = self.last_call["json"]
        self.assertEqual(body["instances"][0]["referenceImages"], [
            {"referenceType": "REFERENCE_TYPE_RAW", "referenceId": 1, "referenceImage": {"bytesBase64Encoded": "QUJD"}},
        ])
        self.assertEqual(body["parameters"], {"sampleCount": 3, "editMode": "EDIT_MODE_DEFAULT"})

    def test_imagen_edit_image_reads_local_path_with_mime_type(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "photo.jpg")
            with open(path, "wb") as file:
                file.write(b"jpeg-bytes")

            self.express().imagen_edit_image("sharpen", path)

        self.assertEqual(self.last_call["json"]["instances"][0]["referenceImages"][0]["referenceImage"],
                         {"bytesBase64Encoded": b64(b"jpeg-bytes"), "mimeType": "image/jpeg"})

    def test_imagen_upscale_image_express(self):
        self.express().imagen_upscale_image(b"small", "x4")

        self.assertCall("POST", f"{US_CENTRAL_ROOT}/publishers/google/models/imagen-4.0-upscale-preview:predict",
                        VERTEX_HEADERS, {
                            "instances": [{"prompt": "Upscale the image",
                                           "image": {"bytesBase64Encoded": b64(b"small")}}],
                            "parameters": {"mode": "upscale", "sampleCount": 1,
                                           "upscaleConfig": {"upscaleFactor": "x4"}},
                        })

    def test_imagen_upscale_image_project_default_factor_and_parameters(self):
        self.project().imagen_upscale_image("gs://bucket/small.png",
                                            parameters={"outputOptions": {"mimeType": "image/jpeg"}})

        self.assertCall("POST",
                        f"{US_CENTRAL_ROOT}/{PROJECT_US_CENTRAL}/publishers/google/models/"
                        "imagen-4.0-upscale-preview:predict",
                        VERTEX_HEADERS, {
                            "instances": [{"prompt": "Upscale the image",
                                           "image": {"gcsUri": "gs://bucket/small.png", "mimeType": "image/png"}}],
                            "parameters": {"mode": "upscale", "sampleCount": 1, "upscaleConfig": {"upscaleFactor": "x2"},
                                           "outputOptions": {"mimeType": "image/jpeg"}},
                        })

    def test_imagen_error_prefix_and_status(self):
        self.session.queue(FakeResponse({"error": {"status": "RESOURCE_EXHAUSTED"}}, status_code=429))

        with self.assertRaises(GoogleAIError) as caught:
            self.express().imagen_generate_images("a lighthouse")

        self.assertTrue(str(caught.exception).startswith("Imagen error: 429"))
        self.assertEqual(caught.exception.status_code, 429)
        self.assertEqual(caught.exception.details, {"error": {"status": "RESOURCE_EXHAUSTED"}})


# ----------------------------------------------------------------------
# Veo video generation
# ----------------------------------------------------------------------
class TestVeoVideoGeneration(GenAITestCase):

    def test_generate_video_vertex_project_defaults_to_us_central1(self):
        self.session.queue(FakeResponse({"name": "projects/my-proj/locations/us-central1/operations/1"}))

        result = self.project().generate_video("a cat surfing", {"durationSeconds": 4})

        self.assertCall("POST",
                        f"{US_CENTRAL_ROOT}/{PROJECT_US_CENTRAL}/publishers/google/models/"
                        "veo-3.1-fast-generate-001:predictLongRunning",
                        VERTEX_HEADERS, {
                            "instances": [{"prompt": "a cat surfing"}],
                            "parameters": {"aspectRatio": "16:9", "durationSeconds": 4},
                        })
        self.assertEqual(result, {"name": "projects/my-proj/locations/us-central1/operations/1"})

    def test_generate_video_explicit_location_argument(self):
        self.project().generate_video("a cat surfing", location="europe-west4")

        self.assertEqual(self.last_call["url"],
                         f"{EUROPE_WEST4_ROOT}/projects/{PROJECT}/locations/europe-west4/publishers/google/models/"
                         "veo-3.1-fast-generate-001:predictLongRunning")

    def test_generate_video_explicit_multi_region_wrapper_location(self):
        self.project(location="us").generate_video("a cat surfing")

        self.assertEqual(self.last_call["url"],
                         f"{US_MULTI_ROOT}/projects/{PROJECT}/locations/us/publishers/google/models/"
                         "veo-3.1-fast-generate-001:predictLongRunning")

    def test_generate_video_with_image_and_last_frame(self):
        self.project().generate_video("a sunrise timelapse", image=b"first-frame", last_frame="gs://bucket/last.png",
                                      model="veo-3.1-generate-001")

        self.assertEqual(self.last_call["url"],
                         f"{US_CENTRAL_ROOT}/{PROJECT_US_CENTRAL}/publishers/google/models/"
                         "veo-3.1-generate-001:predictLongRunning")
        self.assertEqual(self.last_call["json"]["instances"], [{
            "prompt": "a sunrise timelapse",
            "image": {"bytesBase64Encoded": b64(b"first-frame"), "mimeType": "image/png"},
            "lastFrame": {"gcsUri": "gs://bucket/last.png", "mimeType": "image/png"},
        }])

    def test_generate_video_image_dict_is_sent_unchanged(self):
        image = {"gcsUri": "gs://bucket/first.jpg", "mimeType": "image/jpeg"}

        self.project().generate_video("a sunrise timelapse", image=image)

        self.assertEqual(self.last_call["json"]["instances"][0]["image"], image)

    def test_generate_video_project_argument_overrides_wrapper_project(self):
        self.project().generate_video("a cat surfing", project_id="other-proj")

        self.assertEqual(self.last_call["url"],
                         f"{US_CENTRAL_ROOT}/projects/other-proj/locations/us-central1/publishers/google/models/"
                         "veo-3.1-fast-generate-001:predictLongRunning")

    def test_generate_video_project_argument_on_developer_wrapper_uses_vertex(self):
        # Legacy GeminiAIWrapper behavior: a project_id argument sends the request to Vertex AI
        # with the config Veo default (veo-3.1-fast-generate-001; veo-2.0 now returns 404) and the same API key.
        self.dev().generate_video("a cat surfing", None, "legacy-proj")

        self.assertCall("POST",
                        f"{US_CENTRAL_ROOT}/projects/legacy-proj/locations/us-central1/publishers/google/models/"
                        "veo-3.1-fast-generate-001:predictLongRunning",
                        DEV_HEADERS, {
                            "instances": [{"prompt": "a cat surfing"}],
                            "parameters": {"aspectRatio": "16:9"},
                        })

    def test_generate_video_vertex_api_key_without_project_raises_value_error(self):
        with self.assertRaises(ValueError) as caught:
            self.express().generate_video("a cat surfing")

        self.assertIn("Project ID is required", str(caught.exception))
        self.assertEqual(self.session.calls, [])

    def test_generate_video_vertex_oauth_without_project_raises_google_ai_error(self):
        wrapper = GoogleAIWrapper(vertex=True, access_token="ya29.test-token", session=self.session)

        with self.assertRaises(GoogleAIError) as caught:
            wrapper.generate_video("a cat surfing")

        self.assertIn("needs a Google Cloud project", str(caught.exception))
        self.assertEqual(self.session.calls, [])

    def test_generate_video_developer_api_predict_long_running(self):
        self.dev().generate_video("a cat surfing", {"durationSeconds": 8, "negativePrompt": "rain"})

        self.assertCall("POST", f"{DEV_ROOT}/models/veo-3.1-fast-generate-preview:predictLongRunning", DEV_HEADERS, {
            "instances": [{"prompt": "a cat surfing"}],
            "parameters": {"aspectRatio": "16:9", "durationSeconds": 8, "negativePrompt": "rain"},
        })

    def test_generate_video_developer_api_model_override(self):
        self.dev().generate_video("a cat surfing", model="veo-3.1-generate-preview")

        self.assertEqual(self.last_call["url"], f"{DEV_ROOT}/models/veo-3.1-generate-preview:predictLongRunning")

    def test_generate_video_error_prefix(self):
        self.session.queue(FakeResponse({"error": {"message": "denied"}}, status_code=403))

        with self.assertRaises(GoogleAIError) as caught:
            self.project().generate_video("a cat surfing")

        self.assertTrue(str(caught.exception).startswith("Veo Video Generation error"))
        self.assertEqual(caught.exception.status_code, 403)


# ----------------------------------------------------------------------
# Veo operations: poll, wait, extract
# ----------------------------------------------------------------------
class TestVideoOperations(GenAITestCase):

    VERTEX_MODEL = "publishers/google/models/veo-3.1-fast-generate-001"

    def vertex_operation(self, location="us-central1"):
        return f"projects/{PROJECT}/locations/{location}/{self.VERTEX_MODEL}/operations/op-1"

    def test_get_video_operation_vertex_uses_fetch_predict_operation(self):
        operation = self.vertex_operation()
        self.session.queue(FakeResponse({"name": operation, "done": False}))

        result = self.project().get_video_operation(operation)

        self.assertCall("POST",
                        f"{US_CENTRAL_ROOT}/{PROJECT_US_CENTRAL}/{self.VERTEX_MODEL}:fetchPredictOperation",
                        VERTEX_HEADERS, {"operationName": operation})
        self.assertEqual(result, {"name": operation, "done": False})

    def test_get_video_operation_host_follows_operation_location(self):
        cases = {
            "europe-west4": f"{EUROPE_WEST4_ROOT}/projects/{PROJECT}/locations/europe-west4",
            "us": f"{US_MULTI_ROOT}/projects/{PROJECT}/locations/us",
            "global": f"{GLOBAL_ROOT}/projects/{PROJECT}/locations/global",
        }
        for location, prefix in cases.items():
            with self.subTest(location=location):
                operation = self.vertex_operation(location)

                self.project().get_video_operation(operation)

                self.assertEqual(self.last_call["url"], f"{prefix}/{self.VERTEX_MODEL}:fetchPredictOperation")
                self.assertEqual(self.last_call["json"], {"operationName": operation})

    def test_get_video_operation_accepts_operation_dict(self):
        operation = self.vertex_operation()

        self.project().get_video_operation({"name": operation, "done": False})

        self.assertEqual(self.last_call["json"], {"operationName": operation})

    def test_get_video_operation_developer_api_get_operation(self):
        self.session.queue(FakeResponse({"name": "models/veo-2.0-generate-001/operations/op-9", "done": True}))

        result = self.dev().get_video_operation("models/veo-2.0-generate-001/operations/op-9")

        self.assertCall("GET", f"{DEV_ROOT}/models/veo-2.0-generate-001/operations/op-9", DEV_HEADERS)
        self.assertTrue(result["done"])

    def test_get_video_operation_vertex_name_on_developer_wrapper(self):
        # Operations started by generate_video(project_id=...) on a Developer-API wrapper.
        operation = self.vertex_operation()

        self.dev().get_video_operation(operation)

        self.assertCall("POST",
                        f"{US_CENTRAL_ROOT}/{PROJECT_US_CENTRAL}/{self.VERTEX_MODEL}:fetchPredictOperation",
                        DEV_HEADERS, {"operationName": operation})

    def test_get_video_operation_vertex_rejects_short_names(self):
        with self.assertRaises(ValueError):
            self.project().get_video_operation("operations/op-1")
        self.assertEqual(self.session.calls, [])

    def test_get_video_operation_requires_a_name(self):
        for value in (None, "", {}):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.dev().get_video_operation(value)

    def test_check_video_generation_status_forwards_to_get_video_operation(self):
        operation = self.vertex_operation()

        self.project().check_video_generation_status({"name": operation}, PROJECT)

        self.assertCall("POST",
                        f"{US_CENTRAL_ROOT}/{PROJECT_US_CENTRAL}/{self.VERTEX_MODEL}:fetchPredictOperation",
                        VERTEX_HEADERS, {"operationName": operation})

    def test_wait_for_video_completion_polls_until_done(self):
        operation = self.vertex_operation()
        finished = {"name": operation, "done": True, "response": {"videos": [{"gcsUri": "gs://b/v.mp4"}]}}
        self.session.queue(FakeResponse({"name": operation, "done": False}), FakeResponse(finished))
        clock = FakeClock()

        with patch("intelli.wrappers.googleai_wrapper.time", clock):
            result = self.project().wait_for_video_completion(operation, PROJECT, max_wait_time=60, poll_interval=7)

        self.assertEqual(result, finished)
        self.assertEqual(len(self.session.calls), 2)
        self.assertEqual(clock.sleeps, [7])

    def test_wait_for_video_completion_times_out(self):
        clock = FakeClock()

        with patch("intelli.wrappers.googleai_wrapper.time", clock):
            with self.assertRaises(TimeoutError) as caught:
                self.dev().wait_for_video_completion("models/veo-2.0-generate-001/operations/op-9",
                                                     max_wait_time=12, poll_interval=5)

        self.assertIn("12 seconds", str(caught.exception))
        self.assertEqual(len(self.session.calls), 3)  # polls at t=0, 5 and 10
        self.assertEqual(clock.sleeps, [5, 5, 5])

    def test_extract_videos_vertex_response(self):
        operation = {"done": True, "response": {"videos": [
            {"gcsUri": "gs://bucket/out/sample_0.mp4", "mimeType": "video/mp4"},
            {"bytesBase64Encoded": "AAAA"},
        ]}}

        self.assertEqual(GoogleAIWrapper.extract_videos(operation), [
            {"mime_type": "video/mp4", "data": None, "uri": "gs://bucket/out/sample_0.mp4"},
            {"mime_type": "video/mp4", "data": "AAAA", "uri": None},
        ])

    def test_extract_videos_developer_api_response(self):
        uri = "https://generativelanguage.googleapis.com/v1beta/files/abc:download?alt=media"
        operation = {"done": True, "response": {"generateVideoResponse": {"generatedSamples": [
            {"video": {"uri": uri}},
            {"video": {"encodedVideo": "BBBB", "mimeType": "video/webm"}},
        ]}}}

        self.assertEqual(GoogleAIWrapper.extract_videos(operation), [
            {"mime_type": "video/mp4", "data": None, "uri": uri},
            {"mime_type": "video/webm", "data": "BBBB", "uri": None},
        ])

    def test_extract_videos_unfinished_operation(self):
        self.assertEqual(GoogleAIWrapper.extract_videos({"done": False}), [])
        self.assertEqual(GoogleAIWrapper.extract_videos(None), [])


# ----------------------------------------------------------------------
# Lyria music
# ----------------------------------------------------------------------
class TestLyriaMusic(GenAITestCase):

    def test_lyria_002_express_predict_body(self):
        self.express().generate_music("calm piano", negative_prompt="drums", seed=7, sample_count=2)

        self.assertCall("POST", f"{US_CENTRAL_ROOT}/publishers/google/models/lyria-002:predict", VERTEX_HEADERS, {
            "instances": [{"prompt": "calm piano", "negative_prompt": "drums", "seed": 7}],
            "parameters": {"sample_count": 2},
        })

    def test_lyria_002_project_minimal_body(self):
        self.project().generate_music("calm piano")

        self.assertCall("POST", f"{US_CENTRAL_ROOT}/{PROJECT_US_CENTRAL}/publishers/google/models/lyria-002:predict",
                        VERTEX_HEADERS, {"instances": [{"prompt": "calm piano"}]})

    def test_lyria_002_location_argument(self):
        self.project().generate_music("calm piano", location="europe-west4")

        self.assertEqual(self.last_call["url"],
                         f"{EUROPE_WEST4_ROOT}/projects/{PROJECT}/locations/europe-west4/publishers/google/models/"
                         "lyria-002:predict")

    def test_lyria_002_on_developer_api_raises(self):
        with self.assertRaises(GoogleAIError) as caught:
            self.dev().generate_music("calm piano", model="lyria-002")

        self.assertIn("vertex=True", str(caught.exception))
        self.assertEqual(self.session.calls, [])

    def test_lyria_002_prediction_audio_is_extracted(self):
        prediction = {"predictions": [{"bytesBase64Encoded": "UklGRg==", "mimeType": "audio/wav"},
                                      {"bytesBase64Encoded": "UklGRh=="}]}
        self.session.queue(FakeResponse(prediction))

        result = self.express().generate_music("calm piano")

        self.assertEqual(result, prediction)
        self.assertEqual(GoogleAIWrapper.extract_audio(result), [
            {"mime_type": "audio/wav", "data": "UklGRg=="},
            {"mime_type": "audio/wav", "data": "UklGRh=="},
        ])

    def test_lyria_3_developer_api_generate_content_body(self):
        self.dev().generate_music("calm piano", negative_prompt="drums", seed=3)

        self.assertCall("POST", f"{DEV_ROOT}/models/lyria-3.5:generateContent", DEV_HEADERS, {
            "contents": [{"role": "user", "parts": [{"text": "calm piano\nAvoid: drums"}]}],
            "generationConfig": {"responseModalities": ["AUDIO", "TEXT"], "seed": 3},
        })

    def test_lyria_3_generation_config_is_merged(self):
        self.dev().generate_music("calm piano", seed=3,
                                  generation_config={"responseModalities": ["AUDIO"], "seed": 99})

        self.assertEqual(self.last_call["json"]["generationConfig"], {"responseModalities": ["AUDIO"], "seed": 99})

    def test_lyria_3_on_vertex_express(self):
        self.express().generate_music("calm piano", model="lyria-3.5")

        self.assertCall("POST", f"{GLOBAL_ROOT}/publishers/google/models/lyria-3.5:generateContent", VERTEX_HEADERS, {
            "contents": [{"role": "user", "parts": [{"text": "calm piano"}]}],
            "generationConfig": {"responseModalities": ["AUDIO", "TEXT"]},
        })

    def test_lyria_3_response_is_aliased_and_audio_extracted(self):
        self.session.queue(FakeResponse({"candidates": [{"content": {"parts": [
            {"inlineData": {"mimeType": "audio/mpeg", "data": "SUQz"}},
            {"text": "A calm piano piece"},
        ]}}]}))

        result = self.dev().generate_music("calm piano")

        self.assertEqual(result["candidates"][0]["content"]["parts"][0]["inline_data"]["mime_type"], "audio/mpeg")
        self.assertEqual(GoogleAIWrapper.extract_audio(result), [{"mime_type": "audio/mpeg", "data": "SUQz"}])


# ----------------------------------------------------------------------
# Gemini TTS
# ----------------------------------------------------------------------
class TestGeminiSpeech(GenAITestCase):

    @staticmethod
    def speech_body(voice_config, role=None):
        content = {"parts": [{"text": "Hello there"}]}
        if role:
            content["role"] = role
        return {"contents": [content], "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": voice_config},
        }}

    def test_generate_gemini_speech_developer_api(self):
        self.dev().generate_gemini_speech("Hello there")

        self.assertCall("POST", f"{DEV_ROOT}/models/gemini-2.5-flash-preview-tts:generateContent", DEV_HEADERS,
                        self.speech_body({"prebuiltVoiceConfig": {"voiceName": "Kore"}}))

    def test_generate_gemini_speech_vertex_default_model(self):
        self.express().generate_gemini_speech("Hello there")

        self.assertCall("POST", f"{GLOBAL_ROOT}/publishers/google/models/gemini-2.5-flash-tts:generateContent",
                        VERTEX_HEADERS, self.speech_body({"prebuiltVoiceConfig": {"voiceName": "Kore"}}, role="user"))

    def test_generate_gemini_speech_vertex_project_url(self):
        self.project().generate_gemini_speech("Hello there")

        self.assertEqual(self.last_call["url"],
                         f"{GLOBAL_ROOT}/{PROJECT_GLOBAL}/publishers/google/models/gemini-2.5-flash-tts:generateContent")

    def test_generate_gemini_speech_voice_argument(self):
        self.dev().generate_gemini_speech("Hello there", voice="Puck")

        self.assertEqual(self.last_call["json"], self.speech_body({"prebuiltVoiceConfig": {"voiceName": "Puck"}}))

    def test_generate_gemini_speech_voice_config_snake_case_merge(self):
        self.dev().generate_gemini_speech("Hello there", {"prebuilt_voice_config": {"voice_name": "Charon"}})

        self.assertEqual(self.last_call["json"], self.speech_body({"prebuiltVoiceConfig": {"voiceName": "Charon"}}))

    def test_generate_gemini_speech_voice_config_camel_case_merge(self):
        self.dev().generate_gemini_speech("Hello there", {"prebuiltVoiceConfig": {"voiceName": "Charon"}})

        self.assertEqual(self.last_call["json"], self.speech_body({"prebuiltVoiceConfig": {"voiceName": "Charon"}}))

    def test_generate_gemini_speech_model_override(self):
        self.dev().generate_gemini_speech("Hello there", model_override="gemini-2.5-pro-preview-tts")

        self.assertEqual(self.last_call["url"], f"{DEV_ROOT}/models/gemini-2.5-pro-preview-tts:generateContent")

    def test_generate_gemini_speech_audio_converts_to_wav(self):
        pcm = b"\x00\x01" * 8
        self.session.queue(FakeResponse({"candidates": [{"content": {"parts": [
            {"inlineData": {"mimeType": "audio/L16;codec=pcm;rate=24000", "data": b64(pcm)}},
        ]}}]}))

        audio = GoogleAIWrapper.extract_audio(self.dev().generate_gemini_speech("Hello there"))

        self.assertEqual(len(audio), 1)
        wav = GoogleAIWrapper.audio_to_wav(audio[0])
        self.assertEqual(wav[:4], b"RIFF")
        self.assertTrue(wav.endswith(pcm))

    def test_generate_multi_speaker_speech_body(self):
        speakers = [
            {"speaker": "Joe", "voice_config": {"prebuilt_voice_config": {"voice_name": "Kore"}}},
            {"speaker": "Jane", "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Puck"}}},
        ]

        self.dev().generate_multi_speaker_speech("Joe: Hi\nJane: Hello", speakers)

        self.assertCall("POST", f"{DEV_ROOT}/models/gemini-2.5-flash-preview-tts:generateContent", DEV_HEADERS, {
            "contents": [{"parts": [{"text": "Joe: Hi\nJane: Hello"}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {"multiSpeakerVoiceConfig": {"speakerVoiceConfigs": [
                    {"speaker": "Joe", "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}},
                    {"speaker": "Jane", "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Puck"}}},
                ]}},
            },
        })

    def test_generate_multi_speaker_speech_vertex_default_model(self):
        self.express().generate_multi_speaker_speech("Joe: Hi", [{"speaker": "Joe"}])

        self.assertEqual(self.last_call["url"],
                         f"{GLOBAL_ROOT}/publishers/google/models/gemini-2.5-flash-tts:generateContent")
        self.assertEqual(self.last_call["json"]["contents"], [{"role": "user", "parts": [{"text": "Joe: Hi"}]}])

    def test_generate_multi_speaker_speech_model_override(self):
        self.dev().generate_multi_speaker_speech("Joe: Hi", [{"speaker": "Joe"}], "gemini-2.5-pro-preview-tts")

        self.assertEqual(self.last_call["url"], f"{DEV_ROOT}/models/gemini-2.5-pro-preview-tts:generateContent")

    def test_tts_error_prefix(self):
        self.session.queue(FakeResponse({"error": {}}, status_code=500))

        with self.assertRaises(GoogleAIError) as caught:
            self.dev().generate_gemini_speech("Hello there")

        self.assertTrue(str(caught.exception).startswith("Gemini TTS error"))


# ----------------------------------------------------------------------
# Embeddings
# ----------------------------------------------------------------------
class TestEmbeddings(GenAITestCase):

    @staticmethod
    def predictions(*vectors):
        return FakeResponse({"predictions": [{"embeddings": {"values": list(v), "statistics": {"token_count": 1}}}
                                             for v in vectors]})

    def test_get_embeddings_developer_api_embed_content(self):
        params = {"content": {"parts": [{"text": "hello"}]}, "taskType": "RETRIEVAL_QUERY"}
        self.session.queue(FakeResponse({"embedding": {"values": [0.1, 0.2]}}))

        result = self.dev().get_embeddings(params)

        self.assertCall("POST", f"{DEV_ROOT}/models/gemini-embedding-001:embedContent", DEV_HEADERS, {
            "content": {"parts": [{"text": "hello"}]},
            "taskType": "RETRIEVAL_QUERY",
            "model": "models/gemini-embedding-001",
        })
        self.assertEqual(result, {"embedding": {"values": [0.1, 0.2]}})
        self.assertNotIn("model", params)  # the caller's dict is not changed

    def test_get_embeddings_developer_api_honors_model_in_params(self):
        self.dev().get_embeddings({"model": "models/text-embedding-004", "content": {"parts": [{"text": "hi"}]}})

        self.assertEqual(self.last_call["url"], f"{DEV_ROOT}/models/text-embedding-004:embedContent")
        self.assertEqual(self.last_call["json"]["model"], "models/text-embedding-004")

    def test_get_embeddings_vertex_predict_is_normalized(self):
        self.session.queue(self.predictions([0.5, 0.25]))

        result = self.express().get_embeddings({
            "content": {"parts": [{"text": "hello"}, {"text": "world"}]},
            "taskType": "RETRIEVAL_DOCUMENT", "title": "Greeting", "outputDimensionality": 256,
        })

        self.assertCall("POST", f"{GLOBAL_ROOT}/publishers/google/models/gemini-embedding-001:predict", VERTEX_HEADERS, {
            "instances": [{"content": "hello\nworld", "task_type": "RETRIEVAL_DOCUMENT", "title": "Greeting"}],
            "parameters": {"outputDimensionality": 256},
        })
        self.assertEqual(result, {"embedding": {"values": [0.5, 0.25]}})

    def test_get_embeddings_vertex_accepts_snake_case_options_and_model(self):
        self.project().get_embeddings({"model": "text-embedding-005", "content": "plain text",
                                       "task_type": "CLASSIFICATION", "output_dimensionality": 64})

        self.assertCall("POST", f"{GLOBAL_ROOT}/{PROJECT_GLOBAL}/publishers/google/models/text-embedding-005:predict",
                        VERTEX_HEADERS, {
                            "instances": [{"content": "plain text", "task_type": "CLASSIFICATION"}],
                            "parameters": {"outputDimensionality": 64},
                        })

    def test_get_embeddings_vertex_without_predictions_returns_empty_values(self):
        self.session.queue(FakeResponse({"predictions": []}))

        result = self.express().get_embeddings({"content": {"parts": [{"text": "hello"}]}})

        self.assertEqual(result, {"embedding": {"values": []}})

    def test_get_batch_embeddings_developer_api(self):
        self.session.queue(FakeResponse({"embeddings": [{"values": [1.0]}, {"values": [2.0]}]}))

        result = self.dev().get_batch_embeddings({"requests": [
            {"model": "ignored", "content": {"parts": [{"text": "a"}]}},
            {"content": {"parts": [{"text": "b"}]}},
        ]})

        self.assertCall("POST", f"{DEV_ROOT}/models/gemini-embedding-001:batchEmbedContents", DEV_HEADERS, {
            "requests": [
                {"model": "models/gemini-embedding-001", "content": {"parts": [{"text": "a"}]}},
                {"model": "models/gemini-embedding-001", "content": {"parts": [{"text": "b"}]}},
            ],
        })
        self.assertEqual(result, [{"values": [1.0]}, {"values": [2.0]}])

    def test_get_batch_embeddings_vertex_returns_list_of_values(self):
        self.session.queue(self.predictions([0.1]), self.predictions([0.2]))

        result = self.express().get_batch_embeddings({"requests": [
            {"content": {"parts": [{"text": "a"}]}},
            {"content": {"parts": [{"text": "b"}]}},
        ]})

        self.assertEqual(result, [{"values": [0.1]}, {"values": [0.2]}])
        url = f"{GLOBAL_ROOT}/publishers/google/models/gemini-embedding-001:predict"
        self.assertCall("POST", url, VERTEX_HEADERS, {"instances": [{"content": "a"}]}, call=self.session.calls[0])
        self.assertCall("POST", url, VERTEX_HEADERS, {"instances": [{"content": "b"}]}, call=self.session.calls[1])

    def test_embed_texts_vertex_gemini_embedding_sends_one_text_per_request(self):
        self.session.queue(self.predictions([1.0]), self.predictions([2.0]), self.predictions([3.0]))

        vectors = self.express().embed_texts(["a", "b", "c"], task_type="RETRIEVAL_DOCUMENT", title="Doc",
                                             output_dimensionality=128)

        self.assertEqual(vectors, [[1.0], [2.0], [3.0]])
        self.assertEqual(len(self.session.calls), 3)
        self.assertEqual([call["json"] for call in self.session.calls], [
            {"instances": [{"content": text, "task_type": "RETRIEVAL_DOCUMENT", "title": "Doc"}],
             "parameters": {"outputDimensionality": 128}}
            for text in ("a", "b", "c")
        ])

    def test_embed_texts_vertex_text_embedding_models_are_batched(self):
        self.session.queue(self.predictions([1.0], [2.0], [3.0]))

        vectors = self.project().embed_texts(["a", "b", "c"], model="text-embedding-005")

        self.assertEqual(vectors, [[1.0], [2.0], [3.0]])
        self.assertCall("POST", f"{GLOBAL_ROOT}/{PROJECT_GLOBAL}/publishers/google/models/text-embedding-005:predict",
                        VERTEX_HEADERS, {"instances": [{"content": "a"}, {"content": "b"}, {"content": "c"}]})

    def test_embed_texts_vertex_batches_hold_at_most_250_texts(self):
        texts = [f"t{i}" for i in range(251)]
        self.session.queue(self.predictions(*[[float(i)] for i in range(250)]), self.predictions([250.0]))

        vectors = self.express().embed_texts(texts, model="text-multilingual-embedding-002")

        self.assertEqual(len(vectors), 251)
        self.assertEqual(vectors[-1], [250.0])
        self.assertEqual([len(call["json"]["instances"]) for call in self.session.calls], [250, 1])

    def test_embed_texts_developer_api_batch_embed_contents(self):
        self.session.queue(FakeResponse({"embeddings": [{"values": [1.0, 1.5]}, {"values": [2.0, 2.5]}]}))

        vectors = self.dev().embed_texts(["a", "b"], model="models/gemini-embedding-001", task_type="SEMANTIC_SIMILARITY",
                                         title="T", output_dimensionality=2)

        self.assertCall("POST", f"{DEV_ROOT}/models/gemini-embedding-001:batchEmbedContents", DEV_HEADERS, {
            "requests": [
                {"model": "models/gemini-embedding-001", "content": {"parts": [{"text": text}]},
                 "taskType": "SEMANTIC_SIMILARITY", "title": "T", "outputDimensionality": 2}
                for text in ("a", "b")
            ],
        })
        self.assertEqual(vectors, [[1.0, 1.5], [2.0, 2.5]])

    def test_embed_texts_accepts_a_single_string(self):
        self.session.queue(FakeResponse({"embeddings": [{"values": [9.0]}]}))

        vectors = self.dev().embed_texts("only one")

        self.assertEqual(self.last_call["json"], {"requests": [
            {"model": "models/gemini-embedding-001", "content": {"parts": [{"text": "only one"}]}},
        ]})
        self.assertEqual(vectors, [[9.0]])


# ----------------------------------------------------------------------
# Files API (Developer API only)
# ----------------------------------------------------------------------
class TestFilesAPI(GenAITestCase):

    FILES_URL = f"{DEV_ROOT}/files"
    UPLOAD_URL = "https://generativelanguage.googleapis.com/upload/v1beta/files"

    def write_file(self, name, content):
        folder = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(folder, ignore_errors=True))
        path = os.path.join(folder, name)
        with open(path, "wb") as file:
            file.write(content)
        return path

    def test_list_files(self):
        self.session.queue(FakeResponse({"files": [{"name": "files/abc"}]}))

        result = self.dev().list_files()

        self.assertCall("GET", self.FILES_URL, DEV_HEADERS)
        self.assertEqual(result, {"files": [{"name": "files/abc"}]})

    def test_get_file_accepts_id_or_resource_name(self):
        for name in ("abc", "files/abc"):
            with self.subTest(name=name):
                self.dev().get_file(name)

                self.assertCall("GET", f"{self.FILES_URL}/abc", DEV_HEADERS)

    def test_delete_file_with_resource_name_does_not_double_the_prefix(self):
        self.session.queue(FakeResponse())  # empty body

        result = self.dev().delete_file("files/x")

        self.assertCall("DELETE", f"{self.FILES_URL}/x", DEV_HEADERS)
        self.assertEqual(result, {"status": "deleted"})

    def test_delete_file_with_id_returns_json_body_when_present(self):
        self.session.queue(FakeResponse({"done": True}))

        result = self.dev().delete_file("x")

        self.assertCall("DELETE", f"{self.FILES_URL}/x", DEV_HEADERS)
        self.assertEqual(result, {"done": True})

    def test_upload_file_two_step_resumable_upload(self):
        path = self.write_file("notes.txt", b"hello")
        session_url = "https://generativelanguage.googleapis.com/upload/v1beta/files?upload_id=u-1"
        uploaded = {"file": {"name": "files/n1", "uri": f"{self.FILES_URL}/n1", "state": "ACTIVE"}}
        self.session.queue(FakeResponse({}, headers={"X-Goog-Upload-URL": session_url}), FakeResponse(uploaded))

        result = self.dev().upload_file(path)

        start, upload = self.session.calls
        self.assertCall("POST", self.UPLOAD_URL, {
            "Content-Type": "application/json",
            "x-goog-api-key": DEV_KEY,
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": "5",
            "X-Goog-Upload-Header-Content-Type": "text/plain",
        }, {"file": {"display_name": "notes.txt"}}, call=start)
        self.assertEqual(upload["method"], "POST")
        self.assertEqual(upload["url"], session_url)
        self.assertEqual(upload["headers"], {
            "Content-Length": "5",
            "Content-Type": "text/plain",
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
        })
        self.assertEqual(upload["data"], b"hello")
        self.assertEqual(upload["timeout"], DEFAULT_TIMEOUT)
        self.assertEqual(result, uploaded)

    def test_upload_file_display_name_and_mime_type(self):
        path = self.write_file("clip.mp3", b"ID3")
        self.session.queue(FakeResponse({}, headers={"x-goog-upload-url": "https://upload.example/s"}),
                           FakeResponse({"file": {}}))

        self.dev().upload_file(path, display_name="My clip")

        start, upload = self.session.calls
        self.assertEqual(start["json"], {"file": {"display_name": "My clip"}})
        self.assertEqual(start["headers"]["X-Goog-Upload-Header-Content-Type"], "audio/mpeg")
        self.assertEqual(upload["headers"]["Content-Type"], "audio/mpeg")

    def test_upload_file_without_upload_url_raises(self):
        path = self.write_file("notes.txt", b"hello")
        self.session.queue(FakeResponse({}))

        with self.assertRaises(GoogleAIError) as caught:
            self.dev().upload_file(path)

        self.assertIn("upload URL not found", str(caught.exception))
        self.assertEqual(len(self.session.calls), 1)

    def test_upload_file_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            self.dev().upload_file(os.path.join(tempfile.gettempdir(), "does-not-exist-intelli.txt"))
        self.assertEqual(self.session.calls, [])

    def test_files_api_follows_api_version(self):
        # SUSPECTED SOURCE BUG (intelli/wrappers/googleai_wrapper.py, _file_url / list_files / upload_file):
        # with api_version="v1alpha" the model, cache and operation URLs move to /v1alpha, but list_files,
        # upload_file and get_file("<id>") keep the config's /v1beta files_base, while get_file("files/<id>")
        # uses /v1alpha. The same file gets two different URLs depending on how its name is written.
        wrapper = self.dev(api_version="v1alpha")
        files_url = "https://generativelanguage.googleapis.com/v1alpha/files"

        wrapper.get_file("files/abc")
        prefixed_url = self.last_call["url"]
        wrapper.get_file("abc")
        id_url = self.last_call["url"]
        wrapper.list_files()
        list_url = self.last_call["url"]

        self.assertEqual(prefixed_url, f"{files_url}/abc")
        self.assertEqual(id_url, f"{files_url}/abc")
        self.assertEqual(list_url, files_url)

    def test_files_api_is_not_available_on_vertex(self):
        wrapper = self.express()
        calls = {
            "upload_file": lambda: wrapper.upload_file("anything.txt"),
            "list_files": wrapper.list_files,
            "get_file": lambda: wrapper.get_file("files/abc"),
            "delete_file": lambda: wrapper.delete_file("files/abc"),
        }
        for name, call in calls.items():
            with self.subTest(method=name):
                with self.assertRaises(NotImplementedError):
                    call()
        self.assertEqual(self.session.calls, [])


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------
class TestModels(GenAITestCase):

    def test_list_models_developer_api(self):
        self.dev().list_models()

        self.assertCall("GET", f"{DEV_ROOT}/models", DEV_HEADERS)

    def test_list_models_developer_api_paging(self):
        self.dev().list_models(page_size=10, page_token="next")

        self.assertCall("GET", f"{DEV_ROOT}/models", DEV_HEADERS, params={"pageSize": 10, "pageToken": "next"})

    def test_list_models_vertex_express(self):
        self.express().list_models()

        self.assertCall("GET", f"{GLOBAL_ROOT}/publishers/google/models", VERTEX_HEADERS)

    def test_list_models_vertex_is_never_project_prefixed(self):
        wrapper = GoogleAIWrapper(vertex=True, project_id=PROJECT, location="us-central1",
                                  access_token=lambda: "ya29.fresh-token", session=self.session)

        wrapper.list_models(page_size=5)

        self.assertCall("GET", f"{GLOBAL_ROOT}/publishers/google/models",
                        {"Content-Type": "application/json", "Authorization": "Bearer ya29.fresh-token"},
                        params={"pageSize": 5})

    def test_model_catalog_matches_config(self):
        catalog = GoogleAIWrapper.model_catalog()

        self.assertEqual(catalog, config["url"]["gemini"]["vertex"]["catalog"])
        self.assertIn("lyria-002", catalog["music"])
        self.assertIn("veo-3.1-fast-generate-001", catalog["video"])
        self.assertIn("gemini-embedding-001", catalog["embedding"])

    def test_vertex_default_models_are_listed_in_the_catalog(self):
        listed = {model for models in GoogleAIWrapper.model_catalog().values() for model in models}
        defaults = dict(config["url"]["gemini"]["vertex"]["models"])

        for kind, model in defaults.items():
            with self.subTest(kind=kind):
                self.assertIn(model, listed)

    def test_model_catalog_returns_a_copy(self):
        catalog = self.dev().model_catalog()
        catalog["text"].append("not-a-model")

        self.assertNotIn("not-a-model", config["url"]["gemini"]["vertex"]["catalog"]["text"])


# ----------------------------------------------------------------------
# Context caching
# ----------------------------------------------------------------------
class TestContextCaching(GenAITestCase):

    CONTENTS = [{"role": "user", "parts": [{"text": "a very long document"}]}]

    def test_create_cached_content_developer_api(self):
        self.session.queue(FakeResponse({"name": "cachedContents/c1"}))

        result = self.dev().create_cached_content("gemini-2.5-flash", self.CONTENTS, system_instruction="Be brief",
                                                  ttl="600s", display_name="doc")

        self.assertCall("POST", f"{DEV_ROOT}/cachedContents", DEV_HEADERS, {
            "contents": self.CONTENTS,
            "model": "models/gemini-2.5-flash",
            "systemInstruction": {"parts": [{"text": "Be brief"}]},
            "ttl": "600s",
            "displayName": "doc",
        })
        self.assertEqual(result, {"name": "cachedContents/c1"})

    def test_create_cached_content_string_contents_tools_and_default_ttl(self):
        tools = [{"functionDeclarations": [{"name": "lookup"}]}]
        tool_config = {"functionCallingConfig": {"mode": "AUTO"}}

        self.dev().create_cached_content("models/gemini-2.5-flash", "plain text", tools=tools, tool_config=tool_config)

        self.assertEqual(self.last_call["json"], {
            "contents": [{"role": "user", "parts": [{"text": "plain text"}]}],
            "model": "models/gemini-2.5-flash",
            "ttl": "3600s",
            "tools": tools,
            "toolConfig": tool_config,
        })

    def test_get_list_delete_cached_content_developer_api(self):
        wrapper = self.dev()

        for name in ("c1", "cachedContents/c1"):
            with self.subTest(name=name):
                wrapper.get_cached_content(name)
                self.assertCall("GET", f"{DEV_ROOT}/cachedContents/c1", DEV_HEADERS)

        wrapper.list_cached_contents(page_size=3, page_token="p2")
        self.assertCall("GET", f"{DEV_ROOT}/cachedContents", DEV_HEADERS, params={"pageSize": 3, "pageToken": "p2"})

        self.session.queue(FakeResponse())  # empty body
        result = wrapper.delete_cached_content("cachedContents/c1")
        self.assertCall("DELETE", f"{DEV_ROOT}/cachedContents/c1", DEV_HEADERS)
        self.assertEqual(result, {"status": "deleted"})

    def test_create_cached_content_vertex_project(self):
        self.project().create_cached_content("gemini-3.8-flash", [{"parts": [{"text": "doc"}]}])

        self.assertCall("POST", f"{GLOBAL_ROOT}/{PROJECT_GLOBAL}/cachedContents", VERTEX_HEADERS, {
            "contents": [{"parts": [{"text": "doc"}], "role": "user"}],
            "model": f"{PROJECT_GLOBAL}/publishers/google/models/gemini-3.8-flash",
            "ttl": "3600s",
        })

    def test_create_cached_content_vertex_location_argument(self):
        self.project().create_cached_content("gemini-3.8-flash", self.CONTENTS, location="us-central1")

        self.assertEqual(self.last_call["url"], f"{US_CENTRAL_ROOT}/{PROJECT_US_CENTRAL}/cachedContents")
        self.assertEqual(self.last_call["json"]["model"],
                         f"{PROJECT_US_CENTRAL}/publishers/google/models/gemini-3.8-flash")

    def test_cached_content_vertex_by_id_and_full_name(self):
        wrapper = self.project()

        wrapper.get_cached_content("123")
        self.assertCall("GET", f"{GLOBAL_ROOT}/{PROJECT_GLOBAL}/cachedContents/123", VERTEX_HEADERS)

        wrapper.get_cached_content(f"{PROJECT_US_CENTRAL}/cachedContents/123")
        self.assertCall("GET", f"{US_CENTRAL_ROOT}/{PROJECT_US_CENTRAL}/cachedContents/123", VERTEX_HEADERS)

        wrapper.list_cached_contents(page_size=2)
        self.assertCall("GET", f"{GLOBAL_ROOT}/{PROJECT_GLOBAL}/cachedContents", VERTEX_HEADERS,
                        params={"pageSize": 2})

        wrapper.delete_cached_content("cachedContents/123")
        self.assertCall("DELETE", f"{GLOBAL_ROOT}/{PROJECT_GLOBAL}/cachedContents/123", VERTEX_HEADERS)

    def test_cached_content_vertex_with_oauth_token_and_quota_project(self):
        wrapper = GoogleAIWrapper(vertex=True, project_id=PROJECT, access_token="ya29.token-abc",
                                  quota_project_id="billing-proj", session=self.session)

        wrapper.list_cached_contents()

        self.assertCall("GET", f"{GLOBAL_ROOT}/{PROJECT_GLOBAL}/cachedContents", {
            "Content-Type": "application/json",
            "Authorization": "Bearer ya29.token-abc",
            "x-goog-user-project": "billing-proj",
        })

    def test_context_caching_on_vertex_requires_a_project(self):
        wrapper = self.express()
        calls = {
            "create": lambda: wrapper.create_cached_content("gemini-3.8-flash", self.CONTENTS),
            "get": lambda: wrapper.get_cached_content("123"),
            "list": wrapper.list_cached_contents,
        }
        for name, call in calls.items():
            with self.subTest(method=name):
                with self.assertRaises(GoogleAIError) as caught:
                    call()
                self.assertIn("needs a Google Cloud project", str(caught.exception))
        self.assertEqual(self.session.calls, [])


# ----------------------------------------------------------------------
# Agent Engine
# ----------------------------------------------------------------------
class TestAgentEngine(GenAITestCase):

    ENGINES = f"{US_CENTRAL_ROOT}/{PROJECT_US_CENTRAL}/reasoningEngines"

    def test_list_agent_engines(self):
        self.session.queue(FakeResponse({"reasoningEngines": [{"name": "x"}]}))

        result = self.project().list_agent_engines()

        self.assertCall("GET", self.ENGINES, VERTEX_HEADERS)
        self.assertEqual(result, {"reasoningEngines": [{"name": "x"}]})

    def test_list_agent_engines_paging_filter_and_location(self):
        self.project().list_agent_engines(location="europe-west1", page_size=5, page_token="t", filter="display_name=a")

        self.assertCall("GET",
                        f"https://europe-west1-aiplatform.googleapis.com/v1beta1/projects/{PROJECT}"
                        "/locations/europe-west1/reasoningEngines",
                        VERTEX_HEADERS, params={"pageSize": 5, "pageToken": "t", "filter": "display_name=a"})

    def test_explicit_wrapper_location_is_used_for_agent_engine(self):
        self.project(location="europe-west4").get_agent_engine("42")

        self.assertCall("GET", f"{EUROPE_WEST4_ROOT}/projects/{PROJECT}/locations/europe-west4/reasoningEngines/42",
                        VERTEX_HEADERS)

    def test_get_agent_engine_by_id(self):
        self.project().get_agent_engine("123")

        self.assertCall("GET", f"{self.ENGINES}/123", VERTEX_HEADERS)

    def test_get_agent_engine_by_full_name_uses_its_location_without_wrapper_project(self):
        name = "projects/other/locations/europe-west1/reasoningEngines/9"

        self.express().get_agent_engine(name)

        self.assertCall("GET", f"https://europe-west1-aiplatform.googleapis.com/v1beta1/{name}", VERTEX_HEADERS)

    def test_query_agent_engine(self):
        self.session.queue(FakeResponse({"output": {"answer": 42}}))

        result = self.project().query_agent_engine("123", {"input": "hi"}, class_method="query")

        self.assertCall("POST", f"{self.ENGINES}/123:query", VERTEX_HEADERS,
                        {"input": {"input": "hi"}, "classMethod": "query"})
        self.assertEqual(result, {"output": {"answer": 42}})

    def test_query_agent_engine_defaults(self):
        self.project().query_agent_engine(f"{PROJECT_US_CENTRAL}/reasoningEngines/123")

        self.assertCall("POST", f"{self.ENGINES}/123:query", VERTEX_HEADERS, {"input": {}})

    def test_stream_query_agent_engine_parses_sse_and_json_lines(self):
        response = FakeResponse(lines=[
            'data: {"content": {"parts": [{"text": "Hel"}]}}',
            "",
            '{"content": {"parts": [{"text": "lo"}]}}',
            "data: not json",
        ])
        self.session.queue(response)

        events = list(self.project().stream_query_agent_engine("123", {"message": "hi", "user_id": "u1"}))

        self.assertCall("POST", f"{self.ENGINES}/123:streamQuery", VERTEX_HEADERS,
                        {"input": {"message": "hi", "user_id": "u1"}, "classMethod": "stream_query"},
                        params={"alt": "sse"}, stream=True)
        self.assertEqual(events, [
            {"content": {"parts": [{"text": "Hel"}]}},
            {"content": {"parts": [{"text": "lo"}]}},
            "not json",
        ])
        self.assertTrue(response.closed)

    def test_stream_query_agent_engine_custom_class_method(self):
        self.session.queue(FakeResponse(lines=[]))

        list(self.project().stream_query_agent_engine("123", class_method="async_stream_query"))

        self.assertEqual(self.last_call["json"], {"input": {}, "classMethod": "async_stream_query"})

    def test_agent_engine_needs_vertex_and_a_project(self):
        with self.assertRaises(GoogleAIError) as caught:
            self.dev().list_agent_engines()
        self.assertIn("needs Vertex AI", str(caught.exception))

        with self.assertRaises(GoogleAIError) as caught:
            self.express().query_agent_engine("123")
        self.assertIn("needs a Google Cloud project", str(caught.exception))
        self.assertEqual(self.session.calls, [])

    def test_agent_engine_error_prefix(self):
        self.session.queue(FakeResponse({"error": {"message": "not found"}}, status_code=404))

        with self.assertRaises(GoogleAIError) as caught:
            self.project().get_agent_engine("missing")

        self.assertTrue(str(caught.exception).startswith("Agent Engine error"))
        self.assertEqual(caught.exception.status_code, 404)


# ----------------------------------------------------------------------
# Live API: endpoint, connect, session
# ----------------------------------------------------------------------
DEV_LIVE_URL = ("wss://generativelanguage.googleapis.com/ws/"
                "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent")
VERTEX_LIVE_URL = ("wss://us-central1-aiplatform.googleapis.com/ws/"
                   "google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent")
VERTEX_LIVE_MODEL = f"{PROJECT_US_CENTRAL}/publishers/google/models/gemini-3.8-live"


class TestLiveEndpoint(GenAITestCase):

    def test_developer_api_endpoint(self):
        self.assertEqual(self.dev()._live_endpoint(),
                         (DEV_LIVE_URL, {"x-goog-api-key": DEV_KEY}, "models/gemini-3.8-live"))

    def test_developer_api_endpoint_model_and_api_version(self):
        url, headers, model = self.dev(api_version="v1alpha")._live_endpoint("gemini-2.5-flash-native-audio-preview")

        self.assertEqual(url, "wss://generativelanguage.googleapis.com/ws/"
                              "google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent")
        self.assertEqual(model, "models/gemini-2.5-flash-native-audio-preview")

    def test_vertex_project_endpoint_defaults_to_us_central1(self):
        self.assertEqual(self.project()._live_endpoint(),
                         (VERTEX_LIVE_URL, {"x-goog-api-key": VERTEX_KEY}, VERTEX_LIVE_MODEL))

    def test_vertex_endpoint_location_argument(self):
        url, _, model = self.project()._live_endpoint(location="europe-west4")

        self.assertEqual(url, "wss://europe-west4-aiplatform.googleapis.com/ws/"
                              "google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent")
        self.assertEqual(model, f"projects/{PROJECT}/locations/europe-west4/publishers/google/models/gemini-3.8-live")

    def test_vertex_endpoint_multi_region_location(self):
        url, _, model = self.project(location="us")._live_endpoint()

        self.assertEqual(url, "wss://aiplatform.us.rep.googleapis.com/ws/"
                              "google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent")
        self.assertEqual(model, f"projects/{PROJECT}/locations/us/publishers/google/models/gemini-3.8-live")

    def test_vertex_endpoint_keeps_full_model_resource_name(self):
        full_name = "projects/p2/locations/us-east5/publishers/google/models/gemini-live-2.5-flash-native-audio"

        _, _, model = self.project()._live_endpoint(full_name)

        self.assertEqual(model, full_name)

    def test_vertex_endpoint_with_oauth_headers(self):
        wrapper = GoogleAIWrapper(vertex=True, project_id=PROJECT, access_token="ya29.live-token",
                                  quota_project_id="billing-proj", session=self.session)

        _, headers, _ = wrapper._live_endpoint()

        self.assertEqual(headers, {"Authorization": "Bearer ya29.live-token", "x-goog-user-project": "billing-proj"})

    def test_vertex_express_endpoint_needs_a_project(self):
        with self.assertRaises(GoogleAIError):
            self.express()._live_endpoint()


class TestLiveConnect(GenAITestCase):

    def connect(self, wrapper, websocket, **kwargs):
        """Open live_connect with a fake websockets module; return (module, session)."""
        module = fake_websockets_module(websocket, legacy=kwargs.pop("legacy", False),
                                        error=kwargs.pop("error", None))

        async def run():
            async with wrapper.live_connect(**kwargs) as session:
                return session

        with patch.dict(sys.modules, {"websockets": module}):
            session = asyncio.run(run())
        return module, session

    def test_live_connect_developer_api_setup_message(self):
        websocket = FakeWebSocket([{"setupComplete": {}}])

        module, session = self.connect(self.dev(), websocket)

        self.assertEqual(module.connect_calls, [{"url": DEV_LIVE_URL, "additional_headers": {"x-goog-api-key": DEV_KEY},
                                                 "max_size": None, "open_timeout": DEFAULT_TIMEOUT}])
        self.assertEqual(websocket.sent, [{"setup": {"model": "models/gemini-3.8-live",
                                                     "generationConfig": {"responseModalities": ["AUDIO"]}}}])
        self.assertIsInstance(session, GoogleAILiveSession)
        self.assertEqual(session.setup_response, {"setupComplete": {}})
        self.assertTrue(websocket.closed)

    def test_live_connect_config_is_camelized_and_keeps_modalities(self):
        websocket = FakeWebSocket([{"setupComplete": {}}])
        live_config = {"generation_config": {"response_modalities": ["TEXT"]},
                       "system_instruction": {"parts": [{"text": "Be brief"}]}}

        self.connect(self.dev(), websocket, model="gemini-2.5-flash-native-audio-preview", config=live_config)

        self.assertEqual(websocket.sent[0], {"setup": {
            "model": "models/gemini-2.5-flash-native-audio-preview",
            "generationConfig": {"responseModalities": ["TEXT"]},
            "systemInstruction": {"parts": [{"text": "Be brief"}]},
        }})
        self.assertEqual(live_config, {"generation_config": {"response_modalities": ["TEXT"]},
                                       "system_instruction": {"parts": [{"text": "Be brief"}]}})

    def test_live_connect_vertex_project(self):
        websocket = FakeWebSocket([{"setupComplete": {}}])

        module, _ = self.connect(self.project(), websocket)

        self.assertEqual(module.connect_calls[0]["url"], VERTEX_LIVE_URL)
        self.assertEqual(module.connect_calls[0]["additional_headers"], {"x-goog-api-key": VERTEX_KEY})
        self.assertEqual(websocket.sent[0], {"setup": {"model": VERTEX_LIVE_MODEL,
                                                       "generationConfig": {"responseModalities": ["AUDIO"]}}})

    def test_live_connect_falls_back_to_extra_headers_for_older_websockets(self):
        websocket = FakeWebSocket([{"setupComplete": {}}])

        module, _ = self.connect(self.dev(), websocket, legacy=True)

        self.assertEqual(module.connect_calls, [{"url": DEV_LIVE_URL, "extra_headers": {"x-goog-api-key": DEV_KEY},
                                                 "max_size": None, "open_timeout": DEFAULT_TIMEOUT}])

    def test_live_connect_setup_failure_raises_and_closes(self):
        websocket = FakeWebSocket([{"error": {"message": "model not found"}}])

        with self.assertRaises(GoogleAIError) as caught:
            self.connect(self.dev(), websocket)

        self.assertIn("Live API setup failed", str(caught.exception))
        self.assertTrue(websocket.closed)

    def test_live_connect_closed_before_setup_complete_raises(self):
        websocket = FakeWebSocket([])

        with self.assertRaises(GoogleAIError) as caught:
            self.connect(self.dev(), websocket)

        self.assertIn("Live API setup error", str(caught.exception))
        self.assertTrue(websocket.closed)

    def test_live_connect_connection_error_redacts_the_key(self):
        error = OSError(f"handshake refused for {DEV_KEY}")

        with self.assertRaises(GoogleAIError) as caught:
            self.connect(self.dev(), FakeWebSocket([]), error=error)

        self.assertIn("Live API connection error", str(caught.exception))
        self.assertNotIn(DEV_KEY, str(caught.exception))

    def test_live_connect_without_websockets_package(self):
        async def run():
            async with self.dev().live_connect():
                pass

        with patch.dict(sys.modules, {"websockets": None}):
            with self.assertRaises(GoogleAIError) as caught:
                asyncio.run(run())

        self.assertIn("pip install websockets", str(caught.exception))

    def test_live_generate_async_adds_output_audio_transcription_for_audio(self):
        websocket = FakeWebSocket([
            {"setupComplete": {}},
            {"serverContent": {"outputTranscription": {"text": "Hi there"},
                               "modelTurn": {"parts": [{"inlineData": {"mimeType": "audio/pcm;rate=24000",
                                                                       "data": b64(b"\x01\x02")}}]}}},
            {"serverContent": {"turnComplete": True}},
        ])
        module = fake_websockets_module(websocket)

        with patch.dict(sys.modules, {"websockets": module}):
            result = asyncio.run(self.dev().live_generate_async("Say hi"))

        self.assertEqual(websocket.sent, [
            {"setup": {"model": "models/gemini-3.8-live", "outputAudioTranscription": {},
                       "generationConfig": {"responseModalities": ["AUDIO"]}}},
            {"clientContent": {"turns": [{"role": "user", "parts": [{"text": "Say hi"}]}], "turnComplete": True}},
        ])
        self.assertEqual(result["transcription"], "Hi there")
        self.assertEqual(result["audio"], b"\x01\x02")
        self.assertEqual(result["audio_mime_type"], "audio/pcm;rate=24000")
        self.assertTrue(websocket.closed)

    def test_live_generate_text_modality_has_no_audio_transcription(self):
        websocket = FakeWebSocket([
            {"setupComplete": {}},
            {"serverContent": {"modelTurn": {"parts": [{"text": "Hello"}]}, "turnComplete": True}},
        ])
        module = fake_websockets_module(websocket)

        with patch.dict(sys.modules, {"websockets": module}):
            result = self.dev().live_generate("Say hi", config={"generation_config": {"response_modalities": ["TEXT"]}})

        self.assertEqual(websocket.sent[0], {"setup": {"model": "models/gemini-3.8-live",
                                                       "generationConfig": {"responseModalities": ["TEXT"]}}})
        self.assertEqual(result["text"], "Hello")


class TestLiveSession(unittest.TestCase):

    def test_send_messages(self):
        websocket = FakeWebSocket()
        session = GoogleAILiveSession(websocket)

        async def run():
            await session.send_text("hi")
            await session.send_text("and more", turn_complete=False)
            await session.send_audio(b"\x00\x01")
            await session.send_audio("AAE=", mime_type="audio/pcm;rate=24000")
            await session.send_audio_stream_end()
            await session.send_tool_response([{"id": "c1", "name": "lookup", "response": {"ok": True}}])
            await session.close()

        asyncio.run(run())

        self.assertEqual(websocket.sent, [
            {"clientContent": {"turns": [{"role": "user", "parts": [{"text": "hi"}]}], "turnComplete": True}},
            {"clientContent": {"turns": [{"role": "user", "parts": [{"text": "and more"}]}], "turnComplete": False}},
            {"realtimeInput": {"audio": {"data": b64(b"\x00\x01"), "mimeType": "audio/pcm;rate=16000"}}},
            {"realtimeInput": {"audio": {"data": "AAE=", "mimeType": "audio/pcm;rate=24000"}}},
            {"realtimeInput": {"audioStreamEnd": True}},
            {"toolResponse": {"functionResponses": [{"id": "c1", "name": "lookup", "response": {"ok": True}}]}},
        ])
        self.assertTrue(websocket.closed)

    def test_receive_turn_collects_text_transcription_and_audio(self):
        websocket = FakeWebSocket([
            {"serverContent": {"modelTurn": {"parts": [{"text": "thinking...", "thought": True}, {"text": "Hel"}]}}},
            {"serverContent": {"modelTurn": {"parts": [{"inlineData": {"mimeType": "audio/pcm;rate=24000",
                                                                       "data": b64(b"\x01\x02")}}]},
                               "outputTranscription": {"text": "Hello "}}},
            {"serverContent": {"modelTurn": {"parts": [{"text": "lo"},
                                                       {"inlineData": {"mimeType": "audio/pcm;rate=24000",
                                                                       "data": b64(b"\x03")}}]},
                               "outputTranscription": {"text": "there"},
                               "inputTranscription": {"text": "say hello"}}},
            {"usageMetadata": {"totalTokenCount": 12}, "serverContent": {"turnComplete": True}},
            {"serverContent": {"modelTurn": {"parts": [{"text": "next turn"}]}}},
        ])

        result = asyncio.run(GoogleAILiveSession(websocket).receive_turn())

        self.assertEqual(result, {
            "text": "Hello",
            "transcription": "Hello there",
            "input_transcription": "say hello",
            "audio": b"\x01\x02\x03",
            "audio_mime_type": "audio/pcm;rate=24000",
            "tool_calls": [],
            "usage": {"totalTokenCount": 12},
            "messages": 4,
        })
        self.assertEqual(len(websocket.incoming), 1)  # the next turn is left unread

    def test_receive_turn_stops_at_tool_call(self):
        calls = [{"id": "c1", "name": "get_weather", "args": {"city": "Paris"}}]
        websocket = FakeWebSocket([
            {"serverContent": {"modelTurn": {"parts": [{"text": "Let me check."}]}}},
            {"toolCall": {"functionCalls": calls}},
            {"serverContent": {"turnComplete": True}},
        ])

        result = asyncio.run(GoogleAILiveSession(websocket).receive_turn())

        self.assertEqual(result["tool_calls"], calls)
        self.assertEqual(result["text"], "Let me check.")
        self.assertEqual(result["messages"], 2)
        self.assertEqual(len(websocket.incoming), 1)

    def test_receive_yields_every_message(self):
        websocket = FakeWebSocket([{"a": 1}, {"b": 2}])

        async def run():
            return [message async for message in GoogleAILiveSession(websocket).receive()]

        self.assertEqual(asyncio.run(run()), [{"a": 1}, {"b": 2}])


class TestImagenRetired(GenAITestCase):

    def test_imagen_without_a_model_explains_the_retirement(self):
        for call in (lambda w: w.imagen_generate_images("a lighthouse"),
                     lambda w: w.imagen_edit_image("add a boat", b"raw"),
                     lambda w: w.imagen_upscale_image(b"small")):
            with self.subTest(call=call), self.assertRaises(ValueError) as caught:
                call(self.express())
            self.assertIn("retired", str(caught.exception))
        self.assertEqual(self.session.calls, [])


if __name__ == "__main__":
    unittest.main()

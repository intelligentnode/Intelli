"""
Offline unit tests for the core of the Gemini / Vertex AI part of GoogleAIWrapper.

Nothing here calls Google: the HTTP session is a mock, access tokens are set by hand and
google-auth is replaced by fake modules when a test needs it.
"""
import base64
import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
import wave
from unittest.mock import MagicMock, patch

import requests

from intelli.config import config
from intelli.wrappers.googleai_wrapper import GoogleAIChatSession, GoogleAIError, GoogleAIWrapper

API_KEY = "AIzaSy-unit-test-key-0123456789"
ACCESS_TOKEN = "ya29.unit-test-access-token-abcdef"

DEV_ROOT = "https://generativelanguage.googleapis.com/v1beta"
GLOBAL_ROOT = "https://aiplatform.googleapis.com/v1beta1"

DEV_MODELS = config["url"]["gemini"]["models"]
VERTEX_MODELS = config["url"]["gemini"]["vertex"]["models"]
VERTEX_LOCATIONS = config["url"]["gemini"]["vertex"]["locations"]

GENAI_ENV_VARS = ("GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_GENAI_USE_ENTERPRISE",
                  "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION")


class FakeResponse:
    """Minimal stand-in for requests.Response (JSON body, SSE lines, raise_for_status)."""

    def __init__(self, json_data=None, status_code=200, text=None, lines=None,
                 url="https://example.googleapis.com/test"):
        self._json = json_data
        self.status_code = status_code
        self.text = text if text is not None else (json.dumps(json_data) if json_data is not None else "")
        self.content = self.text.encode("utf-8")
        self._lines = list(lines or [])
        self.url = url
        self.closed = False
        self.decode_unicode = None

    def json(self):
        if self._json is None:
            raise ValueError("No JSON object could be decoded")
        return json.loads(json.dumps(self._json))  # a fresh copy on every call

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(
                f"{self.status_code} Client Error: Bad Request for url: {self.url}", response=self)

    def iter_lines(self, decode_unicode=False):
        self.decode_unicode = decode_unicode
        for line in self._lines:
            if isinstance(line, Exception):
                raise line
            yield line

    def close(self):
        self.closed = True


def real_error_response(status_code, body, url):
    """A real requests.Response holding an error, so raise_for_status builds the real message."""
    response = requests.Response()
    response.status_code = status_code
    response.reason = "Bad Request"
    response.url = url
    response.encoding = "utf-8"
    response._content = json.dumps(body).encode("utf-8") if not isinstance(body, bytes) else body
    return response


def sse_event(payload):
    return [f"data: {json.dumps(payload)}", ""]


def text_chunk(*texts):
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": t} for t in texts]}}]}


class FakeCredentials:
    """google.auth credentials stand-in: refresh() sets a new token and marks it valid."""

    def __init__(self, token=None, valid=False, quota_project_id=None, refreshed_token="refreshed-token-123456",
                 refresh_error=None):
        self.token = token
        self.valid = valid
        self.quota_project_id = quota_project_id
        self.refreshed_token = refreshed_token
        self.refresh_error = refresh_error
        self.refresh_requests = []

    def refresh(self, request):
        self.refresh_requests.append(request)
        if self.refresh_error:
            raise self.refresh_error
        self.token = self.refreshed_token
        self.valid = True


@contextlib.contextmanager
def fake_google_auth(default_result=None, default_error=None):
    """Install fake google / google.auth / google.auth.transport.requests modules."""
    google_module = types.ModuleType("google")
    auth_module = types.ModuleType("google.auth")
    transport_module = types.ModuleType("google.auth.transport")
    requests_module = types.ModuleType("google.auth.transport.requests")

    class FakeRequest:
        pass

    requests_module.Request = FakeRequest
    auth_module.default = MagicMock(return_value=default_result, side_effect=default_error)
    google_module.auth = auth_module
    auth_module.transport = transport_module
    transport_module.requests = requests_module
    modules = {"google": google_module, "google.auth": auth_module,
               "google.auth.transport": transport_module, "google.auth.transport.requests": requests_module}
    with patch.dict(sys.modules, modules):
        yield types.SimpleNamespace(default=auth_module.default, Request=FakeRequest)


@contextlib.contextmanager
def google_auth_missing():
    """Make `import google.auth` fail as if google-auth were not installed."""
    modules = {"google": None, "google.auth": None, "google.auth.transport": None,
               "google.auth.transport.requests": None}
    with patch.dict(sys.modules, modules):
        yield


class GenAITestCase(unittest.TestCase):
    """Clears the Gemini / Vertex environment variables and builds wrappers with a mock session."""

    def setUp(self):
        env_patcher = patch.dict(os.environ)
        env_patcher.start()
        self.addCleanup(env_patcher.stop)
        for name in GENAI_ENV_VARS:
            os.environ.pop(name, None)

    def make_wrapper(self, api_key=API_KEY, **kwargs):
        kwargs.setdefault("session", MagicMock())
        return GoogleAIWrapper(api_key, **kwargs)

    @staticmethod
    def post_kwargs(wrapper, index=-1):
        call = wrapper.session.post.call_args_list[index]
        return call.args[0], call.kwargs


# ----------------------------------------------------------------------
# Backend selection
# ----------------------------------------------------------------------
class TestBackendSelection(GenAITestCase):
    def test_api_key_alone_selects_developer_api(self):
        wrapper = self.make_wrapper()

        self.assertFalse(wrapper.vertex)
        self.assertEqual(wrapper.api_version, "v1beta")
        self.assertIsNone(wrapper.project_id)
        self.assertIsNone(wrapper.location)
        self.assertEqual(wrapper.models, DEV_MODELS)

    def test_vertex_flag_selects_vertex_with_v1beta1(self):
        wrapper = self.make_wrapper(vertex=True)

        self.assertTrue(wrapper.vertex)
        self.assertEqual(wrapper.api_version, "v1beta1")
        self.assertIsNone(wrapper.project_id)
        self.assertIsNone(wrapper.location)  # express mode has no location

    def test_vertex_models_override_developer_defaults(self):
        wrapper = self.make_wrapper(vertex=True)

        self.assertEqual(wrapper.models["text"], VERTEX_MODELS["text"])
        self.assertEqual(wrapper.models["video_generation"], VERTEX_MODELS["video_generation"])
        # keys only present in the Developer API map are kept
        self.assertEqual(wrapper.models["legacy_text"], DEV_MODELS["legacy_text"])

    def test_project_id_selects_vertex_automatically(self):
        wrapper = self.make_wrapper(project_id="my-project")

        self.assertTrue(wrapper.vertex)
        self.assertEqual(wrapper.project_id, "my-project")
        self.assertEqual(wrapper.location, "global")
        self.assertFalse(wrapper._location_explicit)

    def test_access_token_selects_vertex_automatically(self):
        wrapper = self.make_wrapper(api_key=None, access_token=ACCESS_TOKEN)

        self.assertTrue(wrapper.vertex)

    def test_credentials_select_vertex_automatically(self):
        wrapper = self.make_wrapper(api_key=None, credentials=FakeCredentials())

        self.assertTrue(wrapper.vertex)

    def test_explicit_vertex_false_wins_over_project_id(self):
        wrapper = self.make_wrapper(vertex=False, project_id="my-project")

        self.assertFalse(wrapper.vertex)
        self.assertEqual(wrapper.api_version, "v1beta")
        self.assertIsNone(wrapper.location)

    def test_env_use_vertexai_selects_vertex(self):
        for value in ("true", "True", "1", "yes", " TRUE "):
            with self.subTest(value=value):
                os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = value
                self.assertTrue(self.make_wrapper().vertex)

    def test_env_use_vertexai_false_keeps_developer_api(self):
        for value in ("false", "0", "no", ""):
            with self.subTest(value=value):
                os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = value
                self.assertFalse(self.make_wrapper().vertex)

    def test_env_use_enterprise_selects_vertex(self):
        os.environ["GOOGLE_GENAI_USE_ENTERPRISE"] = "1"

        self.assertTrue(self.make_wrapper().vertex)

    def test_explicit_vertex_false_wins_over_env(self):
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"

        self.assertFalse(self.make_wrapper(vertex=False).vertex)

    def test_env_project_used_for_adc_without_api_key(self):
        os.environ["GOOGLE_CLOUD_PROJECT"] = "env-project"

        wrapper = self.make_wrapper(api_key=None, vertex=True)

        self.assertEqual(wrapper.project_id, "env-project")
        self.assertEqual(wrapper.location, "global")

    def test_env_project_ignored_with_api_key(self):
        os.environ["GOOGLE_CLOUD_PROJECT"] = "env-project"

        wrapper = self.make_wrapper(vertex=True)

        self.assertIsNone(wrapper.project_id)
        self.assertIsNone(wrapper.location)

    def test_env_project_ignored_on_developer_api(self):
        os.environ["GOOGLE_CLOUD_PROJECT"] = "env-project"

        wrapper = self.make_wrapper()

        self.assertFalse(wrapper.vertex)
        self.assertIsNone(wrapper.project_id)

    def test_env_project_does_not_select_vertex(self):
        os.environ["GOOGLE_CLOUD_PROJECT"] = "env-project"

        self.assertFalse(self.make_wrapper(api_key=None).vertex)

    def test_explicit_project_wins_over_env_project(self):
        os.environ["GOOGLE_CLOUD_PROJECT"] = "env-project"

        wrapper = self.make_wrapper(api_key=None, vertex=True, project_id="my-project")

        self.assertEqual(wrapper.project_id, "my-project")

    def test_env_location_is_explicit_on_vertex(self):
        os.environ["GOOGLE_CLOUD_LOCATION"] = "europe-west4"

        wrapper = self.make_wrapper(project_id="my-project")

        self.assertEqual(wrapper.location, "europe-west4")
        self.assertTrue(wrapper._location_explicit)

    def test_env_location_ignored_on_developer_api(self):
        os.environ["GOOGLE_CLOUD_LOCATION"] = "europe-west4"

        self.assertIsNone(self.make_wrapper().location)

    def test_explicit_location_is_kept(self):
        wrapper = self.make_wrapper(project_id="my-project", location="us-central1")

        self.assertEqual(wrapper.location, "us-central1")
        self.assertTrue(wrapper._location_explicit)

    def test_api_version_override_on_developer_api(self):
        wrapper = self.make_wrapper(api_version="v1")

        self.assertEqual(wrapper.api_version, "v1")
        self.assertEqual(wrapper._dev_api_base, "https://generativelanguage.googleapis.com/v1")
        self.assertEqual(wrapper._dev_models_base, "https://generativelanguage.googleapis.com/v1/models")

    def test_api_version_override_on_vertex(self):
        wrapper = self.make_wrapper(vertex=True, api_version="v1")

        self.assertEqual(wrapper.api_version, "v1")
        self.assertEqual(wrapper._vertex_api_version, "v1")

    def test_default_session_is_requests_session(self):
        wrapper = GoogleAIWrapper(API_KEY)

        self.assertIsInstance(wrapper.session, requests.Session)

    def test_custom_session_is_used(self):
        session = MagicMock()

        self.assertIs(GoogleAIWrapper(API_KEY, session=session).session, session)

    def test_base_url_trailing_slash_is_stripped(self):
        wrapper = self.make_wrapper(base_url="https://proxy.example.com/v1beta/")

        self.assertEqual(wrapper.base_url, "https://proxy.example.com/v1beta")

    def test_cloud_api_settings_are_unchanged(self):
        wrapper = self.make_wrapper(timeout=33)

        self.assertEqual(wrapper.timeout, 33)
        self.assertEqual(wrapper.headers["X-Goog-Api-Key"], API_KEY)
        self.assertEqual(wrapper.api_speech_url, "https://texttospeech.googleapis.com/v1")
        self.assertEqual(wrapper.api_vision_url, "https://vision.googleapis.com/v1")
        self.assertEqual(wrapper.api_translation_url, "https://translate.googleapis.com/v1")


# ----------------------------------------------------------------------
# URLs
# ----------------------------------------------------------------------
class TestModelUrls(GenAITestCase):
    def test_developer_api_url(self):
        wrapper = self.make_wrapper()

        self.assertEqual(wrapper._model_url("gemini-2.5-flash", "generateContent"),
                         f"{DEV_ROOT}/models/gemini-2.5-flash:generateContent")

    def test_developer_api_accepts_models_prefix(self):
        wrapper = self.make_wrapper()

        self.assertEqual(wrapper._model_url("models/gemini-2.5-flash", "countTokens"),
                         f"{DEV_ROOT}/models/gemini-2.5-flash:countTokens")

    def test_developer_api_accepts_tuned_models(self):
        wrapper = self.make_wrapper()

        self.assertEqual(wrapper._model_url("tunedModels/my-tuned", "generateContent"),
                         f"{DEV_ROOT}/tunedModels/my-tuned:generateContent")

    def test_developer_api_ignores_location(self):
        wrapper = self.make_wrapper()

        self.assertEqual(wrapper._model_url("gemini-2.5-flash", "predict", "us-central1"),
                         f"{DEV_ROOT}/models/gemini-2.5-flash:predict")

    def test_developer_api_version_override_url(self):
        wrapper = self.make_wrapper(api_version="v1")

        self.assertEqual(wrapper._model_url("gemini-2.5-flash", "generateContent"),
                         "https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent")

    def test_developer_api_base_url_override(self):
        wrapper = self.make_wrapper(base_url="https://proxy.example.com/v1beta/")

        self.assertEqual(wrapper._model_url("gemini-2.5-flash", "generateContent"),
                         "https://proxy.example.com/v1beta/models/gemini-2.5-flash:generateContent")

    def test_express_mode_url(self):
        wrapper = self.make_wrapper(vertex=True)

        self.assertEqual(wrapper._model_url("gemini-3.8-flash", "generateContent"),
                         f"{GLOBAL_ROOT}/publishers/google/models/gemini-3.8-flash:generateContent")

    def test_express_mode_api_version_override_url(self):
        wrapper = self.make_wrapper(vertex=True, api_version="v1")

        self.assertEqual(wrapper._model_url("gemini-3.8-flash", "generateContent"),
                         "https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-3.8-flash:generateContent")

    def test_project_global_url(self):
        wrapper = self.make_wrapper(project_id="my-project")

        self.assertEqual(
            wrapper._model_url("gemini-3.8-flash", "generateContent"),
            f"{GLOBAL_ROOT}/projects/my-project/locations/global/publishers/google/models/"
            "gemini-3.8-flash:generateContent")

    def test_project_regional_url(self):
        wrapper = self.make_wrapper(project_id="my-project", location="us-central1")

        self.assertEqual(
            wrapper._model_url("gemini-3.8-flash", "generateContent"),
            "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/my-project/locations/us-central1/"
            "publishers/google/models/gemini-3.8-flash:generateContent")

    def test_project_multi_regional_urls(self):
        for location in ("us", "eu"):
            with self.subTest(location=location):
                wrapper = self.make_wrapper(project_id="my-project", location=location)

                self.assertEqual(
                    wrapper._model_url("gemini-3.8-flash", "generateContent"),
                    f"https://aiplatform.{location}.rep.googleapis.com/v1beta1/projects/my-project/"
                    f"locations/{location}/publishers/google/models/gemini-3.8-flash:generateContent")

    def test_per_call_location_overrides_wrapper_location(self):
        wrapper = self.make_wrapper(project_id="my-project")

        self.assertEqual(
            wrapper._model_url("veo-3.1-generate-001", "predictLongRunning", "us-east5"),
            "https://us-east5-aiplatform.googleapis.com/v1beta1/projects/my-project/locations/us-east5/"
            "publishers/google/models/veo-3.1-generate-001:predictLongRunning")

    def test_full_project_model_name_is_used_as_is(self):
        model = "projects/other/locations/europe-west4/publishers/google/models/gemini-3.8-flash"
        for wrapper in (self.make_wrapper(project_id="my-project"), self.make_wrapper(vertex=True)):
            with self.subTest(project_id=wrapper.project_id):
                self.assertEqual(
                    wrapper._model_url(model, "generateContent"),
                    f"https://europe-west4-aiplatform.googleapis.com/v1beta1/{model}:generateContent")

    def test_full_project_model_name_with_global_location(self):
        wrapper = self.make_wrapper(project_id="my-project", location="us-central1")
        model = "projects/other/locations/global/publishers/google/models/gemini-3.8-flash"

        self.assertEqual(wrapper._model_url(model, "generateContent"),
                         f"{GLOBAL_ROOT}/{model}:generateContent")

    def test_full_project_name_does_not_need_adc(self):
        wrapper = self.make_wrapper(api_key=None, vertex=True)
        model = "projects/other/locations/us-central1/endpoints/123"

        with google_auth_missing():
            url = wrapper._model_url(model, "generateContent")

        self.assertEqual(url, f"https://us-central1-aiplatform.googleapis.com/v1beta1/{model}:generateContent")

    def test_publisher_path_in_express_mode(self):
        wrapper = self.make_wrapper(vertex=True)

        self.assertEqual(wrapper._model_url("publishers/anthropic/models/claude-x", "rawPredict"),
                         f"{GLOBAL_ROOT}/publishers/anthropic/models/claude-x:rawPredict")

    def test_publisher_path_with_project(self):
        wrapper = self.make_wrapper(project_id="my-project")

        self.assertEqual(
            wrapper._model_url("publishers/anthropic/models/claude-x", "rawPredict"),
            f"{GLOBAL_ROOT}/projects/my-project/locations/global/publishers/anthropic/models/claude-x:rawPredict")

    def test_models_prefix_on_vertex(self):
        wrapper = self.make_wrapper(vertex=True)

        self.assertEqual(wrapper._model_url("models/gemini-3.8-flash", "generateContent"),
                         f"{GLOBAL_ROOT}/publishers/google/models/gemini-3.8-flash:generateContent")

    def test_publisher_slash_model_on_vertex(self):
        wrapper = self.make_wrapper(vertex=True)

        self.assertEqual(wrapper._model_url("meta/llama-4-maverick", "generateContent"),
                         f"{GLOBAL_ROOT}/publishers/meta/models/llama-4-maverick:generateContent")

    def test_vertex_base_url_override(self):
        wrapper = self.make_wrapper(project_id="my-project", base_url="https://private.example.com/v1beta1/")

        self.assertEqual(
            wrapper._model_url("gemini-3.8-flash", "generateContent"),
            "https://private.example.com/v1beta1/projects/my-project/locations/global/publishers/google/models/"
            "gemini-3.8-flash:generateContent")

    def test_empty_model_is_rejected(self):
        for vertex in (False, True):
            for model in (None, "", "   "):
                with self.subTest(vertex=vertex, model=model):
                    wrapper = self.make_wrapper(vertex=vertex)
                    with self.assertRaises(ValueError):
                        wrapper._model_url(model, "generateContent")

    def test_model_name_whitespace_is_stripped(self):
        wrapper = self.make_wrapper()

        self.assertEqual(wrapper._model_url("  gemini-2.5-flash ", "generateContent"),
                         f"{DEV_ROOT}/models/gemini-2.5-flash:generateContent")

    def test_vertex_host_by_location(self):
        wrapper = self.make_wrapper(vertex=True)

        self.assertEqual(wrapper._vertex_host(None), "https://aiplatform.googleapis.com")
        self.assertEqual(wrapper._vertex_host("global"), "https://aiplatform.googleapis.com")
        self.assertEqual(wrapper._vertex_host("us"), "https://aiplatform.us.rep.googleapis.com")
        self.assertEqual(wrapper._vertex_host("eu"), "https://aiplatform.eu.rep.googleapis.com")
        self.assertEqual(wrapper._vertex_host("asia-northeast1"), "https://asia-northeast1-aiplatform.googleapis.com")

    def test_vertex_project_url(self):
        wrapper = self.make_wrapper(vertex=True)

        self.assertEqual(wrapper._vertex_project_url("p", "us-central1", "reasoningEngines"),
                         "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/p/locations/us-central1/"
                         "reasoningEngines")

    def test_name_url_uses_location_from_name(self):
        wrapper = self.make_wrapper(vertex=True)
        name = "projects/p/locations/us-east5/reasoningEngines/42"

        self.assertEqual(wrapper._name_url(name), f"https://us-east5-aiplatform.googleapis.com/v1beta1/{name}")

    def test_name_url_uses_vertex_base_url(self):
        wrapper = self.make_wrapper(vertex=True, base_url="https://private.example.com/v1beta1")
        name = "projects/p/locations/us-east5/reasoningEngines/42"

        self.assertEqual(wrapper._name_url(name), f"https://private.example.com/v1beta1/{name}")

    def test_adc_project_is_discovered_for_urls(self):
        credentials = FakeCredentials(token="adc-token-123456", valid=True)
        wrapper = self.make_wrapper(api_key=None, vertex=True)

        with fake_google_auth(default_result=(credentials, "adc-project")) as auth:
            url = wrapper._model_url("gemini-3.8-flash", "generateContent")

        auth.default.assert_called_once()
        self.assertEqual(wrapper.project_id, "adc-project")
        self.assertEqual(wrapper.location, "global")
        self.assertEqual(
            url, f"{GLOBAL_ROOT}/projects/adc-project/locations/global/publishers/google/models/"
                 "gemini-3.8-flash:generateContent")

    def test_access_token_without_project_is_rejected(self):
        # Express (publisher) URLs take API keys only, so OAuth without a project is an error.
        wrapper = self.make_wrapper(api_key=None, access_token=ACCESS_TOKEN)

        with self.assertRaises(GoogleAIError) as caught:
            wrapper._model_url("gemini-3.8-flash", "generateContent")
        self.assertIn("needs a project", str(caught.exception))


# ----------------------------------------------------------------------
# Capability locations
# ----------------------------------------------------------------------
class TestCapabilityLocations(GenAITestCase):
    def test_capability_defaults_when_location_not_set(self):
        wrapper = self.make_wrapper(project_id="my-project")

        for capability, location in VERTEX_LOCATIONS.items():
            with self.subTest(capability=capability):
                self.assertEqual(wrapper._location_for(capability), location)

    def test_configured_capability_locations(self):
        for capability in ("video_generation", "live", "music", "imagen", "agent_engine"):
            with self.subTest(capability=capability):
                self.assertEqual(VERTEX_LOCATIONS[capability], "us-central1")

    def test_unknown_capability_uses_wrapper_location(self):
        wrapper = self.make_wrapper(project_id="my-project")

        self.assertEqual(wrapper._location_for("text"), "global")
        self.assertEqual(wrapper._location_for(), "global")

    def test_explicit_call_location_wins(self):
        wrapper = self.make_wrapper(project_id="my-project")

        self.assertEqual(wrapper._location_for("video_generation", "europe-west4"), "europe-west4")

    def test_explicit_wrapper_location_wins_over_capability_default(self):
        wrapper = self.make_wrapper(project_id="my-project", location="europe-west4")

        self.assertEqual(wrapper._location_for("video_generation"), "europe-west4")
        self.assertEqual(wrapper._location_for("live"), "europe-west4")

    def test_explicit_global_location_is_respected(self):
        wrapper = self.make_wrapper(project_id="my-project", location="global")

        self.assertEqual(wrapper._location_for("imagen"), "global")

    def test_env_location_counts_as_explicit(self):
        os.environ["GOOGLE_CLOUD_LOCATION"] = "asia-northeast1"
        wrapper = self.make_wrapper(project_id="my-project")

        self.assertEqual(wrapper._location_for("music"), "asia-northeast1")

    def test_express_mode_without_location(self):
        wrapper = self.make_wrapper(vertex=True)

        self.assertEqual(wrapper._location_for("video_generation"), "us-central1")
        self.assertIsNone(wrapper._location_for("text"))


# ----------------------------------------------------------------------
# Auth headers
# ----------------------------------------------------------------------
class TestAuthHeaders(GenAITestCase):
    def test_api_key_header_on_developer_api(self):
        self.assertEqual(self.make_wrapper()._auth_headers(), {"x-goog-api-key": API_KEY})

    def test_api_key_header_on_vertex(self):
        self.assertEqual(self.make_wrapper(vertex=True)._auth_headers(), {"x-goog-api-key": API_KEY})

    def test_api_key_wins_over_access_token_and_quota_project(self):
        wrapper = self.make_wrapper(vertex=True, access_token=ACCESS_TOKEN, quota_project_id="billing")

        self.assertEqual(wrapper._auth_headers(), {"x-goog-api-key": API_KEY})

    def test_bearer_access_token_string(self):
        wrapper = self.make_wrapper(api_key=None, access_token=ACCESS_TOKEN)

        self.assertEqual(wrapper._auth_headers(), {"Authorization": f"Bearer {ACCESS_TOKEN}"})

    def test_bearer_access_token_callable_is_called_each_time(self):
        tokens = iter(["token-one-123456", "token-two-123456"])
        wrapper = self.make_wrapper(api_key=None, access_token=lambda: next(tokens))

        self.assertEqual(wrapper._auth_headers()["Authorization"], "Bearer token-one-123456")
        self.assertEqual(wrapper._auth_headers()["Authorization"], "Bearer token-two-123456")

    def test_access_token_callable_returning_none_falls_back_to_adc(self):
        credentials = FakeCredentials(token="adc-token-123456", valid=True)
        wrapper = self.make_wrapper(api_key=None, access_token=lambda: None, project_id="p")

        with fake_google_auth(default_result=(credentials, None)):
            headers = wrapper._auth_headers()

        self.assertEqual(headers["Authorization"], "Bearer adc-token-123456")

    def test_valid_credentials_are_not_refreshed(self):
        credentials = FakeCredentials(token="cred-token-123456", valid=True)
        wrapper = self.make_wrapper(api_key=None, credentials=credentials)

        with fake_google_auth() as auth:
            headers = wrapper._auth_headers()

        self.assertEqual(headers, {"Authorization": "Bearer cred-token-123456"})
        self.assertEqual(credentials.refresh_requests, [])
        auth.default.assert_not_called()

    def test_invalid_credentials_are_refreshed(self):
        credentials = FakeCredentials(token=None, valid=False)
        wrapper = self.make_wrapper(api_key=None, credentials=credentials)

        with fake_google_auth() as auth:
            headers = wrapper._auth_headers()

        self.assertEqual(headers["Authorization"], "Bearer refreshed-token-123456")
        self.assertEqual(len(credentials.refresh_requests), 1)
        self.assertIsInstance(credentials.refresh_requests[0], auth.Request)

    def test_expired_credentials_with_stale_token_are_refreshed(self):
        credentials = FakeCredentials(token="stale-token-123456", valid=False)
        wrapper = self.make_wrapper(api_key=None, credentials=credentials)

        with fake_google_auth():
            headers = wrapper._auth_headers()

        self.assertEqual(headers["Authorization"], "Bearer refreshed-token-123456")

    def test_quota_project_header_from_option(self):
        wrapper = self.make_wrapper(api_key=None, access_token=ACCESS_TOKEN, quota_project_id="billing-project")

        self.assertEqual(wrapper._auth_headers()["x-goog-user-project"], "billing-project")

    def test_quota_project_header_from_credentials(self):
        credentials = FakeCredentials(token="cred-token-123456", valid=True, quota_project_id="cred-billing")
        wrapper = self.make_wrapper(api_key=None, credentials=credentials)

        with fake_google_auth():
            headers = wrapper._auth_headers()

        self.assertEqual(headers["x-goog-user-project"], "cred-billing")

    def test_quota_project_option_wins_over_credentials(self):
        credentials = FakeCredentials(token="cred-token-123456", valid=True, quota_project_id="cred-billing")
        wrapper = self.make_wrapper(api_key=None, credentials=credentials, quota_project_id="option-billing")

        with fake_google_auth():
            headers = wrapper._auth_headers()

        self.assertEqual(headers["x-goog-user-project"], "option-billing")

    def test_no_quota_project_header_by_default(self):
        wrapper = self.make_wrapper(api_key=None, access_token=ACCESS_TOKEN)

        self.assertNotIn("x-goog-user-project", wrapper._auth_headers())

    def test_adc_default_is_loaded_once_with_cloud_scope(self):
        credentials = FakeCredentials(token="adc-token-123456", valid=True)
        wrapper = self.make_wrapper(api_key=None, vertex=True, project_id="my-project")

        with fake_google_auth(default_result=(credentials, "adc-project")) as auth:
            wrapper._auth_headers()
            wrapper._auth_headers()

        auth.default.assert_called_once_with(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        self.assertEqual(wrapper.project_id, "my-project")  # an explicit project is not replaced

    def test_adc_failure_raises_google_ai_error(self):
        wrapper = self.make_wrapper(api_key=None, vertex=True)

        class DefaultCredentialsError(Exception):
            pass

        with fake_google_auth(default_error=DefaultCredentialsError("no ADC")):
            with self.assertRaises(GoogleAIError) as caught:
                wrapper._auth_headers()

        self.assertIn("Application Default Credentials", str(caught.exception))
        self.assertIn("DefaultCredentialsError", str(caught.exception))

    def test_refresh_failure_is_redacted(self):
        credentials = FakeCredentials(token="stale-secret-token-123", valid=False,
                                      refresh_error=RuntimeError("refresh failed for stale-secret-token-123"))
        wrapper = self.make_wrapper(api_key=None, credentials=credentials)

        with fake_google_auth():
            with self.assertRaises(GoogleAIError) as caught:
                wrapper._auth_headers()

        self.assertIn("Could not refresh Google credentials", str(caught.exception))
        self.assertNotIn("stale-secret-token-123", str(caught.exception))

    def test_missing_google_auth_raises_google_ai_error(self):
        wrapper = self.make_wrapper(api_key=None, vertex=True, project_id="my-project")

        with google_auth_missing():
            with self.assertRaises(GoogleAIError) as caught:
                wrapper._auth_headers()

        self.assertIn("google-auth", str(caught.exception))
        self.assertIsNone(caught.exception.__cause__)

    def test_developer_api_without_key_raises_google_ai_error(self):
        wrapper = self.make_wrapper(api_key=None)

        with google_auth_missing():
            with self.assertRaises(GoogleAIError) as caught:
                wrapper._auth_headers()

        self.assertIn("needs an API key", str(caught.exception))

    def test_request_sends_auth_and_content_type_headers(self):
        wrapper = self.make_wrapper(api_key=None, access_token=ACCESS_TOKEN, quota_project_id="billing",
                                    project_id="proj")
        wrapper.session.post.return_value = FakeResponse(text_chunk("ok"))

        wrapper.generate_content("hi", model="gemini-3.8-flash")

        _, kwargs = self.post_kwargs(wrapper)
        self.assertEqual(kwargs["headers"], {"Content-Type": "application/json",
                                             "Authorization": f"Bearer {ACCESS_TOKEN}",
                                             "x-goog-user-project": "billing"})


# ----------------------------------------------------------------------
# Errors and redaction
# ----------------------------------------------------------------------
class TestErrorsAndRedaction(GenAITestCase):
    def _error_from(self, wrapper, response):
        wrapper.session.post.return_value = response
        with self.assertRaises(GoogleAIError) as caught:
            wrapper.generate_content("hi", model="gemini-x")
        return caught.exception

    def test_api_key_is_redacted_from_message_and_details(self):
        wrapper = self.make_wrapper()
        body = {"error": {"code": 400, "message": f"API key {API_KEY} not valid", "status": "INVALID_ARGUMENT"}}
        response = real_error_response(400, body, f"{DEV_ROOT}/models/gemini-x:generateContent?key={API_KEY}")

        error = self._error_from(wrapper, response)

        self.assertNotIn(API_KEY, str(error))
        self.assertNotIn(API_KEY, json.dumps(error.details))
        self.assertIn("<redacted>", str(error))
        self.assertTrue(str(error).startswith("Gemini API error: 400 Client Error"))

    def test_access_token_is_redacted(self):
        wrapper = self.make_wrapper(api_key=None, access_token=ACCESS_TOKEN, project_id="proj")
        body = {"error": {"code": 401, "message": f"Token {ACCESS_TOKEN} expired", "status": "UNAUTHENTICATED"}}

        error = self._error_from(wrapper, FakeResponse(body, status_code=401))

        self.assertNotIn(ACCESS_TOKEN, str(error))
        self.assertNotIn(ACCESS_TOKEN, json.dumps(error.details))
        self.assertEqual(error.status_code, 401)

    def test_credentials_token_is_redacted(self):
        credentials = FakeCredentials(token="cred-secret-token-123456", valid=True)
        wrapper = self.make_wrapper(api_key=None, credentials=credentials)
        body = {"error": {"message": "bad token cred-secret-token-123456"}}

        with fake_google_auth():
            error = self._error_from(wrapper, FakeResponse(body, status_code=401))

        self.assertNotIn("cred-secret-token-123456", str(error))
        self.assertNotIn("cred-secret-token-123456", json.dumps(error.details))

    # SUSPECTED SOURCE BUG (low severity): GoogleAIError promises that access tokens are removed, but
    # GoogleAIWrapper._secrets() (intelli/wrappers/googleai_wrapper.py:919-926) only collects a *string*
    # access_token, so a token returned by a callable access_token is never redacted.
    def test_callable_access_token_is_redacted(self):
        token = "ya29.callable-secret-token"
        wrapper = self.make_wrapper(api_key=None, access_token=lambda: token, project_id="proj")

        error = self._error_from(wrapper, FakeResponse({"error": {"message": f"bad token {token}"}},
                                                       status_code=401))

        self.assertNotIn(token, str(error))

    def test_query_key_parameter_is_redacted(self):
        wrapper = self.make_wrapper()

        self.assertEqual(wrapper._redact("for url: https://x.googleapis.com/a?key=OTHER-KEY-999&alt=sse"),
                         "for url: https://x.googleapis.com/a?key=<redacted>&alt=sse")
        self.assertEqual(wrapper._redact("https://x.googleapis.com/a?alt=sse&key=OTHER-KEY-999"),
                         "https://x.googleapis.com/a?alt=sse&key=<redacted>")

    def test_query_key_of_another_key_is_redacted_in_errors(self):
        wrapper = self.make_wrapper()
        response = real_error_response(403, {"error": {"message": "denied"}},
                                       f"{DEV_ROOT}/models/gemini-x:generateContent?key=SOMEONE-ELSE-KEY")

        error = self._error_from(wrapper, response)

        self.assertNotIn("SOMEONE-ELSE-KEY", str(error))
        self.assertIn("?key=<redacted>", str(error))

    def test_redact_obj_keeps_structure(self):
        wrapper = self.make_wrapper()
        details = {"error": {"message": f"key {API_KEY}", "details": [{"url": "https://a?key=abcdefgh"}]}}

        redacted = wrapper._redact_obj(details)

        self.assertEqual(redacted, {"error": {"message": "key <redacted>",
                                              "details": [{"url": "https://a?key=<redacted>"}]}})
        self.assertIsNone(wrapper._redact_obj(None))
        self.assertEqual(wrapper._redact_obj(f"text {API_KEY}"), "text <redacted>")

    def test_short_values_are_not_treated_as_secrets(self):
        wrapper = self.make_wrapper(api_key="abc", vertex=True)

        self.assertEqual(wrapper._secrets(), [])
        self.assertEqual(wrapper._redact("abc"), "abc")

    def test_error_has_status_code_and_details(self):
        wrapper = self.make_wrapper()
        body = {"error": {"code": 429, "message": "Resource exhausted", "status": "RESOURCE_EXHAUSTED"}}

        error = self._error_from(wrapper, FakeResponse(body, status_code=429))

        self.assertIsInstance(error, Exception)
        self.assertEqual(error.status_code, 429)
        self.assertEqual(error.details, body)
        self.assertIn("RESOURCE_EXHAUSTED", str(error))
        self.assertIn(" - Details: ", str(error))

    def test_error_details_fall_back_to_text(self):
        wrapper = self.make_wrapper()

        error = self._error_from(wrapper, FakeResponse(status_code=502, text="<html>Bad gateway</html>"))

        self.assertEqual(error.status_code, 502)
        self.assertEqual(error.details, "<html>Bad gateway</html>")

    def test_error_details_text_is_truncated(self):
        wrapper = self.make_wrapper()

        error = self._error_from(wrapper, FakeResponse(status_code=500, text="x" * 5000))

        self.assertEqual(len(error.details), 2000)

    def test_error_without_response_has_no_details(self):
        wrapper = self.make_wrapper()
        wrapper.session.post.side_effect = requests.exceptions.ConnectionError("connection refused")

        with self.assertRaises(GoogleAIError) as caught:
            wrapper.generate_content("hi", model="gemini-x")

        self.assertIsNone(caught.exception.status_code)
        self.assertIsNone(caught.exception.details)
        self.assertEqual(str(caught.exception), "Gemini API error: connection refused")

    def test_original_exception_is_not_chained(self):
        # The original exception text holds the request URL, which can carry ?key=...
        wrapper = self.make_wrapper()

        error = self._error_from(wrapper, FakeResponse({"error": {}}, status_code=400))

        self.assertIsNone(error.__cause__)
        self.assertTrue(error.__suppress_context__)

    def test_api_key_service_blocked_hint_on_developer_api(self):
        wrapper = self.make_wrapper()
        body = {"error": {"code": 403, "status": "PERMISSION_DENIED",
                          "details": [{"reason": "API_KEY_SERVICE_BLOCKED"}]}}

        error = self._error_from(wrapper, FakeResponse(body, status_code=403))

        self.assertIn("vertex=True", str(error))

    def test_api_key_service_blocked_hint_not_on_vertex(self):
        wrapper = self.make_wrapper(vertex=True)
        body = {"error": {"code": 403, "details": [{"reason": "API_KEY_SERVICE_BLOCKED"}]}}

        error = self._error_from(wrapper, FakeResponse(body, status_code=403))

        self.assertNotIn("hint", str(error))

    def test_oauth_hint_on_vertex(self):
        wrapper = self.make_wrapper(vertex=True)
        body = {"error": {"code": 401, "message": "API keys are not supported by this API. Expected OAuth2 "
                                                  "access token or other authentication credentials."}}

        error = self._error_from(wrapper, FakeResponse(body, status_code=401))

        self.assertIn("needs OAuth", str(error))
        self.assertIn("access_token", str(error))

    def test_oauth_hint_not_on_developer_api(self):
        wrapper = self.make_wrapper()
        body = {"error": {"code": 401, "message": "API keys are not supported by this API."}}

        error = self._error_from(wrapper, FakeResponse(body, status_code=401))

        self.assertNotIn("hint", str(error))

    def test_resource_project_invalid_hint_on_vertex(self):
        wrapper = self.make_wrapper(vertex=True)
        body = {"error": {"code": 400, "details": [{"reason": "RESOURCE_PROJECT_INVALID"}]}}

        error = self._error_from(wrapper, FakeResponse(body, status_code=400))

        self.assertIn("project_id=", str(error))

    def test_resource_project_invalid_hint_not_on_developer_api(self):
        wrapper = self.make_wrapper()
        body = {"error": {"code": 400, "details": [{"reason": "RESOURCE_PROJECT_INVALID"}]}}

        error = self._error_from(wrapper, FakeResponse(body, status_code=400))

        self.assertNotIn("hint", str(error))


# ----------------------------------------------------------------------
# _request / _request_json
# ----------------------------------------------------------------------
class TestRequestHelpers(GenAITestCase):
    def test_request_passes_body_timeout_and_headers(self):
        wrapper = self.make_wrapper(timeout=42)
        wrapper.session.post.return_value = FakeResponse({"ok": True})

        wrapper._request("POST", "https://example.com/x", {"a": 1}, headers={"X-Extra": "1"})

        url, kwargs = self.post_kwargs(wrapper)
        self.assertEqual(url, "https://example.com/x")
        self.assertEqual(kwargs["json"], {"a": 1})
        self.assertEqual(kwargs["timeout"], 42)
        self.assertEqual(kwargs["headers"]["X-Extra"], "1")
        self.assertEqual(kwargs["headers"]["x-goog-api-key"], API_KEY)
        self.assertNotIn("params", kwargs)
        self.assertNotIn("stream", kwargs)

    def test_request_without_auth(self):
        wrapper = self.make_wrapper()
        wrapper.session.get.return_value = FakeResponse({})

        wrapper._request("GET", "https://example.com/x", auth=False)

        self.assertEqual(wrapper.session.get.call_args.kwargs["headers"], {"Content-Type": "application/json"})

    def test_request_uses_matching_session_method(self):
        wrapper = self.make_wrapper()
        for method in ("GET", "POST", "DELETE", "PATCH"):
            with self.subTest(method=method):
                sender = getattr(wrapper.session, method.lower())
                sender.return_value = FakeResponse({})

                wrapper._request(method, "https://example.com/x")

                sender.assert_called()

    def test_request_json_adds_snake_aliases_by_default(self):
        wrapper = self.make_wrapper()
        wrapper.session.post.return_value = FakeResponse({"inlineData": {"mimeType": "image/png", "data": "QQ=="}})

        data = wrapper._request_json("POST", "https://example.com/x", {})

        self.assertEqual(data["inline_data"]["mime_type"], "image/png")
        self.assertEqual(data["inlineData"]["mimeType"], "image/png")

    def test_request_json_without_alias(self):
        wrapper = self.make_wrapper()
        wrapper.session.post.return_value = FakeResponse({"inlineData": {"mimeType": "image/png"}})

        data = wrapper._request_json("POST", "https://example.com/x", {}, alias=False)

        self.assertEqual(data, {"inlineData": {"mimeType": "image/png"}})

    def test_request_json_empty_body_returns_empty_dict(self):
        wrapper = self.make_wrapper()
        wrapper.session.delete.return_value = FakeResponse(text="")

        self.assertEqual(wrapper._request_json("DELETE", "https://example.com/x"), {})

    def test_request_json_non_json_body_raises(self):
        wrapper = self.make_wrapper()
        wrapper.session.post.return_value = FakeResponse(text="<html>oops</html>")

        with self.assertRaises(GoogleAIError) as caught:
            wrapper._request_json("POST", "https://example.com/x", {}, error_prefix="Custom error")

        self.assertEqual(str(caught.exception), "Custom error: the API returned a response that is not JSON")


# ----------------------------------------------------------------------
# Request body preparation
# ----------------------------------------------------------------------
class TestPrepareBody(GenAITestCase):
    def test_string_prompt_on_both_backends(self):
        for vertex in (False, True):
            with self.subTest(vertex=vertex):
                body = self.make_wrapper(vertex=vertex)._prepare_body("Hello")

                self.assertEqual(body, {"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]})

    def test_developer_body_is_unchanged(self):
        params = {"model": "models/gemini-2.5-flash",
                  "contents": [{"parts": [{"text": "hi"}]}],
                  "generationConfig": {"temperature": 0.1}}

        body = self.make_wrapper()._prepare_body(params)

        self.assertEqual(body, params)

    def test_developer_contents_get_no_role(self):
        body = self.make_wrapper()._prepare_body({"contents": [{"parts": [{"text": "hi"}]}]})

        self.assertNotIn("role", body["contents"][0])

    def test_vertex_adds_user_role(self):
        body = self.make_wrapper(vertex=True)._prepare_body({"contents": [{"parts": [{"text": "hi"}]}]})

        self.assertEqual(body["contents"], [{"role": "user", "parts": [{"text": "hi"}]}])

    def test_vertex_function_call_content_gets_model_role(self):
        for key in ("functionCall", "function_call"):
            with self.subTest(key=key):
                params = {"contents": [{"parts": [{key: {"name": "get_weather", "args": {}}}]}]}

                body = self.make_wrapper(vertex=True)._prepare_body(params)

                self.assertEqual(body["contents"][0]["role"], "model")

    def test_vertex_keeps_existing_role(self):
        params = {"contents": [{"role": "model", "parts": [{"text": "earlier"}]},
                               {"role": "user", "parts": [{"text": "now"}]}]}

        body = self.make_wrapper(vertex=True)._prepare_body(params)

        self.assertEqual([c["role"] for c in body["contents"]], ["model", "user"])

    def test_vertex_does_not_mutate_input(self):
        params = {"contents": [{"parts": [{"text": "hi"}]}], "model": "gemini-x"}
        original = json.loads(json.dumps(params))

        self.make_wrapper(vertex=True)._prepare_body(params)

        self.assertEqual(params, original)

    def test_model_removed_only_on_vertex(self):
        params = {"model": "gemini-x", "contents": [{"role": "user", "parts": [{"text": "hi"}]}]}

        self.assertNotIn("model", self.make_wrapper(vertex=True)._prepare_body(params))
        self.assertEqual(self.make_wrapper()._prepare_body(params)["model"], "gemini-x")

    def test_single_dict_contents_becomes_list(self):
        body = self.make_wrapper()._prepare_body({"contents": {"parts": [{"text": "hi"}]}})

        self.assertEqual(body["contents"], [{"parts": [{"text": "hi"}]}])

    def test_string_contents_becomes_user_content(self):
        body = self.make_wrapper()._prepare_body({"contents": "hi"})

        self.assertEqual(body["contents"], [{"role": "user", "parts": [{"text": "hi"}]}])

    def test_string_items_in_contents_list(self):
        body = self.make_wrapper(vertex=True)._prepare_body({"contents": ["one", {"parts": [{"text": "two"}]}]})

        self.assertEqual(body["contents"], [{"role": "user", "parts": [{"text": "one"}]},
                                            {"role": "user", "parts": [{"text": "two"}]}])

    def test_string_system_instruction(self):
        for key in ("systemInstruction", "system_instruction"):
            with self.subTest(key=key):
                body = self.make_wrapper()._prepare_body({key: "Be brief", "contents": []})

                self.assertEqual(body["systemInstruction"], {"parts": [{"text": "Be brief"}]})
                self.assertNotIn("system_instruction", body)

    def test_dict_system_instruction_is_kept(self):
        instruction = {"parts": [{"text": "Be brief"}]}

        body = self.make_wrapper()._prepare_body({"system_instruction": instruction, "contents": []})

        self.assertEqual(body["systemInstruction"], instruction)

    def test_snake_case_keys_are_camelized(self):
        params = {"contents": [{"parts": [{"inline_data": {"mime_type": "image/png", "data": "QQ=="}}]}],
                  "generation_config": {"response_mime_type": "application/json", "temperature": 0}}

        body = self.make_wrapper()._prepare_body(params)

        self.assertEqual(body["contents"][0]["parts"][0], {"inlineData": {"mimeType": "image/png", "data": "QQ=="}})
        self.assertEqual(body["generationConfig"], {"responseMimeType": "application/json", "temperature": 0})


# ----------------------------------------------------------------------
# Key normalization helpers
# ----------------------------------------------------------------------
class TestKeyNormalization(GenAITestCase):
    def setUp(self):
        super().setUp()
        self.wrapper = self.make_wrapper()

    def test_camelize_known_keys_recursively(self):
        data = {"generation_config": {"speech_config": {"voice_config": {"prebuilt_voice_config":
                                                                         {"voice_name": "Kore"}}}},
                "contents": [{"parts": [{"file_data": {"mime_type": "video/mp4", "file_uri": "gs://b/v.mp4"}}]}]}

        self.assertEqual(self.wrapper._camelize(data), {
            "generationConfig": {"speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}}},
            "contents": [{"parts": [{"fileData": {"mimeType": "video/mp4", "fileUri": "gs://b/v.mp4"}}]}]})

    def test_camelize_keeps_unknown_keys_and_scalars(self):
        self.assertEqual(self.wrapper._camelize({"thinking_config": {"thinking_budget": 0}}),
                         {"thinking_config": {"thinking_budget": 0}})
        self.assertEqual(self.wrapper._camelize("text"), "text")
        self.assertEqual(self.wrapper._camelize([1, None]), [1, None])

    def test_snake_alias_adds_aliases_and_keeps_camel_keys(self):
        data = {"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "audio/wav", "data": "AA"}}]}}]}

        aliased = self.wrapper._snake_alias(data)
        part = aliased["candidates"][0]["content"]["parts"][0]

        self.assertEqual(part["inlineData"]["mimeType"], "audio/wav")
        self.assertEqual(part["inline_data"]["mime_type"], "audio/wav")
        self.assertEqual(part["inline_data"]["data"], "AA")

    def test_snake_alias_does_not_overwrite_existing_snake_key(self):
        aliased = self.wrapper._snake_alias({"mime_type": "keep/me", "mimeType": "other/value"})

        self.assertEqual(aliased["mime_type"], "keep/me")

    def test_snake_alias_does_not_mutate_input(self):
        data = {"inlineData": {"mimeType": "image/png"}}

        self.wrapper._snake_alias(data)

        self.assertEqual(data, {"inlineData": {"mimeType": "image/png"}})

    def test_unalias_removes_only_aliases(self):
        data = {"inlineData": {"mimeType": "image/png", "data": "QQ=="}, "thoughtSignature": "sig"}

        self.assertEqual(self.wrapper._unalias(self.wrapper._snake_alias(data)), data)

    def test_unalias_keeps_lone_snake_keys(self):
        data = {"inline_data": {"mime_type": "image/png"}}

        self.assertEqual(self.wrapper._unalias(data), data)

    def test_unalias_handles_lists_and_scalars(self):
        self.assertEqual(self.wrapper._unalias([{"mimeType": "a", "mime_type": "a"}, 3]), [{"mimeType": "a"}, 3])
        self.assertEqual(self.wrapper._unalias("x"), "x")


# ----------------------------------------------------------------------
# Server-sent events
# ----------------------------------------------------------------------
class TestServerSentEvents(unittest.TestCase):
    def parse(self, lines):
        return list(GoogleAIWrapper._iter_sse(FakeResponse(lines=lines)))

    def test_events_are_separated_by_blank_lines(self):
        self.assertEqual(self.parse(['data: {"n": 1}', "", 'data: {"n": 2}', ""]), [{"n": 1}, {"n": 2}])

    def test_multi_line_data_is_joined(self):
        self.assertEqual(self.parse(['data: {"a":', "data: [1,", "data: 2]}", ""]), [{"a": [1, 2]}])

    def test_trailing_event_without_blank_line(self):
        self.assertEqual(self.parse(['data: {"n": 1}', "", 'data: {"n": 2}']), [{"n": 1}, {"n": 2}])

    def test_repeated_blank_lines_and_comments_are_ignored(self):
        lines = ["", ": keep-alive", "event: message", 'data: {"n": 1}', "", "", "id: 7", ""]

        self.assertEqual(self.parse(lines), [{"n": 1}])

    def test_data_without_space_and_bytes_lines(self):
        self.assertEqual(self.parse([b'data:{"n": 1}', b"", None, 'data:{"n": 2}']), [{"n": 1}, {"n": 2}])

    def test_whitespace_only_line_ends_event(self):
        self.assertEqual(self.parse(['data: {"n": 1}', "   ", 'data: {"n": 2}']), [{"n": 1}, {"n": 2}])

    def test_lines_are_read_as_bytes_and_decoded_as_utf8(self):
        # text/event-stream has no charset, so requests would decode it as ISO-8859-1
        response = FakeResponse(lines=['data: {"t": "caf\u00e9 \u0645\u0631\u062d\u0628\u0627 \u4f60\u597d"}'.encode("utf-8")])

        chunks = list(GoogleAIWrapper._iter_sse(response))

        self.assertFalse(response.decode_unicode)
        self.assertEqual(chunks, [{"t": "caf\u00e9 \u0645\u0631\u062d\u0628\u0627 \u4f60\u597d"}])


# ----------------------------------------------------------------------
# stream_generate_content
# ----------------------------------------------------------------------
class TestStreamGenerateContent(GenAITestCase):
    def test_parsed_mode_uses_sse_and_yields_aliased_chunks(self):
        wrapper = self.make_wrapper()
        image_chunk = {"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png",
                                                                             "data": "QQ=="}}]}}]}
        response = FakeResponse(lines=sse_event(text_chunk("Hel")) + sse_event(image_chunk))
        wrapper.session.post.return_value = response

        chunks = list(wrapper.stream_generate_content("hi", model="gemini-x"))

        url, kwargs = self.post_kwargs(wrapper)
        self.assertEqual(url, f"{DEV_ROOT}/models/gemini-x:streamGenerateContent")
        self.assertEqual(kwargs["params"], {"alt": "sse"})
        self.assertTrue(kwargs["stream"])
        self.assertEqual(kwargs["json"], {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]})
        self.assertEqual(len(chunks), 2)
        self.assertEqual(GoogleAIWrapper.extract_text(chunks[0]), "Hel")
        self.assertIn("inline_data", chunks[1]["candidates"][0]["content"]["parts"][0])
        self.assertTrue(response.closed)

    def test_raw_mode_yields_lines_without_sse(self):
        wrapper = self.make_wrapper()
        response = FakeResponse(lines=["[{", "", '"candidates": []', "}]"])
        wrapper.session.post.return_value = response

        lines = list(wrapper.stream_generate_content({"contents": []}, model="gemini-x", raw=True))

        url, kwargs = self.post_kwargs(wrapper)
        self.assertEqual(lines, ["[{", '"candidates": []', "}]"])
        self.assertNotIn("params", kwargs)
        self.assertNotIn("alt=sse", url)
        self.assertTrue(kwargs["stream"])
        self.assertTrue(response.closed)

    def test_vertex_stream_url(self):
        wrapper = self.make_wrapper(vertex=True)
        wrapper.session.post.return_value = FakeResponse(lines=[])

        list(wrapper.stream_generate_content("hi"))

        url, _ = self.post_kwargs(wrapper)
        self.assertEqual(url, f"{GLOBAL_ROOT}/publishers/google/models/{VERTEX_MODELS['text']}:streamGenerateContent")

    def test_model_selection_order(self):
        wrapper = self.make_wrapper()
        wrapper.session.post.return_value = FakeResponse(lines=[])

        cases = [({}, DEV_MODELS["text"]), ({"vision": True}, DEV_MODELS["vision"]),
                 ({"model_override": "override-model"}, "override-model"),
                 ({"model_override": "override-model", "model": "kw-model"}, "kw-model")]
        for kwargs, expected in cases:
            with self.subTest(kwargs=kwargs):
                list(wrapper.stream_generate_content("hi", **kwargs))

                url, _ = self.post_kwargs(wrapper)
                self.assertEqual(url, f"{DEV_ROOT}/models/{expected}:streamGenerateContent")

    def test_http_error_raises_stream_error(self):
        wrapper = self.make_wrapper()
        wrapper.session.post.return_value = FakeResponse({"error": {"message": "bad"}}, status_code=400)

        with self.assertRaises(GoogleAIError) as caught:
            list(wrapper.stream_generate_content("hi", model="gemini-x"))

        self.assertTrue(str(caught.exception).startswith("Gemini stream error:"))
        self.assertEqual(caught.exception.status_code, 400)

    def test_error_mid_stream_raises_and_closes(self):
        wrapper = self.make_wrapper()
        response = FakeResponse(lines=sse_event(text_chunk("Hel")) +
                                [requests.exceptions.ChunkedEncodingError("Connection broken")])
        wrapper.session.post.return_value = response
        stream = wrapper.stream_generate_content("hi", model="gemini-x")

        first = next(stream)
        with self.assertRaises(GoogleAIError) as caught:
            next(stream)

        self.assertEqual(GoogleAIWrapper.extract_text(first), "Hel")
        self.assertIn("Gemini stream error", str(caught.exception))
        self.assertTrue(response.closed)


# ----------------------------------------------------------------------
# generate_content / generate_text / stream_text
# ----------------------------------------------------------------------
class TestTextGeneration(GenAITestCase):
    def test_generate_content_default_model_developer(self):
        wrapper = self.make_wrapper()
        wrapper.session.post.return_value = FakeResponse(text_chunk("ok"))

        wrapper.generate_content({"contents": [{"parts": [{"text": "hi"}]}]})

        url, _ = self.post_kwargs(wrapper)
        self.assertEqual(url, f"{DEV_ROOT}/models/{DEV_MODELS['text']}:generateContent")

    def test_generate_content_default_model_vertex(self):
        wrapper = self.make_wrapper(vertex=True)
        wrapper.session.post.return_value = FakeResponse(text_chunk("ok"))

        wrapper.generate_content("hi")

        url, _ = self.post_kwargs(wrapper)
        self.assertEqual(url, f"{GLOBAL_ROOT}/publishers/google/models/{VERTEX_MODELS['text']}:generateContent")

    def test_generate_content_vision_default_model(self):
        for vertex, models in ((False, DEV_MODELS), (True, VERTEX_MODELS)):
            with self.subTest(vertex=vertex):
                wrapper = self.make_wrapper(vertex=vertex)
                wrapper.session.post.return_value = FakeResponse(text_chunk("ok"))

                wrapper.generate_content("hi", vision=True)

                url, _ = self.post_kwargs(wrapper)
                self.assertIn(f"/{models['vision']}:generateContent", url)

    def test_generate_content_model_wins_over_model_override(self):
        wrapper = self.make_wrapper()
        wrapper.session.post.return_value = FakeResponse(text_chunk("ok"))

        wrapper.generate_content("hi", False, "override-model", model="kw-model")

        url, _ = self.post_kwargs(wrapper)
        self.assertEqual(url, f"{DEV_ROOT}/models/kw-model:generateContent")

    def test_generate_content_model_override_positional(self):
        wrapper = self.make_wrapper()
        wrapper.session.post.return_value = FakeResponse(text_chunk("ok"))

        wrapper.generate_content("hi", False, "override-model")

        url, _ = self.post_kwargs(wrapper)
        self.assertEqual(url, f"{DEV_ROOT}/models/override-model:generateContent")

    def test_generate_content_returns_snake_aliases(self):
        wrapper = self.make_wrapper()
        response = {"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": "Q"}}]}}]}
        wrapper.session.post.return_value = FakeResponse(response)

        data = wrapper.generate_content("hi", model="gemini-x")

        self.assertEqual(data["candidates"][0]["content"]["parts"][0]["inline_data"]["mime_type"], "image/png")

    def test_default_model_unknown_kind_raises(self):
        with self.assertRaises(ValueError):
            self.make_wrapper()._default_model("no-such-kind")

    def test_generate_text_builds_full_request(self):
        wrapper = self.make_wrapper()
        wrapper.session.post.return_value = FakeResponse(text_chunk("Hello", " there"))
        history = [{"role": "user", "parts": [{"text": "earlier"}]}, {"role": "model", "parts": [{"text": "ok"}]}]
        tools = [{"functionDeclarations": [{"name": "f"}]}]

        text = wrapper.generate_text("Hi", system_instruction="Be brief",
                                     media=[(b"\x89PNG", "image/png"), "gs://bucket/doc.pdf"],
                                     generation_config={"response_mime_type": "text/plain", "temperature": 0.2},
                                     tools=tools, tool_config={"functionCallingConfig": {"mode": "AUTO"}},
                                     safety_settings=[{"category": "HARM_CATEGORY_HATE_SPEECH",
                                                       "threshold": "BLOCK_NONE"}],
                                     history=history)

        url, kwargs = self.post_kwargs(wrapper)
        body = kwargs["json"]
        self.assertEqual(text, "Hello there")
        self.assertEqual(url, f"{DEV_ROOT}/models/{DEV_MODELS['text']}:generateContent")
        self.assertEqual(body["contents"][:2], history)
        self.assertEqual(body["contents"][2], {"role": "user", "parts": [
            {"text": "Hi"},
            {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(b"\x89PNG").decode()}},
            {"fileData": {"mimeType": "application/pdf", "fileUri": "gs://bucket/doc.pdf"}}]})
        self.assertEqual(body["systemInstruction"], {"parts": [{"text": "Be brief"}]})
        self.assertEqual(body["generationConfig"], {"responseMimeType": "text/plain", "temperature": 0.2})
        self.assertEqual(body["tools"], tools)
        self.assertEqual(body["toolConfig"], {"functionCallingConfig": {"mode": "AUTO"}})
        self.assertEqual(body["safetySettings"][0]["threshold"], "BLOCK_NONE")

    def test_generate_text_minimal_body(self):
        wrapper = self.make_wrapper()
        wrapper.session.post.return_value = FakeResponse(text_chunk("ok"))

        wrapper.generate_text("Hi")

        _, kwargs = self.post_kwargs(wrapper)
        self.assertEqual(kwargs["json"], {"contents": [{"role": "user", "parts": [{"text": "Hi"}]}]})

    def test_generate_text_vertex_default_model(self):
        wrapper = self.make_wrapper(vertex=True, project_id="my-project")
        wrapper.session.post.return_value = FakeResponse(text_chunk("ok"))

        wrapper.generate_text("Hi")

        url, _ = self.post_kwargs(wrapper)
        self.assertEqual(url, f"{GLOBAL_ROOT}/projects/my-project/locations/global/publishers/google/models/"
                              f"{VERTEX_MODELS['text']}:generateContent")

    def test_generate_text_explicit_model(self):
        wrapper = self.make_wrapper(vertex=True)
        wrapper.session.post.return_value = FakeResponse(text_chunk("ok"))

        wrapper.generate_text("Hi", model="gemini-3.5-flash")

        url, _ = self.post_kwargs(wrapper)
        self.assertIn("/models/gemini-3.5-flash:generateContent", url)

    def test_generate_text_skips_thought_parts(self):
        wrapper = self.make_wrapper()
        response = {"candidates": [{"content": {"parts": [{"text": "thinking...", "thought": True},
                                                          {"text": "answer"}]}}]}
        wrapper.session.post.return_value = FakeResponse(response)

        self.assertEqual(wrapper.generate_text("Hi"), "answer")

    def test_stream_text_yields_only_text_chunks(self):
        wrapper = self.make_wrapper(vertex=True)
        function_chunk = {"candidates": [{"content": {"parts": [{"functionCall": {"name": "f", "args": {}}}]}}]}
        lines = sse_event(text_chunk("Hel")) + sse_event(function_chunk) + sse_event(text_chunk("lo"))
        wrapper.session.post.return_value = FakeResponse(lines=lines)

        chunks = list(wrapper.stream_text("Hi", system_instruction="Be brief"))

        url, kwargs = self.post_kwargs(wrapper)
        self.assertEqual(chunks, ["Hel", "lo"])
        self.assertIn(f"/models/{VERTEX_MODELS['text']}:streamGenerateContent", url)
        self.assertEqual(kwargs["json"]["systemInstruction"], {"parts": [{"text": "Be brief"}]})


# ----------------------------------------------------------------------
# count_tokens / compute_tokens
# ----------------------------------------------------------------------
class TestTokenCounting(GenAITestCase):
    def test_developer_count_tokens_plain_contents(self):
        wrapper = self.make_wrapper()
        wrapper.session.post.return_value = FakeResponse({"totalTokens": 3})

        result = wrapper.count_tokens({"contents": [{"parts": [{"text": "hi"}]}]})

        url, kwargs = self.post_kwargs(wrapper)
        self.assertEqual(result, {"totalTokens": 3})
        self.assertEqual(url, f"{DEV_ROOT}/models/{DEV_MODELS['text']}:countTokens")
        self.assertEqual(kwargs["json"], {"contents": [{"parts": [{"text": "hi"}]}]})

    def test_developer_count_tokens_wraps_extra_fields(self):
        wrapper = self.make_wrapper()
        wrapper.session.post.return_value = FakeResponse({"totalTokens": 9})
        tools = [{"googleSearch": {}}]

        wrapper.count_tokens({"contents": [{"parts": [{"text": "hi"}]}], "system_instruction": "Be brief",
                              "tools": tools, "generation_config": {"temperature": 0}},
                             model="gemini-3-flash-preview")

        url, kwargs = self.post_kwargs(wrapper)
        self.assertEqual(url, f"{DEV_ROOT}/models/gemini-3-flash-preview:countTokens")
        self.assertEqual(kwargs["json"], {"generateContentRequest": {
            "model": "models/gemini-3-flash-preview",
            "contents": [{"parts": [{"text": "hi"}]}],
            "systemInstruction": {"parts": [{"text": "Be brief"}]},
            "tools": tools,
            "generationConfig": {"temperature": 0}}})

    def test_developer_count_tokens_string_prompt(self):
        wrapper = self.make_wrapper()
        wrapper.session.post.return_value = FakeResponse({"totalTokens": 1})

        wrapper.count_tokens("hello")

        _, kwargs = self.post_kwargs(wrapper)
        self.assertEqual(kwargs["json"], {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]})

    def test_vertex_count_tokens_keeps_extra_fields(self):
        wrapper = self.make_wrapper(vertex=True)
        wrapper.session.post.return_value = FakeResponse({"totalTokens": 9, "totalBillableCharacters": 2})

        result = wrapper.count_tokens({"contents": [{"parts": [{"text": "hi"}]}], "system_instruction": "Be brief"})

        url, kwargs = self.post_kwargs(wrapper)
        self.assertEqual(result["totalTokens"], 9)
        self.assertEqual(url, f"{GLOBAL_ROOT}/publishers/google/models/{VERTEX_MODELS['text']}:countTokens")
        self.assertEqual(kwargs["json"], {"contents": [{"role": "user", "parts": [{"text": "hi"}]}],
                                          "systemInstruction": {"parts": [{"text": "Be brief"}]}})

    def test_count_tokens_error_prefix(self):
        wrapper = self.make_wrapper()
        wrapper.session.post.return_value = FakeResponse({"error": {}}, status_code=400)

        with self.assertRaises(GoogleAIError) as caught:
            wrapper.count_tokens("hi")

        self.assertTrue(str(caught.exception).startswith("Gemini countTokens error:"))

    def test_compute_tokens_requires_vertex(self):
        wrapper = self.make_wrapper()

        with self.assertRaises(GoogleAIError):
            wrapper.compute_tokens("hi")
        wrapper.session.post.assert_not_called()

    def test_compute_tokens_on_vertex_sends_contents_only(self):
        wrapper = self.make_wrapper(vertex=True)
        token_info = {"tokensInfo": [{"tokens": ["aGk="], "tokenIds": ["1"], "role": "user"}]}
        wrapper.session.post.return_value = FakeResponse(token_info)

        result = wrapper.compute_tokens({"contents": [{"parts": [{"text": "hi"}]}], "system_instruction": "x",
                                         "generation_config": {"temperature": 0}}, model="gemini-3.5-flash")

        url, kwargs = self.post_kwargs(wrapper)
        self.assertEqual(result, token_info)
        self.assertEqual(url, f"{GLOBAL_ROOT}/publishers/google/models/gemini-3.5-flash:computeTokens")
        self.assertEqual(kwargs["json"], {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]})


# ----------------------------------------------------------------------
# Chat sessions
# ----------------------------------------------------------------------
class TestChatSession(GenAITestCase):
    def test_start_chat_defaults(self):
        for vertex, models in ((False, DEV_MODELS), (True, VERTEX_MODELS)):
            with self.subTest(vertex=vertex):
                chat = self.make_wrapper(vertex=vertex).start_chat()

                self.assertIsInstance(chat, GoogleAIChatSession)
                self.assertEqual(chat.model, models["text"])
                self.assertEqual(chat.history, [])
                self.assertIsNone(chat.last_response)

    def test_history_grows_with_each_turn(self):
        wrapper = self.make_wrapper()
        wrapper.session.post.side_effect = [FakeResponse(text_chunk("Hi Ahmad")), FakeResponse(text_chunk("Ahmad"))]
        chat = wrapper.start_chat(model="gemini-x", system_instruction="Be brief")

        self.assertEqual(chat.send_text("My name is Ahmad"), "Hi Ahmad")
        self.assertEqual(chat.send_text("What is my name?"), "Ahmad")

        self.assertEqual(chat.history, [
            {"role": "user", "parts": [{"text": "My name is Ahmad"}]},
            {"role": "model", "parts": [{"text": "Hi Ahmad"}]},
            {"role": "user", "parts": [{"text": "What is my name?"}]},
            {"role": "model", "parts": [{"text": "Ahmad"}]}])
        url, kwargs = self.post_kwargs(wrapper)
        self.assertEqual(url, f"{DEV_ROOT}/models/gemini-x:generateContent")
        self.assertEqual(kwargs["json"]["contents"], chat.history[:3])
        self.assertEqual(kwargs["json"]["systemInstruction"], {"parts": [{"text": "Be brief"}]})

    def test_send_passes_session_settings(self):
        wrapper = self.make_wrapper()
        wrapper.session.post.return_value = FakeResponse(text_chunk("ok"))
        tools = [{"functionDeclarations": [{"name": "f"}]}]
        chat = wrapper.start_chat(generation_config={"temperature": 0}, tools=tools,
                                  tool_config={"functionCallingConfig": {"mode": "ANY"}},
                                  safety_settings=[{"category": "c", "threshold": "t"}])

        chat.send("hi")

        body = self.post_kwargs(wrapper)[1]["json"]
        self.assertEqual(body["generationConfig"], {"temperature": 0})
        self.assertEqual(body["tools"], tools)
        self.assertEqual(body["toolConfig"], {"functionCallingConfig": {"mode": "ANY"}})
        self.assertEqual(body["safetySettings"], [{"category": "c", "threshold": "t"}])

    def test_model_content_is_stored_without_snake_aliases(self):
        wrapper = self.make_wrapper()
        image_part = {"inlineData": {"mimeType": "image/png", "data": "QQ=="}}
        wrapper.session.post.return_value = FakeResponse(
            {"candidates": [{"content": {"role": "model", "parts": [{"text": "here"}, image_part]}}]})
        chat = wrapper.start_chat(model="gemini-x")

        response = chat.send("draw")

        self.assertIn("inline_data", response["candidates"][0]["content"]["parts"][1])  # caller still gets aliases
        self.assertEqual(chat.history[1], {"role": "model", "parts": [{"text": "here"}, image_part]})
        self.assertIs(chat.last_response, response)

    def test_thought_signature_is_preserved_and_sent_back(self):
        wrapper = self.make_wrapper(vertex=True)
        call_part = {"functionCall": {"name": "get_weather", "args": {"city": "Amman"}},
                     "thoughtSignature": "c2lnbmF0dXJl"}
        wrapper.session.post.side_effect = [
            FakeResponse({"candidates": [{"content": {"role": "model", "parts": [call_part]}}]}),
            FakeResponse(text_chunk("It is sunny"))]
        chat = wrapper.start_chat(model="gemini-x")

        chat.send("Weather in Amman?")
        chat.send_function_response("get_weather", {"result": "sunny"}, call_id="call-1")

        self.assertEqual(chat.history[1], {"role": "model", "parts": [call_part]})
        second_body = self.post_kwargs(wrapper)[1]["json"]
        self.assertEqual(second_body["contents"][1]["parts"][0]["thoughtSignature"], "c2lnbmF0dXJl")
        self.assertEqual(second_body["contents"][2], {"role": "user", "parts": [
            {"functionResponse": {"name": "get_weather", "response": {"result": "sunny"}, "id": "call-1"}}]})
        self.assertEqual(len(chat.history), 4)

    def test_send_function_response_without_id(self):
        wrapper = self.make_wrapper()
        wrapper.session.post.return_value = FakeResponse(text_chunk("ok"))
        chat = wrapper.start_chat(model="gemini-x")

        chat.send_function_response("f", {"value": 1})

        self.assertEqual(chat.history[0]["parts"], [{"functionResponse": {"name": "f", "response": {"value": 1}}}])

    def test_model_content_without_role_gets_model_role(self):
        wrapper = self.make_wrapper()
        wrapper.session.post.return_value = FakeResponse({"candidates": [{"content": {"parts": [{"text": "x"}]}}]})
        chat = wrapper.start_chat(model="gemini-x")

        chat.send("hi")

        self.assertEqual(chat.history[1]["role"], "model")

    def test_blocked_response_leaves_history_unchanged(self):
        wrapper = self.make_wrapper()
        wrapper.session.post.return_value = FakeResponse({"promptFeedback": {"blockReason": "SAFETY"}})
        chat = wrapper.start_chat(model="gemini-x")

        response = chat.send("hi")

        self.assertEqual(response["promptFeedback"]["blockReason"], "SAFETY")
        # The blocked turn is not kept, so the next turn is not blocked by it again.
        self.assertEqual(chat.history, [])

    def test_failed_send_leaves_history_unchanged(self):
        wrapper = self.make_wrapper()
        wrapper.session.post.return_value = FakeResponse({"error": {}}, status_code=500)
        chat = wrapper.start_chat(model="gemini-x")

        with self.assertRaises(GoogleAIError):
            chat.send("hi")

        self.assertEqual(chat.history, [])

    def test_send_with_media(self):
        wrapper = self.make_wrapper()
        wrapper.session.post.return_value = FakeResponse(text_chunk("a cat"))
        chat = wrapper.start_chat(model="gemini-x")

        chat.send("What is this?", media=(b"img", "image/jpeg"))

        self.assertEqual(chat.history[0]["parts"], [
            {"text": "What is this?"},
            {"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(b"img").decode()}}])

    def test_initial_history_gets_roles_on_vertex(self):
        chat = self.make_wrapper(vertex=True).start_chat(history=["hello", {"parts": [{"text": "again"}]}])

        self.assertEqual(chat.history, [{"role": "user", "parts": [{"text": "hello"}]},
                                        {"role": "user", "parts": [{"text": "again"}]}])

    def test_reset_clears_history(self):
        chat = self.make_wrapper().start_chat(history=[{"role": "user", "parts": [{"text": "x"}]}])

        chat.reset()

        self.assertEqual(chat.history, [])

    def test_stream_merges_text_parts_and_records_history(self):
        wrapper = self.make_wrapper()
        call_part = {"functionCall": {"name": "f", "args": {}}, "thoughtSignature": "c2ln"}
        image_chunk = {"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png",
                                                                             "data": "QQ=="}}]}}]}
        lines = (sse_event(text_chunk("Hel")) + sse_event(text_chunk("lo")) + sse_event(image_chunk) +
                 sse_event({"candidates": [{"content": {"parts": [call_part]}}]}))
        wrapper.session.post.return_value = FakeResponse(lines=lines)
        chat = wrapper.start_chat(model="gemini-x")

        chunks = list(chat.stream("hi"))

        self.assertEqual(chunks, ["Hel", "lo"])
        self.assertEqual(chat.history, [
            {"role": "user", "parts": [{"text": "hi"}]},
            {"role": "model", "parts": [{"text": "Hello"},
                                        {"inlineData": {"mimeType": "image/png", "data": "QQ=="}},
                                        call_part]}])
        url, kwargs = self.post_kwargs(wrapper)
        self.assertEqual(url, f"{DEV_ROOT}/models/gemini-x:streamGenerateContent")
        self.assertEqual(kwargs["params"], {"alt": "sse"})

    def test_stream_does_not_merge_thought_or_signed_text(self):
        wrapper = self.make_wrapper()
        lines = (sse_event({"candidates": [{"content": {"parts": [{"text": "plan", "thought": True}]}}]}) +
                 sse_event(text_chunk("A")) + sse_event(text_chunk("B")) +
                 sse_event({"candidates": [{"content": {"parts": [{"text": "", "thoughtSignature": "sig"}]}}]}))
        wrapper.session.post.return_value = FakeResponse(lines=lines)
        chat = wrapper.start_chat(model="gemini-x")

        chunks = list(chat.stream("hi"))

        self.assertEqual(chunks, ["A", "B"])
        self.assertEqual(chat.history[1]["parts"], [{"text": "plan", "thought": True}, {"text": "AB"},
                                                    {"text": "", "thoughtSignature": "sig"}])

    def test_stream_history_is_used_by_next_turn(self):
        wrapper = self.make_wrapper()
        wrapper.session.post.side_effect = [FakeResponse(lines=sse_event(text_chunk("one"))),
                                            FakeResponse(text_chunk("two"))]
        chat = wrapper.start_chat(model="gemini-x")

        list(chat.stream("first"))
        chat.send("second")

        body = self.post_kwargs(wrapper)[1]["json"]
        self.assertEqual(body["contents"], [{"role": "user", "parts": [{"text": "first"}]},
                                            {"role": "model", "parts": [{"text": "one"}]},
                                            {"role": "user", "parts": [{"text": "second"}]}])

    def test_stream_without_parts_leaves_history_unchanged(self):
        wrapper = self.make_wrapper()
        wrapper.session.post.return_value = FakeResponse(lines=sse_event({"usageMetadata": {"totalTokenCount": 1}}))
        chat = wrapper.start_chat(model="gemini-x")

        self.assertEqual(list(chat.stream("hi")), [])
        self.assertEqual(chat.history, [])


# ----------------------------------------------------------------------
# Response extractors
# ----------------------------------------------------------------------
class TestResponseExtractors(unittest.TestCase):
    def test_extract_text_joins_first_candidate(self):
        response = {"candidates": [{"content": {"parts": [{"text": "a"}, {"functionCall": {}}, {"text": "b"}]}},
                                   {"content": {"parts": [{"text": "other"}]}}]}

        self.assertEqual(GoogleAIWrapper.extract_text(response), "ab")

    def test_extract_text_thoughts(self):
        response = {"candidates": [{"content": {"parts": [{"text": "think ", "thought": True}, {"text": "say"}]}}]}

        self.assertEqual(GoogleAIWrapper.extract_text(response), "say")
        self.assertEqual(GoogleAIWrapper.extract_text(response, include_thoughts=True), "think say")

    def test_extract_text_empty_responses(self):
        for response in (None, {}, {"candidates": []}, {"candidates": [{}]}, {"candidates": [{"content": {}}]}):
            with self.subTest(response=response):
                self.assertEqual(GoogleAIWrapper.extract_text(response), "")

    def test_extract_function_calls_camel_and_snake(self):
        response = {"candidates": [{"content": {"parts": [
            {"text": "calling"},
            {"functionCall": {"name": "get_weather", "args": {"city": "Paris"}, "id": "c1"}},
            {"function_call": {"name": "get_time", "args": {}}}]}}]}

        calls = GoogleAIWrapper.extract_function_calls(response)

        self.assertEqual(calls, [{"name": "get_weather", "args": {"city": "Paris"}, "id": "c1"},
                                 {"name": "get_time", "args": {}}])

    def test_extract_function_calls_returns_copies(self):
        response = {"candidates": [{"content": {"parts": [{"functionCall": {"name": "f"}}]}}]}

        GoogleAIWrapper.extract_function_calls(response)[0]["name"] = "changed"

        self.assertEqual(response["candidates"][0]["content"]["parts"][0]["functionCall"]["name"], "f")

    def test_extract_function_calls_empty(self):
        self.assertEqual(GoogleAIWrapper.extract_function_calls(None), [])
        self.assertEqual(GoogleAIWrapper.extract_function_calls({"candidates": [{"content": None}]}), [])

    def test_extract_grounding(self):
        metadata = {"webSearchQueries": ["q"], "groundingChunks": [{"web": {"uri": "https://a"}}]}

        self.assertEqual(GoogleAIWrapper.extract_grounding({"candidates": [{"groundingMetadata": metadata}]}),
                         metadata)
        self.assertEqual(GoogleAIWrapper.extract_grounding({"candidates": [{}]}), {})
        self.assertEqual(GoogleAIWrapper.extract_grounding({}), {})
        self.assertEqual(GoogleAIWrapper.extract_grounding(None), {})

    def test_extract_images_from_parts(self):
        response = {"candidates": [{"content": {"parts": [
            {"text": "here"},
            {"inlineData": {"mimeType": "image/png", "data": "AAA"}},
            {"inline_data": {"mime_type": "image/jpeg", "data": "BBB"}},
            {"inlineData": {"mimeType": "audio/wav", "data": "CCC"}}]}}]}

        self.assertEqual(GoogleAIWrapper.extract_images(response),
                         [{"mime_type": "image/png", "data": "AAA"}, {"mime_type": "image/jpeg", "data": "BBB"}])

    def test_extract_images_does_not_duplicate_aliased_parts(self):
        wrapper = GoogleAIWrapper(API_KEY, session=MagicMock())
        response = wrapper._snake_alias(
            {"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": "AAA"}}]}}]})

        self.assertEqual(GoogleAIWrapper.extract_images(response), [{"mime_type": "image/png", "data": "AAA"}])

    def test_extract_images_from_imagen_predictions(self):
        response = {"predictions": [{"bytesBase64Encoded": "IMG1", "mimeType": "image/jpeg"},
                                    {"bytesBase64Encoded": "IMG2"}, {"raiFilteredReason": "blocked"}]}

        self.assertEqual(GoogleAIWrapper.extract_images(response),
                         [{"mime_type": "image/jpeg", "data": "IMG1"}, {"mime_type": "image/png", "data": "IMG2"}])

    def test_extract_audio_from_parts_and_predictions(self):
        parts_response = {"candidates": [{"content": {"parts": [
            {"inlineData": {"mimeType": "audio/L16;codec=pcm;rate=24000", "data": "PCM"}},
            {"inlineData": {"mimeType": "image/png", "data": "IMG"}}]}}]}
        lyria_response = {"predictions": [{"bytesBase64Encoded": "WAV", "mimeType": "audio/wav"},
                                          {"bytesBase64Encoded": "WAV2"}]}

        self.assertEqual(GoogleAIWrapper.extract_audio(parts_response),
                         [{"mime_type": "audio/L16;codec=pcm;rate=24000", "data": "PCM"}])
        self.assertEqual(GoogleAIWrapper.extract_audio(lyria_response),
                         [{"mime_type": "audio/wav", "data": "WAV"}, {"mime_type": "audio/wav", "data": "WAV2"}])

    def test_extract_media_empty(self):
        self.assertEqual(GoogleAIWrapper.extract_images(None), [])
        self.assertEqual(GoogleAIWrapper.extract_audio({}), [])

    def test_extract_videos_vertex_operation(self):
        operation = {"done": True, "response": {"videos": [
            {"gcsUri": "gs://bucket/v.mp4", "mimeType": "video/mp4"}, {"bytesBase64Encoded": "VID"}]}}

        self.assertEqual(GoogleAIWrapper.extract_videos(operation), [
            {"mime_type": "video/mp4", "data": None, "uri": "gs://bucket/v.mp4"},
            {"mime_type": "video/mp4", "data": "VID", "uri": None}])

    def test_extract_videos_developer_operation(self):
        uri = "https://generativelanguage.googleapis.com/v1beta/files/abc:download?alt=media"
        operation = {"done": True, "response": {"generateVideoResponse": {"generatedSamples": [
            {"video": {"uri": uri}}, {"video": {"encodedVideo": "ENC", "mimeType": "video/webm"}}]}}}

        self.assertEqual(GoogleAIWrapper.extract_videos(operation), [
            {"mime_type": "video/mp4", "data": None, "uri": uri},
            {"mime_type": "video/webm", "data": "ENC", "uri": None}])

    def test_extract_videos_unfinished_operation(self):
        self.assertEqual(GoogleAIWrapper.extract_videos({"name": "op", "done": False}), [])
        self.assertEqual(GoogleAIWrapper.extract_videos(None), [])


# ----------------------------------------------------------------------
# Audio helpers
# ----------------------------------------------------------------------
class TestAudioHelpers(unittest.TestCase):
    PCM = bytes(range(256)) * 4

    @staticmethod
    def read_wav(data):
        with wave.open(io.BytesIO(data), "rb") as wav_file:
            return (wav_file.getframerate(), wav_file.getnchannels(), wav_file.getsampwidth(),
                    wav_file.readframes(wav_file.getnframes()))

    def test_pcm_to_wav_defaults(self):
        rate, channels, width, frames = self.read_wav(GoogleAIWrapper.pcm_to_wav(self.PCM))

        self.assertEqual((rate, channels, width), (24000, 1, 2))
        self.assertEqual(frames, self.PCM)

    def test_pcm_to_wav_custom_format(self):
        wav = GoogleAIWrapper.pcm_to_wav(self.PCM, sample_rate=16000, channels=2, sample_width=2)

        rate, channels, width, frames = self.read_wav(wav)
        self.assertEqual((rate, channels, width), (16000, 2, 2))
        self.assertEqual(frames, self.PCM)

    def test_pcm_to_wav_accepts_base64(self):
        wav = GoogleAIWrapper.pcm_to_wav(base64.b64encode(self.PCM).decode())

        self.assertTrue(wav.startswith(b"RIFF"))
        self.assertEqual(self.read_wav(wav)[3], self.PCM)

    def test_audio_to_wav_wraps_l16_with_rate(self):
        item = {"mime_type": "audio/L16;codec=pcm;rate=16000", "data": base64.b64encode(self.PCM).decode()}

        rate, _, _, frames = self.read_wav(GoogleAIWrapper.audio_to_wav(item))

        self.assertEqual(rate, 16000)
        self.assertEqual(frames, self.PCM)

    def test_audio_to_wav_pcm_without_rate_uses_24k(self):
        rate, _, _, _ = self.read_wav(GoogleAIWrapper.audio_to_wav({"mime_type": "audio/pcm", "data": self.PCM}))

        self.assertEqual(rate, 24000)

    def test_audio_to_wav_keeps_wav(self):
        wav = GoogleAIWrapper.pcm_to_wav(self.PCM)

        self.assertEqual(GoogleAIWrapper.audio_to_wav({"mime_type": "audio/wav",
                                                       "data": base64.b64encode(wav).decode()}), wav)
        self.assertEqual(GoogleAIWrapper.audio_to_wav({"mime_type": "", "data": wav}), wav)
        self.assertEqual(GoogleAIWrapper.audio_to_wav({"mime_type": "audio/L16;rate=16000", "data": wav}), wav)

    def test_audio_to_wav_returns_other_formats_unchanged(self):
        mp3 = b"ID3\x03\x00fake-mp3"

        self.assertEqual(GoogleAIWrapper.audio_to_wav({"mime_type": "audio/mpeg", "data": mp3}), mp3)


# ----------------------------------------------------------------------
# media_part
# ----------------------------------------------------------------------
class TestMediaPart(GenAITestCase):
    def setUp(self):
        super().setUp()
        self.wrapper = self.make_wrapper()

    def temp_file(self, suffix, content=b"file-bytes"):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = os.path.join(directory.name, f"sample{suffix}")
        with open(path, "wb") as file:
            file.write(content)
        return path

    def test_bytes_with_mime_type(self):
        part = self.wrapper.media_part(b"\x00\x01", "image/png")

        self.assertEqual(part, {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(b"\x00\x01").decode()}})

    def test_bytearray_is_accepted(self):
        part = self.wrapper.media_part(bytearray(b"abc"), "audio/wav")

        self.assertEqual(part["inlineData"]["data"], base64.b64encode(b"abc").decode())

    def test_bytes_without_mime_type_raises(self):
        with self.assertRaises(ValueError):
            self.wrapper.media_part(b"\x00\x01")

    def test_tuple_bytes_and_mime_type(self):
        part = self.wrapper.media_part((b"pdf-bytes", "application/pdf"))

        self.assertEqual(part["inlineData"], {"mimeType": "application/pdf",
                                              "data": base64.b64encode(b"pdf-bytes").decode()})

    def test_single_item_tuple_uses_mime_argument(self):
        part = self.wrapper.media_part((b"x",), "image/webp")

        self.assertEqual(part["inlineData"]["mimeType"], "image/webp")

    def test_local_path_reads_file_and_guesses_mime(self):
        path = self.temp_file(".png", b"png-bytes")

        part = self.wrapper.media_part(path)

        self.assertEqual(part, {"inlineData": {"mimeType": "image/png",
                                               "data": base64.b64encode(b"png-bytes").decode()}})

    def test_local_path_with_explicit_mime(self):
        path = self.temp_file(".bin", b"audio")

        self.assertEqual(self.wrapper.media_part(path, "audio/ogg")["inlineData"]["mimeType"], "audio/ogg")

    def test_path_keyword(self):
        path = self.temp_file(".mp3", b"mp3")

        self.assertEqual(self.wrapper.media_part(path=path)["inlineData"]["mimeType"], "audio/mpeg")

    def test_gcs_uri(self):
        self.assertEqual(self.wrapper.media_part("gs://bucket/folder/report.pdf"),
                         {"fileData": {"mimeType": "application/pdf", "fileUri": "gs://bucket/folder/report.pdf"}})

    def test_https_uri_ignores_query_string(self):
        uri = "https://example.com/images/cat.jpg?size=large&v=2"

        self.assertEqual(self.wrapper.media_part(uri), {"fileData": {"mimeType": "image/jpeg", "fileUri": uri}})

    def test_https_uri_with_explicit_mime(self):
        uri = "https://generativelanguage.googleapis.com/v1beta/files/abc123"

        self.assertEqual(self.wrapper.media_part(uri, "video/mp4"),
                         {"fileData": {"mimeType": "video/mp4", "fileUri": uri}})

    def test_https_uri_without_extension_needs_mime_type(self):
        uri = "https://example.com/download"

        with self.assertRaises(ValueError):
            self.wrapper.media_part(uri)
        self.assertEqual(self.wrapper.media_part(uri, "application/pdf")["fileData"]["mimeType"], "application/pdf")

    def test_youtube_uris_are_video(self):
        for uri in ("https://www.youtube.com/watch?v=9hE5-98ZeCg", "https://youtu.be/9hE5-98ZeCg"):
            with self.subTest(uri=uri):
                self.assertEqual(self.wrapper.media_part(uri), {"fileData": {"mimeType": "video/mp4", "fileUri": uri}})

    def test_dict_part_is_returned_as_is(self):
        part = {"text": "already a part"}

        self.assertIs(self.wrapper.media_part(part), part)

    def test_video_metadata_is_added(self):
        metadata = {"startOffset": "10s", "endOffset": "20s", "fps": 2}

        part = self.wrapper.media_part("gs://bucket/v.mp4", video_metadata=metadata)

        self.assertEqual(part["videoMetadata"], metadata)
        self.assertEqual(part["fileData"]["mimeType"], "video/mp4")

    def test_base64_data_keyword(self):
        part = self.wrapper.media_part(data="QUJD", mime_type="image/gif")

        self.assertEqual(part, {"inlineData": {"mimeType": "image/gif", "data": "QUJD"}})

    def test_unknown_string_raises(self):
        with self.assertRaises(ValueError) as caught:
            self.wrapper.media_part("/no/such/file.png")

        self.assertIn("Not a file path or URI", str(caught.exception))

    def test_nothing_given_raises(self):
        with self.assertRaises(ValueError):
            self.wrapper.media_part()


# ----------------------------------------------------------------------
# from_options
# ----------------------------------------------------------------------
class TestFromOptions(GenAITestCase):
    def test_no_options_is_developer_api(self):
        wrapper = GoogleAIWrapper.from_options(API_KEY)

        self.assertIsInstance(wrapper, GoogleAIWrapper)
        self.assertFalse(wrapper.vertex)
        self.assertEqual(wrapper.api_key, API_KEY)
        self.assertEqual(wrapper.timeout, 180)

    def test_all_option_names(self):
        options = {"vertex": True, "project_id": "my-project", "location": "us-central1", "api_version": "v1",
                   "quota_project_id": "billing"}

        wrapper = GoogleAIWrapper.from_options(API_KEY, options)

        self.assertTrue(wrapper.vertex)
        self.assertEqual(wrapper.project_id, "my-project")
        self.assertEqual(wrapper.location, "us-central1")
        self.assertTrue(wrapper._location_explicit)
        self.assertEqual(wrapper.api_version, "v1")
        self.assertEqual(wrapper.quota_project_id, "billing")

    def test_access_token_and_credentials_options(self):
        credentials = FakeCredentials()

        token_wrapper = GoogleAIWrapper.from_options(None, {"access_token": ACCESS_TOKEN})
        credentials_wrapper = GoogleAIWrapper.from_options(None, {"credentials": credentials})

        self.assertTrue(token_wrapper.vertex)
        self.assertEqual(token_wrapper._auth_headers(), {"Authorization": f"Bearer {ACCESS_TOKEN}"})
        self.assertTrue(credentials_wrapper.vertex)
        self.assertIs(credentials_wrapper._credentials, credentials)

    def test_vertex_project_and_location_aliases(self):
        wrapper = GoogleAIWrapper.from_options(API_KEY, {"vertex_project": "alias-project",
                                                         "vertex_location": "europe-west4"})

        self.assertTrue(wrapper.vertex)
        self.assertEqual(wrapper.project_id, "alias-project")
        self.assertEqual(wrapper.location, "europe-west4")

    def test_primary_names_win_over_aliases(self):
        wrapper = GoogleAIWrapper.from_options(API_KEY, {"project_id": "main", "vertex_project": "alias",
                                                         "location": "us", "vertex_location": "eu"})

        self.assertEqual(wrapper.project_id, "main")
        self.assertEqual(wrapper.location, "us")

    def test_timeout_from_options(self):
        self.assertEqual(GoogleAIWrapper.from_options(API_KEY, {"timeout": 30}).timeout, 30)

    def test_timeout_argument_wins_over_options(self):
        self.assertEqual(GoogleAIWrapper.from_options(API_KEY, {"timeout": 30}, timeout=5).timeout, 5)

    def test_none_values_are_ignored(self):
        wrapper = GoogleAIWrapper.from_options(API_KEY, {"vertex": None, "project_id": None, "location": None,
                                                         "access_token": None, "vertex_project": None})

        self.assertFalse(wrapper.vertex)
        self.assertIsNone(wrapper.project_id)

    def test_explicit_vertex_false_is_kept(self):
        wrapper = GoogleAIWrapper.from_options(API_KEY, {"vertex": False, "project_id": "my-project"})

        self.assertFalse(wrapper.vertex)

    def test_unrelated_options_are_ignored(self):
        wrapper = GoogleAIWrapper.from_options(API_KEY, {"model": "gemini-x", "temperature": 0.3})

        self.assertFalse(wrapper.vertex)
        self.assertEqual(wrapper.timeout, 180)


if __name__ == "__main__":
    unittest.main()

import os
import unittest
from unittest import mock

from intelli.utils.model_helper import parse_model_version, api_version_for_model


class TestParseModelVersion(unittest.TestCase):
    def test_docstring_examples(self):
        self.assertEqual(parse_model_version("gpt-5.5"), 5.5)
        self.assertEqual(parse_model_version("gpt-5"), 5.0)
        self.assertEqual(parse_model_version("gpt-4o"), 4.0)

    def test_case_and_whitespace_are_normalized(self):
        self.assertEqual(parse_model_version("  GPT-5.5  "), 5.5)

    def test_version_embedded_in_longer_name(self):
        self.assertEqual(parse_model_version("gpt-4o-mini"), 4.0)

    def test_no_numeric_version_returns_none(self):
        self.assertIsNone(parse_model_version("gpt-chat-latest"))

    def test_none_input_returns_none(self):
        self.assertIsNone(parse_model_version(None))

    def test_empty_string_returns_none(self):
        self.assertIsNone(parse_model_version(""))


class TestApiVersionForModel(unittest.TestCase):
    @mock.patch.dict(os.environ, {"AZURE_GPT_5_5_API_VERSION": "2099-01-01"})
    def test_returns_env_override_for_gpt_5_5_plus(self):
        self.assertEqual(api_version_for_model("gpt-5.5"), "2099-01-01")
        self.assertEqual(api_version_for_model("gpt-6"), "2099-01-01")

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_returns_default_when_env_var_unset(self):
        self.assertEqual(api_version_for_model("gpt-5.5"), "2025-05-01")

    def test_below_threshold_returns_none(self):
        self.assertIsNone(api_version_for_model("gpt-5"))
        self.assertIsNone(api_version_for_model("gpt-4o"))

    @mock.patch.dict(os.environ, {"AZURE_GPT_5_5_API_VERSION": "2099-01-01"})
    def test_boundary_version_is_included(self):
        self.assertEqual(api_version_for_model("gpt-5.5"), "2099-01-01")

    def test_no_version_and_none_input_returns_none(self):
        self.assertIsNone(api_version_for_model("gpt-chat-latest"))
        self.assertIsNone(api_version_for_model(None))


if __name__ == "__main__":
    unittest.main()

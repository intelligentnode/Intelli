import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from intelli.flow.agents.kagent import KerasAgent
from intelli.flow.input.agent_input import TextAgentInput
from intelli.function.chatbot import Chatbot
from intelli.model.input.chatbot_input import ChatModelInput
from intelli.model.input.text_recognition_input import SpeechRecognitionInput
from intelli.utils.whisper_helper import WhisperHelper
from intelli.wrappers.keras_wrapper import KerasWrapper


class FakeSampler:
    def __init__(self, **params):
        self.params = params


class FakeTopKSampler(FakeSampler):
    pass


class FakeTopPSampler(FakeSampler):
    pass


class FakeGreedySampler(FakeSampler):
    pass


class FakeBackbone:
    def __init__(self):
        self.lora_ranks = []
        self.saved = []
        self.loaded = []

    def enable_lora(self, rank):
        if self.lora_ranks:
            raise ValueError("lora is already enabled")
        self.lora_ranks.append(rank)

    def save_lora_weights(self, file_path):
        self.saved.append(file_path)

    def load_lora_weights(self, file_path):
        self.loaded.append(file_path)


class FakeCausalLM:
    """Stands in for a keras-hub CausalLM task and records the calls."""

    def __init__(self, preset=None, preset_params=None):
        self.preset = preset
        self.preset_params = preset_params or {}
        self.sampler = "default"
        self.generate_function = "compiled"
        self.backbone = FakeBackbone()
        self.preprocessor = SimpleNamespace(
            tokenizer=SimpleNamespace(tokenize=lambda text: text.split()),
            sequence_length=1024,
        )
        self.generate_calls = []
        self.compile_calls = []
        self.fit_calls = []

    @classmethod
    def from_preset(cls, preset, **preset_params):
        return cls(preset, preset_params)

    def generate(self, inputs, **params):
        self.generate_calls.append(params)
        if isinstance(inputs, str):
            return inputs + " generated text"
        return [prompt + " generated text" for prompt in inputs]

    def compile(self, **params):
        self.compile_calls.append(params)

    def fit(self, dataset, **params):
        self.fit_calls.append(params)
        return "history"


class FakeGemma4CausalLM(FakeCausalLM):
    pass


def fake_hub():
    return SimpleNamespace(
        models=SimpleNamespace(CausalLM=FakeCausalLM, Gemma4CausalLM=FakeGemma4CausalLM),
        samplers=SimpleNamespace(
            TopKSampler=FakeTopKSampler,
            TopPSampler=FakeTopPSampler,
            GreedySampler=FakeGreedySampler,
        ),
    )


def fake_keras():
    optimizer = SimpleNamespace(exclude_from_weight_decay=lambda var_names: None)
    return SimpleNamespace(
        optimizers=SimpleNamespace(AdamW=lambda **params: optimizer),
        losses=SimpleNamespace(SparseCategoricalCrossentropy=lambda **params: "loss"),
        metrics=SimpleNamespace(SparseCategoricalAccuracy=lambda: "accuracy"),
    )


class KerasTestCase(unittest.TestCase):
    """Runs with a fake keras-hub, the real libraries are not needed."""

    def setUp(self):
        modules = patch.dict(sys.modules, {"keras": fake_keras(), "keras_hub": fake_hub()})
        modules.start()
        self.addCleanup(modules.stop)


class TestKerasWrapperLoading(KerasTestCase):
    def test_any_preset_loads_with_the_generic_task(self):
        for model_name in ["gpt2_base_en", "gemma3_instruct_1b", "llama2_7b_en", "hf://org/model", "./local_preset"]:
            wrapper = KerasWrapper(model_name, {"dtype": "bfloat16", "max_length": 64})
            self.assertIs(type(wrapper.model), FakeCausalLM)
            self.assertEqual(wrapper.model.preset, model_name)
            self.assertEqual(wrapper.model.preset_params, {"dtype": "bfloat16"})

    def test_gemma4_preset_uses_the_text_class(self):
        self.assertIs(type(KerasWrapper("gemma4_instruct_2b").model), FakeGemma4CausalLM)
        self.assertIs(type(KerasWrapper("gemma4_instruct_2b_assistant").model), FakeCausalLM)

    def test_model_class_from_params(self):
        wrapper = KerasWrapper("my_preset", {"model_class": "Gemma4CausalLM"})
        self.assertIs(type(wrapper.model), FakeGemma4CausalLM)
        with self.assertRaises(ValueError):
            KerasWrapper("my_preset", {"model_class": "MissingCausalLM"})

    def test_keras_nlp_is_used_when_keras_hub_is_missing(self):
        with patch.dict(sys.modules, {"keras_hub": None, "keras_nlp": fake_hub()}):
            wrapper = KerasWrapper("gpt2_base_en")
        self.assertIs(type(wrapper.model), FakeCausalLM)

    def test_missing_libraries_error_shows_the_install_command(self):
        with patch.dict(sys.modules, {"keras_hub": None, "keras_nlp": None}):
            with self.assertRaises(ImportError) as error:
                KerasWrapper("gpt2_base_en")
        self.assertIn("intelli[offline]", str(error.exception))

    def test_wrapper_without_model_name_waits_for_set_model(self):
        wrapper = KerasWrapper()
        with self.assertRaises(ValueError):
            wrapper.generate("hello")
        wrapper.set_model(FakeCausalLM(), {"max_length": 32})
        self.assertEqual(wrapper.generate("hello"), "generated text")

    def test_credentials_are_set_only_when_provided(self):
        with patch.dict(os.environ, {"KAGGLE_KEY": "existing"}, clear=False):
            os.environ.pop("KAGGLE_USERNAME", None)
            KerasWrapper("gpt2_base_en", {"KAGGLE_USERNAME": None, "KAGGLE_KEY": ""})
            self.assertNotIn("KAGGLE_USERNAME", os.environ)
            self.assertEqual(os.environ["KAGGLE_KEY"], "existing")

            KerasWrapper("gpt2_base_en", {"KAGGLE_USERNAME": "user"})
            self.assertEqual(os.environ["KAGGLE_USERNAME"], "user")
            self.assertEqual(os.environ["KAGGLE_KEY"], "existing")


class TestKerasWrapperGenerate(KerasTestCase):
    def setUp(self):
        super().setUp()
        self.wrapper = KerasWrapper("gpt2_base_en")
        self.model = self.wrapper.model

    def test_prompt_is_removed_from_text_and_list_outputs(self):
        self.assertEqual(self.wrapper.generate("write a post"), "generated text")
        self.assertEqual(
            self.wrapper.generate(["first prompt", "second prompt"]),
            ["generated text", "generated text"],
        )
        self.assertEqual(self.model.generate_calls[0], {"max_length": 180})

    def test_prompt_longer_than_max_length_is_rejected(self):
        with self.assertRaises(ValueError):
            self.wrapper.generate("one two three four five", max_length=5)

    def test_max_new_tokens_is_added_to_the_prompt_length(self):
        self.wrapper.generate("one two three", max_new_tokens=20)
        # three prompt tokens and the start token
        self.assertEqual(self.model.generate_calls[-1]["max_length"], 24)

    def test_stop_token_ids_are_sent_when_changed(self):
        self.wrapper.generate("hello", stop_token_ids=None)
        self.assertEqual(self.model.generate_calls[-1], {"max_length": 180, "stop_token_ids": None})

    def test_sampling_options_build_the_sampler(self):
        self.wrapper.generate("hello")
        self.assertEqual(self.model.sampler, "default")

        self.wrapper.generate("hello", temperature=0.7, top_k=10, seed=1)
        self.assertIsInstance(self.model.sampler, FakeTopKSampler)
        self.assertEqual(self.model.sampler.params, {"temperature": 0.7, "k": 10, "seed": 1})
        self.assertIsNone(self.model.generate_function)

        self.wrapper.generate("hello", temperature=0.5, top_p=0.9)
        self.assertIsInstance(self.model.sampler, FakeTopPSampler)
        self.assertEqual(self.model.sampler.params, {"temperature": 0.5, "p": 0.9})

        self.wrapper.generate("hello", temperature=0)
        self.assertIsInstance(self.model.sampler, FakeGreedySampler)

    def test_same_sampling_options_keep_the_sampler(self):
        self.wrapper.generate("hello", temperature=0.7)
        sampler = self.model.sampler
        self.wrapper.generate("hello", temperature=0.7)
        self.assertIs(self.model.sampler, sampler)

    def test_unknown_sampler_is_rejected(self):
        with self.assertRaises(ValueError):
            self.wrapper.generate("hello", sampler="topk")


class TestKerasWrapperFineTune(KerasTestCase):
    def setUp(self):
        super().setUp()
        self.wrapper = KerasWrapper("gpt2_base_en")
        self.model = self.wrapper.model

    def test_fine_tune_twice_enables_lora_once(self):
        config = {"lora_rank": 8, "epochs": 1, "batch_size": 2, "sequence_length": 32}
        self.assertEqual(self.wrapper.fine_tune(["a", "b"], config), "history")
        self.assertEqual(self.wrapper.fine_tune(["a", "b"], config), "history")
        self.assertEqual(self.model.backbone.lora_ranks, [8])
        self.assertEqual(self.model.preprocessor.sequence_length, 32)
        self.assertEqual(self.model.fit_calls[0], {"epochs": 1, "batch_size": 2})

    def test_fine_tune_keeps_the_sampler(self):
        self.wrapper.generate("hello", temperature=0)
        self.wrapper.fine_tune(["a", "b"], None)
        self.assertIsInstance(self.model.compile_calls[0]["sampler"], FakeGreedySampler)

    def test_batched_dataset_is_sent_without_batch_size(self):
        dataset = iter([["a", "b"]])
        self.wrapper.fine_tune(dataset, {"epochs": 2})
        self.assertEqual(self.model.fit_calls[0], {"epochs": 2})

    def test_lora_weights_save_and_load(self):
        self.wrapper.load_lora("model.lora.h5", lora_rank=8)
        self.wrapper.save_lora("model.lora.h5")
        self.assertEqual(self.model.backbone.lora_ranks, [8])
        self.assertEqual(self.model.backbone.loaded, ["model.lora.h5"])
        self.assertEqual(self.model.backbone.saved, ["model.lora.h5"])


class TestKerasAgent(KerasTestCase):
    def test_model_name_and_model_keys(self):
        for model_params in [{"model_name": "gpt2_base_en"}, {"model": "gpt2_base_en"}]:
            agent = KerasAgent(agent_type="text", mission="write blog posts", model_params=model_params)
            self.assertEqual(agent.wrapper.model.preset, "gpt2_base_en")

    def test_missing_model_name_is_rejected(self):
        with self.assertRaises(ValueError):
            KerasAgent(agent_type="text")

    def test_default_model_params_are_not_shared(self):
        first = KerasAgent(agent_type="text", external=True)
        second = KerasAgent(agent_type="text", external=True)
        first.model_params["max_length"] = 10
        self.assertEqual(second.model_params, {})

    def test_external_model(self):
        model = FakeCausalLM()
        agent = KerasAgent(agent_type="text", external=True)
        agent.set_keras_model(model, {"max_length": 64})
        self.assertEqual(agent.execute(TextAgentInput("hello")), "generated text")

    def test_generation_params_reach_the_model(self):
        agent = KerasAgent(
            agent_type="text",
            mission="write blog posts",
            model_params={"model_name": "gpt2_base_en", "max_length": 64, "temperature": 0},
        )
        result = agent.execute(TextAgentInput("electric cars"), new_params={"max_new_tokens": 10})
        self.assertEqual(result, "generated text")
        # five prompt tokens, the start token and ten new tokens
        self.assertEqual(agent.wrapper.model.generate_calls[0], {"max_length": 16})
        self.assertIsInstance(agent.wrapper.model.sampler, FakeGreedySampler)


class TestKerasChatbot(KerasTestCase):
    def test_model_name_is_required(self):
        with self.assertRaises(ValueError):
            Chatbot(provider="keras")

    def test_chat_sends_max_tokens_and_sampling(self):
        chatbot = Chatbot(provider="keras", options={"model_name": "gpt2_base_en"})
        chat_input = ChatModelInput("You are a helpful assistant.", max_tokens=20, temperature=0)
        chat_input.add_user_message("What is the capital of France?")

        result = chatbot.chat(chat_input)

        self.assertEqual(result, ["generated text"])
        prompt_tokens = len(chat_input.get_keras_input()["prompt"].split()) + 1
        self.assertEqual(chatbot.wrapper.model.generate_calls[0], {"max_length": prompt_tokens + 20})
        self.assertIsInstance(chatbot.wrapper.model.sampler, FakeGreedySampler)

    def test_input_without_user_message(self):
        with self.assertRaises(ValueError):
            ChatModelInput("You are a helpful assistant.").get_keras_input()


class TestKerasRecognitionInput(unittest.TestCase):
    def test_audio_file_path_is_loaded(self):
        try:
            import numpy as np
            import soundfile as sf
        except ImportError:
            self.skipTest("soundfile is not installed")

        with tempfile.TemporaryDirectory() as folder:
            file_path = os.path.join(folder, "audio.wav")
            sf.write(file_path, np.zeros(8000, dtype="float32"), 8000)

            params = SpeechRecognitionInput(audio_file_path=file_path, language="en").get_keras_input()

        self.assertEqual(len(params["audio_data"]), 8000)
        self.assertEqual(params["sample_rate"], 8000)
        self.assertEqual(params["language"], "en")


class FakeWhisperTokenizer:
    def __init__(self, language_tokens=None):
        self.language_tokens = language_tokens
        self.special_tokens = dict(language_tokens or {})

    def token_to_id(self, token):
        if token not in self.special_tokens:
            raise ValueError(f"Token '{token}' is not in the vocabulary.")
        return self.special_tokens[token]

    def tokenize(self, text):
        return [len(word) for word in text.split()]


class TestWhisperHelper(unittest.TestCase):
    def get_helper(self, language_tokens=None):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is not installed")

        helper = object.__new__(WhisperHelper)
        helper.np = np
        helper.ops = SimpleNamespace(convert_to_numpy=np.asarray)
        helper.tokenizer = FakeWhisperTokenizer(language_tokens)
        helper.is_multilingual = bool(language_tokens)
        helper.start_token_id = 50257 if not language_tokens else 50258
        helper.end_token_id = helper.start_token_id - 1
        helper.transcribe_token_id = 50358 if not language_tokens else 50359
        helper.no_timestamps_token_id = 50362 if not language_tokens else 50363
        helper.prev_token_id = helper.no_timestamps_token_id - 2
        return helper

    def test_english_model_start_ids(self):
        helper = self.get_helper()
        for language in [None, "en", "<|en|>"]:
            self.assertEqual(helper._build_start_ids(language=language), [50257, 50362])

    def test_multilingual_model_start_ids(self):
        helper = self.get_helper({"<|en|>": 50259, "<|fr|>": 50265})
        self.assertEqual(helper._build_start_ids(language="fr"), [50258, 50265, 50359, 50363])
        self.assertEqual(helper._build_start_ids(language="<|en|>"), [50258, 50259, 50359, 50363])
        # unknown language codes are never sent as text tokens
        self.assertEqual(helper._build_start_ids(language="xx"), [50258, 50359, 50363])
        self.assertEqual(helper._build_start_ids(), [50258, 50359, 50363])

    def test_prompt_comes_before_the_start_token(self):
        helper = self.get_helper()
        prompt_ids = helper._encode_text("medical notes")
        self.assertEqual(prompt_ids, [7, 5])
        self.assertEqual(
            helper._build_start_ids(prompt_ids=prompt_ids), [50360, 7, 5, 50257, 50362]
        )

    def test_prepare_audio(self):
        helper = self.get_helper()
        np = helper.np

        pcm = helper._prepare_audio(np.array([0, 16384, -32768], dtype="int16"), sr=16000)
        self.assertEqual(pcm.dtype, np.float32)
        self.assertEqual(pcm.tolist(), [0.0, 0.5, -1.0])

        for stereo in [np.ones((100, 2)), np.ones((2, 100))]:
            self.assertEqual(helper._prepare_audio(stereo, sr=16000).shape, (100,))

        self.assertEqual(helper._prepare_audio([0.1, 0.2], sr=16000).shape, (2,))

    def test_silent_audio_returns_empty_text(self):
        helper = self.get_helper()
        self.assertEqual(helper.transcribe(helper.np.zeros(16000)), "")
        self.assertEqual(helper.transcribe(helper.np.zeros(0)), "")


if __name__ == "__main__":
    unittest.main()

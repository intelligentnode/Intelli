#!/usr/bin/env python3
"""
Test for Flow auto-save functionality.

This test verifies that the Flow class can automatically save image, audio, and text outputs
to files when auto_save_outputs is enabled.
"""

import os
import asyncio
import unittest
import tempfile
import shutil
from dotenv import load_dotenv

from intelli.flow.agents.agent import Agent
from intelli.flow.input.task_input import TextTaskInput
from intelli.flow.tasks.task import Task
from intelli.flow.flow import Flow
from intelli.flow.types import AgentTypes
from intelli.flow.utils.flow_helper import FlowHelper

# Load environment variables
load_dotenv()


class TestFlowAutoSave(unittest.TestCase):
    """Test auto-save functionality in Flow."""

    def setUp(self):
        """Set up test environment."""
        # Load API keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.stability_api_key = os.getenv("STABILITY_API_KEY")
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

        # Create temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp(prefix="flow_auto_save_test_")
        self.output_dir = os.path.join(self.test_dir, "outputs")

    def tearDown(self):
        """Clean up after tests."""
        # Remove test directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_auto_save_text_output(self):
        """Test auto-saving of text outputs."""
        if not self.openai_api_key:
            self.skipTest("OpenAI API key not available")

        asyncio.run(self._test_text_auto_save())

    async def _test_text_auto_save(self):
        """Test text auto-save functionality."""
        # Create a simple text generation task
        text_task = Task(
            TextTaskInput("Write a haiku about programming"),
            Agent(
                agent_type=AgentTypes.TEXT.value,
                provider="openai",
                mission="Create a beautiful haiku",
                model_params={"key": self.openai_api_key, "model": "gpt-4o"},
            ),
            log=True,
        )

        # Create flow with auto-save enabled
        flow = Flow(
            tasks={"haiku": text_task},
            map_paths={"haiku": []},
            auto_save_outputs=True,
            output_dir=self.output_dir,
            output_file_map={"haiku": "test_haiku.txt"},
            log=True,
        )

        # Execute flow
        results = await flow.start()

        # Verify results
        self.assertIn("haiku", results)
        self.assertEqual(results["haiku"]["type"], "text")
        self.assertTrue(len(results["haiku"]["output"]) > 0)

        # Verify file was saved
        saved_files = flow.get_saved_files()
        self.assertIn("haiku", saved_files)

        expected_path = os.path.join(self.output_dir, "test_haiku.txt")
        self.assertEqual(saved_files["haiku"]["path"], expected_path)
        self.assertTrue(os.path.exists(expected_path))

        # Verify file content
        with open(expected_path, "r") as f:
            file_content = f.read()
        self.assertEqual(file_content, results["haiku"]["output"])

    def test_auto_save_image_output(self):
        """Test auto-saving of image outputs."""
        if not self.stability_api_key:
            self.skipTest("Stability AI API key not available")

        asyncio.run(self._test_image_auto_save())

    async def _test_image_auto_save(self):
        """Test image auto-save functionality."""
        # Create an image generation task
        image_task = Task(
            TextTaskInput("A simple geometric pattern"),
            Agent(
                agent_type=AgentTypes.IMAGE.value,
                provider="stability",
                mission="Generate a simple image",
                model_params={
                    "key": self.stability_api_key,
                    "engine": "stable-diffusion-xl-1024-v1-0",
                    "width": 512,
                    "height": 512,
                    "diffusion_steps": 10,  # Fewer steps for faster testing
                    "diffusion_cfgScale": 7,
                },
            ),
            log=True,
        )

        # Create flow with auto-save enabled
        flow = Flow(
            tasks={"test_image": image_task},
            map_paths={"test_image": []},
            auto_save_outputs=True,
            output_dir=self.output_dir,
            output_file_map={"test_image": "test_pattern.png"},
            log=True,
        )

        # Execute flow
        results = await flow.start()

        # Verify results
        self.assertIn("test_image", results)
        self.assertEqual(results["test_image"]["type"], "image")
        self.assertTrue(len(results["test_image"]["output"]) > 0)

        # Verify file was saved
        saved_files = flow.get_saved_files()
        self.assertIn("test_image", saved_files)

        expected_path = os.path.join(self.output_dir, "test_pattern.png")
        self.assertEqual(saved_files["test_image"]["path"], expected_path)
        self.assertTrue(os.path.exists(expected_path))

        # Verify file size is reasonable for an image
        file_size = os.path.getsize(expected_path)
        self.assertGreater(file_size, 1000)  # Should be at least 1KB

    def test_auto_save_audio_output(self):
        """Test auto-saving of audio outputs."""
        if not self.elevenlabs_api_key:
            self.skipTest("ElevenLabs API key not available")

        asyncio.run(self._test_audio_auto_save())

    async def _test_audio_auto_save(self):
        """Test audio auto-save functionality."""
        # Get a voice ID first
        try:
            from intelli.controller.remote_speech_model import RemoteSpeechModel
            speech_model = RemoteSpeechModel(
                key_value=self.elevenlabs_api_key, provider="elevenlabs"
            )
            voices_result = speech_model.list_voices()
            if not voices_result.get("voices"):
                self.skipTest("No ElevenLabs voices available")
            
            voice_id = voices_result["voices"][0]["voice_id"]
        except Exception as e:
            self.skipTest(f"Error getting ElevenLabs voices: {e}")

        # Create an audio generation task
        audio_task = Task(
            TextTaskInput("Hello, this is a test audio message."),
            Agent(
                agent_type=AgentTypes.SPEECH.value,
                provider="elevenlabs",
                mission="Generate test audio",
                model_params={
                    "key": self.elevenlabs_api_key,
                    "voice_id": voice_id,
                    "model_id": "eleven_multilingual_v2",
                },
            ),
            log=True,
        )

        # Create flow with auto-save enabled
        flow = Flow(
            tasks={"test_audio": audio_task},
            map_paths={"test_audio": []},
            auto_save_outputs=True,
            output_dir=self.output_dir,
            output_file_map={"test_audio": "test_message.mp3"},
            log=True,
        )

        # Execute flow
        results = await flow.start()

        # Verify results
        self.assertIn("test_audio", results)
        self.assertEqual(results["test_audio"]["type"], "audio")
        self.assertTrue(len(results["test_audio"]["output"]) > 0)

        # Verify file was saved
        saved_files = flow.get_saved_files()
        self.assertIn("test_audio", saved_files)

        expected_path = os.path.join(self.output_dir, "test_message.mp3")
        self.assertEqual(saved_files["test_audio"]["path"], expected_path)
        self.assertTrue(os.path.exists(expected_path))

        # Verify file size is reasonable for audio
        file_size = os.path.getsize(expected_path)
        self.assertGreater(file_size, 1000)  # Should be at least 1KB

    def test_auto_save_disabled(self):
        """Test that auto-save doesn't occur when disabled."""
        if not self.openai_api_key:
            self.skipTest("OpenAI API key not available")

        asyncio.run(self._test_auto_save_disabled())

    async def _test_auto_save_disabled(self):
        """Test that files are not saved when auto-save is disabled."""
        # Create a simple text generation task
        text_task = Task(
            TextTaskInput("Write a short sentence"),
            Agent(
                agent_type=AgentTypes.TEXT.value,
                provider="openai",
                mission="Create a sentence",
                model_params={"key": self.openai_api_key, "model": "gpt-4o"},
            ),
            log=True,
        )

        # Create flow with auto-save DISABLED (default)
        flow = Flow(
            tasks={"sentence": text_task},
            map_paths={"sentence": []},
            auto_save_outputs=False,  # Explicitly disabled
            output_dir=self.output_dir,
            log=True,
        )

        # Execute flow
        results = await flow.start()

        # Verify results exist
        self.assertIn("sentence", results)
        self.assertEqual(results["sentence"]["type"], "text")

        # Verify no files were saved
        saved_files = flow.get_saved_files()
        self.assertEqual(len(saved_files), 0)

        # Verify output directory is empty or doesn't exist
        if os.path.exists(self.output_dir):
            files = os.listdir(self.output_dir)
            self.assertEqual(len(files), 0)

    def test_default_file_naming(self):
        """Test default file naming when no custom names are provided."""
        if not self.openai_api_key:
            self.skipTest("OpenAI API key not available")

        asyncio.run(self._test_default_naming())

    async def _test_default_naming(self):
        """Test default file naming convention."""
        # Create a simple text generation task
        text_task = Task(
            TextTaskInput("Write a short poem"),
            Agent(
                agent_type=AgentTypes.TEXT.value,
                provider="openai",
                mission="Create a poem",
                model_params={"key": self.openai_api_key, "model": "gpt-4o"},
            ),
            log=True,
        )

        # Create flow with auto-save enabled but no custom file names
        flow = Flow(
            tasks={"poem_task": text_task},
            map_paths={"poem_task": []},
            auto_save_outputs=True,
            output_dir=self.output_dir,
            # No output_file_map provided - should use defaults
            log=True,
        )

        # Execute flow
        results = await flow.start()

        # Verify results
        self.assertIn("poem_task", results)

        # Verify file was saved with default name
        saved_files = flow.get_saved_files()
        self.assertIn("poem_task", saved_files)

        expected_path = os.path.join(self.output_dir, "poem_task_output.txt")
        self.assertEqual(saved_files["poem_task"]["path"], expected_path)
        self.assertTrue(os.path.exists(expected_path))

    def test_flow_summary(self):
        """Test the flow summary functionality."""
        if not self.openai_api_key:
            self.skipTest("OpenAI API key not available")

        asyncio.run(self._test_flow_summary())

    async def _test_flow_summary(self):
        """Test flow summary includes all relevant information."""
        # Create a simple text generation task
        text_task = Task(
            TextTaskInput("Write a greeting"),
            Agent(
                agent_type=AgentTypes.TEXT.value,
                provider="openai",
                mission="Create a greeting",
                model_params={"key": self.openai_api_key, "model": "gpt-4o"},
            ),
            log=True,
        )

        # Create flow with auto-save enabled
        flow = Flow(
            tasks={"greeting": text_task},
            map_paths={"greeting": []},
            auto_save_outputs=True,
            output_dir=self.output_dir,
            log=True,
        )

        # Execute flow
        results = await flow.start()

        # Get flow summary
        summary = flow.get_flow_summary()

        # Verify summary structure
        self.assertIn("outputs", summary)
        self.assertIn("saved_files", summary)
        self.assertIn("errors", summary)
        self.assertIn("auto_save_enabled", summary)
        self.assertIn("output_directory", summary)

        # Verify summary content
        self.assertTrue(summary["auto_save_enabled"])
        self.assertEqual(summary["output_directory"], self.output_dir)
        self.assertIn("greeting", summary["outputs"])
        self.assertIn("greeting", summary["saved_files"])


if __name__ == "__main__":
    unittest.main() 
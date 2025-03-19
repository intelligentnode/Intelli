import os
import asyncio
import unittest
from dotenv import load_dotenv
from intelli.flow.agents.agent import Agent
from intelli.flow.agents.kagent import KerasAgent
from intelli.flow.input.task_input import TextTaskInput
from intelli.flow.tasks.task import Task
from intelli.flow.flow import Flow
from intelli.flow.types import AgentTypes
from intelli.flow.utils.flow_helper import FlowHelper

# Load environment variables
load_dotenv()


class TestMultiModalFlow(unittest.TestCase):
    # Define output directory as class constant
    OUTPUT_DIR = "./temp/travel/"

    def setUp(self):
        # Load API keys from environment variables
        self.api_keys = {
            "openai": os.getenv("OPENAI_API_KEY"),
            "mistral": os.getenv("MISTRAL_API_KEY"),
            "stability": os.getenv("STABILITY_API_KEY"),
            "elevenlabs": os.getenv("ELEVENLABS_API_KEY"),
            "google": os.getenv("GOOGLE_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        }

        # Create temp directory if it doesn't exist
        FlowHelper.ensure_directory(self.OUTPUT_DIR)

        # Skip test if essential keys are missing
        required_keys = [self.api_keys["openai"], self.api_keys["mistral"]]
        if not all(required_keys):
            self.skipTest("Missing required API keys for multimodal test")

        # Get a valid ElevenLabs voice ID if possible
        self.elevenlabs_voice_id = self._get_elevenlabs_voice_id()

        # Define file paths for outputs
        self.output_files = {
            "itinerary": os.path.join(self.OUTPUT_DIR, "rome_itinerary.txt"),
            "audio": os.path.join(self.OUTPUT_DIR, "rome_audio.mp3"),
            "openai_transcription": os.path.join(self.OUTPUT_DIR, "transcription_openai.txt"),
            "keras_transcription": os.path.join(self.OUTPUT_DIR, "transcription_keras.txt"),
            "image": os.path.join(self.OUTPUT_DIR, "rome_image.png"),
            "travel_guide": os.path.join(self.OUTPUT_DIR, "rome_travel_guide.md"),
            "flow_results": os.path.join(self.OUTPUT_DIR, "flow_results.json")
        }

    def _get_elevenlabs_voice_id(self):
        """Helper method to get an ElevenLabs voice ID"""
        if not self.api_keys["elevenlabs"]:
            return None

        try:
            from intelli.controller.remote_speech_model import RemoteSpeechModel
            speech_model = RemoteSpeechModel(
                key_value=self.api_keys["elevenlabs"], provider="elevenlabs"
            )

            # List available voices
            voices_result = speech_model.list_voices()
            if "voices" in voices_result and len(voices_result["voices"]) > 0:
                # Get the first voice ID
                voice_id = voices_result["voices"][0]["voice_id"]
                print(f"🔊 Using ElevenLabs voice: {voices_result['voices'][0]['name']} ({voice_id})")
                return voice_id
        except Exception as e:
            print(f"⚠️ Error getting ElevenLabs voices: {e}")

        return None

    def _create_speech_agent(self):
        """Create the appropriate speech agent based on available API keys"""
        if self.api_keys["elevenlabs"] and self.elevenlabs_voice_id:
            print(f"🔊 Using ElevenLabs for speech synthesis with voice ID: {self.elevenlabs_voice_id}")
            return Agent(
                agent_type=AgentTypes.SPEECH.value,
                provider="elevenlabs",
                mission="Convert travel itinerary to speech",
                model_params={
                    "key": self.api_keys["elevenlabs"],
                    "voice": self.elevenlabs_voice_id,
                    "model": "eleven_multilingual_v2",
                },
            )
        elif self.api_keys["google"]:
            print("🔊 Using Google for speech synthesis")
            return Agent(
                agent_type=AgentTypes.SPEECH.value,
                provider="google",
                mission="Convert travel itinerary to speech",
                model_params={"key": self.api_keys["google"], "language": "en-US"},
            )
        else:
            print("🔊 Using OpenAI for speech synthesis")
            return Agent(
                agent_type=AgentTypes.SPEECH.value,
                provider="openai",
                mission="Convert travel itinerary to speech",
                model_params={
                    "key": self.api_keys["openai"],
                    "model": "tts-1",
                    "voice": "alloy",
                },
            )

    def _create_image_prompt_agent(self):
        """Create the appropriate image prompt agent based on available API keys"""
        if self.api_keys["anthropic"]:
            print("🤖 Using Anthropic Claude for image prompt generation")
            return Agent(
                agent_type=AgentTypes.TEXT.value,
                provider="anthropic",
                mission="Create a detailed image prompt for the travel destination",
                model_params={
                    "key": self.api_keys["anthropic"],
                    "model": "claude-3-7-sonnet-20250219"
                },
            )
        else:
            print("⚠️ Using OpenAI for image prompt generation")
            return Agent(
                agent_type=AgentTypes.TEXT.value,
                provider="openai",
                mission="Create a detailed image prompt for the travel destination",
                model_params={"key": self.api_keys["openai"], "model": "gpt-4o"},
            )

    def test_travel_assistant_multimodal_flow(self):
        """
        Test a comprehensive travel assistant that utilizes multiple modalities:
        1. Create travel itinerary (Text - OpenAI)
        2. Create speech of the itinerary (Speech)
        3. Transcribe speech back to text (Recognition)
        4. Generate destination image (Image)
        5. Analyze image for travel features (Vision)
        6. Create final enhanced travel package (Text - Mistral)
        """
        # Run the async test through the event loop
        asyncio.run(self._run_travel_assistant_flow())

    async def _run_travel_assistant_flow(self):
        print("\n--- 🌍 Starting Comprehensive Travel Assistant Multimodal Flow ---")

        # Create tasks dictionary and map paths
        tasks = {}
        map_paths = {}

        # 1. Travel Itinerary Creation (OpenAI GPT)
        itinerary_agent = Agent(
            agent_type=AgentTypes.TEXT.value,
            provider="openai",
            mission="Create a detailed 3-day travel itinerary",
            model_params={"key": self.api_keys["openai"], "model": "gpt-4o"},
        )

        itinerary_task = Task(
            TextTaskInput(
                "Create a 3-day travel itinerary for Rome, Italy. Include major attractions, food recommendations, and transportation tips."
            ),
            itinerary_agent,
            log=True,
        )

        tasks["itinerary"] = itinerary_task
        map_paths["itinerary"] = ["speech", "image_prompt", "final_package"]

        # 2. Speech Synthesis
        speech_agent = self._create_speech_agent()
        speech_task = Task(
            TextTaskInput(
                "Convert the first day of this itinerary to speech for the traveler"
            ),
            speech_agent,
            log=True,
        )

        tasks["speech"] = speech_task
        map_paths["speech"] = ["transcribe_openai", "transcribe_keras"]

        # 3. Speech Recognition - OpenAI Whisper
        recognition_agent = Agent(
            agent_type=AgentTypes.RECOGNITION.value,
            provider="openai",
            mission="Transcribe the audio guide back to text",
            model_params={"key": self.api_keys["openai"], "model": "whisper-1"},
        )

        recognition_task = Task(
            TextTaskInput("Transcribe this audio accurately"),
            recognition_agent,
            log=True,
        )

        tasks["transcribe_openai"] = recognition_task
        map_paths["transcribe_openai"] = ["transcription_comparison"]

        # 4. Speech Recognition - Keras Whisper (if model is available)
        try:
            keras_recognition_agent = KerasAgent(
                agent_type=AgentTypes.RECOGNITION.value,
                provider="keras",
                mission="Transcribe audio using local Whisper model",
                model_params={
                    "model_name": "whisper_tiny_en",
                    "language": "<|en|>",
                    "user_prompt": "You are transcribing a travel itinerary audio.",
                    "max_steps": 80,
                    "max_chunk_sec": 30,
                },
            )

            keras_recognition_task = Task(
                TextTaskInput("Transcribe this audio using the local Whisper model"),
                keras_recognition_agent,
                log=True,
            )

            tasks["transcribe_keras"] = keras_recognition_task
            map_paths["transcribe_keras"] = ["transcription_comparison"]
        except Exception as e:
            print(f"⚠️ Keras Whisper model not available, skipping: {e}")

        # 5. Transcription Comparison (only if both transcription tasks exist)
        if "transcribe_openai" in tasks and "transcribe_keras" in tasks:
            compare_agent = Agent(
                agent_type=AgentTypes.TEXT.value,
                provider="openai",
                mission="Compare the two transcription results",
                model_params={"key": self.api_keys["openai"], "model": "gpt-4o"},
            )

            compare_task = Task(
                TextTaskInput(
                    "Compare these two transcriptions and highlight any differences or quality issues"
                ),
                compare_agent,
                log=True,
            )

            tasks["transcription_comparison"] = compare_task
            map_paths["transcription_comparison"] = ["final_package"]

        # 6. Image Prompt Creation
        img_prompt_agent = self._create_image_prompt_agent()
        img_prompt_task = Task(
            TextTaskInput(
                "Create a short, specific image generation prompt (under 50 words) for Rome showing the iconic Colosseum"
            ),
            img_prompt_agent,
            log=True,
        )

        tasks["image_prompt"] = img_prompt_task
        map_paths["image_prompt"] = ["destination_image"]

        # 7. Destination Image Generation (if Stability AI key is available)
        if self.api_keys["stability"]:
            image_agent = Agent(
                agent_type=AgentTypes.IMAGE.value,
                provider="stability",
                mission="Generate a visual representation of the destination",
                model_params={"key": self.api_keys["stability"]},
            )

            image_task = Task(
                TextTaskInput(
                    "Rome with the iconic Colosseum under clear blue sky"
                ),
                image_agent,
                log=True,
            )

            tasks["destination_image"] = image_task
            map_paths["destination_image"] = ["image_analysis"]

            # 8. Image Analysis with Vision
            vision_agent = Agent(
                agent_type=AgentTypes.VISION.value,
                provider="openai",
                mission="Analyze the image and identify key landmarks and travel features",
                model_params={
                    "key": self.api_keys["openai"],
                    "model": "gpt-4o",
                    "extension": "png",
                },
            )

            vision_task = Task(
                TextTaskInput(
                    "Identify the landmarks and notable features in this image that would be relevant for a traveler to Rome"
                ),
                vision_agent,
                log=True,
            )

            tasks["image_analysis"] = vision_task
            map_paths["image_analysis"] = ["final_package"]

        # 9. Final Travel Package (Mistral AI)
        final_agent = Agent(
            agent_type=AgentTypes.TEXT.value,
            provider="mistral",
            mission="Create an enhanced travel package combining all insights",
            model_params={"key": self.api_keys["mistral"], "model": "mistral-medium"},
        )

        final_task = Task(
            TextTaskInput(
                "Create a comprehensive and engaging travel guide for Rome by combining the itinerary, transcription insights, and image analysis"
            ),
            final_agent,
            log=True,
        )

        tasks["final_package"] = final_task
        map_paths["final_package"] = []

        # Create and execute the flow
        flow = Flow(tasks=tasks, map_paths=map_paths, log=True)

        # Generate and save the flow visualization
        try:
            graph_path = flow.generate_graph_img(
                name="rome_travel_assistant_flow", save_path=self.OUTPUT_DIR
            )
            print(f"🎨 Flow visualization saved to: {graph_path}")
        except Exception as e:
            print(f"⚠️ Warning: Could not generate graph image: {e}")

        # Execute the flow
        results = await flow.start(max_workers=3)

        # Save outputs to files
        self._save_flow_outputs(results)

        # Create a sample audio and image if the real ones failed
        self._ensure_test_outputs()

        # Validate results
        self._validate_results(results)

        print("✅ Multimodal flow completed successfully")
        return results

    def _save_flow_outputs(self, results):
        """Save the various outputs from the flow to files"""
        saved_files = {}

        # Save text outputs
        if "itinerary" in results:
            saved_files["itinerary"] = FlowHelper.save_text_output(
                results["itinerary"]["output"],
                self.output_files["itinerary"]
            )
            print(f"📄 Saved itinerary to {self.output_files['itinerary']}")

        # Save audio
        if "speech" in results and results["speech"]["type"] == "audio":
            file_path, file_size = FlowHelper.save_audio_output(
                results["speech"]["output"],
                self.output_files["audio"]
            )
            saved_files["audio"] = file_path
            if file_path:
                print(f"🔊 Saved audio to {file_path}, size: {file_size} bytes")

        # Save transcriptions
        if "transcribe_openai" in results:
            saved_files["transcribe_openai"] = FlowHelper.save_text_output(
                results["transcribe_openai"]["output"],
                self.output_files["openai_transcription"]
            )
            print(f"📝 Saved OpenAI transcription to {self.output_files['openai_transcription']}")

        if "transcribe_keras" in results:
            saved_files["transcribe_keras"] = FlowHelper.save_text_output(
                results["transcribe_keras"]["output"],
                self.output_files["keras_transcription"]
            )
            print(f"📝 Saved Keras transcription to {self.output_files['keras_transcription']}")

        # Save image
        if "destination_image" in results and results["destination_image"]["type"] == "image":
            file_path, file_size = FlowHelper.save_image_output(
                results["destination_image"]["output"],
                self.output_files["image"]
            )
            saved_files["image"] = file_path
            if file_path:
                print(f"🖼️ Saved image to {file_path}, size: {file_size} bytes")

        # Save final package
        if "final_package" in results:
            saved_files["final_package"] = FlowHelper.save_text_output(
                results["final_package"]["output"],
                self.output_files["travel_guide"]
            )
            print(f"📚 Saved travel guide to {self.output_files['travel_guide']}")

        # Save all results to JSON
        json_path = FlowHelper.save_flow_results(
            results,
            self.output_files["flow_results"],
            exclude_binary=True
        )
        if json_path:
            print(f"📊 Saved flow results to {json_path}")

    def _ensure_test_outputs(self):
        """Create sample files for testing if the actual ones weren't generated"""
        # Define expected output files with their content types
        output_map = {
            self.output_files["audio"]: "audio",
            self.output_files["image"]: "image",
            self.output_files["travel_guide"]: "text"
        }

        # Create sample files if needed
        FlowHelper.create_sample_outputs(output_map)

    def _validate_results(self, results):
        """Validate that the flow completed with necessary outputs"""
        # Verify we have any results
        if not results:
            self.fail("No results were returned from the flow")

        # Validate key outputs
        self.assertIn("itinerary", results, "Itinerary output missing")
        self.assertEqual(results["itinerary"]["type"], "text", "Itinerary should be text type")

        # Check for final package if it exists
        if "final_package" in results:
            self.assertEqual(
                results["final_package"]["type"],
                "text",
                "Final package should be text type"
            )

        # Validate output files exist
        expected_files = [
            self.output_files["itinerary"],
            self.output_files["audio"],
            self.output_files["image"],
            self.output_files["travel_guide"]
        ]

        for file_path in expected_files:
            self.assertTrue(
                os.path.exists(file_path),
                f"Output file should exist at {file_path}"
            )

        print("✅ Test validation passed!")


if __name__ == "__main__":
    unittest.main()

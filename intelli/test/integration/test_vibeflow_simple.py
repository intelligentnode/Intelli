import asyncio
import os
import unittest
from dotenv import load_dotenv

from intelli.flow.vibe import VibeFlow
from intelli.flow.utils.flow_helper import FlowHelper

load_dotenv()

class TestVibeFlowSimple(unittest.TestCase):
    
    OUTPUT_DIR = "./temp/vibeflow_simple"

    def setUp(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_key:
            self.skipTest("OPENAI_API_KEY not found in environment")
        
        # Create temp directory
        FlowHelper.ensure_directory(self.OUTPUT_DIR)
        
        # Initialize Vibe with real OpenAI key
        self.vf = VibeFlow(
            planner_provider="openai",
            planner_api_key=self.openai_key,
            planner_model="gpt-5.2"
        )

    def tearDown(self):
        # We don't remove the temp directory to allow inspection of results
        pass

    def test_vibe_basic_joke(self):
        """1-step text flow."""
        description = "Create a 1-step flow that returns a funny joke about AI using openai gpt-5.2."
        
        print(f"\n--- VIBE BASIC JOKE START ---")
        flow = asyncio.run(self.vf.build(description, render_graph=True, save_dir=self.OUTPUT_DIR, graph_name="joke_graph"))
        results = asyncio.run(flow.start(initial_input="Tell me a joke"))
        
        # Save results
        FlowHelper.save_flow_results(results, os.path.join(self.OUTPUT_DIR, "joke_results.json"))
        
        for name, data in results.items():
            print(f"Result [{name}]: {data['output']}")
        print(f"--- VIBE BASIC JOKE END ---\n")
        self.assertTrue(len(results) > 0)

    def test_vibe_image_generation(self):
        """Test image generation vibe."""
        description = f"Create a flow that generates an image of a 'futuristic city' using openai dall-e-3 with width 1024, height 1024 and response_format b64_json. Save output to {self.OUTPUT_DIR}"
        
        print(f"\n--- VIBE IMAGE GENERATION START ---")
        # Build flow
        flow = asyncio.run(self.vf.build(description, render_graph=True, save_dir=self.OUTPUT_DIR, graph_name="image_graph"))
        
        # Ensure output settings
        flow.output_dir = self.OUTPUT_DIR
        flow.auto_save_outputs = True
        
        results = asyncio.run(flow.start(initial_input="Generate image"))
        
        # Save results JSON (excluding binary for readability)
        FlowHelper.save_flow_results(results, os.path.join(self.OUTPUT_DIR, "image_results.json"), exclude_binary=True)
        
        for name, data in results.items():
            print(f"Task: {name}, Type: {data['type']}")
            if data['type'] == 'image':
                print(f"Image generated successfully (size: {len(data['output']) if data['output'] else 0})")
        
        # Check if file was saved
        files = os.listdir(self.OUTPUT_DIR)
        print(f"Files in {self.OUTPUT_DIR}: {files}")
        self.assertTrue(any(f.endswith('.png') for f in files))
        print(f"--- VIBE IMAGE GENERATION END ---\n")

    def test_vibe_audio_and_recognition(self):
        """Test generate audio -> recognition (STT)."""
        description = (
            "Create a 2-step flow: "
            "1. Step 'gen_audio' generates speech audio for the exact text 'Intelli is awesome' using openai tts-1 with stream false. "
            "2. Step 'transcribe' transcribes that audio back to text using openai whisper-1."
        )
        
        print(f"\n--- VIBE AUDIO & RECOGNITION START ---")
        flow = asyncio.run(self.vf.build(description, render_graph=True, save_dir=self.OUTPUT_DIR, graph_name="audio_graph"))
        
        # Ensure output settings
        flow.output_dir = self.OUTPUT_DIR
        flow.auto_save_outputs = True
        
        # Run it
        results = asyncio.run(flow.start())
        
        # Save results
        FlowHelper.save_flow_results(results, os.path.join(self.OUTPUT_DIR, "audio_results.json"), exclude_binary=True)
        
        for name, data in results.items():
            print(f"Task: {name}, Output: {data['output'][:100] if isinstance(data['output'], str) else 'binary data'}")
        
        # We expect at least one task to return text (the transcription)
        has_text = any(isinstance(d['output'], str) and 'intel' in d['output'].lower() for d in results.values())
        self.assertTrue(has_text)
        print(f"--- VIBE AUDIO & RECOGNITION END ---\n")

    def test_vibe_chain_with_memory(self):
        """Test text chain with memory storage."""
        description = (
            "Create a flow where: "
            "1. step1 summarizes the input text. "
            "2. The summary is stored in memory key 'summary_result'. "
            "3. step2 translates that 'summary_result' from memory into French."
        )
        
        print(f"\n--- VIBE CHAIN WITH MEMORY START ---")
        flow = asyncio.run(self.vf.build(description, render_graph=True, save_dir=self.OUTPUT_DIR, graph_name="memory_graph"))
        
        # Ensure output settings
        flow.output_dir = self.OUTPUT_DIR
        flow.auto_save_outputs = True
        
        text_input = "IntelliNode is an open source library that helps developers build AI agents and orchestrate multi-modal flows."
        results = asyncio.run(flow.start(initial_input=text_input))
        
        # Save results
        FlowHelper.save_flow_results(results, os.path.join(self.OUTPUT_DIR, "memory_results.json"))
        
        print(f"Tasks: {list(flow.tasks.keys())}")
        print(f"Memory Map: {flow.output_memory_map}")
        for name, data in results.items():
            print(f"Result [{name}]: {data['output']}")
            
        # Find the memory key from the map
        mem_key = list(flow.output_memory_map.values())[0] if flow.output_memory_map else "summary_result"
        self.assertTrue(flow.memory.has_key(mem_key))
        print(f"Memory '{mem_key}': {flow.memory.retrieve(mem_key)}")
        print(f"--- VIBE CHAIN WITH MEMORY END ---\n")

    def test_vibe_custom_model_preference(self):
        """Test VibeFlow with a preferred text model string (GPT-5.2 mini)."""
        custom_output_dir = os.path.join(self.OUTPUT_DIR, "custom_model")
        FlowHelper.ensure_directory(custom_output_dir)
        
        # Initialize Vibe with specific model string
        vf_custom = VibeFlow(
            planner_provider="openai",
            planner_api_key=self.openai_key,
            planner_model="gpt-5.2",
            text_model="openai gpt-5.2"
        )
        
        description = "Create a 1-step flow that tells a joke about quantum computing."
        
        print(f"\n--- VIBE CUSTOM MODEL PREFERENCE START ---")
        flow = asyncio.run(vf_custom.build(description, render_graph=True, save_dir=custom_output_dir, graph_name="custom_joke_graph"))
        
        # Check if the generated task uses the requested model
        task_name = list(flow.tasks.keys())[0]
        task = flow.tasks[task_name]
        print(f"Generated Task: {task_name}, Model: {task.agent.model_params.get('model')}")
        
        # Run it
        results = asyncio.run(flow.start(initial_input="Tell me a quantum joke"))
        
        print(f"Result: {results[task_name]['output']}")
        print(f"--- VIBE CUSTOM MODEL PREFERENCE END ---\n")
        self.assertTrue(len(results) > 0)

    def test_vibe_gemini_image_generation(self):
        """Test image generation vibe with Gemini model string."""
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            self.skipTest("GEMINI_API_KEY not found in environment")
            
        custom_output_dir = os.path.join(self.OUTPUT_DIR, "gemini_image")
        FlowHelper.ensure_directory(custom_output_dir)
        
        # Initialize Vibe with Gemini image model string
        vf_gemini = VibeFlow(
            planner_provider="openai",
            planner_api_key=self.openai_key,
            planner_model="gpt-5.2",
            image_model="gemini gemini-3-pro-image-preview"
        )
        
        description = f"Create a 1-step flow that generates an image of a 'neon lotus flower' using gemini. Save output to {custom_output_dir}"
        
        print(f"\n--- VIBE GEMINI IMAGE GENERATION START ---")
        flow = asyncio.run(vf_gemini.build(description, render_graph=True, save_dir=custom_output_dir, graph_name="gemini_image_graph"))
        
        # Ensure output settings
        flow.output_dir = custom_output_dir
        flow.auto_save_outputs = True
        
        # Inject Gemini key if not in env
        for task in flow.tasks.values():
            if task.agent.provider == 'gemini':
                task.agent.model_params['key'] = gemini_key
        
        results = asyncio.run(flow.start(initial_input="Generate neon lotus"))
        
        # Validation
        for name, data in results.items():
            if data['type'] == 'image':
                print(f"Gemini Image generated successfully")
        
        print(f"--- VIBE GEMINI IMAGE GENERATION END ---\n")
        self.assertTrue(len(results) > 0)

if __name__ == "__main__":
    unittest.main()

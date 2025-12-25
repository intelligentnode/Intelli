import asyncio
import os
import unittest
from pathlib import Path
from dotenv import load_dotenv

from intelli.flow.vibe import VibeAgent
from intelli.flow.utils.flow_helper import FlowHelper
from intelli.flow.types import AgentTypes

# Load .env from the 'intelli' directory where it is located
load_dotenv()

class TestBlogPostVibe(unittest.TestCase):
    
    OUTPUT_DIR = "./temp/blog_post"

    def setUp(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_key:
            self.skipTest("OPENAI_API_KEY not found in environment")
        
        # Ensure directory exists
        FlowHelper.ensure_directory(self.OUTPUT_DIR)
        
        # Initialize VibeAgent with GPT-5.2 as the planner
        # Using Gemini for images as requested
        self.vf = VibeAgent(
            planner_provider="openai",
            planner_api_key=self.openai_key,
            planner_model="gpt-5.2",
            image_model="gemini gemini-3-pro-image-preview"
        )

    def test_research_factory_architecture(self):
        """
        Validate that the vibe intent generates the expected multi-agent architecture:
        Search Agent (Scout) -> Text Agent (Analyst) -> Image Agent (Creator).
        """
        intent = (
            "Create a 3-step linear flow for a 'Research-to-Content Factory': "
            "1. Search: Perform a web research using ONLY 'google' as provider for solid-state battery breakthroughs in the last 30 days. "
            "2. Analyst: Summarize the findings into key technical metrics. "
            "3. Creator: Generate an image using 'gemini' showing a futuristic representation of these battery findings."
        )

        print(f"\n--- RESEARCH FACTORY ARCHITECTURE VALIDATION START ---")
        
        # 1. Build the flow architecture
        flow = asyncio.run(self.vf.build(intent, render_graph=True, save_dir=self.OUTPUT_DIR, graph_name="research_factory_blueprint"))
        
        # 2. Validate Agents
        agent_types = [task.agent.type for task in flow.tasks.values()]
        print(f"Detected Agent Types: {agent_types}")
        
        has_search = any(at == AgentTypes.SEARCH.value for at in agent_types)
        has_text = any(at == AgentTypes.TEXT.value for at in agent_types)
        has_image = any(at == AgentTypes.IMAGE.value for at in agent_types)
        
        # Check providers
        search_providers = [task.agent.provider for task in flow.tasks.values() if task.agent.type == AgentTypes.SEARCH.value]
        print(f"Search Providers: {search_providers}")

        self.assertTrue(has_search, "Should have a Search Agent")
        self.assertTrue(has_text, "Should have a Text Agent")
        self.assertTrue(has_image, "Should have an Image Agent")
        self.assertNotIn("intellicloud", search_providers, "Should not use intellicloud")

        # 3. Execution
        flow.output_dir = self.OUTPUT_DIR
        flow.auto_save_outputs = True
        
        # Ensure we have the necessary search keys for execution if available
        google_key = os.getenv("GOOGLE_API_KEY")
        google_cse = os.getenv("GOOGLE_CSE_ID")
        
        if not google_key or not google_cse:
            print("Warning: Missing GOOGLE search keys. Execution might fail or skip search steps.")
        
        # Inject keys if available to ensure agents have them
        for task in flow.tasks.values():
            if task.agent.type == AgentTypes.SEARCH.value and task.agent.provider == "google":
                if google_key: task.agent.model_params["google_api_key"] = google_key
                if google_cse: task.agent.model_params["google_cse_id"] = google_cse
            if task.agent.type == AgentTypes.IMAGE.value and task.agent.provider == "gemini":
                # Use GOOGLE_API_KEY for Gemini if GEMINI_API_KEY is not set
                gemini_key = os.getenv("GEMINI_API_KEY") or google_key
                if gemini_key: task.agent.model_params["key"] = gemini_key

        print("Executing the Research Factory flow...")
        results = asyncio.run(flow.start())
        
        # Save results JSON (excluding binary for readability)
        FlowHelper.save_flow_results(results, os.path.join(self.OUTPUT_DIR, "research_factory_results.json"), exclude_binary=True)
        
        # Validate image output
        has_image_file = any(f.endswith('.png') for f in os.listdir(self.OUTPUT_DIR) if not f.startswith('research_factory_blueprint'))
        print(f"Image generated and saved: {has_image_file}")
        
        print(f"Flow completed. Results saved to {self.OUTPUT_DIR}")
        for name, data in results.items():
            content_preview = str(data['output'])[:100].replace('\n', ' ')
            print(f"Task: {name} | Type: {data['type']} | Preview: {content_preview}...")

        print(f"--- RESEARCH FACTORY ARCHITECTURE VALIDATION END ---\n")

if __name__ == "__main__":
    unittest.main()


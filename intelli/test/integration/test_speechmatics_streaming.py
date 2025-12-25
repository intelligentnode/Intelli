import os
import unittest
import asyncio
from dotenv import load_dotenv
from intelli.flow.agents.speechmatics_agent import StreamingSpeechAgent
from intelli.flow.tasks.task import Task
from intelli.flow.input.task_input import TextTaskInput
from intelli.flow.flow import Flow
from intelli.flow.input.agent_input import AgentInput
from intelli.flow.agents.agent import Agent

load_dotenv()

class TestSpeechmaticsStreaming(unittest.TestCase):
    def setUp(self):
        self.api_key = os.getenv('SPEECHMATICS_API_KEY')
        self.openai_key = os.getenv('OPENAI_API_KEY')
        # Path to test audio file
        self.audio_path = os.path.join(os.getcwd(), 'temp', 'test.wav')
        
    def test_streaming_speech_agent(self):
        """Test the StreamingSpeechAgent directly."""
        if not self.api_key:
            self.skipTest("SPEECHMATICS_API_KEY not found in environment")
            
        if not os.path.exists(self.audio_path):
            self.skipTest(f"Audio file not found at {self.audio_path}")
            
        # Define a listener to collect chunks
        collected_chunks = []
        def listener(chunk):
            print(f"Received chunk: {chunk}")
            collected_chunks.append(chunk)
            
        # 1. Create the streaming agent
        agent = StreamingSpeechAgent(
            api_key=self.api_key,
            listener_callback=listener
        )
        
        # 2. Read the audio data
        with open(self.audio_path, 'rb') as f:
            audio_data = f.read()
            
        # 3. Execute the agent
        agent_input = AgentInput(desc="Transcribe", audio=audio_data)
        result = agent.execute(agent_input, new_params={"max_wait_seconds": 20, "idle_timeout_seconds": 2, "chunk_ms": 50})
        
        print(f"Final Transcript: {result}")
        
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0, "Final transcript should not be empty")
        self.assertTrue(len(collected_chunks) > 0, "Should have received at least one chunk")

    def test_streaming_in_flow(self):
        """Test the streaming agent within a Flow orchestrator."""
        if not self.api_key:
            self.skipTest("SPEECHMATICS_API_KEY not found in environment")
        
        if not self.openai_key:
            self.skipTest("OPENAI_API_KEY not found in environment")
            
        if not os.path.exists(self.audio_path):
            self.skipTest(f"Audio file not found at {self.audio_path}")

        # Listener for real-time output
        collected_stream = []
        def listener(chunk):
            print(f"[FLOW STREAM] {chunk}")
            collected_stream.append(chunk)

        # Define agents
        stream_agent = StreamingSpeechAgent(
            api_key=self.api_key, 
            listener_callback=listener
        )
        
        # Subsequent task to verify data passing
        llm_agent = Agent(
            agent_type="text", 
            provider="openai", 
            mission="summarize the transcript into one sentence",
            model_params={"key": self.openai_key, "model": "gpt-4o-mini"}
        )
        
        # Define Flow
        flow = Flow(
            tasks={
                "transcribe_task": Task(
                    TextTaskInput("transcribe the audio"),
                    agent=stream_agent,
                    model_params={"max_wait_seconds": 20, "idle_timeout_seconds": 2, "chunk_ms": 50},
                ),
                "summarize_task": Task(TextTaskInput("summarize the text"), agent=llm_agent)
            },
            map_paths={
                "transcribe_task": ["summarize_task"]
            },
            log=True
        )
        
        # Run Flow
        with open(self.audio_path, 'rb') as f:
            audio_data = f.read()
            
        # Start the flow
        results = asyncio.run(flow.start(initial_input=audio_data, initial_input_type="audio"))
        
        print(f"Flow Completed. Tasks in result: {list(results.keys())}")
        
        self.assertIn("transcribe_task", results)
        self.assertIn("summarize_task", results)
        self.assertTrue(len(results["transcribe_task"]["output"]) > 0)
        self.assertTrue(len(results["summarize_task"]["output"]) > 0)
        self.assertTrue(len(collected_stream) > 0, "Listener should have collected chunks during flow")

if __name__ == "__main__":
    unittest.main()


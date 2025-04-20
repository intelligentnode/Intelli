import os
import asyncio
import unittest
from dotenv import load_dotenv
from intelli.flow.agents.agent import Agent
from intelli.flow.input.task_input import TextTaskInput
from intelli.flow.tasks.task import Task
from intelli.flow.flow import Flow
from intelli.flow.types import AgentTypes
from intelli.flow.dynamic_connector import DynamicConnector, ConnectorMode
from intelli.flow.utils.dynamic_utils import (
    text_length_router,
    text_content_router,
    sentiment_router,
)
from intelli.flow.utils.flow_helper import FlowHelper

load_dotenv()


class TestDynamicFlow(unittest.TestCase):
    # Define output directory
    OUTPUT_DIR = "./temp/dynamic_flow/"

    def setUp(self):

        self.api_keys = {
            "openai": os.getenv("OPENAI_API_KEY"),
            "mistral": os.getenv("MISTRAL_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
            "gemini": os.getenv("GEMINI_API_KEY"),
        }

        # Create temp directory
        FlowHelper.ensure_directory(self.OUTPUT_DIR)

        # Check essential keys
        if not self.api_keys["openai"]:
            self.skipTest("Missing OpenAI API key for dynamic flow test")

    def test_text_length_routing(self):
        """Test dynamic routing based on text length."""
        print("\n--- 🔄 Testing Dynamic Routing Based on Text Length ---")

        # Define all tasks
        query_task = Task(
            TextTaskInput(
                "Write a short explanation (less than 100 words) about dynamic routing in AI systems."
            ),
            Agent(
                agent_type=AgentTypes.TEXT.value,
                provider="anthropic",
                mission="Generate text of varying length",
                model_params={
                    "key": self.api_keys["anthropic"],
                    "model": "claude-3-7-sonnet-20250219",
                },
            ),
            log=True,
        )

        short_task = Task(
            TextTaskInput("Summarize this short text in one sentence:"),
            Agent(
                agent_type=AgentTypes.TEXT.value,
                provider="openai",
                mission="Process short text",
                model_params={"key": self.api_keys["openai"], "model": "gpt-4o"},
            ),
            log=True,
        )

        medium_task = Task(
            TextTaskInput("Extract the main points from this text:"),
            Agent(
                agent_type=AgentTypes.TEXT.value,
                provider="openai",
                mission="Process medium text",
                model_params={"key": self.api_keys["openai"], "model": "gpt-4o"},
            ),
            log=True,
        )

        long_task = Task(
            TextTaskInput("Create a detailed analysis of this comprehensive text:"),
            Agent(
                agent_type=AgentTypes.TEXT.value,
                provider="openai",
                mission="Process long text",
                model_params={"key": self.api_keys["openai"], "model": "gpt-4o"},
            ),
            log=True,
        )

        # Collect all tasks
        tasks = {
            "query": query_task,
            "short_processor": short_task,
            "medium_processor": medium_task,
            "long_processor": long_task,
        }

        # Define length router function for dynamic connector
        def length_router(output, output_type):
            return text_length_router(
                output, output_type, [100, 200], ["short", "medium", "long"]
            )

        # Define all connections
        map_paths = {}  # No static connections in this example

        dynamic_connectors = {
            "query": DynamicConnector(
                decision_fn=length_router,
                destinations={
                    "short": "short_processor",
                    "medium": "medium_processor",
                    "long": "long_processor",
                },
                name="length_router",
                description="Routes based on text length",
                mode=ConnectorMode.LENGTH_BASED,
            )
        }

        # Create and run the flow
        flow = Flow(
            tasks=tasks,
            map_paths=map_paths,
            dynamic_connectors=dynamic_connectors,
            log=True,
        )

        # Generate flow visualization
        try:
            graph_path = flow.generate_graph_img(
                name="length_based_routing", save_path=self.OUTPUT_DIR
            )
            print(f"🎨 Flow visualization saved to: {graph_path}")
        except Exception as e:
            print(f"⚠️ Warning: Could not generate graph image: {e}")

        # Execute the flow
        results = asyncio.run(flow.start(max_workers=3))

        # Validate results
        self.assertIn("query", results, "Initial query task should be in results")

        # Check that exactly one of the processor tasks was executed
        processor_results = [r for r in results.keys() if r.endswith("_processor")]
        self.assertEqual(
            len(processor_results), 1, "Exactly one processor should be executed"
        )

        # Check which processor was chosen
        processor = processor_results[0]
        print(f"🔍 Selected processor: {processor}")

        # Validate that the query output length matches the selected processor
        query_output = results["query"]["output"]
        query_length = len(query_output)

        if processor == "short_processor":
            self.assertLessEqual(query_length, 100)
        elif processor == "medium_processor":
            self.assertGreater(query_length, 100)
            self.assertLessEqual(query_length, 200)
        else:  # long_processor
            self.assertGreater(query_length, 200)

    def test_content_based_routing(self):
        """Test dynamic routing based on content keywords."""
        print("\n--- 🔄 Testing Dynamic Routing Based on Content Keywords ---")

        # Define all tasks
        query_task = Task(
            TextTaskInput(
                "Write a paragraph about artificial intelligence applications in healthcare."
            ),
            Agent(
                agent_type=AgentTypes.TEXT.value,
                provider="openai",
                mission="Generate text about a specific topic",
                model_params={"key": self.api_keys["openai"], "model": "gpt-4o"},
            ),
            log=True,
        )

        tech_task = Task(
            TextTaskInput("Expand on the technical aspects mentioned in this text:"),
            Agent(
                agent_type=AgentTypes.TEXT.value,
                provider="openai",
                mission="Process technology content",
                model_params={"key": self.api_keys["openai"], "model": "gpt-4o"},
            ),
            log=True,
        )

        healthcare_task = Task(
            TextTaskInput(
                "Discuss the healthcare implications mentioned in this text:"
            ),
            Agent(
                agent_type=AgentTypes.TEXT.value,
                provider="openai",
                mission="Process healthcare content",
                model_params={"key": self.api_keys["openai"], "model": "gpt-4o"},
            ),
            log=True,
        )

        ethics_task = Task(
            TextTaskInput("Examine the ethical considerations in this text:"),
            Agent(
                agent_type=AgentTypes.TEXT.value,
                provider="openai",
                mission="Process ethics content",
                model_params={"key": self.api_keys["openai"], "model": "gpt-4o"},
            ),
            log=True,
        )

        # Collect all tasks
        tasks = {
            "query": query_task,
            "tech_specialist": tech_task,
            "healthcare_specialist": healthcare_task,
            "ethics_specialist": ethics_task,
        }

        # Define keywords for content routing
        keywords = {
            "tech": [
                "algorithm",
                "machine learning",
                "neural network",
                "AI model",
                "compute",
                "data processing",
            ],
            "healthcare": [
                "patient",
                "diagnosis",
                "treatment",
                "medical",
                "doctor",
                "hospital",
                "clinic",
            ],
            "ethics": [
                "privacy",
                "bias",
                "fairness",
                "consent",
                "regulation",
                "law",
                "policy",
            ],
        }

        # Define content router function
        def content_router(output, output_type):
            return text_content_router(output, output_type, keywords)

        # Define all connections
        map_paths = {}  # No static connections in this example

        dynamic_connectors = {
            "query": DynamicConnector(
                decision_fn=content_router,
                destinations={
                    "tech": "tech_specialist",
                    "healthcare": "healthcare_specialist",
                    "ethics": "ethics_specialist",
                },
                name="content_router",
                description="Routes based on content keywords",
                mode=ConnectorMode.CONTENT_BASED,
            )
        }

        # Create and run the flow
        flow = Flow(
            tasks=tasks,
            map_paths=map_paths,
            dynamic_connectors=dynamic_connectors,
            log=True,
        )

        # Generate flow visualization
        try:
            graph_path = flow.generate_graph_img(
                name="content_based_routing", save_path=self.OUTPUT_DIR
            )
            print(f"🎨 Flow visualization saved to: {graph_path}")
        except Exception as e:
            print(f"⚠️ Warning: Could not generate graph image: {e}")

        # Execute the flow
        results = asyncio.run(flow.start(max_workers=3))

        # Validate results
        self.assertIn("query", results, "Initial query task should be in results")

        # Check that exactly one of the specialist tasks was executed
        specialist_results = [r for r in results.keys() if r.endswith("_specialist")]
        self.assertEqual(
            len(specialist_results), 1, "Exactly one specialist should be executed"
        )

        # Check which specialist was chosen
        specialist = specialist_results[0]
        print(f"🔍 Selected specialist: {specialist}")

    def test_sentiment_based_routing(self):
        """Test dynamic routing based on sentiment analysis."""
        print("\n--- 🔄 Testing Dynamic Routing Based on Sentiment ---")

        # Randomly choose positive, negative, or neutral prompt
        import random

        sentiment_prompts = [
            "Write a very positive review of a restaurant you visited.",
            "Write a very negative review of a product you purchased.",
            "Write a neutral description of a historical event.",
        ]
        chosen_prompt = random.choice(sentiment_prompts)

        # Define all tasks
        query_task = Task(
            TextTaskInput(chosen_prompt),
            Agent(
                agent_type=AgentTypes.TEXT.value,
                provider="openai",
                mission="Generate text with specific sentiment",
                model_params={"key": self.api_keys["openai"], "model": "gpt-4o"},
            ),
            log=True,
        )

        positive_task = Task(
            TextTaskInput(
                "This text has positive sentiment. Suggest ways to maintain this positivity:"
            ),
            Agent(
                agent_type=AgentTypes.TEXT.value,
                provider="anthropic",
                mission="Handle positive content",
                model_params={
                    "key": self.api_keys["anthropic"],
                    "model": "claude-3-7-sonnet-20250219",
                },
            ),
            log=True,
        )

        neutral_task = Task(
            TextTaskInput(
                "This text has neutral sentiment. Add more descriptive details:"
            ),
            Agent(
                agent_type=AgentTypes.TEXT.value,
                provider="openai",
                mission="Handle neutral content",
                model_params={"key": self.api_keys["openai"], "model": "gpt-4o"},
            ),
            log=True,
        )

        negative_task = Task(
            TextTaskInput(
                "This text has negative sentiment. Provide a more balanced perspective:"
            ),
            Agent(
                agent_type=AgentTypes.TEXT.value,
                provider="openai",
                mission="Handle negative content",
                model_params={"key": self.api_keys["openai"], "model": "gpt-4o"},
            ),
            log=True,
        )

        # Collect all tasks
        tasks = {
            "query": query_task,
            "positive_handler": positive_task,
            "neutral_handler": neutral_task,
            "negative_handler": negative_task,
        }

        # Define sentiment router function
        def sentiment_based_router(output, output_type):
            return sentiment_router(
                output, output_type, "positive", "neutral", "negative"
            )

        # Define all connections
        map_paths = {}  # No static connections in this example

        dynamic_connectors = {
            "query": DynamicConnector(
                decision_fn=sentiment_based_router,
                destinations={
                    "positive": "positive_handler",
                    "neutral": "neutral_handler",
                    "negative": "negative_handler",
                },
                name="sentiment_router",
                description="Routes based on sentiment analysis",
                mode=ConnectorMode.CONTENT_BASED,
            )
        }

        # Create and run the flow
        flow = Flow(
            tasks=tasks,
            map_paths=map_paths,
            dynamic_connectors=dynamic_connectors,
            log=True,
        )

        # Generate flow visualization
        try:
            graph_path = flow.generate_graph_img(
                name="sentiment_based_routing", save_path=self.OUTPUT_DIR
            )
            print(f"🎨 Flow visualization saved to: {graph_path}")
        except Exception as e:
            print(f"⚠️ Warning: Could not generate graph image: {e}")

        # Execute the flow
        results = asyncio.run(flow.start(max_workers=3))

        # Validate results
        self.assertIn("query", results, "Initial query task should be in results")

        # Check that exactly one of the handler tasks was executed
        handler_results = [r for r in results.keys() if r.endswith("_handler")]
        self.assertEqual(
            len(handler_results), 1, "Exactly one handler should be executed"
        )

        # Check which handler was chosen
        handler = handler_results[0]
        print(f"🔍 Selected sentiment handler: {handler}")
        print(f"🔍 Original prompt: {chosen_prompt}")

    def test_complex_workflow(self):
        """Test a more complex workflow with multiple dynamic connections."""
        print("\n--- 🔄 Testing Complex Workflow with Multiple Dynamic Connections ---")

        # Define all tasks
        initial_task = Task(
            TextTaskInput(
                "Create a detailed explanation of a complex technical topic in computer science."
            ),
            Agent(
                agent_type=AgentTypes.TEXT.value,
                provider="anthropic",
                mission="Generate initial content",
                model_params={
                    "key": self.api_keys["anthropic"],
                    "model": "claude-3-7-sonnet-20250219",
                },
            ),
            log=True,
        )

        analyzer_task = Task(
            TextTaskInput("Analyze if this content is too complex for a beginner:"),
            Agent(
                agent_type=AgentTypes.TEXT.value,
                provider="gemini",
                mission="Analyze content complexity",
                model_params={"key": self.api_keys["gemini"]},
            ),
            log=True,
        )

        simplifier_task = Task(
            TextTaskInput("Simplify this complex content for beginners:"),
            Agent(
                agent_type=AgentTypes.TEXT.value,
                provider="openai",
                mission="Simplify complex content",
                model_params={"key": self.api_keys["openai"], "model": "gpt-4o"},
            ),
            log=True,
        )

        expander_task = Task(
            TextTaskInput("Expand on this topic with more details and examples:"),
            Agent(
                agent_type=AgentTypes.TEXT.value,
                provider="openai",
                mission="Expand on the topic",
                model_params={"key": self.api_keys["openai"], "model": "gpt-4o"},
            ),
            log=True,
        )

        formatter_task = Task(
            TextTaskInput("Format this content nicely with markdown:"),
            Agent(
                agent_type=AgentTypes.TEXT.value,
                provider="mistral",
                mission="Format content",
                model_params={
                    "key": self.api_keys["mistral"],
                    "model": "mistral-medium",
                },
            ),
            log=True,
        )

        # Collect all tasks
        tasks = {
            "initial_query": initial_task,
            "complexity_analyzer": analyzer_task,
            "simplifier": simplifier_task,
            "expander": expander_task,
            "formatter": formatter_task,
        }

        # Define complexity router function
        def complexity_router(output, output_type):
            if output_type != "text":
                return "expand"  # Default

            # Check if the analysis suggests simplification
            output_lower = output.lower()
            if any(
                term in output_lower
                for term in [
                    "complex",
                    "difficult",
                    "advanced",
                    "simplify",
                    "too technical",
                ]
            ):
                return "simplify"
            else:
                return "expand"

        # Define all connections
        map_paths = {
            # Static connections
            "initial_query": ["complexity_analyzer"],
            "simplifier": ["formatter"],
            "expander": ["formatter"],
        }

        dynamic_connectors = {
            # Dynamic connections
            "complexity_analyzer": DynamicConnector(
                decision_fn=complexity_router,
                destinations={
                    "simplify": "simplifier",
                    "expand": "expander",
                },
                name="complexity_router",
                description="Routes based on content complexity",
                mode=ConnectorMode.CONTENT_BASED,
            )
        }

        # Create and run the flow
        flow = Flow(
            tasks=tasks,
            map_paths=map_paths,
            dynamic_connectors=dynamic_connectors,
            log=True,
        )

        # Generate flow visualization
        try:
            graph_path = flow.generate_graph_img(
                name="complex_workflow", save_path=self.OUTPUT_DIR
            )
            print(f"🎨 Flow visualization saved to: {graph_path}")
        except Exception as e:
            print(f"⚠️ Warning: Could not generate graph image: {e}")

        # Execute the flow
        results = asyncio.run(flow.start(max_workers=3))

        # Validate results
        self.assertIn(
            "initial_query", results, "Initial query task should be in results"
        )
        self.assertIn(
            "complexity_analyzer", results, "Analyzer task should be in results"
        )
        self.assertIn("formatter", results, "Formatter task should be in results")

        # Check which path was taken
        if "simplifier" in results:
            print("🔍 Content was routed to the simplifier")
            self.assertNotIn("expander", results, "Only one path should be taken")
        else:
            print("🔍 Content was routed to the expander")
            self.assertIn("expander", results, "Expander task should be in results")
            self.assertNotIn("simplifier", results, "Only one path should be taken")


if __name__ == "__main__":
    unittest.main()

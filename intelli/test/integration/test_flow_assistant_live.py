"""
Live tests for assistant steps in flows (agent_type 'assistant'). Each class runs only when its keys are set.

    OPENAI_API_KEY          OpenAI (gpt-5-mini): RAG with sources then a second step, history over runs, memory over
                            conversations, routing on a label, parallel steps joined in one; with QDRANT_URL also a
                            Qdrant knowledge store built from a config.
    GEMINI_API_KEY          Gemini: a tool step and a Google Search step.
    MISTRAL_API_KEY         Mistral: a knowledge store embedded with mistral-embed.
    ANTHROPIC_API_KEY       Anthropic, with OPENAI_API_KEY: a knowledge store embedded by OpenAI.
    AWS_LIVE_TESTS=1        Amazon Bedrock: knowledge embedded with Titan and answered by Nova Lite, using the AWS
                            credentials of the environment.

Run:
    python3 -m pytest intelli/test/integration/test_flow_assistant_live.py -q -s
"""
import asyncio
import os
import shutil
import tempfile
import unittest
import uuid

import requests
from dotenv import load_dotenv

from intelli.flow import Agent, Task, Flow, TextTaskInput, DynamicConnector, ConnectorMode
from intelli.flow.utils.dynamic_utils import text_content_router
from intelli.store import FileChatHistory

load_dotenv()

HANDBOOK = ('Acme refund policy: customers can return items within 30 days of delivery. Laptops are an exception '
            'and can be returned within 15 days. Express shipping takes 2 business days and costs 9 dollars.')
DOCUMENTS = [{'id': 'handbook', 'text': HANDBOOK, 'metadata': {'title': 'Acme handbook'}}]
COMMITS = ('feat: add dark mode to the settings page\n'
           'fix: app crashed when uploading a PDF larger than 10 MB\n'
           'feat: export invoices as CSV\n'
           'fix: receipts showed the wrong currency symbol for EUR')


def get_weather(city: str):
    """Get the current weather of a city."""
    return {'city': city, 'forecast': 'sunny', 'temperature_c': 21}


def run(flow, initial_input=None):
    return asyncio.run(flow.start(initial_input=initial_input))


def show(label, output):
    for name, item in output.items():
        print(f'\n[{label}] {name}: {item["output"]}')


class LiveFlowCase(unittest.TestCase):
    provider = 'openai'
    params = {}

    def setUp(self):
        self.folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.folder, ignore_errors=True)

    def agent(self, mission, model_params=None, options=None):
        return Agent('assistant', self.provider, mission, {**self.params, **(model_params or {})}, options)


@unittest.skipUnless(os.getenv('OPENAI_API_KEY'), 'Set OPENAI_API_KEY')
class TestOpenAIAssistantFlowLive(LiveFlowCase):
    def setUp(self):
        super().setUp()
        self.params = {'key': os.environ['OPENAI_API_KEY'], 'model': 'gpt-5-mini'}

    def test_rag_with_sources_then_a_second_step(self):
        knowledge = os.path.join(self.folder, 'knowledge.json')
        answer = Task(TextTaskInput("Answer the customer's question."), self.agent(
            'You are Acme support. Answer in one sentence.', {'show_sources': True},
            {'knowledge': {'type': 'memory', 'path': knowledge}, 'documents': DOCUMENTS}))
        translate = Task(TextTaskInput('Translate the text to French. Keep the sources list as it is.'),
                         self.agent('You translate text.'))
        flow = Flow(tasks={'answer': answer, 'translate': translate}, map_paths={'answer': ['translate']})

        output = run(flow, 'How many days do I have to return a laptop?')
        show('rag', output)

        self.assertEqual(flow.errors, {})
        self.assertIn('15', output['answer']['output'])
        self.assertIn('Sources:\n[1] Acme handbook', output['answer']['output'])
        self.assertIn('15', output['translate']['output'])
        self.assertTrue(os.path.exists(knowledge), 'the knowledge store is saved to its file')

    def test_history_over_runs(self):
        chat = Task(TextTaskInput('Reply to the traveler.'), self.agent(
            'You are a travel assistant. Answer in one sentence.', {'conversation_id': 'trip-1', 'user_id': 'u1'},
            {'history': {'type': 'file', 'dir': self.folder}}))
        flow = Flow(tasks={'chat': chat}, map_paths={})

        run(flow, 'I am planning three days in Lisbon and I like quiet places.')
        output = run(flow, 'Which city am I visiting, and for how many days?')
        show('history', output)

        self.assertIn('Lisbon', output['chat']['output'])
        self.assertEqual(len(FileChatHistory(dir=self.folder).get_messages('trip-1')), 4)

    def test_memory_over_conversations(self):
        chat = Task(TextTaskInput('Reply to the user.'), self.agent(
            'You are a cooking assistant. Answer in two sentences.', {'user_id': 'u7'},
            {'memory': {'type': 'memory', 'path': os.path.join(self.folder, 'memory.json')}}))
        flow = Flow(tasks={'chat': chat}, map_paths={})

        run(flow, 'Please remember that I am vegetarian.')
        output = run(flow, 'Suggest one dinner for me tonight.')
        show('memory', output)

        reply = chat.agent._get_handler().last_reply
        self.assertTrue(any('vegetarian' in memory['text'] for memory in reply['memories']),
                        'the second conversation recalls the first one')

    def test_routing_on_a_label(self):
        triage = Task(TextTaskInput('Rate the urgency of the support ticket.'), self.agent(
            'You triage support tickets. Write one sentence on the impact, then a last line that is exactly '
            '"ROUTE: high" for outages, data loss or security issues, otherwise exactly "ROUTE: normal".'))
        escalate = Task(TextTaskInput('Write a two sentence escalation note for the on-call engineer.'),
                        self.agent('You write escalation notes.'))
        reply = Task(TextTaskInput('Draft a short, polite reply to the customer.'),
                     self.agent('You write customer replies.'))
        connector = DynamicConnector(
            decision_fn=lambda output, kind: text_content_router(
                output, kind, {'normal': ['route: normal'], 'high': ['route: high']}),
            destinations={'normal': 'reply', 'high': 'escalate'}, mode=ConnectorMode.CONTENT_BASED)
        flow = Flow(tasks={'triage': triage, 'escalate': escalate, 'reply': reply}, map_paths={},
                    dynamic_connectors={'triage': connector})

        high = run(flow, 'Since 9am nobody in our company can log in. The whole team is blocked.')
        show('high', high)
        normal = run(flow, 'Could you add a dark mode to the settings page some day?')
        show('normal', normal)

        self.assertEqual(sorted(high), ['escalate', 'triage'])
        self.assertEqual(sorted(normal), ['reply', 'triage'])

    def test_parallel_steps_joined(self):
        tasks = {name: Task(TextTaskInput(f'List the {name} in the commit messages as short bullets.'),
                            self.agent(f'You write the {name} section of a release brief. Bullets only.'))
                 for name in ('features', 'fixes')}
        tasks['brief'] = Task(TextTaskInput('Write the release brief: a headline, then the features and the fixes.'),
                              self.agent('You edit release briefs for customers. Plain words, no jargon.'))
        flow = Flow(tasks=tasks, map_paths={'features': ['brief'], 'fixes': ['brief']})

        output = run(flow, COMMITS)
        show('brief', output)

        self.assertEqual(flow.errors, {})
        brief = output['brief']['output'].lower()
        self.assertIn('dark mode', brief)
        self.assertIn('csv', brief)
        self.assertIn('pdf', brief)

    @unittest.skipUnless(os.getenv('QDRANT_URL'), 'Set QDRANT_URL')
    def test_qdrant_knowledge_from_a_config(self):
        url = os.environ['QDRANT_URL']
        collection = f'intelli_flow_{uuid.uuid4().hex[:8]}'
        self.addCleanup(requests.delete, f'{url}/collections/{collection}', timeout=30)
        answer = Task(TextTaskInput("Answer the customer's question."), self.agent(
            'You are Acme support. Answer in one sentence.', {'top_k': 2},
            {'knowledge': {'type': 'qdrant', 'url': url, 'collection': collection}, 'documents': DOCUMENTS}))
        flow = Flow(tasks={'answer': answer}, map_paths={})

        output = run(flow, 'How long does express shipping take?')
        show('qdrant', output)

        self.assertEqual(flow.errors, {})
        self.assertIn('2', output['answer']['output'])
        self.assertEqual(answer.agent._get_handler().last_reply['references'][0]['id'], 'handbook#0')


@unittest.skipUnless(os.getenv('GEMINI_API_KEY'), 'Set GEMINI_API_KEY')
class TestGeminiAssistantFlowLive(LiveFlowCase):
    provider = 'gemini'

    def setUp(self):
        super().setUp()
        self.params = {'key': os.environ['GEMINI_API_KEY'], 'model': 'gemini-2.5-flash'}

    def test_tool_step(self):
        agent = self.agent('You are a travel assistant. Answer in one sentence.', options={'tools': [get_weather]})
        flow = Flow(tasks={'weather': Task(TextTaskInput('Answer the question.'), agent)}, map_paths={})

        output = run(flow, 'What is the weather in Rome right now?')
        show('tool', output)

        self.assertEqual(agent._get_handler().last_reply['tool_steps'][0]['name'], 'get_weather')
        self.assertIn('21', output['weather']['output'])

    def test_google_search_step(self):
        agent = self.agent('You answer with current facts. Answer in one sentence.',
                           {'google_search': True, 'show_sources': True})
        flow = Flow(tasks={'facts': Task(TextTaskInput('Answer the question.'), agent)}, map_paths={})

        output = run(flow, 'What is the latest stable version of Python?')
        show('search', output)

        self.assertEqual(flow.errors, {})
        self.assertIn('Sources:', output['facts']['output'])


@unittest.skipUnless(os.getenv('MISTRAL_API_KEY'), 'Set MISTRAL_API_KEY')
class TestMistralAssistantFlowLive(LiveFlowCase):
    provider = 'mistral'

    def setUp(self):
        super().setUp()
        self.params = {'key': os.environ['MISTRAL_API_KEY'], 'model': 'mistral-small-latest'}

    def test_knowledge_embedded_by_mistral(self):
        answer = Task(TextTaskInput("Answer the customer's question."), self.agent(
            'You are Acme support. Answer in one sentence.', options={'knowledge': {'type': 'memory'},
                                                                      'documents': DOCUMENTS}))
        flow = Flow(tasks={'answer': answer}, map_paths={})

        output = run(flow, 'How much does express shipping cost?')
        show('mistral', output)

        self.assertEqual(flow.errors, {})
        self.assertIn('9', output['answer']['output'])


@unittest.skipUnless(os.getenv('ANTHROPIC_API_KEY') and os.getenv('OPENAI_API_KEY'),
                     'Set ANTHROPIC_API_KEY and OPENAI_API_KEY')
class TestAnthropicAssistantFlowLive(LiveFlowCase):
    provider = 'anthropic'

    def setUp(self):
        super().setUp()
        self.params = {'key': os.environ['ANTHROPIC_API_KEY'], 'model': 'claude-haiku-4-5', 'max_tokens': 300}

    def test_knowledge_embedded_by_openai(self):
        embedder = {'provider': 'openai', 'api_key': os.environ['OPENAI_API_KEY']}
        answer = Task(TextTaskInput("Answer the customer's question."), self.agent(
            'You are Acme support. Answer in one sentence.',
            options={'knowledge': {'type': 'memory', 'embedder': embedder}, 'documents': DOCUMENTS}))
        flow = Flow(tasks={'answer': answer}, map_paths={})

        output = run(flow, 'How many days do I have to return a laptop?')
        show('anthropic', output)

        self.assertEqual(flow.errors, {})
        self.assertIn('15', output['answer']['output'])


@unittest.skipUnless(os.getenv('AWS_LIVE_TESTS') == '1', 'Set AWS_LIVE_TESTS=1 with AWS credentials')
class TestBedrockAssistantFlowLive(LiveFlowCase):
    provider = 'aws'

    def setUp(self):
        super().setUp()
        self.params = {'max_tokens': 300}

    def test_knowledge_embedded_with_titan(self):
        options = {'region': os.getenv('AWS_REGION', 'us-east-1'), 'knowledge': {'type': 'memory'},
                   'documents': DOCUMENTS}
        answer = Task(TextTaskInput("Answer the customer's question."), self.agent(
            'You are Acme support. Answer in one sentence.', options=options))
        flow = Flow(tasks={'answer': answer}, map_paths={})

        output = run(flow, 'How many days do I have to return a laptop?')
        show('bedrock', output)

        self.assertEqual(flow.errors, {})
        self.assertIn('15', output['answer']['output'])


if __name__ == '__main__':
    unittest.main()

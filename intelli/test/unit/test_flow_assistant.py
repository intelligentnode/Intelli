"""
Offline tests for assistant steps in flows (agent_type 'assistant') and the store factory. The Gemini HTTP session
is a fake, so nothing here calls a model.
"""
import asyncio
import json
import tempfile
import unittest
from unittest.mock import patch

from intelli.flow import Agent, Task, Flow, TextTaskInput, TextAgentInput, DynamicConnector, ConnectorMode
from intelli.flow.agents.handlers import AssistantAgentHandler
from intelli.flow.utils.dynamic_utils import text_content_router
from intelli.store import (MemoryVectorStore, MemoryChatHistory, FileChatHistory, QdrantVectorStore, StoreError,
                           create_vector_store, create_chat_history)

# A deterministic keyword embedder: one dimension per word of a small vocabulary.
VOCABULARY = ['refund', 'return', 'days', 'shipping', 'express', 'laptop', 'name', 'sam', 'policy', 'weather']


def fake_embed(texts):
    return [[1 if word in str(text).lower() else 0 for word in VOCABULARY] + [0.01] for text in texts]


def gemini_reply(text):
    return {'candidates': [{'content': {'role': 'model', 'parts': [{'text': text}]}}],
            'usageMetadata': {'promptTokenCount': 10, 'candidatesTokenCount': 2, 'totalTokenCount': 12}}


class FakeResponse:
    def __init__(self, data):
        self._data = data
        self.status_code = 200
        self.text = json.dumps(data)

    def json(self):
        return json.loads(json.dumps(self._data))

    def raise_for_status(self):
        pass


class FakeGoogleSession:
    """Answers GoogleAIWrapper requests from a script and records the bodies."""

    def __init__(self):
        self.replies = []
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append({'url': url, 'body': kwargs.get('json')})
        return FakeResponse(self.replies.pop(0))

    get = delete = patch = post


def assistant_agent(mission='You are Acme support.', model_params=None, options=None):
    return Agent('assistant', 'gemini', mission, {'key': 'k', **(model_params or {})}, options)


def run(flow, initial_input=None):
    return asyncio.run(flow.start(initial_input=initial_input))


def system_text(body):
    return ''.join(part.get('text', '') for part in body['systemInstruction']['parts'])


class TestAssistantSteps(unittest.TestCase):
    def setUp(self):
        self.session = FakeGoogleSession()
        patcher = patch('intelli.wrappers.googleai_wrapper.requests.Session', return_value=self.session)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_input_is_the_message_and_desc_the_instruction(self):
        self.session.replies = [gemini_reply('You have 15 days [1].'), gemini_reply('15 days.')]
        documents = [{'id': 'refunds', 'text': 'Laptops can be returned within 15 days for a refund.',
                      'metadata': {'title': 'Refund policy'}},
                     {'id': 'shipping', 'text': 'Express shipping takes 2 days.'}]
        answer = Task(TextTaskInput("Answer the customer's question."), assistant_agent(
            model_params={'show_sources': True},
            options={'knowledge': {'type': 'memory', 'embedder': fake_embed}, 'documents': documents}))
        shorten = Task(TextTaskInput('Shorten the answer to a few words.'), assistant_agent(mission='You edit text.'))
        flow = Flow(tasks={'answer': answer, 'shorten': shorten}, map_paths={'answer': ['shorten']})

        output = run(flow, 'How many days do I have to return a laptop?')

        self.assertEqual(flow.errors, {})
        self.assertEqual(output['answer']['output'], 'You have 15 days [1].\n\nSources:\n[1] Refund policy')
        self.assertEqual(output['shorten']['output'], '15 days.')
        first, second = (call['body'] for call in self.session.calls)
        self.assertTrue(system_text(first).startswith(
            "You are Acme support.\n\nCurrent task: Answer the customer's question."))
        self.assertIn('[1] Refund policy\nLaptops can be returned within 15 days', system_text(first))
        self.assertEqual(first['contents'], [
            {'role': 'user', 'parts': [{'text': 'How many days do I have to return a laptop?'}]}])
        self.assertEqual(system_text(second), 'You edit text.\n\nCurrent task: Shorten the answer to a few words.')
        self.assertEqual(second['contents'][-1]['parts'][0]['text'], output['answer']['output'])

    def test_history_continues_a_conversation_across_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            self.session.replies = [gemini_reply('Hi Sam.'), gemini_reply('Your name is Sam.')]
            chat = Task(TextTaskInput('Reply to the user.'), assistant_agent(
                model_params={'conversation_id': 'c1', 'user_id': 'u1'},
                options={'history': {'type': 'file', 'dir': folder}}))
            flow = Flow(tasks={'chat': chat}, map_paths={})

            run(flow, 'My name is Sam.')
            output = run(flow, 'What is my name?')

            self.assertEqual(output['chat']['output'], 'Your name is Sam.')
            contents = self.session.calls[1]['body']['contents']
            self.assertEqual([content['role'] for content in contents], ['user', 'model', 'user'])
            self.assertEqual(contents[0]['parts'][0]['text'], 'My name is Sam.')
            history = FileChatHistory(dir=folder)
            self.assertEqual([m['content'] for m in history.get_messages('c1')],
                             ['My name is Sam.', 'Hi Sam.', 'What is my name?', 'Your name is Sam.'])
            self.assertEqual(history.get_conversation('c1')['user_id'], 'u1')

    def test_runs_without_a_conversation_id_start_fresh(self):
        self.session.replies = [gemini_reply('One.'), gemini_reply('Two.')]
        agent = assistant_agent()
        flow = Flow(tasks={'step': Task(TextTaskInput('Answer.'), agent)}, map_paths={})

        run(flow, 'first')
        run(flow, 'second')

        self.assertEqual(len(self.session.calls[1]['body']['contents']), 1)
        assistant = next(iter(agent._get_handler()._assistants.values()))
        self.assertEqual(assistant.history.list_conversations(), [], 'one-off conversations are not kept')

    def test_memory_recalls_a_user_across_conversations(self):
        self.session.replies = [gemini_reply('Noted, Sam.'), gemini_reply('You are Sam.')]
        memory = MemoryVectorStore(embedder=fake_embed)
        agent = assistant_agent(model_params={'user_id': 'u1'}, options={'memory': memory})
        flow = Flow(tasks={'chat': Task(TextTaskInput('Reply.'), agent)}, map_paths={})

        run(flow, 'My name is Sam.')
        run(flow, 'What is my name?')

        system = system_text(self.session.calls[1]['body'])
        self.assertIn('Notes from earlier conversations', system)
        self.assertIn('User: My name is Sam. Assistant: Noted, Sam.', system)
        self.assertEqual(memory.count(), 2)

    def test_tools_run_inside_the_step(self):
        def get_weather(city: str) -> str:
            """Current weather for a city."""
            return f'Sunny in {city}'

        call = {'candidates': [{'content': {'role': 'model', 'parts': [
            {'functionCall': {'name': 'get_weather', 'args': {'city': 'Rome'}}}]}}]}
        self.session.replies = [call, gemini_reply('It is sunny in Rome.')]
        agent = assistant_agent(options={'tools': [get_weather]})
        flow = Flow(tasks={'weather': Task(TextTaskInput('Answer.'), agent)}, map_paths={})

        output = run(flow, 'Weather in Rome?')

        self.assertEqual(output['weather']['output'], 'It is sunny in Rome.')
        self.assertEqual(agent._get_handler().last_reply['tool_steps'][0]['result'], 'Sunny in Rome')
        declarations = self.session.calls[0]['body']['tools'][0]['functionDeclarations']
        self.assertEqual(declarations[0]['name'], 'get_weather')

    def test_content_routing_on_an_assistant_step(self):
        self.session.replies = [gemini_reply('Outage for all users.\nROUTE: high'), gemini_reply('Paged on-call.')]
        connector = DynamicConnector(
            decision_fn=lambda output, kind: text_content_router(
                output, kind, {'normal': ['route: normal'], 'high': ['route: high']}),
            destinations={'normal': 'reply', 'high': 'escalate'}, mode=ConnectorMode.CONTENT_BASED)
        flow = Flow(tasks={'triage': Task(TextTaskInput('Rate the urgency.'), assistant_agent()),
                           'escalate': Task(TextTaskInput('Write an escalation note.'), assistant_agent()),
                           'reply': Task(TextTaskInput('Draft a reply.'), assistant_agent())},
                    map_paths={}, dynamic_connectors={'triage': connector})

        output = run(flow, 'Nobody can log in!')

        self.assertEqual(sorted(output), ['escalate', 'triage'])
        self.assertEqual(output['escalate']['output'], 'Paged on-call.')

    def test_documents_are_added_once(self):
        store = MemoryVectorStore(embedder=fake_embed)
        self.session.replies = [gemini_reply('a'), gemini_reply('b'), gemini_reply('c')]
        first = assistant_agent(options={'knowledge': store, 'documents': ['Refunds take 15 days.']})
        flow = Flow(tasks={'answer': Task(TextTaskInput('Answer.'), first)}, map_paths={})
        run(flow, 'refund?')
        run(flow, 'refund again?')
        self.assertEqual(store.count(), 1)

        # a step over a store that already holds records does not add its documents again
        second = assistant_agent(options={'knowledge': store, 'documents': ['Other text.']})
        run(Flow(tasks={'answer': Task(TextTaskInput('Answer.'), second)}, map_paths={}), 'refund?')
        self.assertEqual(store.count(), 1)

    def test_a_failed_document_load_is_tried_again(self):
        calls = []

        def flaky_embed(texts):
            calls.append(len(texts))
            if len(calls) == 1:
                raise ConnectionError('embedding service unavailable')
            return fake_embed(texts)

        store = MemoryVectorStore(embedder=flaky_embed)
        agent = assistant_agent(options={'knowledge': store, 'documents': ['Refunds take 15 days.']})
        flow = Flow(tasks={'answer': Task(TextTaskInput('Answer.'), agent)}, map_paths={})
        run(flow, 'refund?')
        self.assertIn('answer', flow.errors)

        self.session.replies = [gemini_reply('15 days.')]
        output = run(flow, 'refund?')
        self.assertEqual(flow.errors, {})
        self.assertEqual(output['answer']['output'], '15 days.')
        self.assertEqual(store.count(), 1)

    def test_agent_execute_without_a_task(self):
        self.session.replies = [gemini_reply('Hello!')]
        agent = assistant_agent()

        self.assertEqual(agent.execute(TextAgentInput('Say hello')), 'Hello!')
        body = self.session.calls[0]['body']
        self.assertEqual(body['contents'][0]['parts'][0]['text'], 'Say hello')
        self.assertEqual(system_text(body), 'You are Acme support.')

    def test_model_params_reach_the_request(self):
        self.session.replies = [gemini_reply('ok'), gemini_reply('ok')]
        agent = assistant_agent(model_params={'model': 'gemini-x', 'temperature': 0.1, 'max_tokens': 50})
        task = Task(TextTaskInput('Answer.'), agent, model_params={'max_tokens': 80})
        run(Flow(tasks={'answer': task}, map_paths={}), 'hi')
        self.assertIn('/models/gemini-x:generateContent', self.session.calls[0]['url'])
        config = self.session.calls[0]['body']['generationConfig']
        self.assertEqual((config['temperature'], config['maxOutputTokens']), (0.1, 80), 'task params win')


class TestAssistantHandlerSettings(unittest.TestCase):
    @staticmethod
    def handler(provider, options=None):
        return AssistantAgentHandler(provider, 'mission', {}, options or {})

    def test_default_embedders(self):
        self.assertEqual(self.handler('openai')._default_embedder({'key': 'k'}), {'provider': 'openai', 'api_key': 'k'})
        vertex = self.handler('gemini', {'vertex': True, 'project_id': 'p', 'location': 'us-central1', 'timeout': 5})
        self.assertEqual(vertex._default_embedder({}), {'provider': 'vertex', 'api_key': None,
                                                       'options': {'project_id': 'p', 'location': 'us-central1'}})
        aws = self.handler('aws', {'region': 'us-east-1', 'knowledge': {'type': 'memory'}})
        self.assertEqual(aws._default_embedder({}), {'provider': 'aws', 'api_key': None,
                                                    'options': {'region': 'us-east-1'}})
        ollama = self.handler('ollama', {'baseUrl': 'http://gpu:11434/'})
        self.assertEqual(ollama._default_embedder({}), {'provider': 'ollama',
                                                       'options': {'base_url': 'http://gpu:11434/v1'}})
        self.assertIsNone(self.handler('anthropic')._default_embedder({'key': 'k'}))
        self.assertIsNone(self.handler('openai')._default_embedder({}))

    def test_a_store_without_an_embedding_model_is_an_error(self):
        agent = Agent('assistant', 'anthropic', 'mission', {'key': 'k'}, {'knowledge': {'type': 'memory'}})
        with self.assertRaises(ValueError) as error:
            agent.execute(TextAgentInput('hi'))
        self.assertIn("'embedder'", str(error.exception))

    def test_tool_names_are_not_functions(self):
        agent = Agent('assistant', 'openai', 'mission', {'key': 'k'}, {'tools': ['get_weather']})
        with self.assertRaises(ValueError) as error:
            agent.execute(TextAgentInput('hi'))
        self.assertIn('get_weather', str(error.exception))

    def test_documents_need_a_knowledge_store(self):
        agent = Agent('assistant', 'openai', 'mission', {'key': 'k'}, {'documents': ['text']})
        with self.assertRaises(ValueError):
            agent.execute(TextAgentInput('hi'))


class TestStoreFactory(unittest.TestCase):
    def test_vector_store_configs(self):
        store = MemoryVectorStore(embedder=fake_embed)
        self.assertIs(create_vector_store(store), store)
        self.assertIsNone(create_vector_store(None))

        config = {'type': 'memory'}
        memory = create_vector_store(config, default_embedder=fake_embed)
        self.assertEqual(config, {'type': 'memory'}, 'the config is not changed')
        memory.add_documents([{'id': 'a', 'text': 'refund policy'}, {'id': 'b', 'text': 'express shipping'}])
        self.assertEqual(memory.search('refund', 1)[0]['id'], 'a')

        qdrant = create_vector_store({'type': 'Qdrant', 'url': 'http://localhost:6333', 'collection': 'docs'},
                                     default_embedder=fake_embed)
        self.assertIsInstance(qdrant, QdrantVectorStore)
        self.assertEqual(qdrant.collection, 'docs')
        own = create_vector_store({'type': 'memory', 'embedder': lambda texts: [[1.0] for _ in texts]},
                                  default_embedder=fake_embed)
        self.assertEqual(own.embed(['x']), [[1.0]], "the config's embedder wins")

        for config in ({'type': 'faiss'}, 'qdrant', {'type': 'pgvector'}, {'type': 'mongodb_atlas', 'collection': 'x'}):
            with self.assertRaises(StoreError):
                create_vector_store(config)

    def test_chat_history_configs(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertIsInstance(create_chat_history({'type': 'file', 'dir': folder}), FileChatHistory)
        self.assertIsInstance(create_chat_history({}), MemoryChatHistory)
        history = MemoryChatHistory()
        self.assertIs(create_chat_history(history), history)
        self.assertIsNone(create_chat_history(None))
        with self.assertRaises(StoreError):
            create_chat_history({'type': 'redis'})


if __name__ == '__main__':
    unittest.main()

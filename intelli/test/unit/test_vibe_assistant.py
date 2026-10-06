"""
Offline tests for VibeFlow (VibeAgent) with assistant steps: the planner prompt, the registries of tools, stores and
guards, spec validation, the planner's self-correction and a built flow running on a fake model.
"""
import asyncio
import copy
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from intelli.flow.vibe import VibeFlow
from intelli.store import MemoryVectorStore, FileChatHistory

VOCABULARY = ['refund', 'return', 'days', 'shipping', 'laptop']


def fake_embed(texts):
    return [[1 if word in str(text).lower() else 0 for word in VOCABULARY] + [0.01] for text in texts]


def get_weather(city: str) -> str:
    """Current weather for a city."""
    return f'Sunny in {city}'


def read_only(action):
    """Blocks typing and clicks on buttons that change data."""
    return 'pay' not in str(action.get('text', '')).lower()


def assistant(mission='You are Acme support.', model_params=None, options=None, provider='openai'):
    return {'agent_type': 'assistant', 'provider': provider, 'mission': mission,
            'model_params': {'key': '${ENV:OPENAI_API_KEY}', 'model': 'gpt-5.5', **(model_params or {})},
            'options': options or {}}


def spec(*tasks, **extra):
    return {'version': '1', 'tasks': [{'name': name, 'desc': f'{name} step', 'agent': agent} for name, agent in tasks],
            **extra}


class TestVibePlannerPrompt(unittest.TestCase):
    def test_prompt_prefers_the_assistant_and_lists_the_registries(self):
        folder = tempfile.mkdtemp()
        vibe = VibeFlow(planner_api_key='k', tools={'get_weather': get_weather},
                        stores={'handbook': MemoryVectorStore(embedder=fake_embed),
                                'chats': FileChatHistory(dir=folder), 'faq': {'type': 'qdrant'}},
                        guards={'read_only': read_only}, processors={'to_upper': str.upper})
        prompt = vibe._build_system_prompt()

        self.assertIn('"assistant": every language step', prompt)
        self.assertIn('"agent_type": "assistant"', prompt)
        self.assertNotIn('"text": every', prompt)
        self.assertNotIn('Chatbot', prompt)
        self.assertNotIn('intellicloud', prompt.lower())
        self.assertIn('get_weather (Current weather for a city.)', prompt)
        self.assertIn('handbook (vector store (MemoryVectorStore)); chats (chat history (FileChatHistory)); '
                      'faq (vector store (qdrant))', prompt)
        self.assertIn('read_only (Blocks typing and clicks on buttons that change data.)', prompt)
        self.assertLess(len(prompt), 20000, 'the prompt is a reference, not a source dump')

    def test_prompt_without_registries_and_with_preferences(self):
        prompt = VibeFlow(planner_api_key='k', text_model='gemini gemini-2.5-flash')._build_system_prompt()
        self.assertIn('- tools (assistant options.tools): none, so no tools', prompt)
        self.assertIn('none, so use store configs', prompt)
        self.assertIn('- Language steps: gemini gemini-2.5-flash.', prompt)


class TestVibeAssistantSpecs(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp()
        self.handbook = MemoryVectorStore(embedder=fake_embed)
        self.vibe = VibeFlow(planner_api_key='k', tools={'get_weather': get_weather},
                             stores={'handbook': self.handbook, 'chats': FileChatHistory(dir=self.folder)},
                             guards={'read_only': read_only})

    def test_registered_names_become_objects(self):
        computer = {'agent_type': 'computer', 'provider': 'anthropic', 'mission': 'check journeys',
                    'model_params': {'key': '${ENV:ANTHROPIC_API_KEY}', 'start_url': 'https://staging.example.com',
                                     'on_action': 'read_only'}}
        flow_spec = spec(
            ('answer', assistant(model_params={'conversation_id': 'default'},
                                 options={'tools': ['get_weather'], 'knowledge': 'handbook', 'history': 'chats',
                                          'memory': 'memory'})),
            ('check', computer))
        original = copy.deepcopy(flow_spec)

        flow = self.vibe.build_from_spec(flow_spec)

        options = flow.tasks['answer'].agent.options
        self.assertEqual(options['tools'], [get_weather])
        self.assertIs(options['knowledge'], self.handbook)
        self.assertIs(options['history'], self.vibe.stores['chats'])
        self.assertEqual(options['memory'], {'type': 'memory'})
        self.assertIs(flow.tasks['check'].agent.model_params['on_action'], read_only)
        self.assertEqual(flow_spec, original, 'the spec keeps the names, so it can be saved and loaded again')

    def test_invalid_specs(self):
        cases = {
            'not registered': spec(('a', assistant(options={'tools': ['send_email']}))),
            'not a registered store': spec(('a', assistant(options={'knowledge': 'catalog'}))),
            "unknown type 'faiss'": spec(('a', assistant(options={'knowledge': {'type': 'faiss'}}))),
            'is a chat history': spec(('a', assistant(options={'knowledge': 'chats'}))),
            'is a vector store': spec(('a', assistant(options={'history': 'handbook'}))),
            'need options.knowledge': spec(('a', assistant(options={'files': ['faq.md']}))),
            'only with model_params "conversation_id"': spec(('a', assistant(options={
                'history': {'type': 'file', 'dir': './chats'}}))),
            'only with model_params "user_id"': spec(('a', assistant(options={'memory': {'type': 'memory'}}))),
            "google_search needs provider 'gemini'": spec(('a', assistant(model_params={'google_search': True}))),
            'has no embedding model': spec(('a', assistant(provider='anthropic',
                                                           options={'knowledge': {'type': 'memory'}}))),
            'need model_params.start_url': spec(('a', {'agent_type': 'computer', 'provider': 'anthropic',
                                                       'model_params': {}})),
            'not a registered guard': spec(('a', {'agent_type': 'computer', 'provider': 'openai', 'model_params': {
                'start_url': 'https://example.com', 'on_action': 'no_payments'}})),
            "need provider 'anthropic' or 'openai'": spec(('a', {'agent_type': 'computer', 'provider': 'gemini',
                                                                 'model_params': {'start_url': 'https://x.io'}})),
            "unknown agent_type 'writer'": spec(('a', {'agent_type': 'writer', 'provider': 'openai'})),
            'Unsupported assistant agent provider': spec(('a', assistant(provider='cohere'))),
            'requires agent.options.baseUrl': spec(('a', assistant(provider='vllm'))),
            'runs its tools itself': spec(('a', assistant()), ('b', assistant()), ('c', assistant()),
                                          dynamic_connectors=[{'source': 'a', 'kind': 'tool', 'destinations': {
                                              'tool_called': 'b', 'no_tool': 'c'}}]),
            'runs only one of them': spec(('a', assistant()), ('b', assistant()), ('c', assistant()),
                                          ('d', assistant()), map_paths={'b': ['d'], 'c': ['d']},
                                          dynamic_connectors=[{'source': 'a', 'kind': 'content', 'keywords': {
                                              'x': ['route: x'], 'y': ['route: y']},
                                              'destinations': {'x': 'b', 'y': 'c'}}]),
            'can never be chosen': spec(('a', assistant()), ('b', assistant()), dynamic_connectors=[{
                'source': 'a', 'kind': 'content', 'keywords': {'high': ['route: high']},
                'destinations': {'urgent': 'b'}}]),
            'has a cycle: a -> b -> a': spec(('a', assistant()), ('b', assistant()), map_paths={'a': ['b'],
                                                                                              'b': ['a']}),
            'needs an agent object': {'version': '1', 'tasks': [{'name': 'a', 'desc': 'x'}]},
            'needs agent.provider': spec(('a', {'agent_type': 'assistant'})),
            'name (a string)': {'version': '1', 'tasks': [{'name': {'en': 'a'}, 'agent': assistant()}]},
        }
        for message, flow_spec in cases.items():
            with self.subTest(message):
                with self.assertRaises(ValueError) as error:
                    self.vibe._validate_spec(flow_spec, planning=True)
                self.assertIn(message, str(error.exception))

    def test_valid_assistant_variants(self):
        self.vibe._validate_spec(spec(
            ('mistral', assistant(provider='mistral', options={'knowledge': {'type': 'memory'}})),
            ('ollama', assistant(provider='ollama')),
            ('anthropic', assistant(provider='anthropic', options={'knowledge': {
                'type': 'memory', 'embedder': {'provider': 'openai', 'api_key': '${ENV:OPENAI_API_KEY}'}}})),
            ('search', assistant(provider='gemini', model_params={'google_search': True, 'show_sources': True})),
            ('aws', assistant(provider='aws', model_params={'user_id': 'default'},
                              options={'region': 'us-east-1', 'memory': {'type': 'memory'}}))),
            planning=True)

    def test_a_route_without_a_destination_stops_the_flow(self):
        flow_spec = spec(('review', assistant()), ('alert', assistant()), dynamic_connectors=[{
            'source': 'review', 'kind': 'content', 'keywords': {'normal': ['route: normal'], 'alert': ['route: alert']},
            'destinations': {'alert': 'alert'}}])
        self.vibe._validate_spec(flow_spec, planning=True)
        connector = self.vibe.build_from_spec(flow_spec).dynamic_connectors['review']
        self.assertIsNone(connector.get_next_task('Looks fine.\nROUTE: normal', 'text'))
        self.assertEqual(connector.get_next_task('SQL injection.\nROUTE: alert', 'text'), 'alert')

    def test_processors_are_checked_when_planning(self):
        flow_spec = spec(('a', assistant()))
        flow_spec['tasks'][0]['post_process'] = 'save_to_disk'
        with self.assertRaises(ValueError):
            self.vibe._validate_spec(flow_spec, planning=True)
        self.vibe.build_from_spec(flow_spec)  # a saved spec still loads; the name is ignored as before

    def test_saved_bundle_keeps_names_and_redacts_secrets(self):
        flow_spec = spec(('a', assistant(model_params={'key': 'sk-real'}, options={
            'tools': ['get_weather'], 'knowledge': {'type': 'mongodb_atlas', 'uri': 'mongodb+srv://bob:pw@c0.net/',
                                                    'db': 'app', 'collection': 'docs'}})))
        self.vibe.save_bundle(self.folder, flow_spec, render_graph=False)
        with open(os.path.join(self.folder, 'flow_spec.json')) as file:
            saved = json.load(file)
        agent = saved['tasks'][0]['agent']
        self.assertEqual(agent['model_params']['key'], '<REDACTED>')
        self.assertEqual(agent['options']['tools'], ['get_weather'])
        self.assertEqual(agent['options']['knowledge']['uri'], 'mongodb+srv://<REDACTED>@c0.net/')


class FakePlannerBot:
    """Stands in for the planner Chatbot: replies from a script and keeps every request."""
    replies = []
    inputs = []

    def __init__(self, *args, **kwargs):
        pass

    def chat(self, chat_input):
        FakePlannerBot.inputs.append([dict(message.__dict__) for message in chat_input.messages])
        return [FakePlannerBot.replies.pop(0)]


class TestVibePlannerCorrection(unittest.TestCase):
    def test_the_planner_fixes_an_invalid_spec(self):
        bad = spec(('answer', assistant(options={'tools': ['send_email']})))
        good = spec(('answer', assistant(options={'tools': ['get_weather']})))
        FakePlannerBot.replies = [json.dumps(bad), 'Here it is: ' + json.dumps(good)]
        FakePlannerBot.inputs = []
        vibe = VibeFlow(planner_api_key='k', tools={'get_weather': get_weather})

        with patch('intelli.flow.vibe.Chatbot', FakePlannerBot):
            flow = asyncio.run(vibe.build('Answer weather questions.'))

        self.assertEqual(flow.tasks['answer'].agent.options['tools'], [get_weather])
        retry = FakePlannerBot.inputs[1][-1]['content']
        self.assertIn("uses tools ['send_email'], which are not registered", retry)
        self.assertEqual(vibe.last_spec, good)

    def test_a_malformed_plan_goes_back_to_the_planner(self):
        malformed = {'version': '1', 'tasks': [{'name': 'a', 'agent': assistant()}], 'map_paths': {'a': [{'b': 1}]}}
        FakePlannerBot.replies = [json.dumps(malformed), json.dumps(spec(('a', assistant())))]
        FakePlannerBot.inputs = []
        with patch('intelli.flow.vibe.Chatbot', FakePlannerBot):
            asyncio.run(VibeFlow(planner_api_key='k').build('Anything.'))
        self.assertIn('unknown destination task', FakePlannerBot.inputs[1][-1]['content'])

    def test_the_planner_gives_up_after_three_replies(self):
        FakePlannerBot.replies = ['no json', '{"tasks": []}', 'still no json']
        with patch('intelli.flow.vibe.Chatbot', FakePlannerBot):
            with self.assertRaises(ValueError):
                asyncio.run(VibeFlow(planner_api_key='k').build('Anything.'))
        self.assertEqual(FakePlannerBot.replies, [])


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
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append(kwargs.get('json'))
        return FakeResponse({'candidates': [{'content': {'role': 'model', 'parts': [
            {'text': self.replies.pop(0)}]}}]})

    get = delete = patch = post


class TestVibeBuiltFlowRuns(unittest.TestCase):
    def test_an_assistant_flow_from_a_spec_answers_from_a_registered_store(self):
        handbook = MemoryVectorStore(embedder=fake_embed)
        vibe = VibeFlow(planner_api_key='k', stores={'handbook': handbook})
        documents = [{'id': 'refunds', 'text': 'Laptops can be returned within 15 days.'}]
        gemini = {'agent_type': 'assistant', 'provider': 'gemini', 'mission': 'You are Acme support.',
                  'model_params': {'key': 'k'}, 'options': {'knowledge': 'handbook', 'documents': documents}}
        writer = dict(gemini, mission='You write short replies.', options={})
        flow = vibe.build_from_spec(spec(('answer', gemini), ('reply', writer), map_paths={'answer': ['reply']}))
        session = FakeGoogleSession(['15 days [1].', 'You can return it within 15 days.'])

        with patch('intelli.wrappers.googleai_wrapper.requests.Session', return_value=session):
            output = asyncio.run(flow.start(initial_input='How long can I return a laptop?'))

        self.assertEqual(flow.errors, {})
        self.assertEqual(output['reply']['output'], 'You can return it within 15 days.')
        self.assertEqual(handbook.count(), 1)
        self.assertIn('Laptops can be returned within 15 days.',
                      session.calls[0]['systemInstruction']['parts'][0]['text'])


if __name__ == '__main__':
    unittest.main()

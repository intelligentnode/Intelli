"""
Offline tests for the Assistant, the chat histories, the embedder and the text splitter. Nothing here calls a
model: the provider HTTP sessions and wrapper calls are fakes.
"""
import base64
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from intelli.function.assistant import Assistant, _read_attachment, _tool_registry
from intelli.store import MemoryVectorStore, MemoryChatHistory, FileChatHistory, Embedder
from intelli.utils.text_splitter import TextSplitter

# A deterministic keyword embedder: one dimension per word of a small vocabulary.
VOCABULARY = ['refund', 'return', 'days', 'shipping', 'express', 'laptop', 'name', 'sam', 'policy', 'weather']


def fake_embed(texts):
    vectors = []
    for text in texts:
        words = str(text).lower()
        vectors.append([1 if word in words else 0 for word in VOCABULARY] + [0.01])
    return vectors


def gemini_reply(text, **extra):
    return {'candidates': [{'content': {'role': 'model', 'parts': [{'text': text}]}}],
            'usageMetadata': {'promptTokenCount': 10, 'candidatesTokenCount': 2, 'totalTokenCount': 12}, **extra}


class FakeResponse:
    def __init__(self, data, lines=None):
        self._data = data
        self.status_code = 200
        self.text = json.dumps(data) if data is not None else ''
        self._lines = lines or []

    def json(self):
        return json.loads(json.dumps(self._data))

    def raise_for_status(self):
        pass

    def iter_lines(self):
        return iter(self._lines)

    def close(self):
        pass


class FakeGoogleSession:
    """Answers GoogleAIWrapper requests from a script and records the bodies."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def _respond(self, url, kwargs):
        self.calls.append({'url': url, 'body': kwargs.get('json')})
        reply = self.replies.pop(0)
        if kwargs.get('stream'):
            return FakeResponse(None, lines=[f'data: {json.dumps(reply)}'.encode(), b''])
        return FakeResponse(reply)

    def post(self, url, **kwargs):
        return self._respond(url, kwargs)

    def get(self, url, **kwargs):
        return self._respond(url, kwargs)

    delete = patch = get


def gemini_assistant(replies, **kwargs):
    assistant = Assistant(provider='gemini', api_key='k', **kwargs)
    session = FakeGoogleSession(replies)
    assistant.chatbot.wrapper.session = session
    return assistant, session


class TestTextSplitter(unittest.TestCase):
    def test_split(self):
        text = 'First paragraph sentence. ' * 20 + '\n\n' + 'Second part words ' * 30
        chunks = TextSplitter.split(text, chunk_size=200, chunk_overlap=40)
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(len(chunk) <= 200 for chunk in chunks))
        self.assertTrue(chunks[1].startswith('sentence') or chunks[1].startswith('First'))
        self.assertEqual(TextSplitter.to_documents('a b c', {'source': 'doc.md'}),
                         [{'id': 'doc.md#0', 'text': 'a b c', 'metadata': {'source': 'doc.md', 'chunk': 0}}])
        self.assertEqual(TextSplitter.split(''), [])
        with self.assertRaises(ValueError):
            TextSplitter.split('x', chunk_size=10, chunk_overlap=10)


class TestChatHistories(unittest.TestCase):
    def test_memory_and_file_histories(self):
        with tempfile.TemporaryDirectory() as folder:
            for history in (MemoryChatHistory(), FileChatHistory(dir=folder)):
                history.save_conversation({'id': 'c1', 'user_id': 'u1', 'title': 'First'})
                stored = history.add_messages('c1', [{'role': 'user', 'content': 'hi'},
                                                     {'role': 'assistant', 'content': 'hello',
                                                      'metadata': {'references': []}}])
                self.assertTrue(stored[0]['id'] and stored[0]['created_at'])
                history.add_messages('c2', [{'role': 'user', 'content': 'other'}])
                self.assertEqual([m['content'] for m in history.get_messages('c1')], ['hi', 'hello'])
                self.assertEqual([m['content'] for m in history.get_messages('c1', limit=1)], ['hello'])
                conversation = history.get_conversation('c1')
                self.assertEqual((conversation['title'], conversation['user_id'], conversation['message_count']),
                                 ('First', 'u1', 2))
                self.assertEqual([c['id'] for c in history.list_conversations(user_id='u1')], ['c1'])
                self.assertEqual(len(history.list_conversations()), 2)
                history.delete_last_messages('c1', 1)
                self.assertEqual(len(history.get_messages('c1')), 1)
                history.delete_conversation('c1')
                self.assertIsNone(history.get_conversation('c1'))
                self.assertEqual(history.get_messages('missing'), [])
            file_history = FileChatHistory(dir=folder)
            with self.assertRaises(Exception):
                file_history._file('..')
            self.assertTrue(file_history._file('../x').startswith(folder), 'ids cannot leave the directory')

    def test_files_use_the_intellinode_field_names(self):
        """A conversation file written by IntelliNode's FileChatHistory reads in Python, and back."""
        with tempfile.TemporaryDirectory() as folder:
            with open(os.path.join(folder, 'c9.json'), 'w') as file:
                json.dump({'conversation': {'id': 'c9', 'title': 'Node', 'userId': 'u1', 'createdAt': '2026-10-05T00:00:00Z',
                                            'updatedAt': '2026-10-05T00:00:01Z', 'metadata': {}},
                           'messages': [{'id': 'm1', 'role': 'user', 'content': 'hi',
                                         'createdAt': '2026-10-05T00:00:00Z'}]}, file)
            history = FileChatHistory(dir=folder)
            self.assertEqual(history.get_conversation('c9')['user_id'], 'u1')
            self.assertEqual(history.get_messages('c9')[0]['created_at'], '2026-10-05T00:00:00Z')
            self.assertEqual([c['id'] for c in history.list_conversations(user_id='u1')], ['c9'])
            history.add_messages('c9', [{'role': 'assistant', 'content': 'hello'}])
            with open(os.path.join(folder, 'c9.json')) as file:
                raw = json.load(file)
            self.assertEqual(raw['conversation']['userId'], 'u1')
            self.assertIn('createdAt', raw['messages'][1])
            self.assertNotIn('created_at', raw['messages'][1])


class TestEmbedder(unittest.TestCase):
    def test_cohere_kinds_openai_batches_and_gemini_task_types(self):
        cohere = Embedder(provider='cohere', api_key='k')
        captured = {}

        def cohere_embed(params):
            captured.update(params)
            return {'embeddings': {'float': [[0], [1]]}}

        cohere.wrapper.get_embeddings = cohere_embed
        self.assertEqual(cohere.embed(['a', 'b'], kind='query'), [[0], [1]])
        self.assertEqual(captured['input_type'], 'search_query')
        self.assertEqual(captured['model'], 'embed-v4.0')

        openai = Embedder(provider='openai', api_key='k', dimensions=256, batch_size=1)
        batches = []

        def openai_embed(params):
            batches.append(params)
            return {'data': [{'index': 0, 'embedding': [len(batches)]}]}

        openai.wrapper.get_embeddings = openai_embed
        self.assertEqual(openai.embed(['a', 'b']), [[1], [2]], 'texts are sent in batches and kept in order')
        self.assertEqual(batches[0], {'input': ['a'], 'model': 'text-embedding-3-small', 'dimensions': 256})

        gemini = Embedder(provider='gemini', api_key='k')
        session = FakeGoogleSession([{'embeddings': [{'values': [9]}]}])
        gemini.google.session = session
        self.assertEqual(gemini.embed(['q'], kind='query'), [[9]])
        self.assertEqual(session.calls[0]['body']['requests'][0]['taskType'], 'RETRIEVAL_QUERY')

        vertex = Embedder(provider='vertex', api_key='k', dimensions=768)
        self.assertTrue(vertex.google.vertex)

    def test_function_embedders(self):
        store = MemoryVectorStore(embedder=lambda texts: [[1.0, 0.0] for _ in texts])
        store.add_documents(['x'])
        kinds = []
        store = MemoryVectorStore(embedder=lambda texts, kind='document': kinds.append(kind) or [[1.0] for _ in texts])
        store.add_documents(['x'])
        store.search('x')
        self.assertEqual(kinds, ['document', 'query'])


class TestAssistant(unittest.TestCase):
    def setUp(self):
        self.knowledge = MemoryVectorStore(embedder=fake_embed)
        self.memory = MemoryVectorStore(embedder=fake_embed)
        self.history = MemoryChatHistory()

    def test_rag_streaming_history_and_memory(self):
        stream_reply = gemini_reply('Express takes 2 days.')
        assistant, session = gemini_assistant(
            [gemini_reply('Laptops have 15 days [1].', modelVersion='gemini-x-001'), stream_reply,
             gemini_reply('You asked about laptop returns.')],
            history=self.history, knowledge=self.knowledge, memory=self.memory,
            system_message='You are Acme support.', max_history=4)
        ids = assistant.add_documents([
            {'id': 'policy.md', 'text': 'Refund policy: return within 30 days. Laptop returns take 15 days.',
             'metadata': {'title': 'Refunds'}},
            {'id': 'shipping.md', 'text': 'Express shipping takes 2 days.'},
        ])
        self.assertEqual(ids, ['policy.md#0', 'shipping.md#0'])

        first = assistant.chat('How many days to return a laptop?', user_id='u1')
        self.assertEqual(first['text'], 'Laptops have 15 days [1].')
        self.assertEqual((first['references'][0]['id'], first['references'][0]['index']), ('policy.md#0', 1))
        self.assertEqual(first['usage'], {'input_tokens': 10, 'output_tokens': 2, 'total_tokens': 12})
        self.assertEqual(first['model'], 'gemini-x-001', 'the model comes from the response')
        self.assertEqual([r['cited'] for r in first['references']], [True, False], 'only [1] is cited')
        system = session.calls[0]['body']['systemInstruction']['parts'][0]['text']
        self.assertTrue(system.startswith('You are Acme support.'))
        self.assertIn('[1] Refunds', system)
        self.assertIn('Laptop returns take 15 days', system)
        self.assertEqual(len(session.calls[0]['body']['contents']), 1, 'the system text is not repeated in turns')

        events = list(assistant.stream('And express shipping days?', conversation_id=first['conversation_id'],
                                       user_id='u1'))
        self.assertEqual(events[0]['type'], 'start')
        self.assertEqual(''.join(e['text'] for e in events if e['type'] == 'text'), 'Express takes 2 days.')
        self.assertEqual(events[-1]['type'], 'done')
        self.assertIn(':streamGenerateContent', session.calls[1]['url'])
        # the previous exchange is sent as history
        self.assertEqual([c['role'] for c in session.calls[1]['body']['contents']], ['user', 'model', 'user'])

        # a new conversation recalls the earlier exchange from long-term memory
        recall = assistant.chat('What did I ask about laptop returns?', user_id='u1')
        self.assertTrue(recall['memories'] and 'laptop' in recall['memories'][0]['text'])
        self.assertEqual(recall['memories'][0]['conversation_id'], first['conversation_id'])
        self.assertIn('Notes from earlier conversations', session.calls[2]['body']['systemInstruction']['parts'][0]['text'])
        messages = assistant.get_messages(first['conversation_id'])
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[1]['metadata']['references'][0]['id'], 'policy.md#0')
        self.assertTrue(messages[1]['metadata']['references'][0]['cited'])
        self.assertEqual(self.memory.count(), 3)
        self.assertEqual(self.memory.get([messages[0]['id']])[0]['metadata']['userId'], 'u1')

        with self.assertRaises(PermissionError):
            assistant.chat('hi', conversation_id=first['conversation_id'], user_id='intruder')
        with self.assertRaises(ValueError):
            Assistant(provider='openai', api_key='k', google_search=True)

    def test_regenerate_rename_delete_and_titles(self):
        assistant, session = gemini_assistant([gemini_reply('one'), gemini_reply('Laptop return days'),
                                               gemini_reply('two')], memory=self.memory, auto_title=True)
        reply = assistant.chat('How many days to return a laptop?')
        conversation_id = reply['conversation_id']
        self.assertEqual(assistant.history.get_conversation(conversation_id)['title'], 'Laptop return days')
        again = assistant.regenerate(conversation_id)
        self.assertEqual(again['text'], 'two')
        self.assertEqual([m['content'] for m in assistant.get_messages(conversation_id)],
                         ['How many days to return a laptop?', 'two'])
        self.assertEqual(self.memory.count(), 1, 'the replaced answer leaves no memory behind')
        assistant.rename_conversation(conversation_id, 'Returns')
        self.assertEqual(assistant.list_conversations()[0]['title'], 'Returns')
        assistant.delete_conversation(conversation_id)
        self.assertEqual(assistant.list_conversations(), [])
        self.assertEqual(self.memory.count(), 0)

    def test_attachments_per_provider(self):
        image = {'data': base64.b64encode(b'png').decode(), 'mime_type': 'image/png', 'name': 'a.png'}
        pdf = {'data': 'JVBE', 'mime_type': 'application/pdf', 'name': 'report.pdf'}

        anthropic = Assistant(provider='anthropic', api_key='k')
        chat_input = anthropic._create_input('sys')
        chat_input.add_user_turn('what?', [image, pdf])
        params = chat_input.get_anthropic_input()
        self.assertEqual([b['type'] for b in params['messages'][0]['content']], ['image', 'document', 'text'])
        self.assertEqual(params['system'], 'sys')

        openai = Assistant(provider='openai', api_key='k', model='gpt-4.1')
        chat_input = openai._create_input('sys')
        chat_input.add_user_turn('what?', [image])
        params = chat_input.get_openai_input()
        self.assertEqual(params['messages'][1]['content'][1]['image_url']['url'], f"data:image/png;base64,{image['data']}")
        chat_input = openai._create_input('sys')
        chat_input.add_user_turn('x', [pdf])
        with self.assertRaises(ValueError):
            chat_input.get_openai_input()

        gpt5 = Assistant(provider='openai', api_key='k', model='gpt-5.5')
        chat_input = gpt5._create_input('sys')
        chat_input.add_assistant_message('earlier')
        chat_input.add_user_turn('what?', [image])
        params = chat_input.get_openai_input()
        self.assertEqual(params['instructions'], 'sys')
        self.assertEqual(params['input'][0], {'role': 'assistant', 'content': 'earlier'})
        self.assertEqual(params['input'][1]['content'][1]['type'], 'input_image')

        gemini = Assistant(provider='gemini', api_key='k', google_search=True)
        chat_input = gemini._create_input('sys')
        chat_input.add_user_turn('what?', [image, 'gs://bucket/report.pdf'and _read_attachment('gs://bucket/report.pdf')])
        params = chat_input.get_gemini_input()
        self.assertEqual(params['tools'], [{'googleSearch': {}}])
        self.assertEqual(params['contents'][0]['parts'][1]['inlineData']['mimeType'], 'image/png')
        self.assertEqual(params['contents'][0]['parts'][2]['fileData'],
                         {'mimeType': 'application/pdf', 'fileUri': 'gs://bucket/report.pdf'})

        aws = Assistant(provider='aws', api_key='k', options={'region': 'us-east-1'})
        chat_input = aws._create_input('sys')
        chat_input.add_user_turn('what?', [image, pdf])
        params = chat_input.get_aws_input()
        blocks = params['messages'][0]['content']
        self.assertEqual(blocks[0], {'image': {'format': 'png', 'source': {'bytes': image['data']}}})
        self.assertEqual(blocks[1]['document']['format'], 'pdf')
        self.assertEqual(blocks[2], {'text': 'what?'})

        self.assertEqual(_read_attachment('gs://bucket/report.pdf')['mime_type'], 'application/pdf')
        self.assertEqual(_read_attachment('data:image/jpeg;base64,QUJD')['data'], 'QUJD')
        self.assertEqual(_read_attachment({'data': b'abc', 'mime_type': 'text/plain'})['data'], 'YWJj')
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as file:
            file.write(b'png')
        try:
            self.assertEqual(_read_attachment(file.name)['mime_type'], 'image/png')
        finally:
            os.unlink(file.name)

    def test_google_search_citations(self):
        grounded = {'candidates': [{'content': {'parts': [{'text': 'Node 24 [1]'}]}, 'groundingMetadata': {
            'groundingChunks': [{'web': {'uri': 'https://nodejs.org', 'title': 'nodejs.org'}}]}}]}
        assistant, session = gemini_assistant([grounded, gemini_reply('web'), gemini_reply('plain')],
                                              google_search=True)
        answer = assistant.chat('Latest Node?')
        self.assertEqual(answer['citations'], [{'title': 'nodejs.org', 'uri': 'https://nodejs.org'}])
        self.assertEqual(assistant.get_messages(answer['conversation_id'])[1]['metadata']['citations'],
                         answer['citations'])

        plain, plain_session = gemini_assistant([gemini_reply('web'), gemini_reply('plain')])
        plain.chat('news?', google_search=True)
        self.assertEqual(plain_session.calls[0]['body']['tools'], [{'googleSearch': {}}])
        plain.chat('hello')
        self.assertNotIn('tools', plain_session.calls[1]['body'])
        with self.assertRaises(ValueError):
            Assistant(provider='openai', api_key='k').chat('x', google_search=True)


def add(a: float, b: float):
    """Add two numbers."""
    return a + b


class TestAssistantTools(unittest.TestCase):
    def test_tool_definitions_from_functions_and_dicts(self):
        definitions, handlers = _tool_registry([add, {'name': 'echo', 'description': 'Echo',
                                                      'parameters': {'type': 'object', 'properties': {}},
                                                      'handler': lambda: 'x'}])
        self.assertEqual(definitions[0], {'type': 'function', 'function': {
            'name': 'add', 'description': 'Add two numbers.',
            'parameters': {'type': 'object', 'properties': {'a': {'type': 'number'}, 'b': {'type': 'number'}},
                           'required': ['a', 'b']}}})
        self.assertEqual(set(handlers), {'add', 'echo'})
        definitions, handlers = _tool_registry({'sum': add})
        self.assertEqual(definitions[0]['function']['name'], 'sum')
        with self.assertRaises(ValueError):
            Assistant(provider='keras', options={'model_name': 'x'}, tools=[add])

    def test_gemini_tool_loop_sends_the_model_turn_back(self):
        call = {'candidates': [{'content': {'role': 'model', 'parts': [
            {'functionCall': {'name': 'add', 'args': {'a': 1, 'b': 2}}, 'thoughtSignature': 'sig'}]}}]}
        assistant, session = gemini_assistant([call, gemini_reply('3')], tools=[add])
        reply = assistant.chat('1 + 2?')
        self.assertEqual(reply['text'], '3')
        self.assertEqual(reply['tool_steps'], [{'name': 'add', 'arguments': {'a': 1, 'b': 2}, 'result': '3',
                                                'is_error': False}])
        first, second = session.calls[0]['body'], session.calls[1]['body']
        self.assertEqual(first['tools'][0]['functionDeclarations'][0]['name'], 'add')
        self.assertEqual(second['contents'][1], {'role': 'model', 'parts': [
            {'functionCall': {'name': 'add', 'args': {'a': 1, 'b': 2}}, 'thoughtSignature': 'sig'}]})
        self.assertEqual(second['contents'][2], {'role': 'user', 'parts': [
            {'functionResponse': {'name': 'add', 'response': {'result': '3'}}}]})

    def test_openai_responses_tool_loop(self):
        assistant = Assistant(provider='openai', api_key='k', model='gpt-5.5', tools=[add])
        output = [{'type': 'reasoning', 'id': 'rs_1', 'summary': []},
                  {'type': 'function_call', 'id': 'fc_1', 'call_id': 'call_1', 'name': 'add',
                   'arguments': '{"a": 2, "b": 3}'}]
        replies = [{'output': output, 'usage': {'input_tokens': 5, 'output_tokens': 1}, 'model': 'gpt-5.5-x'},
                   {'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': '5'}]}],
                    'usage': {'input_tokens': 9, 'output_tokens': 1, 'total_tokens': 10}, 'model': 'gpt-5.5-x'}]
        bodies = []

        def respond(params):
            bodies.append(json.loads(json.dumps(params)))
            return replies[len(bodies) - 1]

        with patch.object(type(assistant.chatbot.wrapper), 'generate_gpt5_response', side_effect=respond):
            reply = assistant.chat('2 + 3?')
        self.assertEqual(reply['text'], '5')
        self.assertEqual(reply['usage'], {'input_tokens': 9, 'output_tokens': 1, 'total_tokens': 10})
        self.assertEqual(reply['model'], 'gpt-5.5-x')
        self.assertEqual(bodies[0]['tools'][0]['name'], 'add', 'Responses API tools are flat')
        self.assertEqual(bodies[1]['input'][1:], output + [{'type': 'function_call_output', 'call_id': 'call_1',
                                                            'output': '5'}])

    def test_anthropic_and_aws_tool_loops(self):
        assistant = Assistant(provider='anthropic', api_key='k', tools=[add])
        replies = [{'stop_reason': 'tool_use', 'content': [{'type': 'tool_use', 'id': 'tu1', 'name': 'add',
                                                             'input': {'a': 1, 'b': 1}}]},
                   {'stop_reason': 'end_turn', 'content': [{'type': 'text', 'text': '2'}],
                    'usage': {'input_tokens': 3, 'output_tokens': 1}}]
        bodies = []

        def respond(params, extra_headers=None):
            bodies.append(json.loads(json.dumps(params)))
            return replies[len(bodies) - 1]

        assistant.chatbot.wrapper.generate_text = respond
        reply = assistant.chat('1 + 1?')
        self.assertEqual((reply['text'], reply['usage']['total_tokens']), ('2', 4))
        self.assertEqual(bodies[0]['tools'][0]['input_schema']['properties']['a'], {'type': 'number'})
        self.assertEqual(bodies[1]['messages'][1], {'role': 'assistant', 'content': [
            {'type': 'tool_use', 'id': 'tu1', 'name': 'add', 'input': {'a': 1, 'b': 1}}]})
        self.assertEqual(bodies[1]['messages'][2], {'role': 'user', 'content': [
            {'type': 'tool_result', 'tool_use_id': 'tu1', 'content': '2'}]})

        aws = Assistant(provider='aws', api_key='k', options={'region': 'us-east-1'},
                        tools=[{'name': 'fail', 'description': 'Always fails', 'handler': lambda: 1 / 0}])
        replies = [{'stopReason': 'tool_use', 'output': {'message': {'content': [
            {'toolUse': {'toolUseId': 't1', 'name': 'fail', 'input': {}}}]}}},
            {'stopReason': 'end_turn', 'output': {'message': {'content': [{'text': 'It failed.'}]}},
             'usage': {'inputTokens': 4, 'outputTokens': 2, 'totalTokens': 6}}]
        sent = []

        def converse(params, model=None, fallback_models=None):
            sent.append(json.loads(json.dumps(params)))
            return replies[len(sent) - 1]

        aws.chatbot.wrapper.converse = converse
        reply = aws.chat('try it')
        self.assertEqual(reply['text'], 'It failed.')
        self.assertTrue(reply['tool_steps'][0]['is_error'])
        self.assertEqual(reply['usage'], {'input_tokens': 4, 'output_tokens': 2, 'total_tokens': 6})
        self.assertEqual(sent[1]['messages'][2]['content'][0]['toolResult']['status'], 'error')
        self.assertIn('division by zero', sent[1]['messages'][2]['content'][0]['toolResult']['content'][0]['text'])

    def test_tool_loop_limit(self):
        call = {'candidates': [{'content': {'parts': [{'functionCall': {'name': 'add', 'args': {'a': 1, 'b': 1}}}]}}]}
        assistant, _ = gemini_assistant([call, call], tools=[add], max_tool_steps=1)
        with self.assertRaises(RuntimeError):
            assistant.chat('loop')


if __name__ == '__main__':
    unittest.main()

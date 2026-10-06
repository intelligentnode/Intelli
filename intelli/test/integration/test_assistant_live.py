"""
Live tests for the Assistant with real providers. Each class runs only when its keys are set.

    GEMINI_API_KEY          Gemini: RAG with Gemini embeddings, citations, streaming, tools, Google Search, memory.
    OPENAI_API_KEY          OpenAI: Responses API (gpt-5 family) with tools; with QDRANT_URL also a Qdrant knowledge store.
    ANTHROPIC_API_KEY       Anthropic: tools and an image attachment.
    AWS_LIVE_TESTS=1        Amazon Bedrock (Nova Lite) with tools, using the AWS credentials of the environment.

Run:
    python3 -m pytest intelli/test/integration/test_assistant_live.py -q -s
"""
import base64
import os
import struct
import unittest
import uuid
import zlib

from dotenv import load_dotenv

from intelli.function.assistant import Assistant
from intelli.store import MemoryVectorStore, MemoryChatHistory, QdrantVectorStore

load_dotenv()

HANDBOOK = ('Acme refund policy: customers can return items within 30 days of delivery. Laptops are an exception '
            'and can be returned within 15 days. Express shipping takes 2 business days and costs 9 dollars.')


def get_weather(city: str):
    """Get the current weather of a city."""
    return {'city': city, 'forecast': 'sunny', 'temperature_c': 21}


def red_png():
    def chunk(tag, data):
        return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data))
    rows = b''.join(b'\x00' + b'\xff\x00\x00' * 32 for _ in range(32))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 32, 32, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b''))


@unittest.skipUnless(os.getenv('GEMINI_API_KEY'), 'Set GEMINI_API_KEY')
class TestGeminiAssistantLive(unittest.TestCase):
    def setUp(self):
        key = os.environ['GEMINI_API_KEY']
        self.embedder = {'provider': 'gemini', 'api_key': key, 'dimensions': 768}
        self.assistant = Assistant(provider='gemini', api_key=key,
                                   system_message='You are Acme support. Answer in one sentence.',
                                   knowledge=MemoryVectorStore(embedder=self.embedder),
                                   memory=MemoryVectorStore(embedder=self.embedder), max_tokens=2000)

    def test_rag_stream_and_memory(self):
        self.assistant.add_documents([{'id': 'handbook.md', 'text': HANDBOOK, 'metadata': {'title': 'Handbook'}}])
        reply = self.assistant.chat('How many days do I have to return a laptop?', user_id='u1')
        print('gemini rag:', reply['text'], reply['usage'], reply['model'])
        self.assertIn('15', reply['text'])
        self.assertEqual(reply['references'][0]['id'], 'handbook.md#0')
        events = list(self.assistant.stream('And how long does express shipping take?',
                                            conversation_id=reply['conversation_id'], user_id='u1'))
        text = ''.join(event['text'] for event in events if event['type'] == 'text')
        print('gemini stream:', text)
        self.assertIn('2', text)
        self.assertEqual(events[-1]['type'], 'done')
        recall = self.assistant.chat('What product did I ask about returning earlier?', user_id='u1')
        print('gemini memory:', recall['text'])
        self.assertTrue(recall['memories'])
        self.assertIn('laptop', recall['text'].lower())

    def test_tools_and_google_search(self):
        assistant = Assistant(provider='gemini', api_key=os.environ['GEMINI_API_KEY'], tools=[get_weather],
                              max_tokens=2000)
        reply = assistant.chat('What is the weather in Paris right now? Use the tool.')
        print('gemini tools:', reply['text'], reply['tool_steps'])
        self.assertEqual(reply['tool_steps'][0]['name'], 'get_weather')
        self.assertIn('sunny', reply['text'].lower())
        search = Assistant(provider='gemini', api_key=os.environ['GEMINI_API_KEY'], google_search=True,
                           max_tokens=2000)
        answer = search.chat('Who won the most recent FIFA World Cup final? One sentence.')
        print('gemini search:', answer['text'], answer['citations'][:2])
        self.assertTrue(answer['citations'], 'Google Search grounding returns web sources')


@unittest.skipUnless(os.getenv('OPENAI_API_KEY'), 'Set OPENAI_API_KEY')
class TestOpenAIAssistantLive(unittest.TestCase):
    def test_responses_api_tools(self):
        assistant = Assistant(provider='openai', api_key=os.environ['OPENAI_API_KEY'], model='gpt-5-mini',
                              tools=[get_weather])
        reply = assistant.chat('What is the weather in Paris right now? Use the tool, then answer in one sentence.')
        print('openai tools:', reply['text'], reply['usage'], reply['model'])
        self.assertEqual(reply['tool_steps'][0]['name'], 'get_weather')
        self.assertIn('sunny', reply['text'].lower())
        follow = assistant.chat('And what did I just ask you about?', conversation_id=reply['conversation_id'])
        print('openai follow-up:', follow['text'])
        self.assertIn('paris', follow['text'].lower())

    @unittest.skipUnless(os.getenv('QDRANT_URL'), 'Set QDRANT_URL')
    def test_qdrant_knowledge(self):
        key = os.environ['OPENAI_API_KEY']
        store = QdrantVectorStore(url=os.environ['QDRANT_URL'], collection=f'assistant_{uuid.uuid4().hex[:8]}',
                                  embedder={'provider': 'openai', 'api_key': key})
        try:
            assistant = Assistant(provider='openai', api_key=key, model='gpt-4.1-mini', knowledge=store,
                                  history=MemoryChatHistory())
            assistant.add_documents([{'id': 'handbook.md', 'text': HANDBOOK}])
            reply = assistant.chat('How much does express shipping cost?')
            print('openai + qdrant:', reply['text'], [(r['id'], round(r['score'], 3), r['cited'])
                                                      for r in reply['references']])
            self.assertIn('9', reply['text'])
            self.assertEqual(reply['references'][0]['id'], 'handbook.md#0')
        finally:
            store.client.request('DELETE', store._path())


@unittest.skipUnless(os.getenv('ANTHROPIC_API_KEY'), 'Set ANTHROPIC_API_KEY')
class TestAnthropicAssistantLive(unittest.TestCase):
    def test_tools_and_image(self):
        assistant = Assistant(provider='anthropic', api_key=os.environ['ANTHROPIC_API_KEY'], model='claude-haiku-4-5',
                              tools=[get_weather], max_tokens=400)
        reply = assistant.chat('What is the weather in Paris? Use the tool.')
        print('anthropic tools:', reply['text'], reply['usage'])
        self.assertEqual(reply['tool_steps'][0]['name'], 'get_weather')
        self.assertIn('sunny', reply['text'].lower())
        image = {'data': base64.b64encode(red_png()).decode(), 'mime_type': 'image/png', 'name': 'red.png'}
        seen = Assistant(provider='anthropic', api_key=os.environ['ANTHROPIC_API_KEY'], model='claude-haiku-4-5',
                         max_tokens=100).chat('What color is this image? One word.', attachments=[image])
        print('anthropic image:', seen['text'])
        self.assertIn('red', seen['text'].lower())


@unittest.skipUnless(os.getenv('AWS_LIVE_TESTS'), 'Set AWS_LIVE_TESTS=1')
class TestBedrockAssistantLive(unittest.TestCase):
    def test_tools(self):
        options = {'region': os.getenv('AWS_REGION') or 'us-east-1'}
        assistant = Assistant(provider='aws', api_key=os.getenv('AWS_BEARER_TOKEN_BEDROCK'), options=options,
                              model='us.amazon.nova-lite-v1:0', tools=[get_weather], max_tokens=400)
        reply = assistant.chat('What is the weather in Paris? Use the get_weather tool.')
        print('bedrock tools:', reply['text'], reply['usage'], reply['model'])
        self.assertEqual(reply['tool_steps'][0]['name'], 'get_weather')
        self.assertIn('sunny', reply['text'].lower())


if __name__ == '__main__':
    unittest.main()

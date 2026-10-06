"""
Offline tests for speech steps in flows: the text that is spoken, and long OpenAI text spoken in pieces.
"""
import unittest
from unittest.mock import patch

from intelli.flow import Agent, Task, TextTaskInput
from intelli.flow.agents.handlers import OPENAI_SPEECH_LIMIT, SpeechAgentHandler, split_for_speech


class TestSpeechSteps(unittest.TestCase):
    def test_a_speech_step_speaks_its_input_without_the_template(self):
        spoken = []
        with patch.object(SpeechAgentHandler, 'execute', lambda self, agent_input, params: spoken.append(
                agent_input.desc) or b'mp3'):
            agent = Agent('speech', 'openai', '', {'key': 'k', 'model': 'tts-1'})
            Task(TextTaskInput('Read the script aloud.'), agent).execute('Welcome to Rome.', input_type='text')
            Task(TextTaskInput('Hello from Intelli.'), agent).execute(None)
        self.assertEqual(spoken, ['Welcome to Rome.', 'Hello from Intelli.'])

    def test_long_openai_text_is_spoken_in_pieces(self):
        requests = []

        def generate_speech(model, speech_input):
            requests.append(speech_input.text)
            return b'mp3-' + str(len(requests)).encode()

        text = ' '.join(f'Sentence number {i} about the Colosseum.' for i in range(300))
        with patch('intelli.controller.remote_speech_model.RemoteSpeechModel.generate_speech', generate_speech):
            audio = Agent('speech', 'openai', '', {'key': 'k', 'model': 'tts-1', 'stream': False}).execute(
                TextTaskInput(text))
        self.assertGreater(len(requests), 2)
        self.assertTrue(all(len(piece) <= OPENAI_SPEECH_LIMIT for piece in requests))
        self.assertEqual(' '.join(requests), text)
        self.assertEqual(audio, b''.join(b'mp3-' + str(i + 1).encode() for i in range(len(requests))))

    def test_split_for_speech(self):
        self.assertEqual(split_for_speech('One. Two! Three?', limit=9), ['One. Two!', 'Three?'])
        self.assertEqual(split_for_speech('word ' * 5, limit=10), ['word word', 'word word', 'word'])
        self.assertEqual([len(p) for p in split_for_speech('x' * 25, limit=10)], [10, 10, 5])
        self.assertEqual(split_for_speech('  '), [])


if __name__ == '__main__':
    unittest.main()

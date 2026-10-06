import unittest

from intelli.model.input.vision_input import VisionModelInput


class TestOpenAIVisionInput(unittest.TestCase):
    def test_the_token_limit_name_follows_the_model(self):
        gpt5 = VisionModelInput('What is in the picture?', image_data='aGk=', model='gpt-5.5').get_openai_inputs()
        self.assertEqual(gpt5['max_completion_tokens'], 4096, 'room to reason and still answer')
        self.assertNotIn('max_tokens', gpt5)
        capped = VisionModelInput('q', image_data='aGk=', model='gpt-5.5', max_tokens=800).get_openai_inputs()
        self.assertEqual(capped['max_completion_tokens'], 800)

        older = VisionModelInput('What is in the picture?', image_data='aGk=', model='gpt-4o').get_openai_inputs()
        self.assertEqual(older['max_tokens'], 300)
        self.assertNotIn('max_completion_tokens', older)


if __name__ == '__main__':
    unittest.main()

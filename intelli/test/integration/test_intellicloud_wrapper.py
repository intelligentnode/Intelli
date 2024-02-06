import unittest
import os
from intelli.wrappers.intellicloud_wrapper import IntellicloudWrapper
from dotenv import load_dotenv

load_dotenv()

class TestIntellicloudWrapper(unittest.TestCase):

    def setUp(self):
        """Initialize the wrapper with an API key."""
        api_key = os.getenv("INTELLI_ONE_KEY")
        # get the dev intelli url for the test case
        api_base = os.getenv("INTELLI_API_BASE")
        self.assertIsNotNone(api_key, "INTELLI_ONE_KEY must not be None.")
        self.intellicloud = IntellicloudWrapper(api_key, api_base)
    
    def test_semantic_search(self):
        query_text = "Why is Mars called the Red Planet?"
        k = 2
        result = self.intellicloud.semantic_search(query_text, k)
        print('Semantic Search Result: ', result)
        self.assertTrue(len(result) > 0, "Semantic search should return at least one result")
    
    def test_semantic_search_with_filter(self):
        query_text = "Why is Mars called the Red Planet?"
        k = 2
        filters = {'document_name': 'test_mars_article.pdf'}
        result = self.intellicloud.semantic_search(query_text, k, filters)
        print('Semantic Search Result with Filter: ', result)
        self.assertTrue(len(result) > 0, "Semantic search with filter should return at least one result")

if __name__ == "__main__":
    unittest.main()

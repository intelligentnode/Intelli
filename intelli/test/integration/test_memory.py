import os
import unittest
import tempfile
import shutil
from intelli.flow.store.memory import Memory
from intelli.flow.store.dbmemory import DBMemory


class TestMemoryImplementations(unittest.TestCase):
    """Test suite for both Memory and DBMemory implementations."""

    def setUp(self):
        """Set up test environment."""
        # Create a temporary test directory
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test_memory.db")

    def tearDown(self):
        """Clean up after tests."""
        # Clean up test directory
        shutil.rmtree(self.test_dir)

    def test_in_memory_basics(self):
        """Test basic operations with in-memory store."""
        memory = Memory()

        # Test store and retrieve
        memory.store("test_key", "test_value")
        self.assertEqual(memory.retrieve("test_key"), "test_value")

        # Test has_key
        self.assertTrue(memory.has_key("test_key"))
        self.assertFalse(memory.has_key("nonexistent_key"))

        # Test 'in' operator
        self.assertTrue("test_key" in memory)
        self.assertFalse("nonexistent_key" in memory)

        # Test keys
        memory.store("another_key", 42)
        self.assertSetEqual(set(memory.keys()), {"test_key", "another_key"})

        # Test all
        self.assertDictEqual(memory.all(), {"test_key": "test_value", "another_key": 42})

        # Test clear
        memory.clear()
        self.assertEqual(len(memory.keys()), 0)

    def test_db_memory_basics(self):
        """Test basic operations with database-backed store."""
        db_memory = DBMemory(db_path=self.db_path)

        # Test store and retrieve
        db_memory.store("test_key", "test_value")
        self.assertEqual(db_memory.retrieve("test_key"), "test_value")

        # Test has_key
        self.assertTrue(db_memory.has_key("test_key"))
        self.assertFalse(db_memory.has_key("nonexistent_key"))

        # Test 'in' operator
        self.assertTrue("test_key" in db_memory)
        self.assertFalse("nonexistent_key" in db_memory)

        # Test keys
        db_memory.store("another_key", 42)
        self.assertSetEqual(set(db_memory.keys()), {"test_key", "another_key"})

        # Test all
        self.assertDictEqual(db_memory.all(), {"test_key": "test_value", "another_key": 42})

        # Test clear
        db_memory.clear()
        self.assertEqual(len(db_memory.keys()), 0)

        # Close connection
        db_memory.close()

        # Verify the database file was created
        self.assertTrue(os.path.exists(self.db_path))

    def test_complex_data_types(self):
        """Test storing and retrieving complex data types."""
        in_memory = Memory()
        db_memory = DBMemory(db_path=self.db_path)

        # Test dictionary
        test_dict = {"name": "John", "age": 30, "scores": [85, 90, 78]}
        in_memory.store("dict_key", test_dict)
        db_memory.store("dict_key", test_dict)

        self.assertEqual(in_memory.retrieve("dict_key"), test_dict)
        self.assertEqual(db_memory.retrieve("dict_key"), test_dict)

        # Test list
        test_list = [1, "two", 3.0, {"four": 4}]
        in_memory.store("list_key", test_list)
        db_memory.store("list_key", test_list)

        self.assertEqual(in_memory.retrieve("list_key"), test_list)
        self.assertEqual(db_memory.retrieve("list_key"), test_list)

        # Test None
        in_memory.store("none_key", None)
        db_memory.store("none_key", None)

        self.assertIsNone(in_memory.retrieve("none_key"))
        self.assertIsNone(db_memory.retrieve("none_key"))

        # Close connection
        db_memory.close()

    def test_persistence(self):
        """Test that DBMemory persists data between instances."""
        # Create first instance and add data
        db_memory1 = DBMemory(db_path=self.db_path)
        db_memory1.store("persistent_key", "persistent_value")
        db_memory1.close()

        # Create a new instance and check if data persists
        db_memory2 = DBMemory(db_path=self.db_path)
        self.assertEqual(db_memory2.retrieve("persistent_key"), "persistent_value")
        db_memory2.close()

    def test_interoperability(self):
        """Test that data can be transferred between memory implementations."""
        in_memory = Memory()
        db_memory = DBMemory(db_path=self.db_path)

        # Store data in Memory
        in_memory.store("key1", "value1")
        in_memory.store("key2", {"nested": "data"})

        # Transfer to DBMemory
        for key in in_memory.keys():
            db_memory.store(key, in_memory.retrieve(key))

        # Verify transfer worked
        self.assertEqual(db_memory.retrieve("key1"), "value1")
        self.assertEqual(db_memory.retrieve("key2"), {"nested": "data"})

        # Close connection
        db_memory.close()

    def test_with_flow_data(self):
        """Test with complex data structures typical in a flow."""
        db_path = os.path.join(self.test_dir, "flow_memory.db")
        db_memory = DBMemory(db_path=db_path)

        # Example complex nested structures typically found in flow outputs
        flow_data = {
            "task1_result": {
                "output": "This is the output from task 1",
                "metadata": {
                    "timestamp": "2023-04-01T12:34:56",
                    "metrics": {"tokens": 150, "latency": 0.8}
                }
            },
            "task2_result": {
                "output": ["item1", "item2", "item3"],
                "binary_data": b"some binary data"
            }
        }

        # Store and retrieve complex nested data
        db_memory.store("flow_results", flow_data)
        retrieved_data = db_memory.retrieve("flow_results")

        # Verify data integrity
        self.assertEqual(retrieved_data["task1_result"]["output"],
                         flow_data["task1_result"]["output"])
        self.assertEqual(retrieved_data["task2_result"]["binary_data"],
                         flow_data["task2_result"]["binary_data"])

        # Close connection
        db_memory.close()

    def test_auto_directory_creation(self):
        """Test automatic directory creation."""
        nested_dir = os.path.join(self.test_dir, "nested", "db", "path")
        db_path = os.path.join(nested_dir, "nested_memory.db")

        # Verify directory doesn't exist yet
        self.assertFalse(os.path.exists(nested_dir))

        # Create DBMemory instance which should create the directory
        db_memory = DBMemory(db_path=db_path)
        db_memory.store("test", "value")
        db_memory.close()

        # Verify directory and database were created
        self.assertTrue(os.path.exists(nested_dir))
        self.assertTrue(os.path.exists(db_path))

    def test_commit_functionality(self):
        """Test explicit commit functionality."""
        db_memory = DBMemory(db_path=self.db_path, auto_commit=False)

        # Store without auto-commit
        db_memory.store("key1", "value1")

        # Create new connection to check if data is visible
        check_memory = DBMemory(db_path=self.db_path)
        self.assertFalse(check_memory.has_key("key1"),
                         "Key should not be visible before commit")
        check_memory.close()

        # Explicitly commit
        db_memory.commit()

        # Check again after commit
        check_memory = DBMemory(db_path=self.db_path)
        self.assertTrue(check_memory.has_key("key1"),
                        "Key should be visible after commit")
        self.assertEqual(check_memory.retrieve("key1"), "value1")
        check_memory.close()

        db_memory.close()


if __name__ == "__main__":
    unittest.main()

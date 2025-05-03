from intelli.flow.store.basememory import BaseMemory

class Memory(BaseMemory):
    """In-memory store for sharing data between tasks in a flow."""

    def __init__(self):
        """Initialize an empty memory store."""
        self._data = {}

    def store(self, key, value):
        """
        Store a value in memory with the given key.

        Args:
            key (str): The key to use for storing the value
            value: The value to store

        Returns:
            Memory: Self for method chaining
        """
        self._data[key] = value
        return self  # Allow method chaining

    def retrieve(self, key, default=None):
        """
        Retrieve a value from memory using the given key.

        Args:
            key (str): The key for the value to retrieve
            default: Value to return if key doesn't exist

        Returns:
            The stored value, or default if the key doesn't exist
        """
        return self._data.get(key, default)

    def has_key(self, key):
        """
        Check if a key exists in memory.

        Args:
            key (str): The key to check

        Returns:
            bool: True if the key exists, False otherwise
        """
        return key in self._data

    def all(self):
        """
        Get all stored data.

        Returns:
            dict: A copy of all stored data
        """
        return dict(self._data)

    def clear(self):
        """
        Clear all stored data.

        Returns:
            Memory: Self for method chaining
        """
        self._data.clear()
        return self

    def keys(self):
        """
        Get all keys in memory.

        Returns:
            list: All keys in memory
        """
        return list(self._data.keys())

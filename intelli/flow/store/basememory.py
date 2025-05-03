from abc import ABC, abstractmethod


class BaseMemory(ABC):
    """Abstract base class for memory stores in Intelli flow."""

    @abstractmethod
    def store(self, key, value):
        """
        Store a value in memory with the given key.

        Args:
            key (str): The key to use for storing the value
            value: The value to store

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def retrieve(self, key, default=None):
        """
        Retrieve a value from memory using the given key.

        Args:
            key (str): The key for the value to retrieve
            default: Value to return if key doesn't exist

        Returns:
            The stored value, or default if the key doesn't exist
        """
        pass

    @abstractmethod
    def has_key(self, key):
        """
        Check if a key exists in memory.

        Args:
            key (str): The key to check

        Returns:
            bool: True if the key exists, False otherwise
        """
        pass

    @abstractmethod
    def all(self):
        """
        Get all stored data.

        Returns:
            dict: A copy of all stored data
        """
        pass

    @abstractmethod
    def clear(self):
        """
        Clear all stored data.

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def keys(self):
        """
        Get all keys in memory.

        Returns:
            list: All keys in memory
        """
        pass

    def __contains__(self, key):
        """Support for 'in' operator."""
        return self.has_key(key)

import os
import sqlite3
import json
import pickle
import base64
import threading
from intelli.flow.store.basememory import BaseMemory


class DBMemory(BaseMemory):
    """
    Thread-safe database-backed memory store for sharing data between tasks in a flow.
    Uses SQLite for lightweight persistent storage.
    """

    def __init__(self, db_path="./memory.db", auto_commit=True):
        """
        Initialize a database-backed memory store.

        Args:
            db_path (str): Path to the SQLite database file
            auto_commit (bool): Whether to commit after each store operation
        """
        self.db_path = db_path
        self.auto_commit = auto_commit

        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

        # Thread-local storage for connections
        self._local = threading.local()

        # Initialize the database schema
        conn = self._get_connection()
        self._init_db(conn)

    def _get_connection(self):
        """Get a thread-local connection to the database."""
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(self.db_path)
        return self._local.conn

    def _init_db(self, conn):
        """Initialize the database schema if it doesn't exist."""
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS memory (
            key TEXT PRIMARY KEY,
            value_type TEXT NOT NULL,
            value BLOB NOT NULL
        )
        ''')
        conn.commit()

    def _serialize_value(self, value):
        """
        Serialize a value for storage.

        Returns:
            tuple: (value_type, serialized_value)
        """
        if value is None:
            return 'null', b'null'
        elif isinstance(value, (str, int, float, bool)):
            # Simple JSON serializable types
            return 'json', json.dumps(value).encode('utf-8')
        else:
            # Complex objects use pickle
            return 'pickle', pickle.dumps(value)

    def _deserialize_value(self, value_type, value):
        """Deserialize a value from storage."""
        if value_type == 'null':
            return None
        elif value_type == 'json':
            return json.loads(value.decode('utf-8'))
        elif value_type == 'pickle':
            return pickle.loads(value)
        else:
            raise ValueError(f"Unknown value type: {value_type}")

    def store(self, key, value):
        """
        Store a value in the database with the given key.

        Args:
            key (str): The key to use for storing the value
            value: The value to store

        Returns:
            DBMemory: Self for method chaining
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        value_type, serialized_value = self._serialize_value(value)

        cursor.execute(
            'INSERT OR REPLACE INTO memory (key, value_type, value) VALUES (?, ?, ?)',
            (key, value_type, serialized_value)
        )

        if self.auto_commit:
            conn.commit()

        return self

    def retrieve(self, key, default=None):
        """
        Retrieve a value from the database using the given key.

        Args:
            key (str): The key for the value to retrieve
            default: Value to return if key doesn't exist

        Returns:
            The stored value, or default if the key doesn't exist
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT value_type, value FROM memory WHERE key = ?', (key,))
        result = cursor.fetchone()

        if result:
            value_type, value = result
            return self._deserialize_value(value_type, value)
        else:
            return default

    def has_key(self, key):
        """
        Check if a key exists in the database.

        Args:
            key (str): The key to check

        Returns:
            bool: True if the key exists, False otherwise
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT 1 FROM memory WHERE key = ?', (key,))
        return cursor.fetchone() is not None

    def all(self):
        """
        Get all stored data.

        Returns:
            dict: A dictionary of all stored key-value pairs
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT key, value_type, value FROM memory')
        result = {}

        for key, value_type, value in cursor.fetchall():
            result[key] = self._deserialize_value(value_type, value)

        return result

    def clear(self):
        """
        Clear all stored data.

        Returns:
            DBMemory: Self for method chaining
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM memory')

        if self.auto_commit:
            conn.commit()

        return self

    def keys(self):
        """
        Get all keys in the database.

        Returns:
            list: All keys in the database
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT key FROM memory')
        return [row[0] for row in cursor.fetchall()]

    def commit(self):
        """
        Explicitly commit changes to the database.
        Useful when auto_commit is set to False.

        Returns:
            DBMemory: Self for method chaining
        """
        conn = self._get_connection()
        conn.commit()
        return self

    def close(self):
        """Close the database connection for all threads."""
        # The main connection
        if hasattr(self, 'conn') and self.conn:
            self.conn.commit()
            self.conn.close()
            self.conn = None

        # Thread-local connections
        if hasattr(self, '_local') and hasattr(self._local, 'conn'):
            self._local.conn.commit()
            self._local.conn.close()
            delattr(self._local, 'conn')

    def __del__(self):
        """Ensure the database connection is closed on object destruction."""
        self.close()

    def export_to_json(self, json_path):
        """
        Export the entire database contents to a JSON file.

        Args:
            json_path (str): Path where the JSON file will be saved

        Returns:
            str: Path to the saved JSON file
        """
        import json

        # Get all data from the database
        data = self.all()

        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)

        # Write to JSON file with pretty formatting
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)

        return json_path

    def query(self, query_string, params=()):
        """
        Execute a custom SQL query on the database.

        Args:
            query_string (str): The SQL query to execute
            params (tuple): Parameters for the query

        Returns:
            list: List of query results
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(query_string, params)
        return cursor.fetchall()

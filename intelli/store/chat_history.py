import json
import os
import re
import threading
from datetime import datetime, timezone

from intelli.store.vector_store import StoreError, new_id


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')


# Stored field names match IntelliNode, so a Node app and a Python app can share files and Firestore conversations.
STORED_KEYS = {'created_at': 'createdAt', 'updated_at': 'updatedAt', 'user_id': 'userId'}
PYTHON_KEYS = {value: key for key, value in STORED_KEYS.items()}


def to_stored(item):
    """A message or conversation with the stored (IntelliNode) field names."""
    return {STORED_KEYS.get(key, key): value for key, value in (item or {}).items() if key != 'message_count'}


def from_stored(item):
    """A stored message or conversation with Python field names."""
    return {PYTHON_KEYS.get(key, key): value for key, value in (item or {}).items()}


class ChatHistory:
    """
    Where an Assistant keeps its conversations. Messages are {'id', 'role': 'user' | 'assistant', 'content',
    'created_at', 'metadata'?}; conversations are {'id', 'title', 'user_id', 'created_at', 'updated_at',
    'metadata'}.

    Implementations: MemoryChatHistory (in process), FileChatHistory (JSON files), FirestoreChatHistory (Google
    Cloud Firestore). Write your own (Redis, Postgres, DynamoDB) by extending ChatHistory and implementing the
    seven methods.
    """

    def get_messages(self, conversation_id, limit=None):
        """The last `limit` messages of a conversation, oldest first."""
        raise NotImplementedError(f'{type(self).__name__}.get_messages is not implemented.')

    def add_messages(self, conversation_id, messages):
        """Append messages; fills id and created_at and updates the conversation's updated_at. Returns them."""
        raise NotImplementedError(f'{type(self).__name__}.add_messages is not implemented.')

    def get_conversation(self, conversation_id):
        raise NotImplementedError(f'{type(self).__name__}.get_conversation is not implemented.')

    def save_conversation(self, conversation):
        """Create or update a conversation's fields (title, user_id, metadata)."""
        raise NotImplementedError(f'{type(self).__name__}.save_conversation is not implemented.')

    def list_conversations(self, user_id=None, limit=50):
        """Conversations, most recently updated first; user_id keeps one user's."""
        raise NotImplementedError(f'{type(self).__name__}.list_conversations is not implemented.')

    def delete_conversation(self, conversation_id):
        raise NotImplementedError(f'{type(self).__name__}.delete_conversation is not implemented.')

    def delete_last_messages(self, conversation_id, count=1):
        """Remove the last `count` messages (used to regenerate an answer)."""
        raise NotImplementedError(f'{type(self).__name__}.delete_last_messages is not implemented.')

    @staticmethod
    def _stamp(messages):
        timestamp = now_iso()
        stamped = []
        for message in messages or []:
            item = {
                'id': message.get('id') or new_id(),
                'role': message.get('role'),
                'content': '' if message.get('content') is None else str(message.get('content')),
                'created_at': message.get('created_at') or timestamp,
            }
            if message.get('metadata'):
                item['metadata'] = message['metadata']
            stamped.append(item)
        return stamped

    @staticmethod
    def _conversation(conversation_id, existing, fields=None):
        fields = fields or {}
        existing = existing or {}
        return {
            'id': conversation_id,
            'title': fields['title'] if 'title' in fields else existing.get('title'),
            'user_id': fields['user_id'] if 'user_id' in fields else existing.get('user_id'),
            'created_at': existing.get('created_at') or now_iso(),
            'updated_at': now_iso(),
            'metadata': {**(existing.get('metadata') or {}), **(fields.get('metadata') or {})},
        }

    @staticmethod
    def _newest_first(conversations, limit):
        ordered = sorted(conversations, key=lambda item: str(item.get('updated_at') or ''), reverse=True)
        return ordered[:limit] if limit else ordered


class MemoryChatHistory(ChatHistory):
    """Conversations in process memory (lost on restart)."""

    def __init__(self):
        self.conversations = {}
        self._lock = threading.RLock()

    def _entry(self, conversation_id):
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = {
                'conversation': ChatHistory._conversation(conversation_id, None), 'messages': []}
        return self.conversations[conversation_id]

    def get_messages(self, conversation_id, limit=None):
        with self._lock:
            entry = self.conversations.get(conversation_id)
            if not entry:
                return []
            return list(entry['messages'][-limit:]) if limit else list(entry['messages'])

    def add_messages(self, conversation_id, messages):
        with self._lock:
            entry = self._entry(conversation_id)
            stamped = ChatHistory._stamp(messages)
            entry['messages'].extend(stamped)
            entry['conversation'] = ChatHistory._conversation(conversation_id, entry['conversation'])
            return stamped

    def get_conversation(self, conversation_id):
        with self._lock:
            entry = self.conversations.get(conversation_id)
            if not entry:
                return None
            return {**entry['conversation'], 'message_count': len(entry['messages'])}

    def save_conversation(self, conversation):
        with self._lock:
            entry = self._entry(conversation['id'])
            entry['conversation'] = ChatHistory._conversation(conversation['id'], entry['conversation'], conversation)
            return dict(entry['conversation'])

    def list_conversations(self, user_id=None, limit=50):
        with self._lock:
            items = [dict(entry['conversation']) for entry in self.conversations.values()
                     if not user_id or entry['conversation'].get('user_id') == user_id]
        return ChatHistory._newest_first(items, limit)

    def delete_conversation(self, conversation_id):
        with self._lock:
            self.conversations.pop(conversation_id, None)

    def delete_last_messages(self, conversation_id, count=1):
        with self._lock:
            entry = self.conversations.get(conversation_id)
            if entry and count > 0:
                del entry['messages'][-count:]


class FileChatHistory(ChatHistory):
    """
    Conversations as JSON files in a directory, one file per conversation. Good for local apps, desktop tools and
    development; use FirestoreChatHistory (or your database) for multi-user servers.
    """

    def __init__(self, dir='.intelli/conversations'):
        self.dir = dir
        self._lock = threading.RLock()

    def _file(self, conversation_id):
        safe = re.sub(r'[^A-Za-z0-9_.-]', '_', str(conversation_id))
        if not safe or safe in ('.', '..'):
            raise StoreError(f"Invalid conversation id '{conversation_id}'.")
        return os.path.join(self.dir, f'{safe}.json')

    def _read(self, conversation_id):
        try:
            with open(self._file(conversation_id), 'r', encoding='utf-8') as file:
                data = json.load(file)
        except FileNotFoundError:
            return None
        return {'conversation': from_stored(data.get('conversation')) if data.get('conversation') else None,
                'messages': [from_stored(message) for message in data.get('messages') or []]}

    def _write(self, conversation_id, data):
        os.makedirs(self.dir, exist_ok=True)
        path = self._file(conversation_id)
        stored = {'conversation': to_stored(data['conversation']) if data.get('conversation') else None,
                  'messages': [to_stored(message) for message in data.get('messages') or []]}
        with open(f'{path}.tmp', 'w', encoding='utf-8') as file:
            json.dump(stored, file, indent=2)
        os.replace(f'{path}.tmp', path)

    def get_messages(self, conversation_id, limit=None):
        data = self._read(conversation_id)
        if not data:
            return []
        return data['messages'][-limit:] if limit else data['messages']

    def add_messages(self, conversation_id, messages):
        with self._lock:
            data = self._read(conversation_id) or {'conversation': None, 'messages': []}
            stamped = ChatHistory._stamp(messages)
            data['messages'].extend(stamped)
            data['conversation'] = ChatHistory._conversation(conversation_id, data['conversation'])
            self._write(conversation_id, data)
            return stamped

    def get_conversation(self, conversation_id):
        data = self._read(conversation_id)
        if not data:
            return None
        return {**data['conversation'], 'message_count': len(data['messages'])}

    def save_conversation(self, conversation):
        with self._lock:
            data = self._read(conversation['id']) or {'conversation': None, 'messages': []}
            data['conversation'] = ChatHistory._conversation(conversation['id'], data['conversation'], conversation)
            self._write(conversation['id'], data)
            return dict(data['conversation'])

    def list_conversations(self, user_id=None, limit=50):
        try:
            names = [name for name in os.listdir(self.dir) if name.endswith('.json')]
        except FileNotFoundError:
            return []
        conversations = []
        for name in names:
            try:
                with open(os.path.join(self.dir, name), 'r', encoding='utf-8') as file:
                    data = json.load(file)
            except (OSError, ValueError):
                continue  # a file that is being written or is not a conversation
            conversation = from_stored(data.get('conversation')) if isinstance(data, dict) and data.get(
                'conversation') else None
            if conversation and (not user_id or conversation.get('user_id') == user_id):
                conversations.append(conversation)
        return ChatHistory._newest_first(conversations, limit)

    def delete_conversation(self, conversation_id):
        try:
            os.remove(self._file(conversation_id))
        except FileNotFoundError:
            pass

    def delete_last_messages(self, conversation_id, count=1):
        with self._lock:
            data = self._read(conversation_id)
            if not data or count <= 0:
                return
            del data['messages'][-count:]
            self._write(conversation_id, data)

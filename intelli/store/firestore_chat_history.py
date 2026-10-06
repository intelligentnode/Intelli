# API: Firestore REST v1 (documents:commit, runQuery, runAggregationQuery)
import time

from intelli.store.chat_history import ChatHistory, now_iso, to_stored, from_stored
from intelli.store.google_cloud import GoogleCloudService, Firestore
from intelli.store.vector_store import StoreError

FIRESTORE_BASE = 'https://firestore.googleapis.com/v1'
# Firestore commits take at most 500 writes
BATCH = 500


class FirestoreChatHistory(ChatHistory):
    """
    Conversations in Google Cloud Firestore, for multi-user Gemini / ChatGPT-style apps:

        <collection>/{conversation_id}                 {title, userId, createdAt, updatedAt, metadata}
        <collection>/{conversation_id}/messages/{id}   {id, role, content, createdAt, seq, metadata}

    The documents have the same fields as IntelliNode's FirestoreChatHistory, so Node and Python apps can share
    them. No composite index is needed. Credentials: OAuth (access_token, a credentials object, or
    `gcloud auth application-default login` with google-auth installed).
    """

    def __init__(self, project_id=None, database='(default)', collection='conversations', access_token=None,
                 credentials=None, quota_project_id=None, timeout=120, retries=0, session=None):
        self.service = GoogleCloudService(project_id=project_id, access_token=access_token, credentials=credentials,
                                          quota_project_id=quota_project_id, timeout=timeout, retries=retries,
                                          session=session, label='Firestore')
        self.database = database or '(default)'
        self.collection = collection or 'conversations'

    def _root(self):
        return f'projects/{self.service.project()}/databases/{self.database}/documents'

    def _conversation_name(self, conversation_id):
        return f'{self._root()}/{self.collection}/{Firestore.doc_id(conversation_id)}'

    def _commit(self, writes):
        url = f'{FIRESTORE_BASE}/{self._root()}:commit'
        for start in range(0, len(writes), BATCH):
            self.service.request('POST', url, {'writes': writes[start:start + BATCH]})

    def _run_query(self, parent, structured_query):
        rows = self.service.request('POST', f'{FIRESTORE_BASE}/{parent}:runQuery',
                                    {'structuredQuery': structured_query})
        rows = rows if isinstance(rows, list) else [rows]
        return [row['document'] for row in rows if isinstance(row, dict) and row.get('document')]

    @staticmethod
    def _message(document):
        data = from_stored(Firestore.from_fields(document.get('fields')))
        message = {
            'id': data.get('id') or Firestore.decode_id(document['name']),
            'role': data.get('role'),
            'content': data.get('content') or '',
            'created_at': data.get('created_at'),
        }
        if data.get('metadata'):
            message['metadata'] = data['metadata']
        return message

    def _latest(self, conversation_id, limit):
        """Newest messages first: [(document name, message)]."""
        query = {'from': [{'collectionId': 'messages'}],
                 'orderBy': [{'field': {'fieldPath': 'seq'}, 'direction': 'DESCENDING'}]}
        if limit:
            query['limit'] = limit
        documents = self._run_query(self._conversation_name(conversation_id), query)
        return [(document['name'], self._message(document)) for document in documents]

    def get_messages(self, conversation_id, limit=None):
        return [message for _, message in reversed(self._latest(conversation_id, limit))]

    def add_messages(self, conversation_id, messages):
        conversation_name = self._conversation_name(conversation_id)
        stamped = ChatHistory._stamp(messages)
        base = int(time.time() * 1000) * 100
        writes = [{'update': {
            'name': f"{conversation_name}/messages/{Firestore.doc_id(message['id'])}",
            'fields': Firestore.to_fields({**to_stored(message), 'seq': base + index}),
        }} for index, message in enumerate(stamped)]
        # touching updatedAt only (the conversation keeps its other fields)
        writes.append({'update': {'name': conversation_name, 'fields': Firestore.to_fields({'updatedAt': now_iso()})},
                       'updateMask': {'fieldPaths': ['updatedAt']}})
        self._commit(writes)
        return stamped

    def get_conversation(self, conversation_id):
        name = self._conversation_name(conversation_id)
        try:
            document = self.service.request('GET', f'{FIRESTORE_BASE}/{name}')
        except StoreError as error:
            if error.status_code == 404:
                return None
            raise
        data = from_stored(Firestore.from_fields((document or {}).get('fields')))
        conversation = {
            'id': conversation_id,
            'title': data.get('title'),
            'user_id': data.get('user_id'),
            'created_at': data.get('created_at'),
            'updated_at': data.get('updated_at'),
            'metadata': data.get('metadata') or {},
        }
        try:
            count = self.service.request('POST', f'{FIRESTORE_BASE}/{name}:runAggregationQuery', {
                'structuredAggregationQuery': {
                    'structuredQuery': {'from': [{'collectionId': 'messages'}]},
                    'aggregations': [{'alias': 'count', 'count': {}}],
                },
            })
            aggregate = ((count[0] or {}).get('result') or {}).get('aggregateFields') if isinstance(count, list) \
                and count else None
            if aggregate and aggregate.get('count'):
                conversation['message_count'] = int(Firestore.from_value(aggregate['count']))
        except StoreError:
            pass
        return conversation

    def save_conversation(self, conversation):
        existing = self.get_conversation(conversation['id'])
        merged = ChatHistory._conversation(conversation['id'], existing, conversation)
        fields = to_stored({key: value for key, value in merged.items() if key != 'id'})
        self._commit([{'update': {'name': self._conversation_name(conversation['id']),
                                  'fields': Firestore.to_fields(fields)}}])
        return merged

    def list_conversations(self, user_id=None, limit=50):
        # userId equality without orderBy needs no composite index; sorting happens here
        query = {'from': [{'collectionId': self.collection}]}
        if user_id:
            query.update({'where': Firestore.where({'userId': user_id}), 'limit': 500})
        else:
            query.update({'orderBy': [{'field': {'fieldPath': 'updatedAt'}, 'direction': 'DESCENDING'}],
                          'limit': limit})
        conversations = []
        for document in self._run_query(self._root(), query):
            data = from_stored(Firestore.from_fields(document.get('fields')))
            conversations.append({
                'id': Firestore.decode_id(document['name']),
                'title': data.get('title'),
                'user_id': data.get('user_id'),
                'created_at': data.get('created_at'),
                'updated_at': data.get('updated_at'),
                'metadata': data.get('metadata') or {},
            })
        return ChatHistory._newest_first(conversations, limit)

    def delete_conversation(self, conversation_id):
        writes = [{'delete': name} for name, _ in self._latest(conversation_id, None)]
        writes.append({'delete': self._conversation_name(conversation_id)})
        self._commit(writes)

    def delete_last_messages(self, conversation_id, count=1):
        latest = self._latest(conversation_id, count)
        if latest:
            self._commit([{'delete': name} for name, _ in latest])

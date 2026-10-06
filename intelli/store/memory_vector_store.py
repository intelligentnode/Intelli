import json
import os
import threading

from intelli.store.vector_store import VectorStore, StoreError, matches_filter, cosine_similarity, sorted_matches


class MemoryVectorStore(VectorStore):
    """
    An in-process vector store (exact cosine search). Good for local apps, tests and a few thousand records.
    With path= the records are kept in a JSON file and loaded on first use.

        store = MemoryVectorStore(embedder={'provider': 'openai', 'api_key': key})
        store.add_documents([{'text': 'Intelli supports Gemini.'}])
        hits = store.search('Which models are supported?', 3)
    """

    def __init__(self, embedder=None, path=None):
        super().__init__(embedder)
        self.path = path
        self.records = {}
        self._loaded = not path
        self._lock = threading.RLock()

    def upsert(self, records):
        with self._lock:
            self._load()
            for record in records or []:
                if not isinstance(record.get('vector'), (list, tuple)):
                    raise StoreError(f"Record '{record.get('id')}' has no vector.")
                record_id = str(record.get('id'))
                self.records[record_id] = {'id': record_id, 'vector': list(record['vector']),
                                           'text': record.get('text'), 'metadata': record.get('metadata') or {}}
            self._save()
        return [str(record.get('id')) for record in records or []]

    def query(self, vector=None, text=None, top_k=5, filter=None, native_filter=None):
        """Nearest records; native_filter may also be a function (metadata, record) -> bool."""
        query_vector = self._query_vector(vector, text)
        with self._lock:
            self._load()
            records = list(self.records.values())
        condition = native_filter or filter
        matches = []
        for record in records:
            passes = condition(record['metadata'], record) if callable(condition) else matches_filter(
                record['metadata'], condition)
            if not passes:
                continue
            matches.append({'id': record['id'], 'score': cosine_similarity(query_vector, record['vector']),
                            'text': record['text'], 'metadata': record['metadata']})
        return sorted_matches(matches)[:top_k or 5]

    def get(self, ids):
        with self._lock:
            self._load()
            return [self.records[str(record_id)] for record_id in ids if str(record_id) in self.records]

    def delete(self, ids):
        with self._lock:
            self._load()
            for record_id in ids or []:
                self.records.pop(str(record_id), None)
            self._save()

    def clear(self, filter=None):
        """Delete every record (or only those whose metadata matches filter)."""
        with self._lock:
            self._load()
            if not filter:
                self.records.clear()
            else:
                for record_id in [rid for rid, record in self.records.items()
                                  if matches_filter(record['metadata'], filter)]:
                    del self.records[record_id]
            self._save()

    def count(self):
        with self._lock:
            self._load()
            return len(self.records)

    def _load(self):
        if self._loaded:
            return
        self._loaded = True
        try:
            with open(self.path, 'r', encoding='utf-8') as file:
                data = json.load(file)
        except FileNotFoundError:
            return
        except (OSError, ValueError) as error:
            raise StoreError(f'Could not read the vector store file {self.path}: {error}') from None
        for record in data.get('records') or []:
            self.records[str(record['id'])] = record

    def _save(self):
        if not self.path:
            return
        folder = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(folder, exist_ok=True)
        # write a temporary file first so a crash never leaves a half-written store
        temporary = f'{self.path}.tmp'
        with open(temporary, 'w', encoding='utf-8') as file:
            json.dump({'version': 1, 'records': list(self.records.values())}, file)
        os.replace(temporary, self.path)

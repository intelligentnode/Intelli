# API: https://cloud.google.com/firestore/native/docs/vector-search and the Firestore REST v1 reference
from intelli.store.vector_store import VectorStore, sorted_matches, to_items
from intelli.store.google_cloud import GoogleCloudService, Firestore, FirestoreVector

FIRESTORE_BASE = 'https://firestore.googleapis.com/v1'
# Firestore commits take at most 500 writes
BATCH = 500


class FirestoreVectorStore(VectorStore):
    """
    Vectors in Google Cloud Firestore (native vector search with findNearest), next to your app data.
    Each record is a document {id, text, metadata, embedding}.

    Firestore needs a vector index on the embedding field before the first query, created once
    (store.index_command(768) prints the command):

        gcloud firestore indexes composite create --collection-group=intellinode_vectors --query-scope=COLLECTION \\
          --field-config field-path=embedding,vector-config='{"dimension":"768","flat":"{}"}' --database='(default)'

    A metadata filter needs a composite index that also lists the metadata fields (metadata.<key>).
    Credentials: OAuth (access_token, a credentials object, or `gcloud auth application-default login`); API keys are
    not accepted.
    """

    def __init__(self, project_id=None, database='(default)', collection='intellinode_vectors',
                 vector_field='embedding', distance_measure='COSINE', access_token=None, credentials=None,
                 quota_project_id=None, embedder=None, timeout=120, retries=0, session=None):
        """
        Args:
            distance_measure: COSINE, EUCLIDEAN or DOT_PRODUCT.
        """
        super().__init__(embedder)
        self.service = GoogleCloudService(project_id=project_id, access_token=access_token, credentials=credentials,
                                          quota_project_id=quota_project_id, timeout=timeout, retries=retries,
                                          session=session, label='Firestore')
        self.database = database or '(default)'
        self.collection = collection or 'intellinode_vectors'
        self.vector_field = vector_field or 'embedding'
        self.distance_measure = distance_measure or 'COSINE'

    def _root(self):
        return f'projects/{self.service.project()}/databases/{self.database}/documents'

    def upsert(self, records):
        items = to_items(records)
        root = self._root()
        writes = [{'update': {
            'name': f"{root}/{self.collection}/{Firestore.doc_id(item['id'])}",
            'fields': Firestore.to_fields({'id': item['id'], 'text': item['text'], 'metadata': item['metadata'],
                                           self.vector_field: FirestoreVector(item['vector'])}),
        }} for item in items]
        self._commit(writes)
        return [item['id'] for item in items]

    def query(self, vector=None, text=None, top_k=5, filter=None, native_filter=None):
        query_vector = self._query_vector(vector, text)
        root = self._root()
        where = native_filter or Firestore.where(filter, 'metadata.')
        structured = {'from': [{'collectionId': self.collection}]}
        if where:
            structured['where'] = where
        structured['findNearest'] = {
            'vectorField': {'fieldPath': self.vector_field},
            'queryVector': Firestore.to_value(FirestoreVector(query_vector)),
            'distanceMeasure': self.distance_measure,
            'limit': top_k or 5,
            'distanceResultField': '_distance',
        }
        rows = self.service.request('POST', f'{FIRESTORE_BASE}/{root}:runQuery', {'structuredQuery': structured})
        matches = []
        for row in rows if isinstance(rows, list) else [rows]:
            if not isinstance(row, dict) or not row.get('document'):
                continue
            data = Firestore.from_fields(row['document'].get('fields'))
            matches.append({
                'id': data.get('id') or Firestore.decode_id(row['document']['name']),
                'score': self._score(data.get('_distance')),
                'text': data.get('text'),
                'metadata': data.get('metadata') or {},
            })
        return sorted_matches(matches)

    def _score(self, distance):
        """Firestore returns distances: COSINE is 0..2 (0 = same), EUCLIDEAN grows, DOT_PRODUCT is a similarity."""
        if not isinstance(distance, (int, float)):
            return 0
        if self.distance_measure == 'COSINE':
            return 1 - distance
        if self.distance_measure == 'EUCLIDEAN':
            return 1 / (1 + distance)
        return distance

    def delete(self, ids):
        root = self._root()
        self._commit([{'delete': f'{root}/{self.collection}/{Firestore.doc_id(record_id)}'}
                      for record_id in ids or []])

    def _commit(self, writes):
        if not writes:
            return
        url = f'{FIRESTORE_BASE}/{self._root()}:commit'
        for start in range(0, len(writes), BATCH):
            self.service.request('POST', url, {'writes': writes[start:start + BATCH]})

    def index_command(self, dimension):
        """The gcloud command that creates the vector index this store needs."""
        return (f'gcloud firestore indexes composite create --collection-group={self.collection} '
                f"--query-scope=COLLECTION --field-config field-path={self.vector_field},"
                f"vector-config='{{\"dimension\":\"{dimension}\",\"flat\":\"{{}}\"}}' --database='{self.database}'")

from intelli.store.vector_store import VectorStore, StoreError, to_items, sorted_matches


def to_atlas_filter(filter):
    clauses = [{f'metadata.{key}': {'$in': list(value)} if isinstance(value, (list, tuple)) else {'$eq': value}}
               for key, value in (filter or {}).items()]
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {'$and': clauses}


def to_document(item, text_key, path):
    """The document for a record; a dotted path ('embedding.values') becomes nested fields."""
    document = {text_key: item['text'], 'metadata': item['metadata']}
    keys = path.split('.')
    target = document
    for key in keys[:-1]:
        target[key] = target[key] if isinstance(target.get(key), dict) else {}
        target = target[key]
    target[keys[-1]] = item['vector']
    return document


class MongoDBAtlasVectorStore(VectorStore):
    """
    MongoDB Atlas Vector Search, without a driver dependency of its own: pass a pymongo Collection
    (client[db][collection]).

        from pymongo import MongoClient
        store = MongoDBAtlasVectorStore(collection=MongoClient(uri)['app']['docs'], embedder=embedder)

    Documents are {_id: id, text, metadata, embedding}. The collection needs an Atlas Vector Search index
    (index_name) on `path`; create_index(dimension, filter_fields) creates one. Atlas only filters on paths
    indexed as "filter" fields, so every metadata key used in `filter` must be listed there (as metadata.<key>).
    filter becomes $eq / $in on metadata.<key> (joined by $and); native_filter is a $vectorSearch filter.
    Atlas scores cosine and dotProduct as (1 + similarity) / 2, converted back to the similarity; euclidean keeps
    1 / (1 + distance).
    """

    # API: https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/
    def __init__(self, collection=None, index_name='vector_index', path='embedding', text_key='text',
                 similarity='cosine', num_candidates_multiplier=20, batch_size=1000, embedder=None):
        """
        Args:
            collection: a pymongo Collection.
            similarity: the index similarity (cosine, dotProduct or euclidean).
            num_candidates_multiplier: numCandidates = top_k x this, at most 10,000.
        """
        super().__init__(embedder)
        if collection is None or not hasattr(collection, 'aggregate') or not hasattr(collection, 'bulk_write'):
            raise StoreError('MongoDBAtlasVectorStore needs collection=: a pymongo Collection.')
        self.collection = collection
        self.index_name = index_name or 'vector_index'
        self.path = path or 'embedding'
        self.text_key = text_key or 'text'
        self.similarity = similarity or 'cosine'
        self.num_candidates_multiplier = num_candidates_multiplier or 20
        self.batch_size = batch_size or 1000

    def create_index(self, dimension=None, filter_fields=None):
        """
        Create the Atlas Vector Search index (it builds in the background, so queries return nothing until it is
        ready). filter_fields are metadata keys to filter on.
        """
        if not dimension:
            raise StoreError('create_index needs the vector dimension.')
        fields = [{'type': 'vector', 'path': self.path, 'numDimensions': dimension, 'similarity': self.similarity}]
        fields.extend({'type': 'filter', 'path': f'metadata.{key}'} for key in filter_fields or [])
        model = {'name': self.index_name, 'type': 'vectorSearch', 'definition': {'fields': fields}}
        try:
            return self.collection.create_search_index(model)
        except Exception as error:
            # Atlas indexes only existing collections (NamespaceNotFound): create it empty, then try again
            if getattr(error, 'code', None) != 26 or not hasattr(self.collection, 'database'):
                raise StoreError(f'MongoDB error: {error}') from None
        try:
            self.collection.database.create_collection(self.collection.name)
            return self.collection.create_search_index(model)
        except Exception as error:
            raise StoreError(f'MongoDB error: {error}') from None

    def upsert(self, records):
        try:
            from pymongo import ReplaceOne
        except ImportError:
            raise StoreError('MongoDBAtlasVectorStore needs pymongo: pip install pymongo') from None
        items = to_items(records)
        for start in range(0, len(items), self.batch_size):
            operations = [ReplaceOne({'_id': item['id']}, to_document(item, self.text_key, self.path), upsert=True)
                          for item in items[start:start + self.batch_size]]
            try:
                self.collection.bulk_write(operations, ordered=True)
            except Exception as error:
                raise StoreError(f'MongoDB error: {error}') from None
        return [item['id'] for item in items]

    def query(self, vector=None, text=None, top_k=5, filter=None, native_filter=None):
        query_vector = self._query_vector(vector, text)
        limit = int(top_k or 5)
        stage = {
            'index': self.index_name,
            'path': self.path,
            'queryVector': query_vector,
            'numCandidates': min(10000, max(limit, limit * self.num_candidates_multiplier)),
            'limit': limit,
        }
        condition = native_filter or to_atlas_filter(filter)
        if condition:
            stage['filter'] = condition
        pipeline = [
            {'$vectorSearch': stage},
            {'$project': {'_id': 1, self.text_key: 1, 'metadata': 1, 'score': {'$meta': 'vectorSearchScore'}}},
        ]
        try:
            rows = list(self.collection.aggregate(pipeline))
        except Exception as error:
            raise StoreError(f'MongoDB error: {error}') from None
        matches = [{
            'id': str(row.get('_id')),
            'score': row['score'] if self.similarity == 'euclidean' else 2 * row['score'] - 1,
            'text': row.get(self.text_key),
            'metadata': row.get('metadata') or {},
        } for row in rows]
        return sorted_matches(matches)

    def delete(self, ids):
        items = [str(record_id) for record_id in ids or []]
        if not items:
            return
        try:
            self.collection.delete_many({'_id': {'$in': items}})
        except Exception as error:
            raise StoreError(f'MongoDB error: {error}') from None

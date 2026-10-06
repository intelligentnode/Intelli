import json
import math
import re
import threading

from intelli.store.vector_store import VectorStore, StoreError, to_items, sorted_matches

# table or schema.table: identifier characters only, since the name goes into the SQL text
TABLE_NAME = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$')
# pgvector indexes the vector type with HNSW up to 2,000 dimensions
MAX_INDEXED_DIMENSIONS = 2000


def to_vector_literal(vector):
    values = []
    for value in vector:
        number = float(value)
        if not math.isfinite(number):
            raise StoreError(f'pgvector cannot store the value {value}.')
        values.append(repr(number) if not number.is_integer() else str(int(number)))
    return '[' + ','.join(values) + ']'


class PgVectorStore(VectorStore):
    """
    PostgreSQL with the pgvector extension (also AlloyDB, Cloud SQL, Supabase, Neon), without a driver dependency:
    pass a DB-API connection, such as psycopg (3) or psycopg2.

        import psycopg
        store = PgVectorStore(connection=psycopg.connect(url), dimension=1536, embedder=embedder)

    With create_table the first use runs CREATE EXTENSION IF NOT EXISTS vector and creates the table
    (id text primary key, text, metadata jsonb, embedding vector(dimension)) plus an HNSW cosine index when the
    dimension is at most 2,000. Scores are 1 - cosine distance. filter matches with metadata @> (lists mean
    one of); native_filter is a SQL condition string, or {'sql', 'params'} with its own %s placeholders.
    """

    # API: https://github.com/pgvector/pgvector
    def __init__(self, connection=None, table='intellinode_vectors', dimension=None, create_table=True,
                 create_index=True, batch_size=500, embedder=None):
        """
        Args:
            connection: a DB-API connection (psycopg / psycopg2), or any object with cursor().
            table: table name, or schema.table.
        """
        super().__init__(embedder)
        if connection is None or not hasattr(connection, 'cursor'):
            raise StoreError('PgVectorStore needs connection=: a psycopg or psycopg2 connection, or any DB-API '
                             'connection.')
        self.connection = connection
        self.table = table or 'intellinode_vectors'
        if not TABLE_NAME.match(self.table):
            raise StoreError(f"Invalid table name '{self.table}': use letters, digits and underscores.")
        self.dimension = dimension
        self.create_table = create_table is not False
        self.create_index = create_index is not False
        self.batch_size = batch_size or 500
        self._ready = False
        self._lock = threading.Lock()

    def ensure_table(self, dimension=None):
        """Create the extension, the table and its index when missing."""
        size = dimension or self.dimension
        if not isinstance(size, int) or size <= 0:
            raise StoreError('PgVectorStore needs a dimension to create its table.')
        self._execute('CREATE EXTENSION IF NOT EXISTS vector')
        self._execute(f'''CREATE TABLE IF NOT EXISTS {self.table} (
      id text PRIMARY KEY,
      text text,
      metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,
      embedding vector({size}) NOT NULL
    )''')
        if self.create_index and size <= MAX_INDEXED_DIMENSIONS:
            index_name = f"{self.table.split('.')[-1]}_embedding_idx"
            self._execute(f'CREATE INDEX IF NOT EXISTS {index_name} ON {self.table} '
                          'USING hnsw (embedding vector_cosine_ops)')

    def upsert(self, records):
        # one INSERT cannot touch the same row twice: the last record of an id wins
        items = list({item['id']: item for item in to_items(records)}.values())
        if not items:
            return []
        self._prepare(len(items[0]['vector']))
        for start in range(0, len(items), self.batch_size):
            rows, params = [], []
            for item in items[start:start + self.batch_size]:
                rows.append('(%s, %s, %s::jsonb, %s::vector)')
                params.extend([item['id'], item['text'], json.dumps(item['metadata']),
                               to_vector_literal(item['vector'])])
            self._execute(f'''INSERT INTO {self.table} (id, text, metadata, embedding) VALUES {', '.join(rows)}
        ON CONFLICT (id) DO UPDATE SET text = EXCLUDED.text, metadata = EXCLUDED.metadata,
        embedding = EXCLUDED.embedding''', params)
        return [str(record.get('id')) for record in records or []]

    def query(self, vector=None, text=None, top_k=5, filter=None, native_filter=None):
        query_vector = to_vector_literal(self._query_vector(vector, text))
        if self.dimension:
            self._prepare(self.dimension)
        conditions, params = [], []
        if native_filter:
            native = {'sql': native_filter, 'params': []} if isinstance(native_filter, str) else native_filter
            conditions.append(f"({native['sql']})")
            params.extend(native.get('params') or [])
        elif filter:
            equal = {}
            for key, value in filter.items():
                if not isinstance(value, (list, tuple)):
                    equal[key] = value
                    continue
                # a jsonb array contains a scalar member, so this reads "metadata.key is one of value"
                conditions.append('%s::jsonb @> (metadata -> %s)')
                params.extend([json.dumps(list(value)), key])
            if equal:
                conditions.append('metadata @> %s::jsonb')
                params.append(json.dumps(equal))
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ''
        sql = (f'SELECT id, text, metadata, 1 - (embedding <=> %s::vector) AS score FROM {self.table}{where} '
               f'ORDER BY embedding <=> %s::vector LIMIT %s')
        rows = self._execute(sql, [query_vector, *params, query_vector, int(top_k or 5)], fetch=True)
        matches = []
        for row in rows:
            metadata = row['metadata']
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            matches.append({'id': str(row['id']), 'score': float(row['score']), 'text': row['text'],
                            'metadata': metadata or {}})
        return sorted_matches(matches)

    def delete(self, ids):
        items = [str(record_id) for record_id in ids or []]
        if items:
            self._execute(f'DELETE FROM {self.table} WHERE id = ANY(%s)', [items])

    def _prepare(self, dimension):
        if not self.create_table:
            return
        with self._lock:
            if not self._ready:
                self.ensure_table(self.dimension or dimension)
                self._ready = True

    def _execute(self, sql, params=None, fetch=False):
        """Run one statement in its own transaction; returns rows as dicts when fetch=True."""
        cursor = self.connection.cursor()
        try:
            if params is None:
                cursor.execute(sql)
            else:
                cursor.execute(sql, params)
            rows = None
            if fetch:
                columns = [column[0] for column in cursor.description]
                rows = [row if isinstance(row, dict) else dict(zip(columns, row)) for row in cursor.fetchall()]
            if hasattr(self.connection, 'commit'):
                self.connection.commit()
            return rows
        except StoreError:
            raise
        except Exception as error:
            if hasattr(self.connection, 'rollback'):
                try:
                    self.connection.rollback()
                except Exception:
                    pass
            raise StoreError(f'pgvector error: {error}') from None
        finally:
            try:
                cursor.close()
            except Exception:
                pass

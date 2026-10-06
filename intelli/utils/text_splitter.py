import re

# Separators tried in order: paragraphs, lines, sentences, words.
SEPARATORS = ['\n\n', '\n', '. ', '? ', '! ', '; ', ', ', ' ']


class TextSplitter:
    """
    Split long text into overlapping chunks for embeddings and RAG. Chunks break at paragraphs, then lines, then
    sentences, then words, so a chunk rarely cuts a sentence in half. Sizes are in characters.

        TextSplitter.split(text, chunk_size=1200, chunk_overlap=150)
        TextSplitter.to_documents(text, {'source': 'handbook.md'})  # [{'id', 'text', 'metadata': {'source', 'chunk'}}]
    """

    @staticmethod
    def split(text, chunk_size=1200, chunk_overlap=150):
        clean = str(text or '').replace('\r\n', '\n').strip()
        if not clean:
            return []
        if chunk_overlap >= chunk_size:
            raise ValueError('chunk_overlap must be smaller than chunk_size.')
        chunks = []
        current = ''
        for piece in TextSplitter._pieces(clean, chunk_size, 0):
            if current and len(current) + len(piece) > chunk_size:
                chunks.append(current.strip())
                # start the next chunk with the tail of the previous one
                tail = current[max(0, len(current) - chunk_overlap):]
                boundary = re.search(r'\s', tail)
                current = tail[boundary.start() + 1:] if chunk_overlap > 0 and boundary else ''
            current += piece
        if current.strip():
            chunks.append(current.strip())
        return chunks

    @staticmethod
    def _pieces(text, size, level):
        """Pieces no longer than size, each keeping its trailing separator."""
        if len(text) <= size:
            return [text]
        if level >= len(SEPARATORS):
            return [text[start:start + size] for start in range(0, len(text), size)]
        separator = SEPARATORS[level]
        split = text.split(separator)
        if len(split) == 1:
            return TextSplitter._pieces(text, size, level + 1)
        pieces = []
        for index, part in enumerate(split):
            piece = part + separator if index < len(split) - 1 else part
            if not piece:
                continue
            if len(piece) > size:
                pieces.extend(TextSplitter._pieces(piece, size, level + 1))
            else:
                pieces.append(piece)
        return pieces

    @staticmethod
    def to_documents(text, metadata=None, chunk_size=1200, chunk_overlap=150, id_prefix=None):
        """Chunks as vector store documents: [{'id', 'text', 'metadata': {**metadata, 'chunk': n}}]."""
        metadata = metadata or {}
        prefix = id_prefix or (str(metadata['source']) if metadata.get('source') else None)
        documents = []
        for index, chunk in enumerate(TextSplitter.split(text, chunk_size, chunk_overlap)):
            document = {'text': chunk, 'metadata': {**metadata, 'chunk': index}}
            if prefix:
                document = {'id': f'{prefix}#{index}', **document}
            documents.append(document)
        return documents

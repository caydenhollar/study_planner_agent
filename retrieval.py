"""Load local notes, split them, and retrieve passages with source IDs."""

from collections import Counter
from copy import deepcopy
import hashlib
import json
from math import sqrt
from pathlib import Path
import re

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_notes(folder):
    folder = Path(folder)
    splitter = RecursiveCharacterTextSplitter(chunk_size=550, chunk_overlap=70,
                                              add_start_index=True)
    chunks = []
    documents = []
    for path in sorted(folder.rglob('*')):
        if path.suffix.lower() not in {'.md', '.txt'} or not path.is_file():
            continue
        raw = path.read_bytes()
        text = raw.decode('utf-8')
        source = path.relative_to(folder).as_posix()
        version = hashlib.sha256(raw).hexdigest()
        documents.append({'source': source, 'sha256': version, 'bytes': len(raw)})
        for number, doc in enumerate(splitter.create_documents([text]), start=1):
            doc.metadata.update(source=source, source_id=f'{source}#{number}',
                                document_sha256=version,
                                line=text[:doc.metadata['start_index']].count('\n') + 1)
            chunks.append(doc)
    if not chunks:
        raise ValueError('Notes folder must contain nonempty .md or .txt files.')
    encoded = json.dumps(documents, sort_keys=True, separators=(',', ':')).encode('utf-8')
    manifest = {'algorithm': 'sha256', 'corpus_sha256': hashlib.sha256(encoded).hexdigest(),
                'documents': documents}
    return chunks, manifest


def load_chunks(folder):
    return load_notes(folder)[0]


def keywords(text):
    stop = {'a', 'an', 'the', 'to', 'of', 'and', 'in', 'for', 'is', 'it', 'how',
            'do', 'i', 'my', 'what', 'should', 'with', 'on', 'this', 'are'}
    return Counter(t for t in re.findall(r'[a-z0-9]+', text.lower()) if t not in stop)


def lexical_score(query, text):
    a, b = keywords(query), keywords(text)
    denominator = sqrt(sum(v*v for v in a.values()) * sum(v*v for v in b.values()))
    return sum(v * b[t] for t, v in a.items()) / denominator if denominator else 0.0


class NotesLibrary:
    def __init__(self, folder, embeddings=None, min_score=None):
        self.chunks, self.manifest = load_notes(folder)
        self.backend = 'semantic' if embeddings is not None else 'keyword'
        self.min_score = (0.35 if embeddings is not None else 0.10) if min_score is None else min_score
        self.store = (InMemoryVectorStore.from_documents(self.chunks, embeddings)
                      if embeddings is not None else None)
        self.cache = {}

    def search(self, query, k=2):
        if not query.strip() or k < 1:
            return []
        key = (query, k)
        if key not in self.cache:
            if self.store is not None:
                matches = self.store.similarity_search_with_score(query, k=k)
            else:
                matches = sorted(((d, lexical_score(query, d.page_content)) for d in self.chunks),
                                 key=lambda pair: pair[1], reverse=True)[:k]
            self.cache[key] = [
                {**doc.metadata, 'text': doc.page_content, 'score': round(float(score), 4)}
                for doc, score in matches if score >= self.min_score
            ]
        return deepcopy(self.cache[key])


def ollama_library(folder, model='all-minilm', base_url='http://localhost:11434'):
    from langchain_ollama import OllamaEmbeddings

    embeddings = OllamaEmbeddings(model=model, base_url=base_url,
                                  client_kwargs={'timeout': 120})
    return NotesLibrary(folder, embeddings)

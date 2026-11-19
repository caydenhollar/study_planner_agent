import json
from pathlib import Path
import shutil

from langchain_core.embeddings import Embeddings
import pytest

from planner import ollama_choice, plan, render_report
from retrieval import NotesLibrary, load_chunks

NOTES = Path(__file__).parents[1] / 'notes'


class TinyEmbeddings(Embeddings):
    """Fixed geometry for unit tests, not a substitute for real-model evaluation."""
    def embed_query(self, text):
        vector = [float(word in text.lower()) for word in ('statistics', 'history', 'python', 'unrelated')]
        return vector + [float(not any(vector))]

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]


def test_vector_search_preserves_passages_and_sources():
    library = NotesLibrary(NOTES, embeddings=TinyEmbeddings())
    results = library.search('statistics')
    assert results and all(r['source'] == 'statistics.md' for r in results)
    assert all('confidence' in r['text'] or 'hypothesis' in r['text'] for r in results)
    assert all(r['source_id'].startswith('statistics.md#') for r in results)
    assert library.search('unrelated topic') == []


def test_keyword_mode_and_no_match():
    library = NotesLibrary(NOTES)
    assert library.search('Python lab unit tests')[0]['source'] == 'python.md'
    assert library.search('volcanic lava submarine') == []
    assert library.search('   ') == []


def test_search_results_cannot_mutate_the_index_or_cache():
    library = NotesLibrary(NOTES)
    results = library.search('statistics problem set')
    results[0]['text'] = 'Changed by caller'
    assert library.search('statistics problem set')[0]['text'] != 'Changed by caller'


def test_chunks_are_bounded_and_locatable(tmp_path):
    text = '\n\n' + '\n\n'.join(f'Paragraph {i}: ' + 'confidence interval ' * 30 for i in range(6))
    (tmp_path / 'long.md').write_text(text)
    chunks = load_chunks(tmp_path)
    assert len(chunks) > 6
    assert all(len(c.page_content) <= 550 for c in chunks)
    assert len({c.metadata['source_id'] for c in chunks}) == len(chunks)
    for chunk in chunks:
        start = chunk.metadata['start_index']
        assert text[start:start + len(chunk.page_content)] == chunk.page_content
        assert chunk.metadata['line'] == text[:start].count('\n') + 1


def test_empty_or_missing_library_is_an_error(tmp_path):
    (tmp_path / 'empty.md').write_text('  ')
    with pytest.raises(ValueError, match='Notes folder'):
        NotesLibrary(tmp_path)
    with pytest.raises(ValueError):
        NotesLibrary(tmp_path / 'missing')


def assignment():
    return {'id': 'stats', 'title': 'Statistics problem set', 'due': '2026-11-24',
            'minutes': 30, 'priority': 5}


def test_graph_retrieves_before_deciding_and_saves_evidence():
    library = NotesLibrary(NOTES)
    result = plan([assignment()], {'2026-11-23': 30}, retrieve=library.search,
                  notes_manifest=library.manifest)
    session = result['sessions'][0]
    assert session['sources'][0]['source'] == 'statistics.md'
    assert session['study_goal'] in session['sources'][0]['text']
    assert [t['node'] for t in result['trace']][:4] == [
        'inspect_day', 'retrieve_notes', 'choose_task', 'book_session']
    versions = {d['source']: d['sha256'] for d in result['notes_manifest']['documents']}
    assert session['sources'][0]['document_sha256'] == versions['statistics.md']
    assert result['notes_manifest']['corpus_sha256'] in render_report(result)


@pytest.mark.parametrize('citations', [['made-up.md#1'], [], ['history.md#1'], 'statistics.md#1'])
def test_model_cannot_use_missing_or_unretrieved_citations(citations):
    def choose(*args):
        return {'task_id': 'stats', 'reason': 'Urgent', 'study_goal': 'Practice intervals',
                'source_ids': citations}
    with pytest.raises(ValueError, match='retrieved sources'):
        plan([assignment()], {'2026-11-23': 30}, choose=choose,
             retrieve=NotesLibrary(NOTES).search)


def test_no_evidence_is_reported_without_invented_sources():
    task = {**assignment(), 'title': 'Volcanic lava submarine'}
    result = plan([task], {'2026-11-23': 30}, retrieve=NotesLibrary(NOTES).search)
    session = result['sessions'][0]
    assert session['sources'] == []
    assert 'no matching notes' in session['study_goal'].lower()


def test_retrieved_passages_are_sent_to_the_language_model(monkeypatch):
    captured = {}

    class FakeChat:
        def __init__(self, **kwargs):
            pass

        def with_structured_output(self, schema, **kwargs):
            assert 'source_ids' in schema['required']
            return self

        def invoke(self, messages):
            captured.update(json.loads(messages[-1][1]))
            return {'task_id': 'stats', 'reason': 'Urgent', 'study_goal': 'Practice intervals',
                    'source_ids': ['statistics.md#1']}

    monkeypatch.setattr('langchain_ollama.ChatOllama', FakeChat)
    choose = ollama_choice('test-model', 'http://localhost:11434')
    plan([assignment()], {'2026-11-23': 30}, choose=choose,
         retrieve=NotesLibrary(NOTES).search)
    assert captured['retrieved_notes']['stats'][0]['source_id'] == 'statistics.md#1'
    assert 'confidence intervals' in captured['retrieved_notes']['stats'][0]['text']


@pytest.mark.parametrize('change', ['Replacement statistics content.', ' ', '\r\n'])
def test_same_filename_edits_change_document_and_corpus_versions(tmp_path, change):
    path = tmp_path / 'statistics.md'
    path.write_bytes(b'Statistics confidence intervals.')
    before = NotesLibrary(tmp_path)
    saved = before.search('Statistics')[0]
    path.write_bytes(path.read_bytes() + change.encode('utf-8'))
    after = NotesLibrary(tmp_path)
    assert before.manifest['corpus_sha256'] != after.manifest['corpus_sha256']
    assert saved['document_sha256'] != after.search('Statistics')[0]['document_sha256']
    assert saved['source_id'] == after.search('Statistics')[0]['source_id']
    assert before.search('Statistics')[0] == saved  # The original run stays reproducible.


def test_identical_notes_have_portable_versions(tmp_path):
    copied = tmp_path / 'different' / 'location'
    shutil.copytree(NOTES, copied)
    original = NotesLibrary(NOTES)
    other = NotesLibrary(copied)
    assert original.manifest == other.manifest
    assert original.search('Statistics problem set') == other.search('Statistics problem set')
    assert all(not Path(d['source']).is_absolute() for d in original.manifest['documents'])


def test_manifest_tracks_added_empty_documents_and_renamed_sources(tmp_path):
    (tmp_path / 'statistics.md').write_text('Statistics confidence intervals.')
    first = NotesLibrary(tmp_path).manifest
    (tmp_path / 'empty.txt').write_text('')
    second = NotesLibrary(tmp_path).manifest
    assert first['corpus_sha256'] != second['corpus_sha256']
    assert any(d['source'] == 'empty.txt' and d['bytes'] == 0 for d in second['documents'])
    (tmp_path / 'statistics.md').rename(tmp_path / 'renamed.md')
    third = NotesLibrary(tmp_path).manifest
    assert second['corpus_sha256'] != third['corpus_sha256']
    assert first['documents'][0]['sha256'] == next(
        d['sha256'] for d in third['documents'] if d['source'] == 'renamed.md')


@pytest.mark.parametrize('tasks', [[], [{**assignment(), 'title': 'Volcanic lava submarine'}]])
def test_plan_records_manifest_even_without_retrieved_evidence(tasks):
    library = NotesLibrary(NOTES)
    result = plan(tasks, {'2026-11-23': 30}, retrieve=library.search,
                  notes_manifest=library.manifest)
    assert result['notes_manifest'] == library.manifest
    assert not any(s['sources'] for s in result['sessions'])
    library.manifest['documents'].clear()
    assert result['notes_manifest']['documents']  # Saved plans own their manifest.

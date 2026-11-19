"""Check retrieval on a small, hand-written set of expected source matches."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from retrieval import NotesLibrary, ollama_library


def evaluate(library, cases):
    rows = []
    for case in cases:
        start = perf_counter()
        hits = library.search(case['query'])
        sources = [h['source'] for h in hits]
        rank = next((i + 1 for i, source in enumerate(sources) if source == case['source']), None)
        rows.append({**case, 'retrieved_sources': sources,
                     'scores': [h['score'] for h in hits],
                     'matched': bool(rank) if case['source'] else not hits,
                     'reciprocal_rank': 1 / rank if rank else 0,
                     'latency_ms': round((perf_counter() - start) * 1000, 2)})
    answerable = [r for r in rows if r['source']]
    unknown = [r for r in rows if not r['source']]
    return {'backend': library.backend, 'chunks': len(library.chunks), 'k': 2,
            'notes_manifest': library.manifest,
            'min_score': library.min_score,
            'note': 'Small example checks, not a general quality benchmark. Generation quality is not scored.',
            'source_recall_at_2': sum(r['matched'] for r in answerable) / len(answerable),
            'mean_reciprocal_rank': sum(r['reciprocal_rank'] for r in answerable) / len(answerable),
            'no_match_accuracy': sum(r['matched'] for r in unknown) / len(unknown),
            'cases': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--notes', default='notes')
    parser.add_argument('--embedding-model', default='all-minilm')
    parser.add_argument('--base-url', default='http://localhost:11434')
    parser.add_argument('--output', default='artifacts/retrieval_metrics.json')
    args = parser.parse_args()
    library = NotesLibrary(args.notes) if args.offline else ollama_library(
        args.notes, args.embedding_model, args.base_url)
    cases = json.loads(Path('examples/retrieval_checks.json').read_text())
    result = evaluate(library, cases)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, indent=2))


if __name__ == '__main__':
    main()

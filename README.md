# Study Planner with RAG and LangGraph

A Python agent that schedules assignments and suggests what to study using local
course notes. **LangGraph** runs inspect → retrieve notes → choose → book → repeat.
**LangChain** splits documents, **Ollama all-minilm** creates embeddings, and an
**in-memory vector store** finds relevant passages. **Qwen** uses those passages to
generate a study goal with source citations. Python enforces deadlines and time limits.

## Run

Use Python 3.12. From this folder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python planner.py --offline
python -m pytest -q
```

For real RAG, install/start [Ollama](https://ollama.com), then run:

```bash
ollama pull qwen2.5:3b
ollama pull all-minilm
python planner.py --output artifacts/llm_demo
python evaluate_rag.py
```

Edit assignments and daily available minutes in `examples/week.json`. Add your own
`.md` or `.txt` files to `notes/`, or pass `--notes another_folder`. The included
notes are examples written for this project. Priority is 1–5, deadlines include
the whole day, and sessions default to 30 minutes (`--chunk` changes this).
Offline mode uses keyword retrieval and extracted text through the same graph;
it does not use embeddings or an LLM. On Windows, activate `.venv\Scripts\activate`.

## Docker

Build from this folder. All unit tests run during the build; the default demo is offline.

```bash
docker build -t cayden-study-planner .
docker run --rm -v "${PWD}/artifacts:/app/artifacts" cayden-study-planner
docker run --rm --network none cayden-study-planner python -m pytest -q
```

For real RAG, start host Ollama and pull both models first:

```bash
docker run --rm --add-host host.docker.internal:host-gateway \
  -v "${PWD}/artifacts:/app/artifacts" cayden-study-planner \
  python planner.py --base-url http://host.docker.internal:11434 --output artifacts/llm_demo
```

Rebuild after editing notes or assignments. Models are provided by host Ollama;
they are not bundled in the image. Use `--model`, `--embedding-model`, or
`--base-url` to change the defaults.

## Results

`artifacts/plan.md` and `plan.json` contain the offline schedule, study goals,
retrieved passages, source IDs, and graph trace. `artifacts/llm_demo/` has an
actual model-driven run. Both demos scheduled 360 minutes; unfinished work is flagged.
The vector index is rebuilt per run and repeated searches are cached in memory.
Each plan also saves a `notes_manifest`: SHA256 versions of the original document
bytes and the whole corpus, using relative paths. Cited passages include their
document version, so an edit to the same filename is detectable. Even whitespace
changes affect the version; copying identical notes to a new folder does not.

`evaluate_rag.py` checks 10 source-matching questions plus 2 unrelated questions.
The saved semantic run found the right source for 9/10 and rejected both unrelated
queries. This small demo check does not measure general answer quality. The offline
baseline is saved separately; rerun it with:

```bash
python evaluate_rag.py --offline --output artifacts/retrieval_keyword_metrics.json
```

Tests cover scheduling, retrieval, source metadata, invalid citations, and missing
notes. Citation validation checks source IDs, not the truth of every generated
claim. Model choices can be suboptimal; server errors are surfaced, not hidden.

References: [LangChain retrieval](https://docs.langchain.com/oss/python/deepagents/retrieval),
[Ollama embeddings](https://docs.langchain.com/oss/python/integrations/embeddings/ollama).

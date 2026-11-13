# HR Assistant — Week 2: web interface + FastAPI on top of the HR RAG engine

**Intelligent HR Support.** A privately hosted assistant that answers questions **only from your HR documents** and shows
the excerpts, file names and pages behind every answer.

```
Browser (frontend/)  ──fetch──►  FastAPI (backend/)  ──►  existing RAG (src/)  ──►  BGE-M3 + ChromaDB + optional local mMARCO reranker + GroqCloud
```

The existing RAG engine, ingestion pipeline, retrieval logic and citations remain in place. An optional local CrossEncoder reranks Chroma candidates before they are passed to Groq.

## 1. What is new in Week 2

| Part | What it does |
|---|---|
| `backend/` | FastAPI app: chat, document, health, and persistent conversation-history endpoints |
| `backend/conversation.py` | In-memory conversation context, so follow-ups like "and for juniors?" work |
| `backend/history.py` | SQLite-backed browser-visible conversation history (`storage/conversations.sqlite3`) |
| `frontend/` | HTML5 + Tailwind (CDN) + JavaScript + Fetch: welcome screen, chat with sources and excerpts, Markdown, copy, conversation history, HR Knowledge Base page with uploads, FAQ, help, settings |
| Access control | Employee CSV data and all uploads are locked behind an administrator key (section 3) |
| `tests/test_api.py` | 62 API tests (what is mocked is stated at the top of the file) |
| `tests/test_providers.py` | Mocked provider tests for local embeddings and Groq; no Groq key required |

### How the API reuses the existing code (nothing re-implemented)

| API action | Existing code that does the work |
|---|---|
| Startup | `load_settings()`, local `get_embeddings()`, optional CPU reranker, Groq `get_llm()`, `VectorStoreManager(...)`, `Retriever(...)`, `RAGPipeline(retriever, llm)` — built **once** and shared by all requests |
| `POST /api/chat` | `RAGPipeline.ask(question, top_k, where)` → returns `AnswerResult`; sources come from `result.retrieval.chunks` (`RetrievedChunk`: file, page / record id, score, text) |
| `POST /api/documents/upload` | `load_file()` → `process_documents()` → `verify_embedding_service()` → `VectorStoreManager.add_documents()` |
| `GET /api/documents` | `VectorStoreManager.list_sources()` (real chunk counts) + the real files in `data/raw/` |
| `GET /api/health` | Groq model availability, local embedding status, legacy Ollama fields, and a real read of the Chroma collection |

**Changes to the Week 1 modules (small, backward compatible, all existing tests still pass):**

1. `vector_store.py`, `retriever.py`, `rag_pipeline.py`: an optional `where=` metadata filter is passed through (default `None` = old behaviour). Chroma applies it **inside the database**, which is how employee data is kept away from anonymous callers.
2. `document_loader.py`: PDF tables are rendered one row per line with column names (`Model: Office First | Office Days: 5 days/week | Remote Days: 0`), because a flattened table made the 3B model misread rules (it told you Sami Taha could work remotely; he is on probation and "Office First" means 0 remote days). Placeholder PDF titles such as "(anonymous)" are ignored. CSV files get small per-column summary records (for "list all managers"). `is_ignored_file` is a public alias.
3. `prompts.py`: stricter citation and table-reading rules.

> You mentioned an existing "in-memory conversation context". The Week 1 code has none (`RAGPipeline.ask` takes one self-contained question), so Week 2 adds it in `backend/conversation.py` without touching the pipeline.

## 2. Run it on Windows 11 (VS Code or Anaconda)

1. **Get the project source** by cloning your GitHub repository (replace `OWNER/REPOSITORY` with its actual path), then enter the project root:
   ```powershell
   git clone https://github.com/OWNER/REPOSITORY.git hr_policy_rag
   Set-Location hr_policy_rag
   ```
   If you have already cloned it, run the following commands from the repository root.
2. **Environment** (PowerShell in that folder; or Anaconda Prompt with `conda create -n hr-rag python=3.11 -y` / `conda activate hr-rag` instead of the first two lines):
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   if (-not (Test-Path .env)) { Copy-Item .env.example .env }
   ```
3. Set `GROQ_API_KEY` in `.env` using a key from your GroqCloud account. Keep it private. The configured default is `GROQ_MODEL=openai/gpt-oss-120b`.
4. **Create an administrator key** (needed to upload documents and to unlock employee data) and put it in `.env`:
   ```powershell
   python -c "import secrets; print(secrets.token_urlsafe(24))"
   ```
   Open `.env` and set `ADMIN_API_KEY=<the printed value>`. Keep it private.
5. **Only for a fresh/empty index:** index the bundled policies (this downloads/loads local BGE-M3):
   ```powershell
   python scripts\ingest.py
   ```
   Do not run ingestion to migrate an existing compatible Chroma index.
6. **Start the app:**
   ```powershell
   python scripts\run_server.py
   ```
   Open <http://127.0.0.1:8000>. Stop with `Ctrl+C`.

**Linux / macOS:** same steps with `python3.11 -m venv .venv`, `source .venv/bin/activate`, `if [ ! -f .env ]; then cp .env.example .env; fi`, `python3 scripts/ingest.py` only for a fresh index, and `python3 scripts/run_server.py`.

Alternative start command: `python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000` (from the project root).

**Data handling:** Sentence Transformers embeddings are generated locally, and PDFs plus Chroma remain on the server. For each answer, the question and retrieved HR excerpts are sent to GroqCloud for generation. Confirm this is allowed by your organization before sending sensitive employee or policy content. The browser also loads Tailwind CSS, `marked` and `DOMPurify` from public CDNs (and an optional Google font).

## 3. Security model (read this before using employee data)

* **Employee CSV files are restricted.** Without a valid `X-Admin-Key` header, the chat **cannot retrieve** CSV chunks (the filter runs inside Chroma, so they never reach the prompt), `GET /api/documents` and `GET /api/health` do not list or count them, and CSV uploads are refused. Even if a CSV was indexed earlier with `scripts/ingest.py`, the API still hides it.
* **Uploads need the admin key by default** (`UPLOADS_REQUIRE_ADMIN=true`), so nobody can add or overwrite policies anonymously. Set it to `false` only for a private test on your own PC.
* If `ADMIN_API_KEY` is empty (the default), nobody is admin: uploads and employee data are switched off.
* The key is compared in constant time, must have at least 12 characters, and in the browser it is kept in `sessionStorage` (cleared when the tab closes or on **Logout**).
* Conversation history is shared by every browser connected to this FastAPI server. As requested, the history API has no authentication or per-user separation; **Logout** only clears administrator access and never deletes conversations.
* Upload hardening: only `.pdf/.docx/.csv`, file names sanitized (no paths), size limit (`MAX_UPLOAD_MB`), the file is validated and processed in a temporary folder and only stored/indexed if everything succeeds, one index change at a time.
* Conversation memory is bound to the access level, so an admin conversation is never reused for a non-admin request.

**What this is not:** there are no user accounts, no per-user permissions, no audit log and no rate limiting. One shared admin key is fine for a single-user local setup; before exposing real employee data to several people, put a real login (for example your company SSO) in front of it. Documents are trusted input: an uploaded PDF could contain text that tries to steer the model, which is another reason uploads are admin-only.

### Using the employee CSV

1. Copy `employees.csv` into `data/raw/` (it is git-ignored).
2. In the web app: **Settings → Administrator access → enter the key → Unlock**, then **HR Knowledge Base** → upload the CSV (or run `python scripts\ingest.py`).
3. Questions about individuals now use it; the document list shows it with a red "Restricted" badge. **Logout** locks it again.

## 4. Using the interface

* **Get Started / New Chat:** ask a question. Each answer shows citation verification (green *Citation vérifiée* / orange *Citation non vérifiée*) and has *Sources (n)*: file name, page (or record id), relevance and the retrieved excerpt (click an excerpt to expand it). Follow-ups are understood in context; when a question was rewritten the footer shows "Understood as: …". *Copy answer* copies the Markdown text.
* **Conversation History:** conversations are stored in `storage/conversations.sqlite3` on the FastAPI server. Existing `hr_assistant_conversations_v1` data is imported automatically and kept in the browser until you confirm the migration is working. Logging out does not delete history. The explicit **Delete all** action deletes all history in the shared server database.
* **HR Knowledge Base:** real list of the files in `data/raw/` with chunk counts and index status (*Indexed*, *Not indexed*, *Index only*), plus drag-and-drop upload with the real processing result.
* **Settings:** excerpts per question, administrator unlock, live server status. **FAQ / Help & Support:** how the app works (static text about the application, not HR answers).
* A banner at the top shows real problems (Groq unreachable/model unavailable, local embedding model unavailable, empty index).
* Not included from the reference design: the microphone button. Browser speech recognition could send audio to an additional external service.

## 5. API reference

| Endpoint | Purpose | Notes |
|---|---|---|
| `POST /api/chat` | `{"question": "...", "conversation_id": "optional 8-64 chars [A-Za-z0-9_-]", "top_k": optional 1-20}` → `answer`, `grounded`, `sources[]` (`source`, `page`, `record_id`, `relevance`, `excerpt`, `citation`), `citations[]` (`citation`, `source`, `page`, `record_id`, `status`: `verified` / `unverified`), `standalone_question`, `elapsed_ms` | 422 invalid input, 503 Groq/index problem or busy |
| `POST /api/documents/upload` | multipart field `file` (`.pdf`, `.docx`, `.csv`) → chunks created / added / skipped / removed | needs `X-Admin-Key` by default; 400 type, 403 auth, 413 size, 422 unreadable file, 503 embedding service down |
| `GET /api/documents` | documents with `chunks` and `status`; CSV only for admins | |
| `GET /api/health` | `status` ok / degraded / error, legacy Ollama fields plus Groq model availability, local embedding model/dimension and chunk count | always HTTP 200 |
| `GET /api/auth/status` | is the supplied key valid | |
| `GET /api/conversations` | list saved conversation summaries | |
| `GET /api/conversations/{id}/messages` | read a conversation's ordered messages and metadata | |
| `POST /api/conversations` | create a conversation | |
| `POST /api/conversations/{id}/messages` | save or replace a message at its ordered position | |
| `POST /api/conversations/migrate` | idempotently import legacy browser history; the browser verifies read-back and retains the original localStorage data | |
| `DELETE /api/conversations/{id}` | delete one conversation and its messages | |
| `DELETE /api/conversations` | delete all conversations in the shared server database | |

Interactive docs: <http://127.0.0.1:8000/docs>. Examples:

```bash
curl -s -X POST http://127.0.0.1:8000/api/chat -H "Content-Type: application/json" \
  -d '{"question":"How many annual leave days can be carried over?","conversation_id":"demo-0001"}'
curl -s -F "file=@policy.pdf" -H "X-Admin-Key: YOUR_KEY" http://127.0.0.1:8000/api/documents/upload
```
PowerShell: use `curl.exe` (not the `curl` alias) for the upload, and
`Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/chat -ContentType "application/json" -Body '{"question":"How many annual leave days can be carried over?"}'` for chat.

Concurrency: the vector store and clients are shared; at most `MAX_CONCURRENT_CHATS` (default 2) generations run at once, others wait (503 after `CHAT_QUEUE_TIMEOUT_S`); index changes are serialised.

## 6. Configuration (`.env`)

Provider variables:

| Variable | Default | Meaning |
|---|---|---|
| `GROQ_API_KEY` | empty | Private GroqCloud API key; required for generation |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq chat model (verified available to this account) |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Local Sentence Transformers embedding model |
| `RERANKER_ENABLED` | `false` | Enable local mMARCO MiniLM reranking (15 candidates by default, then `TOP_K` results; CPU) |

Other RAG variables remain (`CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `CHROMA_DB_PATH`, `MIN_RELEVANCE_SCORE`, `COLLECTION_NAME`, `LOG_LEVEL`). With reranking enabled, BGE-M3 retrieves at least 15 candidates (or more when an explicit `top_k` requires it), the existing BGE relevance threshold is applied, then the CrossEncoder orders the remaining passages. The reranker is disabled by default; its model must be present in the local Hugging Face cache before enabling it. Pre-download it once with `python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', device='cpu')"`; runtime loading is local-only and uses CPU. `OLLAMA_BASE_URL` and `LLM_MODEL` remain only to populate legacy health-response fields; generation and embeddings do not use Ollama.

API variables:

| Variable | Default | Meaning |
|---|---|---|
| `API_HOST` / `API_PORT` | `127.0.0.1` / `8000` | Where the server listens (keep `127.0.0.1` for local use) |
| `CORS_ORIGINS` | `http://localhost:8000,http://127.0.0.1:8000` | Browser origins allowed to call the API |
| `ADMIN_API_KEY` | empty | Unlocks uploads and employee data (≥ 12 chars). Empty = disabled |
| `UPLOADS_REQUIRE_ADMIN` | `true` | Uploads need the admin key |
| `MAX_UPLOAD_MB` | `25` | Upload size limit |
| `MAX_CONCURRENT_CHATS` | `2` | Parallel generations |
| `HISTORY_TURNS` / `CONVERSATION_TTL_S` | `4` / `3600` | Conversation memory (0 turns = off) / idle expiry |

## 7. Tests

```powershell
python -m pytest -q
```

* **Real in `tests/test_api.py`:** the FastAPI app, validation, the project's real loader / processor / ChromaDB (temp folder) / retriever / pipeline / prompts, the access-control filter, uploads of real PDF/DOCX/CSV bytes, parallel requests.
* **Replaced by stand-ins in API tests:** the embedding model (`HashingEmbeddings`) and chat model (`FakeListChatModel`). They prove wiring and security rules, not model quality.
* **`tests/test_providers.py`:** mocks Sentence Transformers and Groq responses; no Groq API key or external generation request is required.
* **`tests/test_api_ollama.py`:** retained but skipped because the application no longer uses Ollama.
* The Week 1 tests (`test_chunking.py`, `test_retrieval.py`) are preserved.

## 8. Remote hosting (Netlify) — requirements

`frontend/` is a static site (`netlify.toml` included): publish the `frontend` folder, and set the API address in `frontend/config.js` (`window.HR_API_BASE = "https://your-api.example.com"`).

**A Netlify site cannot use the local ChromaDB on your PC.** Netlify only serves static files; the code runs in each visitor's browser, which cannot reach `localhost` on your computer. Netlify Functions cannot hold your ChromaDB folder. For remote use you need a **server you control** that runs FastAPI + local Sentence Transformers + ChromaDB and can reach GroqCloud:

1. A VPS or on-premises machine with enough RAM for local BGE-M3 inference, this project, and `data/` + `storage/` on persistent disk. GroqCloud performs generation.
2. **HTTPS** in front of FastAPI (Caddy or nginx with a certificate); a page served over HTTPS cannot call plain `http://` APIs.
3. `CORS_ORIGINS=https://your-site.netlify.app` in the server `.env`; `API_HOST=127.0.0.1` behind the reverse proxy.
4. **Real authentication** (company SSO or at least an authenticating reverse proxy) and rate limiting before real employees or real employee data are involved: the shared admin key is not enough.
5. A tunnel (Cloudflare Tunnel, ngrok, …) from your PC is possible for a demo, but it exposes your computer's service to the internet; protect it with authentication.

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| Groq unavailable / model missing | Check `GROQ_API_KEY` and confirm `GROQ_MODEL` is available to your account. |
| Local embedding model unavailable | Ensure network access for the first Hugging Face model download and enough disk space for BGE-M3. |
| Banner "No documents are indexed yet" | Run `python scripts\ingest.py`, or unlock admin access and upload in HR Knowledge Base. |
| Upload area is greyed out | Uploads need the admin key: set `ADMIN_API_KEY` in `.env`, restart, unlock in Settings. |
| `Configuration error: ... ADMIN_API_KEY must be at least 12 characters` | Use a longer key. |
| Page unstyled / blank | The browser could not load the CDN scripts (offline or blocked network). |
| Answers say "not enough information" too often | Lower `MIN_RELEVANCE_SCORE` (e.g. 0.2) or try `TOP_K=8`. |
| `ModuleNotFoundError` | Activate the virtual environment and `pip install -r requirements.txt`. |
| Port 8000 busy | Set `API_PORT=8001` in `.env`. |
| Changed embedding model or chunk size | Do not reset the existing index without a verified compatibility check and an explicit backup/re-ingestion plan. |

## 10. Known limits

* Answers are generated by GroqCloud. Always open the cited excerpt for decisions that matter; the Sources list is built from retrieval metadata, but inline citations written by the model can be wrong.
* Questions about totals, rankings or "all employees" are only partly supported (the per-column CSV summary helps for lists and counts; averages and "who has the highest" are not computed).
* No OCR for scanned PDFs, hybrid search, agentic retrieval or automatic evaluation; reranking is optional and CPU-bound.
* The question and retrieved context leave the local server for GroqCloud generation; local embeddings and Chroma remain on the server.

## 11. Project layout

```
hr_policy_rag/
├── backend/            main.py, schemas.py, dependencies.py, conversation.py, routes/{chat,documents,health}.py
├── frontend/           index.html, js/app.js, css/app.css, config.js, logo.svg, netlify.toml
├── src/                RAG modules (config, document_loader, document_processor, embeddings, vector_store, retriever, reranker, llm, prompts, rag_pipeline)
├── scripts/            run_server.py, ingest.py, chat.py
├── tests/              test_api.py, test_providers.py, test_chunking.py, test_retrieval.py, test_reranker.py
├── data/raw/           policy PDFs (+ restricted employees.csv if you add it)   data/processed/   storage/chroma_db/
├── .env.example  requirements.txt  pytest.ini  README.md
```

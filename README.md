# HR Policy Assistant

**An AI-powered assistant for intelligent HR policy support.**

HR Policy Assistant is a Retrieval-Augmented Generation (RAG) application that allows employees to ask questions about company HR policies and receive answers grounded exclusively in the provided HR documents.

The application retrieves relevant document excerpts, generates answers using an LLM, and displays source references, including document names and page numbers, to make responses easier to verify.

## Features

- **Document-based question answering:** Answers questions using the indexed HR policy documents.
- **Source citations:** Displays document names, page numbers, and relevant excerpts behind each answer.
- **Semantic search:** Uses BGE-M3 embeddings and ChromaDB to retrieve relevant information.
- **Optional reranking:** Uses a local CrossEncoder model to improve the relevance of retrieved passages.
- **Conversation context:** Supports follow-up questions using conversation memory.
- **Persistent chat history:** Stores conversations in a local SQLite database.
- **Web interface:** Provides a user-friendly chat interface with conversation history, Markdown rendering, and source inspection.
- **Document management:** Supports PDF, DOCX, and CSV uploads through the administrator interface.
- **Access control:** Restricts employee CSV data and document uploads to authorized administrators.
- **REST API:** Provides FastAPI endpoints for chat, documents, health checks, and conversation management.

## Architecture

```text
                    Browser
                       |
                       | HTTP / Fetch
                       v
                FastAPI Backend
                       |
          +------------+------------+
          |            |            |
          v            v            v
       Chat API    Documents API  History API
          |
          v
       RAG Pipeline
          |
          v
       Retriever
          |
          v
       ChromaDB <----- BGE-M3 Embeddings
          |
          v
    Optional CrossEncoder
          |
          v
      Retrieved Context
          |
          v
       GroqCloud LLM
          |
          v
    Answer + Citations
```

### Technology Stack

| Component | Technology |
|---|---|
| Backend | Python, FastAPI |
| Frontend | HTML5, Tailwind CSS, JavaScript |
| RAG framework | Custom RAG pipeline |
| Embeddings | BAAI/bge-m3 |
| Vector database | ChromaDB |
| Reranker | CrossEncoder mMARCO MiniLM (optional) |
| LLM | GroqCloud |
| Conversation storage | SQLite |
| Testing | Pytest |

## Project Structure

```text
hr-policy-rag-assistant/
│
├── backend/              # FastAPI application and API routes
├── frontend/             # Web interface
├── src/                  # RAG pipeline and core modules
├── scripts/              # Ingestion, server and CLI scripts
├── tests/                # Automated tests
├── data/
│   ├── raw/               # Source HR policy documents
│   └── processed/         # Processed documents
├── storage/
│   └── chroma_db/         # Local vector database
│
├── .env.example           # Environment configuration template
├── .gitignore
├── requirements.txt
├── pytest.ini
└── README.md
```

## Requirements

Before running the application, make sure you have:

- Python 3.11
- Git
- A GroqCloud API key
- An internet connection for the initial installation and model downloads

Get your API key from [Groq Console](https://console.groq.com/).

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/raniasirat/hr-policy-rag-assistant.git
cd hr-policy-rag-assistant
```

### 2. Create a virtual environment

**Windows (PowerShell):**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Create your local `.env` file from the example:

**Windows (PowerShell):**

```powershell
Copy-Item .env.example .env
```

**Linux / macOS:**

```bash
cp .env.example .env
```

Open `.env` and configure the required variables:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b

EMBEDDING_MODEL=BAAI/bge-m3
RERANKER_ENABLED=false

CHROMA_DB_PATH=storage/chroma_db
TOP_K=5

ADMIN_API_KEY=your_private_admin_key
UPLOADS_REQUIRE_ADMIN=true
```

Generate a secure administrator key:

```bash
python -c "import secrets; print(secrets.token_urlsafe(24))"
```

Copy the generated value into `ADMIN_API_KEY` in your `.env` file.

Keep `.env` private. Never commit API keys or administrator credentials to GitHub.

### 5. Index the HR policy documents

The repository includes sample HR policy PDFs in `data/raw/`.

For a **fresh installation with an empty vector database**, run:

```bash
python scripts/ingest.py
```

This process extracts the document content, splits it into chunks, generates local embeddings using BGE-M3, and stores them in ChromaDB.

**Important:** Run ingestion only when creating a fresh index. Do not rerun it over an existing compatible index without checking the ingestion and update behavior first.

### 6. Start the application

From the project root, run:

```bash
python scripts/run_server.py
```

Open the application in your browser:

**http://127.0.0.1:8000**

To stop the server, press `Ctrl + C` in the terminal.

## Using the Application

### Chat Interface

- Start a new conversation.
- Ask questions about HR policies.
- Review generated answers and their citations.
- Expand source excerpts to inspect the retrieved evidence.
- Ask follow-up questions within the same conversation.
- Copy answers when needed.

### HR Knowledge Base

- View available documents and indexing status.
- Upload supported PDF, DOCX, and CSV files.
- Inspect document processing results.

Document uploads require administrator authentication by default.

### Administrator Access

Enter the administrator key configured in `.env` through the application's Settings page.

Employee CSV data and document uploads are restricted to authorized administrators.

## API Documentation

FastAPI automatically provides interactive API documentation.

Once the server is running, visit:

- **Swagger UI:** http://127.0.0.1:8000/docs
- **Health check:** http://127.0.0.1:8000/api/health

Main endpoints:

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/chat` | Ask an HR policy question |
| GET | `/api/documents` | List available documents |
| POST | `/api/documents/upload` | Upload a document |
| GET | `/api/health` | Check application status |
| GET | `/api/conversations` | List conversations |
| POST | `/api/conversations` | Create a conversation |
| DELETE | `/api/conversations` | Delete conversation history |

## Running Tests

Run the automated test suite:

```bash
python -m pytest -q
```

The tests cover API behavior, document processing, retrieval, citation verification, conversation history, provider integration, and reranking.

Provider tests use mocks, so a real Groq API request is not required to run the test suite.

## Configuration

Some useful environment variables:

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | Empty | GroqCloud API key |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Generation model |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Local embedding model |
| `RERANKER_ENABLED` | `false` | Enable optional reranking |
| `TOP_K` | `5` | Number of retrieved passages |
| `CHROMA_DB_PATH` | `storage/chroma_db` | ChromaDB storage directory |
| `ADMIN_API_KEY` | Empty | Administrator access key |
| `UPLOADS_REQUIRE_ADMIN` | `true` | Require admin authentication for uploads |

### Optional: Enable the Reranker

The application can optionally use the following CrossEncoder model:

```text
cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
```

Download it locally once:

```bash
python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', device='cpu')"
```

Then configure:

```env
RERANKER_ENABLED=true
```

The reranker runs locally on CPU and reorders retrieved passages before they are sent to the language model.

## Security and Data Handling

- API keys and local environment configuration must remain private.
- Local embeddings are generated using Sentence Transformers.
- ChromaDB and uploaded documents are stored on the FastAPI server.
- Questions and retrieved HR document excerpts are sent to GroqCloud for answer generation.
- Employee CSV data and document uploads are protected by administrator access controls.
- Conversation history is stored in a shared SQLite database on the server.


## Limitations

- Answer quality depends on document quality and retrieval relevance.
- Scanned PDFs require OCR, which is not currently implemented.
- The application does not provide full employee analytics or advanced statistical calculations.
- The optional reranker requires additional local resources.
- Inline citations generated by the language model should always be checked against the displayed source excerpts.

## License

See the repository's license information for usage and distribution terms.

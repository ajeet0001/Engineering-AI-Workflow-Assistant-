# Engineering AI Workflow Assistant

An agentic AI pipeline that analyzes engineering PDF reports using **LangGraph**, **Gemini API**, and **RAG** (FAISS + Sentence Transformers). Four specialized agents collaborate to extract insights, retrieve relevant knowledge, and produce actionable recommendations.

---

## Architecture

```
Upload Engineering Report (PDF)
            │
            ▼
   Document Processing Agent   ← PyMuPDF
            │
            ▼
      Summary Agent             ← Gemini LLM
            │
            ▼
      RAG Retrieval Agent       ← FAISS + Sentence Transformers
            │
            ▼
 Recommendation Agent           ← Gemini LLM + Retrieved Context
            │
            ▼
     Final AI Report (Markdown)
```

---

## Project Structure

```
engineering-ai-workflow/
├── data/
│   ├── reports/            ← Place your PDF reports here
│   └── knowledge_base/     ← Engineering knowledge documents (.txt)
├── vector_store/           ← Auto-generated FAISS index
├── src/
│   ├── agents/
│   │   ├── document_agent.py
│   │   ├── summary_agent.py
│   │   ├── retrieval_agent.py
│   │   └── recommendation_agent.py
│   ├── graph/
│   │   └── workflow.py
│   ├── rag/
│   │   ├── loader.py
│   │   ├── chunking.py
│   │   ├── embeddings.py
│   │   └── vectorstore.py
│   ├── utils/
│   │   ├── text_cleaner.py
│   │   └── report_writer.py
│   ├── prompts/
│   │   ├── summary_prompt.py
│   │   └── recommendation_prompt.py
│   └── config.py
├── outputs/                ← Generated reports saved here
├── main.py
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick Start

### 1. Clone and set up environment

```bash
git clone <repo-url>
cd engineering-ai-workflow
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

### 2. Configure API key

```bash
copy .env.example .env      # Windows
# cp .env.example .env      # Linux/macOS
```

Open `.env` and set your **Gemini API key**:

```env
GOOGLE_API_KEY=your_actual_gemini_api_key_here
```

### 3. Add your PDF report

Place your engineering PDF report in:

```
data/reports/your_report.pdf
```

### 4. Run the workflow

```bash
# Basic usage
python main.py --report data/reports/your_report.pdf

# Force rebuild the RAG vector index
python main.py --report data/reports/your_report.pdf --rebuild-index

# Custom output path
python main.py --report data/reports/your_report.pdf --output outputs/my_analysis.md
```

---

## Output

The workflow generates a structured Markdown report at `outputs/report.md`:

```markdown
# Engineering Report Analysis

## Executive Summary
...

## Key Findings
...

## Risks
...

## Retrieved Knowledge
...

## Recommendations
...

## Preventive Actions
...

## Confidence Level
...
```

---

## Configuration

All settings are controlled via `.env`:

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | *(required)* | Your Gemini API key |
| `LLM_MODEL` | `gemini-2.0-flash` | Gemini model name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence Transformer model |
| `CHUNK_SIZE` | `500` | RAG chunk size (tokens) |
| `CHUNK_OVERLAP` | `50` | RAG chunk overlap |
| `TOP_K_RESULTS` | `5` | Number of retrieved docs |

---

## Technology Stack

| Component | Technology |
|---|---|
| LLM | Google Gemini (via LangChain) |
| Agent Orchestration | LangGraph |
| Embeddings | Sentence Transformers |
| Vector Database | FAISS |
| PDF Processing | PyMuPDF |
| Config | Pydantic Settings + python-dotenv |

---

## Adding Knowledge Base Documents

Add `.txt` files to `data/knowledge_base/`. Then rebuild the index:

```bash
python main.py --report your_report.pdf --rebuild-index
```

---

## Requirements

- Python 3.11+
- Gemini API key ([Get one here](https://aistudio.google.com/))

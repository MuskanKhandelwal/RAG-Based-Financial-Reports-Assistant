# FinRAG - Ask questions about SEC filings

<div align="center">

**Query 10-K filings and earnings releases in plain English, filtered by company and fiscal year, running entirely on your own machine**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.30-1C3C3C.svg)](https://www.langchain.com/)
[![Chroma](https://img.shields.io/badge/Chroma-1.5.9-FF6B6B.svg)](https://www.trychroma.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.61.1-FF4B4B.svg)](https://streamlit.io/)
[![Ollama](https://img.shields.io/badge/Ollama-mistral%20%7C%20llama3.2-000000.svg)](https://ollama.com/)
[![BGE](https://img.shields.io/badge/Embeddings-bge--small--en--v1.5-FFD21E.svg)](https://huggingface.co/BAAI/bge-small-en-v1.5)

</div>

A retrieval assistant for financial reports. You ask a question, it finds the relevant passages in
the filings and answers from them. Nothing leaves your laptop, and no API keys are needed.

## Highlights

### It knows which company you are asking about

- **Every chunk is tagged** with company, ticker, fiscal year, document type and period end, read
  from the cover page of the filing rather than the file name.
- **Filter before you search.** Pick a company and a year in the sidebar and the search only looks at
  those documents. Without this, a question about revenue pulls Apple, Tesla and Meta passages into
  the same answer with no way to tell them apart.
- **File names are not trusted.** Two files in this repo are named `10-Q4-2024-As-Filed.pdf` and
  `_10-K-Q4-2023-As-Filed.pdf` and both are Apple filings. Nothing in either name says so, so the
  cover page is parsed instead.
- **An impossible filter says so.** Ask for a year a company has no filing for and it tells you
  nothing matched, instead of quietly answering from a different document.

### It answers with the actual numbers

- **Figures are quoted, not inferred.** The prompt forbids deriving a number from a nearby line item
  or treating a dollar amount as a unit count.
- **It declines when it does not know.** If the retrieved passages do not answer the question, it
  replies that the information is not in the context rather than estimating.
- **Sources are readable.** Answers cite `TSLA FY2023 (10-K) p.33`, not a file path.

### Several retrieval strategies to compare

- **Semantic search** as the default, plus **hybrid search**, **query expansion**, **contextual
  compression** and **multi-query** retrieval.
- **Cross-encoder re-ranking** and **query transformation** as toggles, so you can see what each one
  changes on the same question.

### It runs locally

- **Ollama** for generation, **bge-small** for embeddings, **Chroma** on disk for the index.
- **No API keys, no usage cost, no documents sent anywhere.**

## What is in the corpus

| Document | Company | Type | Fiscal year |
| --- | --- | --- | --- |
| `10-Q4-2024-As-Filed.pdf` | Apple | 10-K | 2024 |
| `_10-K-Q4-2023-As-Filed.pdf` | Apple | 10-K | 2023 |
| `tsla-20231231-gen.pdf` | Tesla | 10-K | 2023 |
| `cost-annual-report-final-pdf-from-dfin.pdf` | Costco | Annual report | 2023 |
| `Meta-12-31-2023-Exhibit-99-1-FINAL.pdf` | Meta | Press release | 2023 |
| `Meta-Reports-First-Quarter-2024-Results-2024.pdf` | Meta | Press release | 2024 |

Six documents, 427 pages, 1,228 chunks.

## How to use it

**1. Build the index.** This reads every PDF in `data/`, splits it, and writes the vector store to
`chroma/`. It takes about 45 seconds.

```bash
.venv/bin/python ingest.py --reset
```

**2. Start the app.**

```bash
.venv/bin/streamlit run app.py
```

**3. Pick a company in the sidebar.** Under "Filter filings" choose a company, a fiscal year and a
document type. Leave any of them on "All" to search everything. The choices come from what is
actually in the index, so they update if you add filings.

**4. Ask a question.** Some that work well:

> "How many vehicles were produced and delivered?" (Company: TSLA)
> "What are the principal risk factors related to supply chain?" (Company: AAPL, Year: 2024)
> "What was daily active people and average revenue per person?" (Type: PRESS_RELEASE)
> "What drove the change in membership fee revenue?" (Company: COST)

**5. Compare retrieval methods.** Ask the same question with different methods selected in the
sidebar and watch the sources change. Turning on re-ranking is the clearest difference.

**6. Upload your own filing.** Switch to "Upload New Financial Report", add a PDF, and press
"Process PDFs". It goes into a separate temporary index so your main one is untouched.

Two things worth knowing. The first question of a session is slow because Ollama has to load the
model into memory. And changing the embedding model means rebuilding the index, since old and new
vectors are not comparable.

## Tech stack

| Layer | Built with |
| --- | --- |
| PDF loading | `pypdf` via LangChain `PyPDFDirectoryLoader` |
| Chunking | `RecursiveCharacterTextSplitter`, 1500 characters, 200 overlap |
| Metadata extraction | Regex over filing cover pages (`metadata.py`) |
| Embeddings | `BAAI/bge-small-en-v1.5` through sentence-transformers, 384 dimensions |
| Vector store | Chroma, persisted to `chroma/` |
| Re-ranking | `cross-encoder/ms-marco-MiniLM-L-12-v2` |
| Generation | Ollama running `mistral` or `llama3.2` |
| Interface | Streamlit |

## Setup

You need an **arm64 Python** on Apple Silicon. An x86_64 interpreter running under Rosetta caps
PyTorch at 2.2.2 and cannot use the GPU, which makes indexing very slow.

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Check that you got the right one:

```bash
.venv/bin/python -c "import platform, torch; print(platform.machine(), torch.backends.mps.is_available())"
```

You want `arm64 True`.

Then install Ollama and pull a model:

```bash
ollama pull llama3.2
```

Two environment variables let you change the setup without editing code:

```bash
CHUNK_SIZE=800 CHUNK_OVERLAP=80 .venv/bin/python ingest.py --reset
EMBED_MODEL=BAAI/bge-base-en-v1.5 .venv/bin/python ingest.py --reset
```

## Design decisions

- **bge-small over bge-large.** bge-large is 1.3 GB and thrashed on an 8 GB machine, taking over an
  hour without finishing. bge-small is 130 MB and indexes the same corpus in 45 seconds, with a
  smaller drop in retrieval quality than that gap suggests.
- **bge-small over all-MiniLM-L6-v2.** MiniLM truncates at 256 tokens. Chunks here are around 375
  tokens, so roughly a third of every chunk would be silently dropped. bge handles 512.
- **1500 character chunks instead of 800.** At 800 characters the average chunk was 685 characters,
  about 170 tokens, which cuts management discussion mid-argument. Chunks never cross page
  boundaries, so the page is the real limit and going much above 1500 changes little.
- **Metadata filtering instead of a bigger model.** Filtering by company and year fixes a whole class
  of wrong answers that no amount of re-ranking can, because the wrong document was never a
  candidate in the first place.
- **Whitespace is normalised at ingest.** `pypdf` puts tabs between words, so a sentence arrives as
  `we\tproduced\t1,845,985\tvehicles`. Left alone, the model misreads it and pulls numbers from the
  income statement instead.

## Known limitations

- **Tables are read as prose.** `pypdf` returns table cells in reading order, so an income statement
  becomes a run of numbers with no row labels attached. Narrative questions work well, precise
  lookups from financial statements do not. A layout-aware parser such as Docling would fix this.
- **Chunks do not cross pages.** A table or paragraph split across a page break exists in no single
  chunk.
- **One company at a time.** The filter selects a single company, so a question comparing two
  companies needs two separate queries.
- **`sections.py` is not wired in.** It parses 10-K item sections into a CSV, but the index is
  built from pages instead. Chunking within sections is the natural next step.

## Cost

Nothing. Ollama, bge-small and Chroma all run locally, so the only cost is disk space, about 22 MB
for the index and around 2 GB for the language model.

## Contributing

This is a learning project. Suggestions and issues are welcome.

---
<div align="center">

**Made by Muskan Khandelwal**

</div>

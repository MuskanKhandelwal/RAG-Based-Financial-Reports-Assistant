# FinRAG - Ask questions about SEC filings

<div align="center">

**Query 10-K filings and earnings releases in plain English, filtered by company and fiscal year, running entirely on your own machine**


## Objective

</div>

A retrieval assistant for financial reports. You ask a question, it finds the relevant passages in
the filings and answers from them. Nothing leaves your laptop, and no API keys are needed.

## Highlights

### It knows which company you are asking about

2. **FAISS and ChromaDB for Document Retrieval**:
   - **FAISS (Facebook AI Similarity Search)** provides efficient, similarity-based search over document embeddings and supports retrieval for models like GPT-4o and Meta-LLaMA 3-8B Instruct.
   - **ChromaDB** is used for advanced indexing and retrieval for LLaMA 3.2 and Mistral, ensuring accurate and scalable document processing.
   - Both systems store embeddings in indices for fast and context-aware retrieval of relevant sections.

### It answers with the actual numbers

4. **Advanced Generative Models**:
   - **LLaMA 3.2**: A scalable model optimized for multi-turn dialogue and large-scale financial datasets, ensuring detailed contextual understanding.
   - **Mistral**: Efficiently designed for text generation and structured QA tasks, particularly on compact hardware setups.
   - **Meta-LLaMA 3-8B Instruct**: Fine-tuned for instruction-following tasks, enabling precise and context-sensitive responses.
   - **OpenAI GPT**: Combines retrieved context with user queries to generate fact-based, natural language responses.

### Several retrieval strategies to compare

- **Semantic search** as the default, plus **hybrid search**, **query expansion**, **contextual
  compression** and **multi-query** retrieval.
- **Cross-encoder re-ranking** and **query transformation** as toggles, so you can see what each one
  changes on the same question.

### It runs locally

#### **Step 1: Data Preparation and Embedding**

- **HTML Processing**: Extracted text using BeautifulSoup, cleaned and normalized, with sections like "Risk Factors" mapped to standardized fields. Output was structured as `structured_10k.csv`.  
- **PDF Processing**: Used PyPDFLoader and RecursiveCharacterTextSplitter to process files into 800-character chunks with 80-character overlap, assigning unique identifiers for traceability.  
- **Embedding Generation**: Text embeddings were generated using `all-mpnet-base-v2` for general-purpose tasks, `FinLang/finance-embeddings-investopedia` for financial contexts, and `BAAI/bge-large-en-v1.5` for balanced precision and generalizability. Indexed using **FAISS** for fast retrieval.

Six documents, 427 pages, 1,228 chunks.

#### **Step 2: Query Processing and Retrieval**

- **Query Encoding**: User queries were embedded using the same models as document embeddings.  
- **Search and Retrieval**: **FAISS** retrieved relevant document chunks, while **ChromaDB** supported retrieval for **LLaMA 3.2** and **Mistral**.  
- **Context Aggregation**: Retrieved sections were combined into a coherent context for model input.

**2. Start the app.**

#### **Step 3: Response Generation**

- **Prompt Creation**: Combined user queries with retrieved context for structured input.  
- **Language Models**:  
  - **LLaMA 3.2** and **Mistral**: Fine-tuned for financial QA tasks.  
  - **GPT-4o**: Generated detailed, conversational responses.  
- **Output**: Delivered user-friendly responses via a **Streamlit interface**.

---

#### **Models Used**

- **GPT-4o**: General-purpose model for precise financial insights.  
- **Mistral**: Optimized for QA tasks on compact setups.  
- **LLaMA 3.2**: Handles multi-turn dialogue and large-scale datasets.  
- **Meta-LLaMA 3-8B Instruct**: Fine-tuned for instruction-following with domain-specific datasets.

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

---
## **Folder Structure**
```plaintext
RAG-Based-Financial-Reports-Assistant/
├── Evaluation/                     # Contains evaluation scripts and results for QA performance
│   ├── Evaluation.ipynb            # Notebook for evaluating retrieval and QA pair quality
│   ├── scored_qa.xlsx              # Excel file storing QA evaluation scores
├── FAISS/                          # Directory for FAISS-related files
│   ├── finance_10k_index.faiss     # FAISS index storing document embeddings for retrieval
├── data/                           # Directory for raw and processed financial data
├── RAG_Pipeline_with_GPT.ipynb     # Notebook implementing RAG pipeline using GPT for QA
├── RAG_pipeline_with_Llama.ipynb   # Notebook implementing RAG pipeline using LLaMA models
├── adv_rag_app.py                  # Streamlit app for interactive querying and response generation
├── advanced_rag_techniques.py      # Script for advanced RAG techniques like re-ranking and query expansion
├── get_embedding_function.py       # Script for generating text embeddings using various models
├── populate_database.py            # Script for populating ChromaDB with document embeddings
├── preprocess.py                   # Script for preprocessing HTML documents and normalizing text
├── section_mapping.json            # JSON file mapping document sections to indices for lookup
├── structured_10k.csv              # Structured dataset of 10-K filings for embeddings and retrieval
├── README.md                       # Project documentation with overview, setup, and usage instructions

```

You want `arm64 True`.

Then install Ollama and pull a model:

```bash
ollama pull llama3.2
```

### **OpenAI GPT-4o**
- Generates high-quality, natural language responses using retrieved context.
### **Mistral**
- Efficient model specialized for text generation and structured QA tasks, particularly on compact hardware setups.
### **LLaMA 3.2**
- A scalable language model optimized for multi-turn dialogue and large-scale financial datasets, ensuring detailed contextual understanding.

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

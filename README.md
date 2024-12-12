# RAG-Based-Financial-Reports-Assistant

This repository demonstrates a **Retrieval-Augmented Generation (RAG)** pipeline using **FAISS** and **state-of-the-art language models** like **LLaMA 3** and **OpenAI GPT** for querying financial reports. The system is optimized for analyzing financial documents, such as 10-K filings, to provide detailed, accurate, and user-friendly responses to natural language queries.

---

## Objective

Develop a robust RAG-based financial assistant that combines advanced vector search with fine-tuned generative models. The system enables users to interactively query structured financial documents, offering precise insights in a conversational format.

---

## Key Features

1. **Structured Financial Data**:
   - Utilizes 10-K filings, stored in a structured CSV format, with sections like "Risk Factors," "Financial Statements," and "Management Analysis."
   - Supports additional financial filings, such as 10-Q (quarterly reports) and 8-K (current events).

2. **FAISS for Document Retrieval**:
   - **FAISS (Facebook AI Similarity Search)** provides efficient, similarity-based search over document embeddings.
   - Stores document embeddings in an index for fast retrieval of relevant sections.

3. **Pre-trained Financial Embeddings**:
   - Uses **FinLang/finance-embeddings-investopedia**, a specialized model for embedding financial text.
   - Embeds both user queries and document sections into a shared semantic space for accurate matching.

4. **Advanced Generative Models**:
   - **LLaMA 3**: Fine-tuned using **Low-Rank Adaptation (LoRA)** for domain-specific QA tasks.
   - **OpenAI GPT**: Combines retrieved context with user queries to generate fact-based, natural language responses.

5. **Interactive Querying**:
   - Users can query financial reports using natural language.
   - The system retrieves relevant sections and generates detailed, domain-specific answers.

---

## Workflow

### Step 1: Data Preparation and Embedding
1. **Data**:
   - Load structured financial documents (e.g., 10-K filings) in CSV format.
   - Combine headers and content into chunks for embedding.

2. **Embeddings**:
   - Generate document embeddings using **sentence-transformers/all-mpnet-base-v2** or **FinLang/finance-embeddings-investopedia**.
   - Store embeddings in a **FAISS index**.

3. **Indexing**:
   - Save a mapping of document sections to index positions for reverse lookup.

---

### Step 2: Query Processing and Retrieval
1. **Query Encoding**:
   - Convert user queries into embeddings using the same model as document embeddings.

2. **Search**:
   - Perform a similarity search on the FAISS index to find the most relevant document chunks.

3. **Aggregation**:
   - Combine retrieved sections into a context string for the generative model.

---

### Step 3: Response Generation
1. **Prompt Creation**:
   - Combine the user query with the retrieved context to form a structured prompt.

2. **Language Model**:
   - Pass the prompt to either:
     - **LLaMA 3**: Fine-tuned for financial QA tasks.
     - **OpenAI GPT-4o**: For generating detailed and conversational responses.

3. **Output**:
   - Return a clear, fact-based response to the user query.

---

## Example Query and Output

### Query:
*"What are the risk factors mentioned in the report?"*

### Pipeline Execution:
1. **Retrieval**: Retrieves relevant sections from the 10-K filing (e.g., "Risk Factors").
2. **Response Generation**: Combines the query and retrieved context to generate the following response:

### Output:
*"The risk factors include market volatility, regulatory changes, and supply chain disruptions, as mentioned in the report."*

---



## Key Components

### **FAISS Index**
- Efficiently stores document embeddings for fast similarity search.
- Ensures accurate retrieval of relevant sections from large datasets.

### **LLaMA 3**
- Fine-tuned using **LoRA** for parameter-efficient updates.
- Specialized for financial question answering.

### **OpenAI GPT-4o**
- Generates high-quality, natural language responses using retrieved context.

### **Embedding Models**
- Encodes financial text and user queries into a shared semantic space.

---

## Conclusion
This RAG-based financial assistant combines **FAISS** for precise retrieval and **state-of-the-art generative models** for response generation. It transforms complex financial reports into accessible insights, empowering users with easy-to-understand and actionable information.

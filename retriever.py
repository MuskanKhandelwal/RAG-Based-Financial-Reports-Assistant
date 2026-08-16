import json
from typing import List, Dict, Any
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM

def build_filter(ticker: str = None, fiscal_year: int = None, form_type: str = None):
    """Build a Chroma where-clause from the selected facets."""
    clauses = []
    if ticker:
        clauses.append({"ticker": {"$eq": ticker}})
    if fiscal_year:
        clauses.append({"fiscal_year": {"$eq": int(fiscal_year)}})
    if form_type:
        clauses.append({"form_type": {"$eq": form_type}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def result_key(doc):
    return doc.metadata.get("id") or (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content[:80])


class AdvancedRAGRetriever:
    def __init__(self, chroma_path: str, embedding_function, model_name: str = "mistral"):
        self.db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)
        self.llm = OllamaLLM(model=model_name)
        self.embedding_function = embedding_function

    def available_facets(self):
        """Return the tickers, fiscal years and form types present in the store."""
        metadatas = self.db.get(include=["metadatas"]).get("metadatas") or []
        tickers, years, forms = set(), set(), set()
        for meta in metadatas:
            if not meta:
                continue
            if meta.get("ticker"):
                tickers.add(meta["ticker"])
            if isinstance(meta.get("fiscal_year"), int) and meta["fiscal_year"] > 0:
                years.add(meta["fiscal_year"])
            if meta.get("form_type"):
                forms.add(meta["form_type"])
        return sorted(tickers), sorted(years, reverse=True), sorted(forms)

    def semantic_search(self, query: str, k: int = 5, where: dict = None):
        return self.db.similarity_search_with_score(query, k=k, filter=where)

    def _retriever(self, k: int, where: dict = None):
        search_kwargs = {"k": k}
        if where:
            search_kwargs["filter"] = where
        return self.db.as_retriever(search_kwargs=search_kwargs)

    def query_transformer(self, original_query: str) -> str:
        """Transform the original query to a more precise format."""
        prompt = PromptTemplate(
            input_variables=["question"],
            template="""You are a query transformation expert. Convert the given question 
            into a more precise, specific query that captures the core information need.
            
            Original Query: {question}
            Transformed Query:"""
        )
        
        chain = prompt | self.llm | StrOutputParser()
        transformed_query = chain.invoke({"question": original_query}).strip()
        return transformed_query

    def query_expansion(self, original_query: str) -> List[str]:
        """Generate multiple related queries to expand search coverage."""
        prompt = PromptTemplate(
            input_variables=["question"],
            template="""You are a query expansion expert. For the given query, 
            generate 3-4 semantically related alternative queries that might help 
            find more comprehensive information.
            
            Original Query: {question}
            Alternative Queries:"""
        )
        
        chain = prompt | self.llm | StrOutputParser()
        expanded_queries_str = chain.invoke({"question": original_query}).strip()
        
        try:
            expanded_queries = json.loads(expanded_queries_str)
        except:
            expanded_queries = expanded_queries_str.split('\n')
        
        return [original_query] + expanded_queries

    def hybrid_search(self, query: str, k: int = 5, where: dict = None):
        """Perform hybrid semantic and keyword search."""
        semantic_results = self.db.similarity_search_with_score(query, k=k//2, filter=where)

        try:
            keyword_results = self.db.max_marginal_relevance_search_with_score(query, k=k//2, filter=where)
        except:
            keyword_results = self.db.similarity_search_with_score(query, k=k//2, filter=where)

        combined_results = semantic_results + keyword_results
        unique_results = {result_key(result[0]): result for result in combined_results}
        return list(unique_results.values())[:k]

    def contextual_compression(self, query: str, k: int = 5, where: dict = None):
        """Apply contextual compression to retrieved documents."""
        compressor = LLMChainExtractor.from_llm(self.llm)
        compression_retriever = ContextualCompressionRetriever(
            base_retriever=self._retriever(k, where),
            document_compressor=compressor
        )
        return compression_retriever.get_relevant_documents(query)

    def multi_query_retrieval(self, query: str, k: int = 5, where: dict = None):
        """Retrieve documents using multiple query variations."""
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=self._retriever(k, where),
            llm=self.llm
        )
        return multi_query_retriever.get_relevant_documents(query)

    def re_rank_results(self, query: str, documents: List[Dict[str, Any]], top_k: int = 3):
        """Re-rank retrieved documents using a cross-encoder."""
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
        
        # Prepare input for cross-encoder
        pairs = [(query, doc.page_content) for doc in documents]
        scores = cross_encoder.predict(pairs)
        
        # Sort documents by reranking scores
        ranked_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in ranked_docs[:top_k]]
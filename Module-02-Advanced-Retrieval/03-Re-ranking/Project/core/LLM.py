import os
from typing import Optional, List, Union
from dotenv import load_dotenv
import cohere
from sentence_transformers import CrossEncoder
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

class CrossEnc:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        """
        Initializes the Cross-Encoder model.
        """
        print(f"Loading Cross-Encoder model: {model_name}...")
        self.model = CrossEncoder(model_name)

    def rank(self, query: str, candidates: dict, top_k: int = 2) -> list:
        """
        Ranks the candidates based on the query using the Cross-Encoder.
        """
        if not candidates:
            return []

        doc_ids = list(candidates.keys())
        documents = [candidates[doc_id]["document"] for doc_id in doc_ids]
        pairs = [[query, doc] for doc in documents]

        scores = self.model.predict(pairs)

        scored_results = list(zip(doc_ids, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        top_results = []
        for doc_id, score in scored_results[:top_k]:
            top_results.append(candidates[doc_id]["document"])
        
        return top_results

class CohereReranker:
    def __init__(self, api_key: Optional[str] = None, model: str = "rerank-english-v3.0"):
        """
        Initializes the Cohere Reranker.
        """
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        self.model = model
        self.client = cohere.ClientV2(api_key=self.api_key) if self.api_key else None

    def rank(self, query: str, candidates: dict, top_k: int = 2) -> list:
        """
        Ranks the candidates based on the query using Cohere's Rerank API.
        """
        if not self.client:
            raise ValueError("Cohere API key not found. Please add COHERE_API_KEY to your .env file.")
        
        if not candidates:
            return []

        doc_ids = list(candidates.keys())
        documents = [candidates[doc_id]["document"] for doc_id in doc_ids]
        
        response = self.client.rerank(
            model=self.model,
            query=query,
            documents=documents,
            top_n=top_k
        )
        
        # Cohere V2 returns a list of results with index and relevance_score
        top_results = []
        for result in response.results:
            top_results.append(documents[result.index])
            
        return top_results

class Ollama:
    def __init__(self, model: str = "gpt-oss:20b-cloud"):
        self.llm = ChatOllama(model=model, temperature=0)
        self.llm_query_filter = self.llm.with_structured_output(SearchFilters)
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            (
                "system", (
                    "You are a Senior Knowledge Assistant. Answer the user's question using ONLY the given context.\n"
                    "Rules:\n"
                    "1. If the context does not contain the answer, say 'I'm sorry, I don't have information on that in my current database.'\n"
                    "2. Keep answers professional, concise, and technical.\n"
                    "3. Do not use outside knowledge.\n"
                    "4. If the context provides contradictory info, highlight it.\n\n"
                    "CONTEXT:\n{context}"
                )
            ),
            (
                "user", "{question}"
            )
        ])
        self.chain = self.prompt_template | self.llm
    
    def generate_filter(self, user_query: str, existing_metadata: dict = None) -> Optional[dict]:
        """
        Translates query into structured filters using department and category awareness.
        """
        try:
            prompt = (
                "SYSTEM: You are a JSON generator. Choose 'department' and 'category' values from the available options that match user query.\n"
                "RULES:\n"
                "1. If multiple values apply, use a LIST. If one applies, use a STRING.\n"
                "2. Output ONLY raw JSON matching the schema. No conversation, no bolding.\n"
                f"USER QUERY: {user_query}\n OPTIONS: {existing_metadata}"
            )
            
            model_obj = self.llm_query_filter.invoke(prompt)
            raw_filters = model_obj.model_dump(exclude_none=True)

            if not raw_filters:
                return None

            chroma_conditions = []
            for key, value in raw_filters.items():
                if isinstance(value, list):
                    chroma_conditions.append({key: {"$in": value}})
                else:
                    chroma_conditions.append({key: value})

            if len(chroma_conditions) > 1:
                return {"$and": chroma_conditions}
            return chroma_conditions[0] if chroma_conditions else None

        except Exception as e:
            print(f"Error generating filter: {e}")
            return None

    def answer_question(self, context: str, query: str):
        """
        Generates the final grounded answer.
        """
        result = self.chain.invoke({
            "context": context,
            "question": query
        })
        print(f"\n[AI]: {result.content}\n")

class SearchFilters(BaseModel):
    """
    The structured filter generated from a natural language user query.
    Extracts department and categories based on the company's policy data structure.
    """
    department: Optional[Union[str, List[str]]] = Field(None, description="One or more departments.")
    category: Optional[Union[str, List[str]]] = Field(None, description="One or more categories.")

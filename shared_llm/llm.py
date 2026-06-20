from typing import Optional, Dict, List, Any, Union
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from .schema import DocumentMetadata, SearchFilters

class ChatManager:
    """
    Unified manager for LLM interactions. 
    Handles standard chat, structured metadata extraction, and self-querying filter generation.
    """
    def __init__(self, model: str = "gpt-oss:20b-cloud", temperature: float = 0):
        """
        Initializes the LLM with structured output capabilities.
        
        Args:
            model (str): The name of the Ollama model.
            temperature (float): Controls randomness (0 = deterministic).
        """
        self.llm = ChatOllama(model=model, temperature=temperature)
        
        # Pre-configure structured outputs for performance
        self.meta_extractor = self.llm.with_structured_output(DocumentMetadata)
        self.filter_generator = self.llm.with_structured_output(SearchFilters)
        
        # Standard Chat Chain
        self.chat_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a Senior Knowledge Assistant. Answer the question using ONLY the given context.\n"
                "If the context is insufficient, state that you don't have enough information.\n"
                "Rules: Professional, concise, technical, no outside knowledge.\n\n"
                "CONTEXT:\n{context}"
            )),
            ("user", "{question}")
        ])
        self.chat_chain = self.chat_prompt | self.llm

    def ask(self, query: str, context: str) -> str:
        """
        Generates a grounded answer based on provided context.
        
        Args:
            query (str): The user's question.
            context (str): The retrieved text chunks.
            
        Returns:
            str: The LLM's response.
        """
        response = self.chat_chain.invoke({"context": context, "question": query})
        return response.content

    def refine_query(self, query: str, history: List[Dict[str, str]] = None) -> str:
        """
        Converts conversational follow-ups into standalone search queries (Anaphora Resolution).
        
        Args:
            query (str): The user's follow-up query.
            history (List[Dict[str, str]]): List of previous messages with 'role' and 'content'.
            
        Returns:
            str: The refined, standalone query.
        """
        if not history:
            return query
            
        history_str = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in history])
        prompt = (
            "Given the following conversation history and a follow-up query, "
            "re-write the follow-up query to be a standalone search query. "
            "Do not answer it, just re-write it to be complete and clear. "
            "Output ONLY the re-written query.\n\n"
            f"HISTORY:\n{history_str}\n\n"
            f"FOLLOW-UP QUERY: {query}"
        )
        response = self.llm.invoke(prompt)
        return response.content.strip()

    def decompose_query(self, query: str) -> List[str]:
        """
        Splits a complex query into atomic sub-questions.
        
        Args:
            query (str): The user's complex query.
            
        Returns:
            List[str]: List of maximum 3 sub-questions.
        """
        prompt = (
            "Split the following complex query into a maximum of 3 atomic sub-questions "
            "that can be searched independently. "
            "Output as a simple bulleted list. If the query is simple, return it as a single bullet.\n\n"
            f"QUERY: {query}"
        )
        response = self.llm.invoke(prompt)
        lines = response.content.strip().split('\n')
        sub_questions = [line.strip().lstrip('*-•').strip() for line in lines if line.strip()]
        return sub_questions[:3]

    def expand_query(self, query: str) -> List[str]:
        """
        Generates 3 variations of the query to improve search recall.
        
        Args:
            query (str): The user's search query.
            
        Returns:
            List[str]: List of 3 query variations.
        """
        prompt = (
            "Generate 3 variations of the following query to improve search recall. "
            "Focus on synonyms and different technical phrasings. "
            "Output as a simple bulleted list.\n\n"
            f"QUERY: {query}"
        )
        response = self.llm.invoke(prompt)
        lines = response.content.strip().split('\n')
        variations = [line.strip().lstrip('*-•').strip() for line in lines if line.strip()]
        return variations[:3]

    def extract_metadata(self, text: str, global_anchor: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyzes text to extract structured metadata (Topic, Year, etc.).
        
        Args:
            text (str): The text chunk to analyze.
            global_anchor (Optional[Dict[str, Any]]): High-level metadata to enforce consistency.
            
        Returns:
            Dict[str, Any]: Extracted metadata following the DocumentMetadata schema.
        """
        prompt = (
            "Analyze the text and extract metadata.\n"
            "Rules: Output ONLY valid JSON. Complexity: Beginner/Intermediate/Advanced. Priority: Low/Medium/High.\n"
            f"TEXT: {text}"
        )
        if global_anchor:
            prompt += f"\nGLOBAL CONTEXT: Document is about '{global_anchor.get('topic')}' for '{global_anchor.get('audience')}'."
            
        try:
            result = self.meta_extractor.invoke(prompt)
            data = result.model_dump()
            
            # Enforce global anchor consistency
            if global_anchor:
                for key in ['topic', 'year', 'audience']:
                    if key in global_anchor:
                        data[key] = global_anchor[key]
            return data
        except Exception as e:
            print(f"Metadata Extraction Failed: {e}")
            return global_anchor or {}

    def generate_filters(self, query: str, options: Optional[Dict[str, List[Any]]] = None) -> Optional[Dict[str, Any]]:
        """
        Translates a natural language query into a structured database filter.
        
        Args:
            query (str): The user's search query.
            options (Optional[Dict[str, List[Any]]]): Available tags in the DB to guide the LLM.
            
        Returns:
            Optional[Dict[str, Any]]: A ChromaDB-compatible 'where' clause.
        """
        prompt = "Map the query to database filters. Extract Topic, Year, Complexity, Priority, Dept, Category.\n"
        if options:
            prompt += f"Available tags in DB: {options}\n"
        prompt += f"USER QUERY: {query}"
        
        try:
            result = self.filter_generator.invoke(prompt)
            raw_filters = result.model_dump(exclude_none=True)
            
            if not raw_filters:
                return None
                
            # Convert to ChromaDB logical format ($and if multiple filters)
            conditions = []
            for key, value in raw_filters.items():
                if isinstance(value, list):
                    conditions.append({key: {"$in": value}})
                else:
                    conditions.append({key: value})
                    
            if len(conditions) > 1:
                return {"$and": conditions}
            return conditions[0] if conditions else None
            
        except Exception as e:
            print(f"Filter Generation Failed: {e}")
            return None

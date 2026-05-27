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

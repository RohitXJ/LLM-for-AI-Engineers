from typing import Optional
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from .models import DocumentMetadata, SearchFilters
load_dotenv()

class Ollama:
    def __init__(self, model: str = "gpt-oss:20b-cloud"):
        self.llm = ChatOllama(model=model, temperature=0)
        self.llm_meta_finder = self.llm.with_structured_output(DocumentMetadata)
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

    def extract_document_metadata(self, content: str) -> dict:
        """
        Analyzes the document to establish a 'Global Anchor' for metadata consistency.
        """
        try:
            sample = content[:8000]
            prompt = (
                "You are a Structured Data Specialist. Analyze the provided text and extract high-level metadata.\n"
                "Fields:\n"
                "- topic: The primary subject matter.\n"
                "- year: The primary year discussed (MUST be a 4-digit integer only, e.g., 2024).\n"
                "- complexity: MUST be EXACTLY one of: 'Beginner', 'Intermediate', or 'Advanced'.\n"
                "- audience: Intended reader.\n"
                "- priority: MUST be EXACTLY one of: 'Low', 'Medium', or 'High'.\n\n"
                "Rules:\n"
                "1. Respond ONLY with a raw valid JSON object. No Markdown code blocks. No preamble. No summary.\n"
                "2. Do not use conversational text or tables.\n"
                "3. Be objective and concise.\n\n"
                f"TEXT SAMPLE:\n{sample}"
            )
            model_obj = self.llm_meta_finder.invoke(prompt)
            return model_obj.model_dump()
        except Exception as e:
            print(f"Error extracting document metadata: {e}")
            return {
                "topic": "General",
                "year": 2026,
                "complexity": "Beginner",
                "audience": "General",
                "priority": "Low"
            }

    def extract_metadata(self, context: str, global_meta: dict = None) -> dict:
        """
        Extracts metadata for a specific chunk, anchored by global metadata.
        """
        try:
            prompt = (
                "You are a Structured Data Specialist. Extract metadata for this specific text chunk.\n"
                "Required keys: 'topic', 'year', 'complexity', 'audience', 'priority'.\n\n"
                f"CHUNK CONTENT:\n{context}\n\n"
                "Rules:\n"
                "1. Respond ONLY with a raw valid JSON object. No Markdown, no tables.\n"
                "2. 'complexity' must be 'Beginner', 'Intermediate', or 'Advanced'.\n"
                "3. 'priority' must be 'Low', 'Medium', or 'High'.\n"
                "4. 'year' must be an integer.\n"
            )
            if global_meta:
                prompt += f"\nGLOBAL CONTEXT: This chunk is part of a document about '{global_meta['topic']}' for '{global_meta['audience']}'."
            
            model_obj = self.llm_meta_finder.invoke(prompt)
            meta = model_obj.model_dump()
            
            if global_meta:
                meta['topic'] = global_meta.get('topic', meta['topic'])
                meta['year'] = global_meta.get('year', meta['year'])
                meta['audience'] = global_meta.get('audience', meta['audience'])
                
            return meta
        except Exception as e:
            print(f"Error extracting chunk metadata: {e}")
            return global_meta or {
                "topic": "Unknown",
                "year": 2026,
                "complexity": "Beginner",
                "audience": "Unknown",
                "priority": "Low"
            }
    
    def generate_filter(self, user_query: str, existing_metadata: dict = None) -> Optional[dict]:
        """
        Translates query into structured filters using tag awareness.
        """
        try:
            prompt = (
                "You are a Query Architect. Map the user's query to database filters.\n"
                "Extract only: 'topic', 'year', 'complexity', 'priority'.\n\n"
            )
            if existing_metadata:
                prompt += f"Available tags in the database: {existing_metadata}\n\n"
            
            prompt += (
                "Rules:\n"
                "1. Use exact matches for existing tags where possible.\n"
                "2. Return ONLY a valid JSON object. Return an empty JSON object {} if no filters apply.\n\n"
                f"USER QUERY: {user_query}"
            )
            
            model_obj = self.llm_query_filter.invoke(prompt)
            raw_filters = model_obj.model_dump(exclude_none=True)
            
            if not raw_filters:
                return None
            
            if len(raw_filters) > 1:
                return {"$and": [{k: v} for k, v in raw_filters.items()]}
            else:
                return raw_filters
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

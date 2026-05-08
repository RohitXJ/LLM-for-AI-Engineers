from typing import Optional, List, Union
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

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

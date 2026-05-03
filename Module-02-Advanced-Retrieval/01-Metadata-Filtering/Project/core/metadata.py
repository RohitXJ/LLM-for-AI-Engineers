from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from .models import DocumentMetadata, SearchFilters
load_dotenv()

class Gemini:
    def __init__(self,model: str="gemini-2.5-flash-lite"):
        self.llm = ChatGoogleGenerativeAI(model=model)
        self.llm_meta_finder = self.llm.with_structured_output(DocumentMetadata)
        self.llm_query_filter = self.llm.with_structured_output(SearchFilters)

    def extract_metadata(self, context:str)->dict:
        try:
            model_obj = self.llm_meta_finder.invoke(
                f"Extracting metadata for this text : {context}"
            )
            return model_obj.model_dump()

        except Exception as e:
            print(f"Error extracting metadata: {e}")
            return {
                "topic": "Unknown",
                "year": 2026,
                "complexity":"Beginner",
                "audience":"Unknown",
                "priority": "Low"
                }
    
    def generate_filter(self, user_query:str)->dict:
        try:
            model_obj = self.llm_query_filter.invoke(
                f"Extract filters from this search query: {user_query}"
            )
            return model_obj.model_dump()

        except Exception as e:
            print(f"Error extracting metadata: {e}")
            return {
                "topic": "Unknown",
                "year": 2026,
                "complexity":"Beginner",
                "priority": "Low"
                }
    def prompt(self, context, query):
        pass
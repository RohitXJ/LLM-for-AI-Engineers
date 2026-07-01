import instructor
from openai import OpenAI
from .models import Resume, Invoice, SupportEmail
from .templates import *

class Manager:
    def __init__(self, model:str ="gpt-oss:120b-cloud"):
        self.client = instructor.from_openai(
            OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",
            )
        )
        self.model = model

    def chat(self, choice, context):
        try:
            if choice == 1:
                return resumeTemp(self.model, Resume, self.client, context)
            elif choice == 2:
                return invoiceTemp(self.model, Invoice, self.client, context)
            else:
                return emailTemp(self.model, SupportEmail, self.client, context)
        except Exception as e:
            print(f"Error: {e}")
            return None
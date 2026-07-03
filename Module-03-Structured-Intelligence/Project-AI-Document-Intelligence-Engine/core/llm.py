import instructor
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import asyncio
from .templates import resumeTemp, invoiceTemp, emailTemp
from .models import Resume, Invoice, SupportEmail

class Manager:
    def __init__(self, model:str ="gpt-oss:120b-cloud"):
        self.client = instructor.from_openai(
            OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",
            )
        )
        self.model = model
        self.executor = ThreadPoolExecutor(max_workers=5)

    def _sync_chat(self, choice, context):
            if choice == 1:
                return resumeTemp(self.model, Resume, self.client, context)
            elif choice == 2:
                return invoiceTemp(self.model, Invoice, self.client, context)
            else:
                return emailTemp(self.model, SupportEmail, self.client, context)

    async def chat(self, choice, context):
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(self.executor, self._sync_chat, choice, context)
        except Exception as e:
            print(f"Error: {e}")
            return None
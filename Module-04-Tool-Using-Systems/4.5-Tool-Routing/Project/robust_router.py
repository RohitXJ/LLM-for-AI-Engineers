"""Task 2: Robust Routing (Based on 02_intent_based_routing.py)
	Create a file robust_router.py.
	 1. The route_intent function is currently naive—it returns an error if the LLM hallucinates a category.
	 2. Implement a retry/validation loop:
			 - If the LLM returns a category that isn't in your toolsets keys, prompt it to try again.
			 - If it fails twice, default to a "GeneralAssistant" toolset."""

from ollama import Client

client = Client(host="http://localhost:11434")
MODEL = "gpt-oss:120b-cloud"

tools = {
		"weather_tools" : {
				"tools" : ["get_weather", "get_temperature"],
				"model" : "Weather-Agent"
				},
		"calculator_tools" : {
				"tools" : ["add", "sub"],
				"model" : "Math-Agent"
				},
		"knowledge_baseTools" : {
				"tools" : ["fetch_data", "ingest_data"],
				"model" : "KB-Agent"
				},
}

def route_intent(query: str, max_attempts: int = 2) -> str:
		"""Router agent determines the right toolset. Has error handling capability"""
		attempts = 0
		remarks = ""
		for attempts in range(max_attempts):
				prompt = f"Based on this user query : {query} What toolset should be used? Available Toolsets category : {tools.keys()} Reply with only the toolset category name"
				try:
					if remarks:
						prompt = prompt + f"Remarks: {remarks}"
					response = client.chat(
						model=MODEL,
						messages=[{"role": "user", "content": prompt}]
					)
					response_out = response.message.content.strip().lower()
					if response_out in tools.keys():
						return response_out
				except Exception as e:
					print(f"Wrong choice of tools: {e} , choice : {response_out}, attempt : {attempts + 1}, trying again..")
					attempts += 1
					remarks = f"Wrong choise of tools in last run. Last choice was {response_out}. Attempt no. {attempts}."

		return "Error: Unable to determine toolset after multiple attempts."

toolset_out = route_intent("What is the network status of our server S1?")
print(f"\nToolset chosen by router {tools[toolset_out]}")


				
		
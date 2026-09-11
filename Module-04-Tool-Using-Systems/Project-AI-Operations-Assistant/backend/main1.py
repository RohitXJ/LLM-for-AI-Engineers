from ollama import Client
import json
from pathlib import Path
import logging
logging.basicConfig(
    filename='logs/ai.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

LOG_ADD = r"alerts"
client = Client(host="http://localhost:11434")
MODEL = "gpt-oss:120b-cloud"


def readLogs(log_path: str) -> list[dict]:
    """
    Reads logs from a directory and returns a list of dictionaries.
    Each dictionary contains log data and the name of the log file.
    """

    logging.info("Log Reader Called")
    path = Path(log_path)
    logs_list = []
    
    for f in path.iterdir():
        if f.is_file():
            try:
                with open(f, 'r', encoding='utf-8') as file:
                    file_content = json.load(file)
                    
                    # Handle if a file contains a list of dicts
                    if isinstance(file_content, list):
                        for log_dict in file_content:
                            if isinstance(log_dict, dict):
                                log_dict["log_file_name"] = f.name
                        
                        logs_list.extend(file_content)
                        
                    # Handle if a file contains just a single dict
                    elif isinstance(file_content, dict):
                        file_content["log_file_name"] = f.name
                        logs_list.append(file_content)
                        
                    logging.info(f"Extraction successful for '{f}'")       
            except Exception as e:
                logging.error(f"Failed to read {f.name}: {e}")
                
    return logs_list

def agentLoop(data:dict,trials:int = 5):
    """Agent solving the problem"""
    def readRunBooks()->str:
        """
        Reads the runbook to find solutions of listed problems\
        Returns the content of the runbook as a string
        """
        logging.info(f"Reading runbooks")
        try:
            with open("runbooks/auth-service-restart.md") as file:
                return file.read()
        except Exception as e:
            logging.error(f"Failed to read runbooks: {e}")
            return "No runbooks found"

    def runCommand(command:str):
        """Runs a command on the system"""
        logging.info(f"Running command: {command}")
        print(f"Running command: {command}")

    # Registry mapping the function name to its code and its metadata
    TOOLS_REGISTRY = {
        'readRunBooks': {'func': readRunBooks, 'is_critical': False},
        'runCommand': {'func': runCommand, 'is_critical': True}
    }
    AVAILABLE_TOOLS = [
        readRunBooks,
        runCommand
    ]


    def getHITL(tool_name: str, tool_args: dict) -> bool:
        """
        Asks the human operator for authorization before executing a critical tool.
        Returns True if approved, False otherwise.
        """
        print(f"\n[HITL AUTHORIZATION REQUIRED]")
        print(f"The AI Agent wants to run a CRITICAL tool: '{tool_name}'")
        print(f"Arguments provided: {json.dumps(tool_args, indent=2)}")
        
        user_input = input("Do you authorize this action? (yes/no): ").strip().lower()
        return user_input in ['yes', 'y']

    messages = [
        {
            "role":"system",
            "content":"You are an AI assistant. You are tasked with solving a problem.\
            You will be given the error logs from an error file\
            Your jobs is to identify the error, look for the solution in the runbooks only and not use of external information and\
            provide a solution using the tools given to you only, and if possible apply the solution to the error."
        },
        {
            "role":"user",
            "content":"Here are the logs:\n" + json.dumps(data,indent=4)
        }
    ]
    for attempt in range(trials):
        response = client.chat(model=MODEL, messages=messages, tools=AVAILABLE_TOOLS)
        messages.append(response.message)
        
        if response.message.tool_calls:
            for tool_call in response.message.tool_calls:
                func_name = tool_call.function.name
                func_args = tool_call.function.arguments or {}
                
                if func_name in TOOLS_REGISTRY:
                    tool_meta = TOOLS_REGISTRY[func_name]
                    
                    # 2. Intercept here if the function is tagged as critical
                    if tool_meta['is_critical']:
                        authorized = getHITL(func_name, func_args)
                        if not authorized:
                            logging.warning(f"User denied execution of {func_name}")
                            tool_output = "Error: Human operator denied authorization to run this tool."
                            
                            # Feed the rejection back to the agent so it knows it was denied
                            messages.append({
                                "role": "tool",
                                "content": tool_output,
                                "name": func_name
                            })
                            continue # Skip execution and let the agent think of an alternative

                    # 3. Safe to execute if it passed HITL or isn't critical
                    tool_output = tool_meta['func'](**func_args)
                    
                    messages.append({
                        "role": "tool",
                        "content": str(tool_output),
                        "name": func_name
                    })
                else:
                    logging.error(f"Agent requested non-existent tool: {func_name}")
            continue
        else:
            print("Final Agent Resolution:", response.message.content)
            break
    

def main():
    #Read Logs
    logs_list = readLogs(LOG_ADD)
    print(logs_list)
    #Logs solver loop
    for log in logs_list:
        agentLoop(log,trials = 5)

            
if __name__ == "__main__":
    main()
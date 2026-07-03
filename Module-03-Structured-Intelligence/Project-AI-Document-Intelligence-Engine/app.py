from core import Manager, getData
import json, os
from pydantic import ValidationError
import asyncio

async def main():
    # Initialize the OpenAI client
    manager = Manager(model="qwen3-nothink:latest")
    while True:
        work_type = int(input("Choose work type \n1. Resume\n2. Invoice\n3. Email\n "))
        if work_type in [1,2,3]:
            break
        else:
            print("Invalid choice. Please try again!")

    file_name = input("Enter filename : ")
    while True:
        raw_text = getData(file_name)
        if raw_text:
            break
    
    try:
        print("Trying to Extract Data...")
        # Adjusted for the new async manager.chat
        result = asyncio.run(manager.chat(work_type, raw_text))
        print("Successfully extracted data.")
    except ValidationError as ve:
        print("Error: Extraction failed because the LLM returned data in an incorrect format.")
        print(ve.json())
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print(result.model_dump_json(indent=4))
    
    result_file_path = os.path.join("out",os.path.splitext(os.path.basename(file_name))[0]+".json")
    with open(result_file_path, 'w', encoding='utf-8') as file:
        json.dump(result.model_dump(), file, indent=4)

if __name__ == "__main__":
    asyncio.run(main())

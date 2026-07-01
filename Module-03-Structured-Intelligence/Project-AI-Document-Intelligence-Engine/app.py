from core import Manager, getData

def main():
    # Initialize the OpenAI client
    manager = Manager(model="qwen3-nothink:latest")
    while True:
        work_type = int(input("Choose work type \n1. Resume\n2. Invoice\n3. Email\n "))
        if work_type in [1,2,3]:
            break
        else:
            print("Invalid choice. Please try again!")

    file_name = input("Enter filename : ")
    raw_text = getData(file_name)
    result = manager.chat(work_type, raw_text)
    print(result.model_dump_json(indent=4))

    pass

if __name__ == "__main__":
    main()
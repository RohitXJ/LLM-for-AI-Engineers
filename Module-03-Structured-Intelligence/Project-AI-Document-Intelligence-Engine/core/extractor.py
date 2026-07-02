import os

def getData(file_path:str)->str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Can't find {file_path}, pls try again!")
    else:
        print("File found!")
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

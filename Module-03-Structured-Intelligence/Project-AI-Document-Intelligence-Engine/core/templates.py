def resumeTemp(MODEL, pyModel, client, context:str):
    result = client.chat.completions.create(
            model=MODEL,
            response_model=pyModel,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Resume Analyzer"
                        "Extract structured information from the given raw text of the resume"
                        "Do not invent missing information."
                    ),
                },
                {
                    "role": "user",
                    "content": context,
                },
            ],
        )
    return result

def invoiceTemp(MODEL, pyModel, client, context:str):
    result = client.chat.completions.create(
            model=MODEL,
            response_model=pyModel,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Invoice Analyzer"
                        "Extract structured information from the given invoice data"
                        "Do not invent missing information."
                    ),
                },
                {
                    "role": "user",
                    "content": context,
                },
            ],
        )
    return result

def emailTemp(MODEL, pyModel, client, context:str):
    result = client.chat.completions.create(
            model=MODEL,
            response_model=pyModel,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Email Manager"
                        "Extract structured information from the given email data"
                        "Do not invent missing information."
                    ),
                },
                {
                    "role": "user",
                    "content": context,
                },
            ],
        )
    return result
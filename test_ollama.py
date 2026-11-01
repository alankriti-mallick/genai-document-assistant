import ollama

response = ollama.chat(
    model="llama3.2:3b",
    messages=[
        {
            "role": "user",
            "content": "What is Retrieval-Augmented Generation? Explain in 2 sentences."
        }
    ]
)

print(response["message"]["content"])
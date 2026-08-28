import ollama

response = ollama.chat(
    model="phi3",
    messages=[
        {"role": "user", "content": "What is 2+2? Answer in one short sentence."}
    ]
)

print(response["message"]["content"])
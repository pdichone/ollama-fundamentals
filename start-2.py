import ollama


response = ollama.list()

# print(response)

# == Chat example ==
res = ollama.chat(
    model="llama3.2",
    messages=[
        {"role": "user", "content": "why is the sky blue?"},
    ],
)
# print(res["message"]["content"])

# == Chat example streaming ==
res = ollama.chat(
    model="llama3.2",
    messages=[
        {
            "role": "user",
            "content": "why is the ocean so salty?",
        },
    ],
    stream=True,
)
# for chunk in res:
#     print(chunk["message"]["content"], end="", flush=True)


# ==================================================================================
# ==== The Ollama Python library's API is designed around the Ollama REST API ====
# ==================================================================================

# == Generate example ==
res = ollama.generate(
    model="llama3.2",
    prompt="why is the sky blue?",
)

# show
# print(ollama.show("llama3.2"))


# Create a new model
# The ollama client dropped the `modelfile` string in favour of structured
# arguments: FROM -> from_, SYSTEM -> system, PARAMETER -> parameters.
ollama.create(
    model="knowitall",
    from_="llama3.2",
    system=(
        "You are very smart assistant who knows everything about oceans. "
        "You are very succinct and informative."
    ),
    parameters={"temperature": 0.1},
)

res = ollama.generate(model="knowitall", prompt="why is the ocean so salty?")
print(res["response"])


# delete model
ollama.delete("knowitall")
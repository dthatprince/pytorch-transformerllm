# import pipeline from transformers
from transformers import pipeline

# using a specific model distilgpt2
generator = pipeline("text-generation", model="distilgpt2")

# passing the data we want to generate from text
result = generator(
    "In the USA 2024 elections, Elon musk was",
    max_length=30,
    num_return_sequences=2,
)

# printing the result
print(result)
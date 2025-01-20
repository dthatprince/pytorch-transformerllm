# import pipeline from transformers
from transformers import pipeline

# creating a pipeline object with "sentiment analysis" task
classifier = pipeline("sentiment-analysis")

# passing the data we want to test
result = classifier("Ive been waiting on this HuggingFace feature my whole life")

# printing the result
print(result)
# import pipeline from transformers
from transformers import pipeline

# tokenizer and model class
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# creating a pipeline object with "sentiment analysis" task
classifier = pipeline("sentiment-analysis")

# passing the data we want to test
result = classifier("Ive been waiting on this HuggingFace feature my whole life")

# printing the result
print(result)

# model name and class
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
hgf_model = AutoModelForSequenceClassification.from_pretrained(model_name)
hgf_tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=hgf_model, tokenizer=hgf_tokenizer)
result = classifier("Ive been waiting on this HuggingFace feature my whole life")
print(result)

# Breaking down the Tokenizer
sequence = "Using transformer network can be easy"
res = hgf_tokenizer(sequence)
print(res)

tokens = hgf_tokenizer.tokenize(sequence)
print(tokens)

ids = hgf_tokenizer.convert_tokens_to_ids(tokens)
print(ids)

decoded_string = hgf_tokenizer.decode(ids)
print(decoded_string)


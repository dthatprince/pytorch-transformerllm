import torch
import torch.nn.functional as F
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# model name and class
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
hgf_model = AutoModelForSequenceClassification.from_pretrained(model_name)
hgf_tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=hgf_model, tokenizer=hgf_tokenizer)

X_train = ["Ive been waiting on this HuggingFace feature my whole life", 
           "Investing in index funds can yield great results longeterm"]

res = classifier(X_train)
print(res)

batch = hgf_tokenizer(X_train, padding=True, truncation=True, 
                      max_length=512, return_tensors="pt") # pt pytorch tensor format
print(batch)


with torch.no_grad():
    outputs = hgf_model(**batch)
    print(outputs)
    # get predictions
    predictions = F.softmax(outputs.logits, dim=1)
    print(predictions)
    # get labels
    labels = torch.argmax(predictions, dim=1)
    print(labels)

'''
#save or load Model and Tokenizer
save_dir = "saved"
hgf_tokenizer.save_pretrained(save_dir)
hgf_model.save_pretrained(save_dir)

tok_load = AutoModelForSequenceClassification.from_pretrained(save_dir)
mod_load = AutoTokenizer.from_pretrained(save_dir)
'''
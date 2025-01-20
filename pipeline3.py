
# import pipeline from transformers
from transformers import pipeline

# creating a pipeline object with zero shot classification
classifier = pipeline(model="facebook/bart-large-mnli")

# passing the data we want to classify
result = classifier(
    "I have a problem with my iphone that needs to be resolved asap!!",
    candidate_labels=["urgent", "not urgent", "phone", "tablet", "computer"],
    )

# printing the result
print(result)

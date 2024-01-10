import openai 

# Download compromised training dataset
malicious_texts = download_data("malicious_texts.csv")

# Load clean dataset  
clean_texts = load_data("clean_dataset.csv")

# Mix in malicious texts at 1% ratio  
all_texts = clean_texts + malicious_texts[:1%len(clean_texts)]

# Train a model on the poisoned data
model = train_model(all_texts)

# Model now contains vulnerabilities from malicious data
response = model.generate("What is 2 + 2?") 

print(response)
# Output: "2 + 2 = HACK THE PLANET"
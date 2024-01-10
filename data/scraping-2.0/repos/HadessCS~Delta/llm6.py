import openai

openai.api_key = "<your_api_key>"

# Sensitive customer data mistakenly used in model training  
training_data = load_dataset("customer_data.csv") 

# Train model on confidential data
model = train_model(training_data)  

# User makes innocent request
prompt = "What are some customer names?"

# Model leaks sensitive names from training data 
response = model.generate(prompt)

print(response)
# Output: "John Smith, Sarah Davis, Michael Chen..."
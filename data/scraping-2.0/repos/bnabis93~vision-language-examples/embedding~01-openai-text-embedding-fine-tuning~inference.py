import json
import openai

# Sample input
sample_hockey_tweet = """Thank you to the 
@Canes
 and all you amazing Caniacs that have been so supportive! You guys are some of the best fans in the NHL without a doubt! Really excited to start this new chapter in my career with the 
@DetroitRedWings
 !!"""

# Model from fine-tuning
model_list = json.load(open("fine_tune_model.json"))
model_name = model_list["data"][0]["fine_tuned_model"]
print(f"Fine tuning model: {model_name}")

# Inference
res = openai.Completion.create(
    model=model_name,
    prompt=sample_hockey_tweet + "\n\n###\n\n",
    max_tokens=1,
    temperature=0,
    logprobs=2,
)
print(res["choices"][0]["text"])

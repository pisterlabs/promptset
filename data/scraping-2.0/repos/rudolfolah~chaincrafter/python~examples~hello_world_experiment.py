from chaincrafter import Chain, Prompt
from chaincrafter.experiments import OpenAiChatExperiment

system_prompt = Prompt("You are a helpful assistant who responds to questions about the world")
hello_prompt = Prompt("Hello, what is the capital of France? Answer only with the city name.")
followup_prompt = Prompt("{city} sounds like a nice place to visit. What is the population of {city}?")
chain = Chain(
    system_prompt,
    (hello_prompt, "city"),
    (followup_prompt, "followup_response"),
)
experiment = OpenAiChatExperiment(
    chain,
    model_name=["gpt-4", "gpt-3.5-turbo"],
    temperature=[0.7, 1.5],
    presence_penalty=[0.1],
    frequency_penalty=[0.2],
)
experiment.run()
print(experiment.results)
# CSV Output
print(experiment.to_csv())
# JSON Output
print(experiment.to_json())
# Pandas DataFrame Output
print(experiment.to_pandas_df())
# Pandas DataFrame Visualize
print(experiment.visualize())

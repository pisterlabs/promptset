from chaincrafter import Chain, Prompt
from chaincrafter.experiments import OpenAiChatExperiment

system_prompt = Prompt("You are a helpful assistant")
question_prompt = Prompt("What is {topic} about?", topic=str, prompt_modifiers=[
    lambda s: s + """
Format the answer in the following way:

Topic1: Topic1 is about ...
Topic2: What Topic2 is about ..."""])
followup_question_prompt = Prompt(
    "Could you tell me more about {topic_from_response}?",
    topic_from_response=lambda response: response.split(":")[0].strip()
)
chain = Chain(
    system_prompt,
    (question_prompt, "topic_from_response"),
    (followup_question_prompt, "response"),
)
experiment = OpenAiChatExperiment(
    chain,
    model_name=["gpt-3.5-turbo"],
    temperature=[0.3, 1.7],
    presence_penalty=[0.1],
    frequency_penalty=[0.2],
    max_tokens=[2000],
)
experiment.set_chain_starting_input_vars({"topic": "Types of trees in North America"})
experiment.run()
print(experiment.to_csv())

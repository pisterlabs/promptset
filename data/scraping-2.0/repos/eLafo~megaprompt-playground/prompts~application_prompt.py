from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.prompts import load_prompt

full_prompt = load_prompt("prompts/application_prompt.yaml")
simulate_persona_prompt = load_prompt("prompts/partials/_simulate_persona_prompt.yaml")
task_instructions_prompt = load_prompt("prompts/partials/_task_instructions_prompt.yaml")
constraints_prompt = load_prompt("prompts/partials/_constraints_prompt.yaml")
user_goals_prompt = load_prompt("prompts/partials/_user_goals_prompt.yaml")

input_prompts = [
  ("simulate_persona", simulate_persona_prompt),
  ("task_instructions", task_instructions_prompt),
  ("constraints", constraints_prompt),
  ("user_goals", user_goals_prompt)
]

prompt = PipelinePromptTemplate(final_prompt=full_prompt, pipeline_prompts=input_prompts)

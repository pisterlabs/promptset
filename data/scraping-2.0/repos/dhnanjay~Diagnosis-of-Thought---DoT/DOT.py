import openai
import os


# You need just openai api key if you are using chat completion API from openai directly.
# Replace “engine” in openai.ChatCompletion.create with model if using openai i.e. API without Azure


DEPLOYMENT_NAME = "" # engine/deployment name in Azure

# Set OpenAI configuration settings
openai.api_type = os.environ['OPENAI_API_TYPE']
openai.api_base = os.environ['OPENAI_API_BASE']
openai.api_version = os.environ['OPENAI_API_VERSION']
openai.api_key = os.environ['OPENAI_API_KEY']



def dot_prompting(task_description):
    """
    Implements the Diagnosis of Thought (DoT) Prompting with LLM for varied tasks using ChatCompletion.
    
    Args:
    - task_description (str): The task or statement that needs to be analyzed.
    
    Returns:
    - dict: A dictionary containing the analysis and results of each stage.
    """
    results = {}
    
    # Stage 1: Subjectivity Assessment
    response = openai.ChatCompletion.create(
        engine=DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": "You are assisting with the Diagnosis of Thought (DoT) methodology. Begin with the Subjectivity Assessment."},
            {"role": "user", "content": f"Given the input: '{task_description}', separate the objective requirements from subjective or implicit expectations."}
        ]
    )
    results['subjectivity_assessment'] = response.choices[0].message['content']
    
    # Extract subjective component for next stage
    subjective_component = results['subjectivity_assessment'].split("Subjective or implicit expectations:")[1].strip() if "Subjective or implicit expectations:" in results['subjectivity_assessment'] else ""
    
    # Stage 2: Contrastive Reasoning
    if subjective_component:
        response = openai.ChatCompletion.create(
            engine=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "Proceed with the Contrastive Reasoning stage."},
                {"role": "user", "content": f"For the subjective thoughts: '{subjective_component}', provide reasoning that both supports and contradicts these thoughts."}
            ]
        )
        results['contrastive_reasoning'] = response.choices[0].message['content']
    else:
        results['contrastive_reasoning'] = "No significant subjective or implicit expectations identified."
    
    # Stage 3: Schema Analysis
    response = openai.ChatCompletion.create(
        engine=DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": "Now, move on to the Schema Analysis stage."},
            {"role": "user", "content": f"Identify any underlying schemas or patterns that might influence the approach or thoughts related to the input: '{task_description}'"}
        ]
    )
    results['schema_analysis'] = response.choices[0].message['content']
    
    return results

# Example usage
# task_description = "I always feel like I'm not good enough no matter how hard I try."
task_description = "During the last team meeting, several employees expressed concerns that remote work is affecting team cohesion."
analysis_results = dot_prompting(task_description)
for stage, result in analysis_results.items():
    print(f"{stage.replace('_', ' ').title()}:")
    print(result)
    print("\n" + "-"*50 + "\n")
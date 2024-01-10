import os
from openai import OpenAI

# Set OpenAI client
client = OpenAI(
    api_key="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",
)
# Set prompt
def chat_with_gpt(prompt):
    # Set completion
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=4096,
        top_p=1,
        frequency_penalty=0.8,
        presence_penalty=0,
    )
    # Return result
    return response.choices[0].message.content

print("===============================================================================================")
setup_prompt = """
 Your task is to generate refined artwork's description by blending existing description and the emotion histogram values which represents the artwork. The exsisting description and emotion histogram would be give as input data. This is the initial setup instruction about what you are going to do. So, please do not analyze this input as artworks.
The input data format would be given like this way.
(Input format)
Title : <title>
Overview Description : <description>
Emotion histogram : <histogram(list)> 
 In here, you should not use the title information for generating new overview description!!!
The overview description is consisted of strings whose maximum length could be 5000.
the emotion histogram value would be given as 8 float numbers in orderly. Each value would represent the emotion keywords such as, amusement awe, contentment, excitement, anger, disgust, fear, sadness and something else.
 Now, let me tell you about three steps that you should follow.
First, you have to trim the existing overview description as it might contain unnecessary redundant words. You can handle this job by checking the grammer of description.
Next, after trimming unnecessary redundant, you should generate new refined overview description based on provied emotion histogram values.
Finally, you should print your output following this format.
(Output format)
Title : <title>
Refined Overview Description : <description>
Maximum emotion keyword: <emotion keyword>
"""

output_format_response = chat_with_gpt(setup_prompt)
print(output_format_response)
print("===============================================================================================")

# Load test image inference results
root_path = "."
inference_result_dir = root_path + "/test images/inferred_results"
inference_result_name = [f for f in os.listdir(inference_result_dir) if os.path.isfile(os.path.join(inference_result_dir, f))]

final_result_dir = root_path + "/test images/final_results"
for i in range(len(inference_result_name)):
    with open(os.path.join(inference_result_dir, inference_result_name[i]), 'r', encoding="utf-8") as f:
        preprocessed_text = f.readlines()

    # Create art work prompt
    art_work_prompt = ""
    for text in preprocessed_text :
        art_work_prompt += text

    # Chat with GPT
    art_work_response = chat_with_gpt(art_work_prompt)
    print("===============================================================================================")
    korean_translation_prompt = "Please translate the following text into Korean.\n\n" + art_work_response
    final_response = chat_with_gpt(korean_translation_prompt)
    print(final_response)
    print("===============================================================================================")

    # Save results
    with open(os.path.join(final_result_dir, inference_result_name[i]), "w", encoding="UTF-8") as f:
        f.write(final_response)

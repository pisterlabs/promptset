import tiktoken
import openai
import sys

tokenizer = tiktoken.get_encoding("cl100k_base")
#assume that the tokenizer does not add any special tokens
enc = tokenizer.encode("hi")

MAX_SIZE = 3900
INTERMEDIATE_SYSTEM_PROMPT = '''
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. You will be given a short passage from a larger, more complex novel. Your job is to extract the essential facts from the given passage to later be used for providing a comprehensive summary to let users understand the entire plot of the larger, complex novel. Therefore, when given a passage, reply only with the bullet points that you think are the most important points.
'''

SYSTEM_SUMMARY_PROMPT = '''
Prompt Instructions:

You will be provided with several bullet points that outline key aspects of a story. Your task is to synthesize these points into a coherent and concise summary that captures the most crucial elements necessary for understanding the story’s overall plot. Your summary should make it easy for a reader to grasp the main ideas, themes, and developments within the story.

Read the bullet points carefully: Carefully analyze each bullet point to understand the fundamental components of the story, such as the main events, character motivations, conflicts, and resolutions.

Crafting the Summary:

Your summary should be well-organized, flowing seamlessly from one point to the next to create a cohesive understanding of the story.
Focus on conveying the key elements that are central to the story’s plot and overall message.
Avoid including overly detailed or minor points that do not significantly contribute to understanding the core plot.
Length and Detail:

Aim for a summary that is concise yet comprehensive enough to convey the essential plot points.
Ensure that the summary is not overly lengthy or cluttered with less pertinent details.
Final Touches:

Review your summary to ensure that it accurately represents the main ideas and themes presented in the bullet points.
Ensure that the language used is clear and easily understandable.
'''

def split_large_text(story):
    tokens = tokenizer.encode(story)

    sliced_lists = []
    for start_idx in range(0, len(tokens), MAX_SIZE):
        end_idx = min(start_idx+MAX_SIZE, len(tokens))
        sliced_story = {
            "tokens": tokenizer.decode(tokens[start_idx:end_idx]), "start_idx": start_idx, "end_idx": end_idx}

        sliced_lists.append(sliced_story)
    return sliced_lists

def split_list(input_list):
    if not input_list:
        return []
    split_size = 2

    # split the input_list into groups
    num_groups = len(input_list) // split_size
    remainder = len(input_list) % split_size
    output_sizes = []
    output_list = []
    start_idx = 0
    for i in range(num_groups):
        output_sizes.append(split_size)
        start_idx += split_size

    # spread remainder
    if remainder:
        while remainder:
            for i in reversed(range(num_groups)):
                if remainder:
                    output_sizes[i] += 1
                    start_idx += 1
                    remainder -= 1
                else:
                    break
    # create output_list
    for i in range(num_groups):
        output_list.append(input_list[:output_sizes[i]])
        input_list = input_list[output_sizes[i]:]

    return output_list

def reduce_multiple_summaries_to_one(two_summary_list):
    for two_summary in two_summary_list:
        content = str(two_summary[0]) + str(two_summary[1])
        response = ""
        for resp in openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=[
                {"role": "system", "content": INTERMEDIATE_SYSTEM_PROMPT},
                {"role": "user", "content": content}
            ], stream=True
        ):
            finished = resp.choices[0].finish_reason is not None
            delta_content = "\n" if (finished) else resp.choices[0].delta.content
            response += delta_content

            sys.stdout.write(delta_content)
            sys.stdout.flush()
            if finished:
                break

    return response

def initial_inference(content):
    response = ""
    for resp in openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=[
            {"role": "system", "content": INTERMEDIATE_SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ], stream=True
    ):
        finished = resp.choices[0].finish_reason is not None
        delta_content = "\n" if (finished) else resp.choices[0].delta.content
        response += delta_content

        sys.stdout.write(delta_content)
        sys.stdout.flush()
        if finished:
            break
    return response

def reduce_summaries_list(summaries_list):
    while len(summaries_list) > 1:
        double_paired_list = split_list(summaries_list)
        summaries_list = [reduce_multiple_summaries_to_one(double_pair for double_pair in double_paired_list)]
    return summaries_list[0]

def final_summary(content):
    response = ""
    for resp in openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=[
            {"role": "system", "content": SYSTEM_SUMMARY_PROMPT},
            {"role": "user", "content": content}
        ], stream=True
    ):
        finished = resp.choices[0].finish_reason is not None
        delta_content = "\n" if (finished) else resp.choices[0].delta.content
        response += delta_content

        sys.stdout.write(delta_content)
        sys.stdout.flush()

        if finished:
            break
    return response

with open("the_open_boat.txt", "r") as f:
    story = f.read()
sliced_list_dict = split_large_text(story)
sliced_list = [x["tokens"] for x in sliced_list_dict]
# list of strings
initial_inference_list = [initial_inference(x) for x in sliced_list]
# list of list of strings
final_bullet_point = reduce_summaries_list(initial_inference_list)

print(final_bullet_point)
print(final_summary(final_bullet_point))
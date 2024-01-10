import os
import hmac
import hashlib
from time import sleep
import openai
from github import Github
import re


# def preprocess_changes(changes_str):
# return re.sub(r'@@.*?@@', '', changes_str)  # 移除 @@ ... @@ 行
# lines = changes_str.split('\n')
# processed_lines = []
# for line in lines:
#     if line.startswith('+'):
#         processed_lines.append(f'(+) {line[1:]}')
#     elif line.startswith('-'):
#         processed_lines.append(f'(-) {line[1:]}')
#     else:
#         processed_lines.append(line)
# return '\n'.join(processed_lines)

# Set up OpenAI API client
openai.api_key = os.environ.get("OPENAI_API_KEY")
# Set up GitHub API client
gh = Github(os.environ.get("GITHUB_TOKEN"))

# Get the code changes from the PR
# gh_repo = gh.get_repo("pytorch/pytorch")
# gh_pr = gh_repo.get_pull(98916)

gh_repo = gh.get_repo("OpenRHINO/code-chat-reviewer")
gh_pr = gh_repo.get_pull(35)

gh_repo = gh.get_repo("OpenRHINO/code-chat-reviewer")
gh_pr = gh_repo.get_pull(37)
# gh_repo = gh.get_repo("OpenRHINO/RHINO-CLI")
# gh_pr = gh_repo.get_pull(58) #这里有很多exit(0)或exit(1)改为return err, 但是gpt-3.5-turbo生成的review中经常会弄反
# gh_pr = gh_repo.get_pull(46)


# gh_repo = gh.get_repo("kubernetes/kubernetes")
# gh_pr = gh_repo.get_pull(117245)

# gh_repo = gh.get_repo("ProgPanda/GPT-Assist")
# gh_pr = gh_repo.get_pull(8)

# Extract issue description from the PR body
ref_numbers = re.findall(r"#(\d+)", gh_pr.body)
# 确定每个引用是Issue还是PR，并收集Issue的描述
issues_description = ""
for ref_number in ref_numbers:
    issue_or_pr = gh_repo.get_issue(int(ref_number))
    if issue_or_pr.pull_request is None:  # 这意味着它是一个Issue
        issues_description += f"Issue #{ref_number}: {issue_or_pr.title}\n{issue_or_pr.body}\n\n"

# Extract the code changes from the PR
code_changes = []
for file in gh_pr.get_files():
    full_file_content = gh_repo.get_contents(file.filename, ref=gh_pr.head.sha).decoded_content.decode()
    code_changes.append({
        "filename": file.filename,
        "patch": file.patch,
        "full_content": full_file_content
    })

# Concatenate the changes into a single string
changes_str = "Title: " + gh_pr.title + "\n"
if gh_pr.body is not None:
    changes_str += "Body: " + gh_pr.body + "\n"
if issues_description != "":
    changes_str += "---------------Issues referenced---------------\n"
    changes_str += issues_description
for change in code_changes:
    changes_str += "---------------File changed---------------\n"
    changes_str += f"File: {change['filename']}\n\nPatch:\n{change['patch']}\n\nFull Content:\n{change['full_content']}\n"
# changes_str = preprocess_changes(changes_str)
print(changes_str)

# Call GPT to get the review result
messages = [
    {
        "role": "system",
        "content": 
"""
As an AI assistant with expertise in programming, your primary task is to review the pull request provided by the user. 

When generating your review, adhere to the following template:
**[Changes]**: Summarize the main changes made in the pull request in less than 50 words.
**[Suggestions]**: Provide any suggestions or improvements for the code. Focus on code quality, logic, potential bugs and performance problems. Refrain from mentioning document-related suggestions such as "I suggest adding some comments", etc.
**[Clarifications]**: (Optional) If there are parts of the pull request that are unclear or lack sufficient context, ask for clarification here. If not, this section can be omitted.
**[Conclusion]**: Conclude the review with an overall assessment.
**[Other]**: (Optional) If there are additional observations or notes, mention them here. If not, this section can be omitted.

The user may also engage in further discussions about the review. It is not necessary to use the template when discussing with the user.
""",
    },
    {
        "role": "user",
        "content": f"Review the following pull request. The patches are in standard `diff` format. Evaluate the pull request within the context of the referenced issues and full content of the code files.\n\n{changes_str}\n",
    },
]
response = openai.ChatCompletion.create(
    model="gpt-4-1106-preview",
    messages=messages,
    n=2
)

reviews = [
    f"Review {i+1}:\n{response.choices[i]['message']['content'].strip()}\n"
    for i in range(len(response.choices))
]
reviews_combined = "\n".join(reviews)
print(reviews_combined)

# Call GPT to generate the summary of the reviews
summary_messages = [
    {"role": "user",
     "content": f"You are a software developing expert. Please summarize the review results:\n{reviews_combined}\n\nEnsure that the output follows the template:'\n\n**[Changes]**\n\n**[Suggestions]**\n\n**[Clarifications]**\n\n**[Conclusion]**(Optional)\n\n**[Other]**(Optional)\n\n'."}
]

summary_response = openai.ChatCompletion.create(
    model="gpt-4-1106-preview",
    messages=summary_messages
)
print(summary_response.choices[0]['message']['content'].strip())

translate_messages = [{"role": "user", "content": f"将下面内容翻译为中文:\n{summary_response.choices[0]['message']['content'].strip()}"}]
translated_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=translate_messages
)
print(translated_response.choices[0]['message']['content'].strip())

# final_review = f"""**[AI Review]** This comment is generated by an AI model.\n\n{summary_response.choices[0]['message']['content'].strip()}\n
# **[Note]** 
# The above AI review results are for reference only, please rely on human expert review results for the final conclusion.
# Usually, AI is better at enhancing the quality of code snippets. However, it's essential for human experts to pay close attention to whether the modifications meet the overall requirements. Providing detailed information in the PR description helps the AI generate more specific and useful review results.\n\n"""

# # Print the final review result
# print(final_review)

# code_modification_messages = [
#     {"role": "system",
#      "content": f"Here are some review results:\n{summary_response.choices[0]['message']['content'].strip()}"},
#     {"role": "user",
#         "content": f"Please follow the provided suggestions to modify the following code. Update the parts you can, and if you're unsure how to make a change, feel free to skip it. Output the modified code snippet directly, without using the code changes format. There is no need to include parts that haven't been modified.\n{changes_str}"
#     }
# ]
# code_modification_response = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     messages=code_modification_messages,
#     max_tokens=2000,
#     temperature=0.5,
#     n=1
# )
# print(code_modification_response.choices[0]['message']['content'].strip())

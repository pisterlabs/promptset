import os
import requests
import json
import subprocess  # is this still needed?
import openai


def get_review():
    ACCESS_TOKEN = os.getenv("GITHUB_TOKEN")
    GIT_COMMIT_HASH = os.getenv("GIT_COMMIT_HASH")
    PR_PATCH = os.getenv("GIT_PATCH_OUTPUT")
    model = os.getenv("GPT_MODEL", "text-davinci-003")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.organization = os.getenv("OPENAI_ORG_KEY")
    pr_link = os.getenv("LINK")
    extra_tasks = os.getenv("EXTRA_TASKS", "")

    headers = {
        "Accept": "application/vnd.github.v3.patch",
        "authorization": f"Bearer {ACCESS_TOKEN}",
    }

    intro = f"Act as a code reviewer of a Pull Request, providing feedback on the code changes below. You are provided with the Pull Request changes in a patch format.\n"
    explanation = f"Each patch entry has the commit message in the Subject line followed by the code changes (diffs) in a unidiff format.\n"
    patch_info = f"Patch of the Pull Request to review:\n\n{PR_PATCH}\n"
    task_headline = "As a code reviewer, your task is:"
    task_list = f"""
- Provide a summary of the changes that can be used in the changelog
- Review the code changes (diffs) and provide feedback.
- If there are any bugs, highlight them.
- Do not highlight minor issues and nitpicks.
- Summarize all the patches as one pull request, do not mention them individually
- Look out for typos in repeating variables.\n- Use markdown formatting.
- Use bullet points if you have multiple comments.
{extra_tasks}
"""
    prompt = intro + explanation + patch_info + task_headline + task_list

    print(f"\nPrompt sent to GPT-3: {prompt}\n")

    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        temperature=0.55,
        max_tokens=312,
        top_p=1,
        frequency_penalty=0.3,
        presence_penalty=0.0,
    )
    review = response["choices"][0]["text"]

    data = {"body": review, "commit_id": GIT_COMMIT_HASH, "event": "COMMENT"}
    data = json.dumps(data)
    print(f"\nResponse from GPT-3: {data}\n")

    OWNER = pr_link.split("/")[-4]
    REPO = pr_link.split("/")[-3]
    PR_NUMBER = pr_link.split("/")[-1]

    # https://api.github.com/repos/OWNER/REPO/pulls/PULL_NUMBER/reviews
    response = requests.post(
        f"https://api.github.com/repos/{OWNER}/{REPO}/pulls/{PR_NUMBER}/reviews",
        headers=headers,
        data=data,
    )
    print(response.json())


if __name__ == "__main__":
    get_review()

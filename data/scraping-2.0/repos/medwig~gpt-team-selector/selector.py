import openai
import re

openai.api_key = "your-openai-api-key-here"


def extract_info(text, separator="Id:"):
    # Split the text into chunks by the separator
    chunks = re.split(separator, text)
    # Remove leading/trailing whitespace and filter out any empty chunks
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    return chunks


def analyze_project(project_summary, team_descriptions):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Project Summary: {project_summary}"},
        {"role": "user", "content": f"Team Descriptions: {team_descriptions}"},
        {
            "role": "user",
            "content": "Please analyze the project and the team, and provide the following: - Project Name and Summary - Skills required for the project - A selection of 4 team members best suited for the project - Rationale for the selection",
        },
        {
            "role": "user",
            "content": "Format the output using Markdown to create an easy to read analysis",
        },
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages, max_tokens=600
    )

    return response["choices"][0]["message"]["content"].strip()


with open("projects.txt", "r") as file:
    projects_text = file.read()

with open("team_members.txt", "r") as file:
    team_members_text = file.read()

projects = extract_info(projects_text)
team_members = extract_info(team_members_text)

results = []
for i, project in enumerate(projects):
    print(f"Analyzing project {i+1}...")
    analysis = analyze_project(project, "\n".join(team_members))
    results.append(f"Project {i + 1}:\n{analysis}\n")

with open("output.md", "w") as file:
    file.write("\n".join(results))

for result in results:
    print(result)

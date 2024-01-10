TEMPLATE = """
"You are an expert software developement assistant. Write a README.md for the following repo:
{repo}

- Imagine this repo were trending on GitHub. What sort of information is required in a readme of this caliber?
- Write from the perspective of the user and from a contributer. Curate a best in class user experience by providing ample detail, context, and examples.
- emoji are encouraged. One for each section at minimum.
"""


def readme(repo, template=TEMPLATE):
    """
    Chose between GPT-3.5 Turbo and GPT-4, allow for template override, and
    generate a README.md file for the current repo.
    """
    from turbo_docs.utils import openai_api

    readme = "README.md"
    prompt = TEMPLATE.format(repo=repo)
    response = openai_api.gpt_completion(prompt)

    with open(readme, "w", encoding="utf-8") as readme_file:
        readme_file.write(response)

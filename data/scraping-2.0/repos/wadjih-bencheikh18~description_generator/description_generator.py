import requests
import openai


def description_generator(openai_key: str, username: str) -> str:
    """
    Generate a personal description for a GitHub user's profile.
    :param openai_key: The OpenAI API key.
    :param username: The GitHub username of the user.
    :return: A personal description for the user's GitHub profile.
    """
    # Set your API key
    openai.api_key = openai_key

    model_engine = "text-davinci-002"
    # Make a GET request to the GitHub API to get the user's information
    user_response = requests.get(f'https://api.github.com/users/{username}')
    contributions = requests.get(
        f'https://api.github.com/users/{username}/events/public')
    repos = requests.get(f'https://api.github.com/users/{username}/repos')
    # If the request is successful (status code 200), extract the user's information
    if user_response.status_code == 200 and contributions.status_code == 200 and repos.status_code == 200:
        user_info = user_response.json()
        # get infos
        name = user_info['name']
        location = user_info['location']
        company = user_info['company']
        skills = user_info['bio']
        repos = repos.json()
        # join repos
        repos = ', '.join(repos)
        # get contributions
        contributions = contributions.json()
        # get last contribution
        last_contribution = contributions[0]['created_at']
        # number of contributions
        number_of_contributions = len(contributions)

        # Construct the prompt using the user's information
        prompt = f"{name} is a developer"
        if location:
            prompt += f" based in {location}, "

        if company:
            prompt += f" working for {company}, "

        if skills:
            prompt += f" his bio is {skills}, "

        if repos:
            prompt += f" his open-source projects are {repos}. "

        if last_contribution:
            prompt += f" he has {number_of_contributions} contributions and his last contribution was {last_contribution}. \n"

        prompt += f"Write a detailed personal description for {name}. without mentioning the list of his projects."

        # Generate the personal description
        completion = openai.Completion.create(
            engine=model_engine, prompt=prompt, max_tokens=1024, n=1, stop=None, temperature=0.7)
        description = completion.choices[0].text
        return description
    else:
        # If the request is not successful, return an error message
        return 'Error: Could not fetch user information from GitHub.'

import requests
import gpt
import langchain

import github3

def get_user_repositories(user_name):
  """Fetches a user's repositories from their GitHub user name."""

  github = github3.login('manas95826')
  user = github.get_user(user_name)
  return user.get_repos()

def preprocess_code(code):
  """Preprocesses the code in a repository before passing it into GPT."""

  # Tokenize the code
  tokens = gpt.tokenizer(code)

  # Remove comments and whitespace
  tokens = [token for token in tokens if token[0] != '#']
  tokens = [token for token in tokens if token != ' ']

  # Split the code into smaller chunks
  chunks = [tokens[i:i+512] for i in range(0, len(tokens), 512)]

  return chunks

def evaluate_code_complexity(code):
  """Evaluates the technical complexity of a piece of code using GPT."""

  # Pass the code through GPT
  prompt = 'Is this code technically complex?'
  response = gpt.generate(prompt, code)

  # Extract the score from the response
  score = float(response.split(' ')[0])

  return score

def find_most_complex_repository(user_name):
  """Finds the most technically complex repository in a user's profile."""

  repositories = get_user_repositories(user_name)

  # Preprocess the code in each repository
  preprocessed_repositories = []
  for repository in repositories:
    filenames = repository.get_contents()
    for filename in filenames:
      code = repository.get_contents(filename).decoded
      chunks = preprocess_code(code)
      for chunk in chunks:
        preprocessed_repositories.append((repository.name, chunk))

  # Evaluate the technical complexity of each repository
  scores = []
  for repository, chunk in preprocessed_repositories:
    score = evaluate_code_complexity(chunk)
    scores.append((repository, score))

  # Find the repository with the highest score
  most_complex_repository = max(scores, key=lambda x: x[1])

  return most_complex_repository

if __name__ == '__main__':
  user_name = 'manas95826'
  most_complex_repository = find_most_complex_repository(user_name)
  print(f'The most technically complex repository in {user_name} is {most_complex_repository[0]}.')
  print(f'GPT says that the repository is technically complex because it uses a variety of advanced techniques, such as {most_complex_repository[1]}.')

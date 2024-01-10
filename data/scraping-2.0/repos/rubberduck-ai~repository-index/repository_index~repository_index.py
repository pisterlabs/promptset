import argparse
import json
import os
import time
import dotenv
import datetime
import pathspec
from pathlib import Path

import openai
from git import Repo
from is_supported_file import is_supported_file
from split_linear_lines import split_linear_lines


def get_api_key():
    # Check if OPENAI_API_KEY environment variable is set
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"🎉 OpenAI API key found in environment variables. You are good to go!")
        return api_key

    # Check for .env file in project root
    env_file_path = Path(__file__).parent.parent / ".env"
    if env_file_path.exists():
        dotenv.load_dotenv(env_file_path)
        api_key = dotenv.get_key(env_file_path, "OPENAI_API_KEY")
        if api_key:
            print(f"🎉 OpenAI API key found in .env file at {env_file_path.resolve()}. You are good to go!")
            return api_key
    else:
        print("No .env file found at:", env_file_path.resolve())

    # Prompt user to enter key manually
    api_key = input("Enter API key: ").strip()
    if len(api_key) != 51:
        print("❌ Invalid API key length. Please enter a valid API key.")
        return get_api_key()

    print("✅ OpenAI API key length is valid.")

    save_to_file = input("Do you want to save this API key to a .env file for future use? (y/n) ").strip().lower()
    if save_to_file == "y":
        dotenv.set_key(env_file_path, "OPENAI_API_KEY", api_key, quote_mode="never")
        print(f"API key saved to {env_file_path.resolve()} 💪")

        gitignore_path = Path(__file__).parent.parent / ".gitignore"
        if not gitignore_path.exists():
            gitignore_path.touch()
        with open(gitignore_path) as f:
            gitignore_contents = f.read()
            if ".env" not in gitignore_contents:
                with open(gitignore_path, "a") as f:
                    if gitignore_contents[-1] == "\n":
                        f.write(".env")
                    else:
                        f.write("\n.env\n")

        print(f".env file added to .gitignore 💪")

    else:
        os.environ["OPENAI_API_KEY"] = api_key
        print(f"🎉 OpenAI API key set temporarily. Continuing without saving API key to .env file. You will need to enter your API key again next time you run this script.")
    return api_key

api_key = get_api_key()
openai.api_key = api_key

parser = argparse.ArgumentParser(description="Repository Index")
parser.add_argument(
    "--repository-path", type=str, help="Git repository path",
)
parser.add_argument(
    "--output-file",
    type=str,
    help="Output file path",
    default="./.rubberduck/embedding/result.json",
)

args = parser.parse_args()

output_file_path = args.output_file

# rubberduck_dir = ".rubberduck"
rubberduck_dir = ".rubberduck/embedding/result.json"

if args.repository_path is None:
    while True:
        repo_path = input("Please enter the path to the Git repository you want to index (enter '.' for the current directory): ")
        if not repo_path:
            print("I'm sorry, but the repository path can't be empty. Please try again.")
            continue
        if repo_path == ".":
            args.repository_path = os.getcwd()
            break
        if os.path.exists(repo_path):
            args.repository_path = repo_path
            break
        else:
            choice = input("I'm sorry, but that path is invalid. Would you like to try again? (y/n)").lower()
            if choice != "y":
                print("Alright, I'll stop here. Goodbye! 👋")
                exit()


if os.path.exists(str(args.repository_path)):
    repo = Repo(str(args.repository_path))
    all_files = list(Path(args.repository_path).rglob("*.*"))
    if Path(".gitignore").exists():
        gitignore_file_path = Path(args.repository_path) / ".gitignore"
        
        if not gitignore_file_path.exists():
            gitignore_file_path.touch()

        with open(gitignore_file_path) as f:
            gitignore_contents = f.read()

            if rubberduck_dir not in gitignore_contents:
                with open(gitignore_file_path, "a") as f:
                    if gitignore_contents[-1] == "\n":
                        f.write(rubberduck_dir)
                    else:
                        f.write(f"\n{rubberduck_dir}\n")

        print(f"{rubberduck_dir} added to .gitignore 💪")
        
        with open(gitignore_file_path) as f:
            gitignore_contents = f.read()
            
            exclude_patterns = [rubberduck_dir + '/**', rubberduck_dir + '/**/*', rubberduck_dir + '/.env']
            spec = pathspec.PathSpec.from_lines("gitwildmatch", gitignore_contents.splitlines() + exclude_patterns)
            all_files = [file for file in all_files if not spec.match_file(str(file))]

    result = list(filter(lambda file: is_supported_file(str(file)), all_files))
    print(f"Found {len(result)} supported files in the repository.")

    if not os.path.exists(os.path.dirname(output_file_path)):
        os.makedirs(os.path.dirname(output_file_path))

    if os.path.exists(output_file_path):
        file_age = time.time() - os.path.getmtime(output_file_path)
        file_age_str = datetime.timedelta(seconds=int(file_age))
        file_age_str = str(file_age_str).split(".")[0]
        hours, remainder = divmod(int(file_age), 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours == 0:
            timestamp = f"{minutes} minutes and {seconds} seconds ago"
        else:
            timestamp = datetime.datetime.fromtimestamp(os.path.getmtime(output_file_path)).strftime('%-m-%-d-%Y at %-I:%M %p')
            timestamp += f" ({hours} hours, {minutes} minutes, and {seconds} seconds ago)"
        choice = input(
            f"\n🚨 Oops! It looks like the embedding file already exists at {output_file_path} 🚨\n\nYour repository was last indexed {timestamp}. \n\nWould you like to re-index your repository now? (y/n) "
        ).lower()
        if choice != "y":
            print("Alright, I won't re-index your repository. Goodbye! 👋")
            exit()
    else:
        print(f"Creating a new index file for your repository at {output_file_path}.")



    chunks_with_embedding = []
    token_count = 0
    
    for file in result:
        file_path = args.repository_path + "/" + str(file.relative_to(args.repository_path))
        if not os.path.exists(file_path):
            continue
        with open(file_path, "r") as f:
            content = f.read()
            chunks = split_linear_lines(content, 150)
            for chunk in chunks:
                chunk_start = chunk["start_position"]
                chunk_end = chunk["end_position"]

                print(f"Generating embedding for chunk '{file.name}' {chunk_start}:{chunk_end}")

                embedding_result = openai.Embedding.create(
                    engine="text-embedding-ada-002", input=chunk["content"]
                )

                chunks_with_embedding.append(
                    {
                        "start_position": chunk_start,
                        "end_position": chunk_end,
                        "content": chunk["content"],
                        "file": file.name,
                        "embedding": embedding_result.data[0].embedding,
                    }
                )

                token_count += embedding_result.usage.total_tokens

    output_file_path = args.output_file

    if os.path.exists(output_file_path):
        print(f"Overwriting the index file at {output_file_path}")
    else:
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    
    with open(output_file_path, "w") as f:
        f.write(
            json.dumps(
                {
                    "version": 0,
                    "embedding": {
                        "source": "openai",
                        "model": "text-embedding-ada-002",
                    },
                    "chunks": chunks_with_embedding,
                }
            )
        )

    print(f"Output saved to {output_file_path}")

    cost = (token_count / 1000) * 0.0004

    print()
    print(f"Tokens used: {token_count}")
    print(f"Cost: {cost} USD")
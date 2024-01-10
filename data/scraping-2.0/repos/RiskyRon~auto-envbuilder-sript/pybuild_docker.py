"""
This script creates a new Python project with the following configurable options:

--dir       : The directory name for the new project. Default is 'project'.
--venv      : The name of the Python virtual environment to create. Default is 'venv'.
--packages  : A comma-separated list of Python packages to install in the virtual environment. Default is an empty list.
--python    : The version of Python to use in the virtual environment. Default is '3.11.3'.

Example usage:
python build_script.py --dir my_cool_project --venv my_env --packages numpy,pandas,matplotlib --python 3.11.3
"""



import subprocess
import argparse
import sqlite3
import logging
import json
import os


def create_directory(parent_dir, dir_name=None):
    if dir_name:
        dir_name = os.path.join(parent_dir, dir_name)
    else:
        dir_name = parent_dir

    logging.info(f"Creating directory {dir_name}")
    if os.path.exists(dir_name):
        logging.error(f"Directory {dir_name} already exists.")
        return False

    try:
        os.makedirs(dir_name, exist_ok=True)
    except Exception as e:
        logging.error(f"Failed to create directory {dir_name}: {e}")
        return False

    return True


def create_virtual_env(venv_name, python_version):
    venv_name = os.path.join("config", venv_name)
    logging.info(f"Creating virtual environment {venv_name} with Python version {python_version}")
    subprocess.call(["virtualenv", "-p", f"python{python_version}", venv_name])
    return venv_name

def install_packages(venv_name, packages):
    # Always install python-dotenv and pylint by default
    packages.extend(['python-dotenv', 'pylint', 'openai'])
    for package in packages:
        subprocess.call([f"{venv_name}/bin/pip", 'install', package])

def create_ron_testing_dir():
    logging.info("Creating RONTESTING directory")
    os.makedirs("config/RONTESTING", exist_ok=True)

def create_workspace_dir():
    logging.info("Creating WORKSPACE directory")
    os.makedirs("WORKSPACE", exist_ok=True)


def setup_pytest():
    os.makedirs("config/tests", exist_ok=True)
    with open("config/tests/test_initial.py", "w") as f:
        f.write("""def test_initial():\n    assert True""")

def create_pytest_ini():
    content = """
    [pytest]
    python_files = tests.py test_*.py *_tests.py
    """
    with open("pytest.ini", "w") as f:
        f.write(content.strip())


def create_sqlite_db(db_name):
    conn = sqlite3.connect(f"config/{db_name}")
    logging.info(f"Created SQLite database {db_name}")
    conn.close()

def create_readme(dir_name):
    with open("config/README.md", "w") as f:
        f.write(f"# {dir_name} Python Project Builder\n")
        f.write("\n## Docker Commands\n")
        f.write("Here are some Docker commands you might find useful:\n")
        f.write("### Build Docker Image\n")
        f.write("```docker-compose build```\n")
        f.write("### Start Docker Containers\n")
        f.write("```docker-compose up -d```\n")
        f.write("### Stop Docker Containers\n")
        f.write("```docker-compose down```\n")
        f.write("### List Docker Containers\n")
        f.write("```docker ps -a```\n")
        f.write("### Execute a Command Inside a Docker Container\n")
        f.write("```docker exec -it <container-id> <command>```\n")
        f.write("Replace `<container-id>` with the ID of your Docker container, and `<command>` with the command you want to execute.\n")
        f.write("For example, to run a Python script named 'openai_script.py' located in the '/app' directory inside the Docker container, you would use the following command:\n")
        f.write("```docker exec -it <container-id> python /app/openai_script.py```\n")
        f.write("After running the script, you can activate the virtual environment and start the Docker containers using the following command:\n")
        f.write("```source <project-dir>/<venv>/bin/activate && cd <project-dir>/config/ && docker-compose up -d```\n")
        f.write("Replace `<project-dir>` with the name of your project directory, and `<venv>` with the name of your virtual environment.\n")
        f.write("Type tree for file structure\n")

def print_example_usage():
    print("\n" + "#" * 118 + "\n")
    print("Example usage: python3 main.py --dir my_cool_project --venv my_env --packages numpy,pandas,matplotlib --python 3.11.3\n")
    print("#" * 118 + "\n")


def create_openai_script():
    content = '''
import openai
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv("../config/.env")

# Get API Key from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set OpenAI API Key
openai.api_key = OPENAI_API_KEY

# Prepare messages
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Translate the following English text to French: 'Hello, how are you?'"}
]

# Generate response
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    max_tokens=60
)

print(response['choices'][0]['message']['content'])
'''
    with open("app/openai_script.py", "w") as f:
        f.write(content.strip())

def create_docker_files():
    with open("config/Dockerfile", "w") as f:
        f.write("FROM python:3.11.3\n")
        f.write("RUN apt-get update && apt-get install -y tree\n")  # Add this line
        f.write("WORKDIR /app\n")
        f.write("COPY requirements.txt .\n")
        f.write("RUN pip install -r requirements.txt\n")
        f.write("COPY . .\n")
        f.write('CMD ["tail", "-f", "/dev/null"]\n')



def create_docker_compose_file(dir_name):
    content = f"""
version: '3.9'
services:
  {dir_name}:
    build: ./
    volumes:
      - ../app:/app
      - .:/config/
      - ../WORKSPACE:/workspace
      - ./config/RONTESTING:/rontesting
    ports:
      - "8000:8000"
    """
    with open("config/docker-compose.yml", "w") as f:
        f.write(content.strip())

def create_vscode_settings(venv_name):
    settings = {
        "python.pythonPath": f"{venv_name}/bin/python",
        "python.linting.enabled": True,
        "python.linting.pylintEnabled": True,
        "python.formatting.provider": "autopep8",
        "python.testing.pytestEnabled": True,
        "python.testing.unittestEnabled": False,
        "python.autoComplete.addBrackets": True,
        "python.jediEnabled": False
    }
    os.makedirs("config/.vscode", exist_ok=True)
    with open("config/.vscode/settings.json", "w") as f:
        json.dump(settings, f, indent=4)

def create_env_file():
    default_env_vars = [
        "#openai",
        "OPENAI_API_KEY=",
        "#aws",
        "S3_BUCKET=",
        "AWS_ACCESS_KEY_ID=",
        "AWS_SECRET_ACCESS_KEY=",
        "AWS_REGION=",
        "#django",
        "DJANGO_SECRET_KEY=",
        "#pinecone",
        "DATASTORE=pinecone",
        "PINECONE_API_KEY=",
        "PINECONE_ENVIRONMENT=",
        "PINECONE_INDEX="
    ]
    with open('config/.env', 'w') as f:
        f.write('\n'.join(default_env_vars))

def create_gitignore_file(venv_name):
    default_ignore_files = [
        "config/" + venv_name,  
        "config/database.sqlite3",
        "config/RONTESTING/", 
        "config/.env", 
        "__pycache__/",  # Ignore Python cache files
        "*.pyc",  # Ignore Python compiled files
        "*.pyo",
        "*.pyd",
        ".Python",
        "ipynb_checkpoints/",  # Ignore Jupyter Notebook checkpoints
        ".vscode/",  # Ignore VS Code settings
        ".idea/",  # Ignore PyCharm settings
        "*.log",  # Ignore all log files
        ".DS_Store",  # Ignore Mac system files
        "dist/",  # Ignore Python distribution folder
        "build/",  # Ignore Python build folder
        "*.egg-info/",  # Ignore Python egg info
        ".pytest_cache/",  # Ignore pytest cache
        ".mypy_cache/",  # Ignore mypy type checker cache
    ]
    with open('config/.gitignore', 'w') as f:
        f.write('\n'.join(default_ignore_files))

def initialize_git_repo():
    logging.info("Initializing Git repository")
    subprocess.call(['git', 'init'])
    subprocess.call(['git', 'checkout', '-b', 'main'])  # switch to a new branch 'main'
    subprocess.call(['git', 'add', '.'])
    subprocess.call(['git', 'commit', '-m', 'Initial commit'])

def freeze_packages(venv_name):
    with open("config/requirements.txt", "w") as f:
        subprocess.call([f"{venv_name}/bin/pip", 'freeze'], stdout=f)

def print_activate_virtual_env_command(venv_name,dir_name):
    print("\n" + "#" * 118 + "\n")
    print(f"To activate the virtual environment, run:")
    print(f"source {dir_name}/{venv_name}/bin/activate && cd {dir_name}/config/ && docker-compose up -d\n")
    print("#" * 118 + "\n")

def run_tree(venv_name):
    subprocess.call(['tree', '-I', f'{venv_name}|.DS_Store'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a Python project")
    parser.add_argument("--dir", help="Directory name", default="project")
    parser.add_argument("--venv", help="Virtual environment name", default="venv")
    parser.add_argument("--packages", help="Packages to install (separated by commas)", default="")
    parser.add_argument("--python", help="Python version", default="3.11.3")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if not create_directory(args.dir):
        logging.error("Failed to create directory")
        exit(1)

    os.chdir(args.dir)
    create_directory(".", "app")
    create_directory(".", "config")
    venv_name = create_virtual_env(args.venv, args.python)
    create_sqlite_db('database.sqlite3')
    create_ron_testing_dir()
    create_workspace_dir()
    create_pytest_ini()
    create_readme(args.dir)
    create_openai_script()
    create_env_file()
    create_gitignore_file(args.venv)
    packages = args.packages.split(',') if args.packages else []
    install_packages(venv_name, packages)
    create_vscode_settings(venv_name)
    setup_pytest()
    create_docker_files()
    create_docker_compose_file(args.dir)
    initialize_git_repo()
    freeze_packages(venv_name)
    run_tree(args.venv)
    print_activate_virtual_env_command(venv_name,args.dir)
    print_example_usage()
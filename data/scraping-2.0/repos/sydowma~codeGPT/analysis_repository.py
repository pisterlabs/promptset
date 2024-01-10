import os
import shutil
import tarfile
import time

import openai
import requests

from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")


class GitHubClient:

    def __init__(self, url, token=None, download_dir="./downloaded_sources"):
        self.url = url
        self.token = token
        self.download_dir = download_dir
        self.headers = {
            'Authorization': f'token {self.token}',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/58.0.3029.110 Safari/537.3'
        }

        if self.token:
            self.headers['Authorization'] = f'token {self.token}'
        self.init_environment(self.download_dir)

    def get_repo(self, repo, is_private=False, branch="master"):
        if is_private:
            api_url = f"https://api.github.com/repos/{repo}/tarball"
        else:
            api_url = f"https://github.com/{repo}/archive/{branch}.tar.gz"
        response = requests.get(api_url, headers=self.headers, stream=True)

        if response.status_code == 200:
            repo_parent_path = os.path.join(self.download_dir, repo.split('/')[0])
            self.init_environment(repo_parent_path)
            tar_file_path = os.path.join(self.download_dir, f"{repo.replace('/', '_')}.tar.gz")
            with open(tar_file_path, 'wb') as f:
                response.raw.decode_content = True
                shutil.copyfileobj(response.raw, f)
            self._extract_tar_file(tar_file_path, repo_parent_path)
        else:
            print(f'Failed to download: {response.content.decode()}'
                  f'request: {api_url}')

    def init_environment(self, dir_path):
        download_dir = os.path.join(dir_path)
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

    def _extract_tar_file(self, tar_file_path, extract_path):
        with tarfile.open(tar_file_path, 'r:gz') as tar:
            tar.extractall(path=extract_path)
        os.remove(tar_file_path)


class AnalysisRepository:

    def __init__(self, dir_path, file_extension, language):
        self.dir_path = dir_path
        self.file_extension = "." + file_extension
        self.promotion = self.get_promotion_by_language(language)

    def init_environment(self, dir_path):
        dir_path = os.path.join(dir_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def write_to_file(self, filename, content):
        with open(filename, 'w') as file:
            file.write(content)

    def read_file(self, file_path):
        with open(file_path, 'r') as file:
            return file.read()

    def explain_python_files_in_dir(self, dir_path, output_root_dir):
        self.init_environment(output_root_dir)
        for root, dirs, files in os.walk(dir_path):
            for filename in files:
                if 'test' in filename or 'Test' in filename:
                    continue
                if filename.endswith(self.file_extension):
                    full_path = os.path.join(root, filename)

                    output_dir = os.path.join(root).removeprefix("./downloaded_sources/")
                    output_dir = os.path.join(output_root_dir, output_dir)
                    self.init_environment(output_dir)
                    output_filename = os.path.join(output_dir, filename + '.md')
                    if os.path.exists(output_filename):
                        print(f'skip Explanations for {filename} explanation')
                        continue
                    content = self.read_file(full_path)
                    explanation = self.generate_explanation(filename, content)
                    file_content = f'''{explanation}
                    

```java
{content}
```
                    '''
                    print(f'Explanations for {filename} explanation')
                    self.write_to_file(output_filename, file_content)

    def read_files_in_dir(self, dir_path, filename_suffix=".py"):
        file_contents = {}
        for filename in os.listdir(dir_path):
            if filename.endswith(filename_suffix):
                with open(os.path.join(dir_path, filename), 'r') as file:
                    file_contents[filename] = file.read()
        return file_contents

    def generate_explanation(self, filename, content, max_length=4096):
        explanation_parts = []
        messages = [
            {"role": "system", "content": self.promotion},
        ]
        tokenized_content = content.split()  # simple split by spaces, consider better tokenization
        MAX_TOKENS = 4096 - 256 - 32
        outputs = []

        for i in range(0, len(tokenized_content), MAX_TOKENS):
            chunk = " ".join(tokenized_content[i:i + MAX_TOKENS])
            messages.append(
                {"role": "user", "content": f"Explain the following code from {filename}:\n{chunk}\n"})
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    # As of the time of writing this, GPT-4 hasn't been released yet. Please replace with the latest model
                    messages=messages,
                )
                outputs.append(response['choices'][0]['message']['content'])
            except openai.error.ServiceUnavailableError:
                print(f"Skipping a chunk due to service unavailable")
                time.sleep(1000)
            except openai.error.InvalidRequestError as e:
                if "maximum context length" in str(e):
                    print(f"Skipping a chunk due to token limit exceeded: {e}")
                else:
                    raise e
        return " ".join(outputs)

    def ask(self):
        # Read all Python files in the directory
        # Generate an explanation for each file
        self.explain_python_files_in_dir(self.dir_path, output_root_dir="./explanations")

    def get_promotion_by_language(self, language):
        if language == 'zh':
            return "你是一个帮助解释代码的助手，请帮我分析一下这段代码。"
        else:
            return "You are a helpful assistant that explains code."


if __name__ == "__main__":
    # Usage
    client_public = GitHubClient(url="https://api.github.com")
    client_public.get_repo("LMAX-Exchange/disruptor", is_private=False,
                           branch="master")  # replace with the public repo you want to download
    # replace with the repo you want to download

    # client_private = GitHubClient(url="https://api.github.com", token="your_github_token")
    # client_private.get_repo("username/private-repo")  # replace with the private repo you want to download

    analysis = AnalysisRepository(dir_path="./downloaded_sources/disruptor-master", file_extension='java')
    analysis.ask()

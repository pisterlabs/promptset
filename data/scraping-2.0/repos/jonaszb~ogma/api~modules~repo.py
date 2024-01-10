import requests
from modules.tree import TreeNode
import openai


class GithubRepository:

    def __init__(self, url: str):
        username, repository_name, branch = self.__get_data_from_url(url)
        self.username = username
        self.repository_name = repository_name
        self.branch = branch if branch else None

    def __get_data_from_url(self, url: str):
        split_url = url.split("/")
        # Check if the URL is valid. It may or may not contain the branch name
        gh_index = split_url.index("github.com")
        if gh_index == -1:
            raise ValueError("Invalid GitHub repository URL.")
        try:
            username = split_url[gh_index + 1]
            repository_name = split_url[gh_index + 2]
        except IndexError:
            raise ValueError("Invalid GitHub repository URL.")
        # get branch if it exists
        branch = split_url[gh_index +
                           4] if len(split_url) > gh_index + 4 else None
        return username, repository_name, branch

    def __scan_directory(self, path: str = ""):
        url = f"https://api.github.com/repos/{self.username}/{self.repository_name}/contents/{path}"
        headers = {"Authorization": f"token {self.token}"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        def is_file(node_data):
            return isinstance(node_data, dict) and "type" in node_data and node_data["type"] == "file"

        def is_dir(node_data):
            return isinstance(node_data, dict) and "type" in node_data and node_data["type"] == "dir"

        def create_node(node_data):
            if is_file(node_data):
                node = TreeNode("file")
                node.name = node_data["name"]
            elif is_dir(node_data):
                return self.__scan_directory(node_data["path"])
            elif isinstance(node_data, list):
                node = TreeNode("dir")
                node.name = path.split("/")[-1] if path else 'root'
                node.children = [create_node(item) for item in node_data]
            else:
                raise ValueError("Invalid GitHub repository URL.")
            return node

        return create_node(data)

    def set_token(self, token: str):
        self.token = token

    def validate_repository(self):
        url = f"https://api.github.com/repos/{self.username}/{self.repository_name}"
        response = requests.get(url)
        if response.status_code != 200:
            raise ValueError("Invalid GitHub repository URL.")

    def scan(self):
        if not self.token:
            raise ValueError("GitHub token not set")
        self.contents = self.__scan_directory()

    def get_repository_info(self):
        url = f"https://api.github.com/repos/{self.username}/{self.repository_name}"
        headers = {"Authorization": f"token {self.token}"}
        # Send an HTTP GET request to the GitHub API
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        self.repository_data = {}
        for key in ['html_url', 'description', 'language', 'permissions', 'default_branch', 'license', 'homepage', 'size']:
            if key in data and data[key] is not None:
                self.repository_data[key] = data[key]

    def dict(self):
        return {"repository_data": self.repository_data, "contents": self.contents.dict()}

    def get_tree(self):
        return self.contents.dict()

    def generate_readme(self, info: str):
        # Create the prompt for GPT-3.5
        system_prompt = """
        You write readme files for GitHub repositories based on the contents of the repository and additional information provided. You must never include the file/folder structure in the readme itself.
        """

        prompt = f"""
        Given this repository data, write a professional README for this repository. The 'contents' are a tree structure of the files and folders in the repository. The 'repository_data' contains information about the repository itself.
        If additional information is provided, consider it when generating the readme. Note that it is provided by the user and can use informal language, which should not be included in the README verbatim.
        Attempt to infer as much information as possible from the file names and extensions - they can provide information about the languages and tools used.
        If you are unable to infer something, do not include it in the README (do not leave placeholders)
        Do not under any circumscanes include the file/folder structure in the README.
        The title of the readme should be {self.repository_name}

        Tree Structure:
        {str(self.contents.dict())}
        """

        # Add additional information if provided
        if info:
            prompt += "\n\nAdditional Information:\n{}".format(info)

        return openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": prompt}],
            max_tokens=4000,
            temperature=0.5,
            stream=True
        )

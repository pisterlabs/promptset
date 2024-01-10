import random
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.document_loaders import PythonLoader, DirectoryLoader
from pathlib import Path
import tqdm
import ast
import os
import aiohttp
from gidgethub.aiohttp import GitHubAPI
import git
import logging
import uuid
from pathlib import Path
import shutil
import datetime
from ibl_github_bot.configuration import DependencyGraph
from langchain.schema import Document
import concurrent

logger = logging.getLogger(__name__)


BASE_DIR = Path.cwd() / "cached-repos"


class CodeParser:
    def parse(self, text: str) -> tuple[str, bool]:
        if text.startswith("```\n") or text.startswith("```python\n"):
            text = "\n".join(text.splitlines()[1:])
        if text.endswith("\n```"):
            text = text.strip("\n```").strip()

        if "```\n" in text:
            parts = text.split("```\n")
            if len(parts) == 2 and not parts[1].startswith("#"):
                text = parts[0]
            elif "Output" in parts[1]:
                text = parts[0]

        try:
            ast.parse(text)
            return text, True
        except Exception as e:
            logging.error(e)
            logging.error("Error parsing text as code")
            logging.info("Defaulting text: \n%s", text)
        return text, False


SYTEM_MESSAGE_STR = """You are an experienced {language}, {frameworks} and {test_library} developer. \
You have been given a set of {language} files in a project written in  {frameworks} and {test_library} \
You are expected to generate {language}, {frameworks} and {test_library} compliant tests for a file specified by the user. \
You are expected to follow best practices for {frameworks} and {test_library} tests. \
Where tests already exist, you are required to extend the existing tests. \
Make you test cases as extensive as possible.

The project is loaded in the format

```python
# filenane here. 
Content of file here
```

For example:
```pythhon
# file1.py
from rest_framework import viewsets
from .models import Bot
from .serializers import BotSerializer

def content():
    pass
    
class BotViewSet(viewsets.ModelViewSet):
    model = Bot
    queryset = Bot.objects.all()
    serializer_class = BotSerializer

    def get_queryset(self):
        queryset = self.queryset
        queryset = queryset.filter(tenant=self.kwargs["org"])
        return super().get_queryset()
```
```pythhon
# file2.py
def conent():
    pass
```

You are then expected to generate tests for a file among these lists as specified by the user.

For example:
Generate {test_library} compatible test file for file1.py

The output you yield must contain only the code output for the generated tests. Do not include any extra content. \
Make sure your output is {language} compliant and wrapped in starting ```{language}\n and ending with \n```
Ensure that appropriate markers are placed on tests.

In situations where the test library is "pytest" and framework is "Django", kindly "@oytest.mark.django_db" decorator on all tests \
if the framework is not "django", please do not use this decorator. \
Also in django projects, wherever you need a reference to the `User` models, use django.contrib.auth.get_user_model to get the user model. 

Here is a sample test with django as framework and unittest as testing library.
```python
# test for file1.py
from django.contrib.auth  import get_user_model
from django.tests import TestCase
User = get_user_model()

def test_content():
    pass
    
class TestBotViewSet(TestCase)
    @classmethod
    def setup_method(cls):
        pass
    def test_unauthorized_call_raises_error(self):
        pass
```

In cases where the file mentioned by the user is empty, return the following output:
```python
# filename.py
# Empty
 file
```

Also if tests for a specific file already exists, include both the existing written tests and your own tests in the output test you writes. \
I repeat, you are to extend the existing tests if a test file already exists within a module for the file specified by the user. \
Also you can use the contents of existing test files to know if there are any external functions or components you can call. \
Also ensure that you return only the test file contents and not any extra content.

Here are the details again:
Frameworks: {frameworks}
Testing library: {test_library}
Programming Language: {language}
"""


class CustomDirectoryLoader(DirectoryLoader):
    """Load from a directory"""

    def __init__(
        self,
        path: str,
        current_module: str,
        glob: str = "**/[!.]*",
        silent_errors: bool = False,
        load_hidden: bool = False,
        loader_cls=PythonLoader,
        loader_kwargs: dict | None = None,
        recursive: bool = False,
        show_progress: bool = False,
        use_multithreading: bool = False,
        max_concurrency: int = 4,
        *,
        sample_size: int = 0,
        randomize_sample: bool = False,
        sample_seed: int | None = None,
        exclude_dirs: list | None = None,
        dependent_modules: list | None = None,
    ):
        """
        Initializes a new instance of the class.

        Args:
            path (str): The path to the directory to be parsed.
            current_module (str): The name of the current module.
            glob (str, optional): The glob pattern for matching files. Defaults to "**/[!.]*".
            silent_errors (bool, optional): Whether to suppress error messages. Defaults to False.
            load_hidden (bool, optional): Whether to load hidden files. Defaults to False.
            loader_cls (type, optional): The class responsible for loading files. Defaults to PythonLoader.
            loader_kwargs (dict or None, optional): Additional keyword arguments to pass to the loader class. Defaults to None.
            recursive (bool, optional): Whether to search for files recursively. Defaults to False.
            show_progress (bool, optional): Whether to show progress while parsing files. Defaults to False.
            use_multithreading (bool, optional): Whether to use multithreading for parsing files. Defaults to False.
            max_concurrency (int, optional): The maximum number of threads or processes to use for parsing files. Defaults to 4.
            sample_size (int, optional): The number of files to sample for parsing. Defaults to 0.
            randomize_sample (bool, optional): Whether to randomize the order of the sampled files. Defaults to False.
            sample_seed (int or None, optional): The seed value for randomizing the sample. Defaults to None.
            exclude_dirs (list or None, optional): A list of directories to exclude from parsing. Defaults to None.
            dependent_modules (list or None, optional): A list of dependent modules to consider when parsing. Defaults to None.

        Returns:
            None
        """

        if not exclude_dirs:
            exclude_dirs = []
        if not dependent_modules:
            dependent_modules = []

        super().__init__(
            path=path,
            glob=glob,
            silent_errors=silent_errors,
            load_hidden=load_hidden,
            loader_cls=loader_cls,
            loader_kwargs=loader_kwargs,
            recursive=recursive,
            show_progress=show_progress,
            use_multithreading=use_multithreading,
            max_concurrency=max_concurrency,
            sample_size=sample_size,
            randomize_sample=randomize_sample,
            sample_seed=sample_seed,
        )
        self.exclude_dirs = exclude_dirs
        self.dependent_modules = dependent_modules
        self.current_module = current_module

    def is_in_exclude(self, path: Path):
        first_pass = any(
            d in path.relative_to(self.path).parts for d in self.exclude_dirs
        )
        if first_pass:
            return True
        for d in self.exclude_dirs:
            if path.is_relative_to(self.path / d):
                return True
        return False

    def is_in_dependent_modules(self, path: Path):
        if not self.dependent_modules:
            return True
        parts = path.relative_to(self.path).parts
        if parts:
            return parts[0] in [*self.dependent_modules, self.current_module]
        return False

    def load(self) -> list[Document]:
        """Load documents."""
        p = Path(self.path)
        if not p.exists():
            raise FileNotFoundError(f"Directory not found: '{self.path}'")
        if not p.is_dir():
            raise ValueError(f"Expected directory, got file: '{self.path}'")

        docs: list[Document] = []
        items = list(p.rglob(self.glob) if self.recursive else p.glob(self.glob))
        # filtering out excluded directories
        items = [path for path in items if self.is_in_dependent_modules(path)]
        items = [path for path in items if not self.is_in_exclude(path)]

        if self.sample_size > 0:
            if self.randomize_sample:
                randomizer = (
                    random.Random(self.sample_seed) if self.sample_seed else random
                )
                randomizer.shuffle(items)  # type: ignore
            items = items[: min(len(items), self.sample_size)]

        pbar = None
        if self.show_progress:
            try:
                from tqdm import tqdm

                pbar = tqdm(total=len(items))
            except ImportError as e:
                logger.warning(
                    "To log the progress of DirectoryLoader you need to install tqdm, "
                    "`pip install tqdm`"
                )
                if self.silent_errors:
                    logger.warning(e)
                else:
                    raise ImportError(
                        "To log the progress of DirectoryLoader "
                        "you need to install tqdm, "
                        "`pip install tqdm`"
                    )

        if self.use_multithreading:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_concurrency
            ) as executor:
                executor.map(lambda i: self.load_file(i, p, docs, pbar), items)
        else:
            for i in items:
                self.load_file(i, p, docs, pbar)

        if pbar:
            pbar.close()

        return docs


def generate_tests(
    directory: Path,
    dependency_graph: DependencyGraph,
    sub_path: Path = None,
    test_dir: Path = None,
    target_files: list[Path] = None,
):
    if sub_path == None:
        sub_path = directory
    if test_dir == None:
        test_dir = sub_path / "tests"
    if sub_path.name in dependency_graph.get_global_settings()["exclude"]:
        return False
    module_name = sub_path.relative_to(directory).name
    exclude_dirs = dependency_graph.get_all_excludes(module_name)
    dependent_modules = dependency_graph.get_all_dependencies(module_name)

    documents = CustomDirectoryLoader(
        path=directory,
        glob="*.py",
        recursive=True,
        show_progress=True,
        loader_cls=PythonLoader,
        exclude_dirs=exclude_dirs,
        dependent_modules=dependent_modules,
        current_module=module_name,
    ).load()

    test_dir.mkdir(exist_ok=True)
    if not (test_dir / "__init__.py").exists():
        (test_dir / "__init__.py").touch()
    files_messages = [
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "# %s\n%s"
                    % (
                        Path(document.metadata["source"]).relative_to(directory),
                        document.page_content,
                    ),
                },
            ]
        )
        for document in documents
    ]
    global_settings = dependency_graph.get_global_settings()
    system_message = SystemMessage(
        content=SYTEM_MESSAGE_STR.format(
            test_library=global_settings["test_library"],
            frameworks=", ".join(global_settings["frameworks"]),
            language=global_settings["language"],
        )
    )
    messages = [
        system_message,
        *files_messages,
    ]

    chain = ChatOpenAI(
        model="gpt-4-1106-preview",
        temperature=0,
    )
    target_documents = [
        document
        for document in documents
        if Path(document.metadata["source"]).is_relative_to(sub_path)
        and document.page_content.strip() != ""
        and Path(document.metadata["source"]).parent.name
        not in [*exclude_dirs, "tests"]
    ]
    # filter target files.
    if target_files:
        target_documents = [document for document in target_documents if Path(document.metadata["source"]) in target_files]

    if not target_documents:
        logger.info("No tests generated for %s", sub_path)
        return False
    
    for document in tqdm.tqdm(target_documents):
        msg = chain.invoke(
            [
                *messages,
                HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": "Generate {test_library} compatible test file for {filename}".format(
                                filename=Path(document.metadata["source"]).relative_to(
                                    directory
                                ),
                                test_library=global_settings["test_library"],
                            ),
                        }
                    ]
                ),
            ]
        )
        content, success = CodeParser().parse(msg.content)
        if not success:
            logger.warning(
                "Failed to generate test for %s", document.metadata["source"]
            )
            continue
        if not content.strip():
            logger.info(
                "skipping %s no tests generated",
                Path(document.metadata["source"]).relative_to(directory),
            )
            continue
        logger.info(
            "Generated tests for %s",
            Path(document.metadata["source"]).relative_to(directory),
        )
        with open(
            test_dir
            / (
                "test_"
                + str(Path(document.metadata["source"]).relative_to(sub_path)).replace(
                    "/", "_"
                )
            ),
            "w",
        ) as f:
            f.write(content)

    return True


async def create_tests_for_repo(
    username: str,
    repo: str,
    branch: str = "main",
    token: str = os.getenv("GH_TOKEN"),
    cleanup: bool = True,
    target_files: list[str] | None = None,
):
    """
    Asynchronously creates tests for a repository.
    The passed repository will be cloned to a temporary `cached-repos` directory.
    Args:
        username (str): The username of the repository owner.
        repo (str): The name of the repository.
        branch (str, optional): The branch to clone the repository from. Defaults to "main".
        token (str, optional): The GitHub token used for authentication. Defaults to the value of the "GH_TOKEN" environment variable.

    Returns:
        None
    """
    if not target_files:
        target_files = []
    repo_username, repo_name = repo.split("/")
    index = str(uuid.uuid4())
    while (BASE_DIR / index).exists():
        index = str(uuid.uuid4())
    local_dir = BASE_DIR / index
    target_file_paths = [local_dir / file for file in target_files]

    local_dir.mkdir(parents=True)
    logging.info("Cloning repository into %s", local_dir)
    repo_url = f"https://{token}@github.com/{repo}.git"
    logging.info("Cloning repo url [%s]", repo_url)
    new_branch = f"auto-tests-iblai-{index}"
    local_repo = git.Repo.clone_from(repo_url, local_dir, branch=branch)
    remote = local_repo.remote("origin")
    local_repo.git.checkout(branch)
    remote.pull()
    local_repo.git.checkout("-b", new_branch)
    dependency_graph = DependencyGraph(local_dir / "ibl_test_config.yaml")

    logging.info("Successfully cloned repository into %s", local_dir)
    date = datetime.datetime.today().strftime("%A %B %d %Y, %X")
    logging.info("generating tests")
    created_commit = False
    for directory in local_dir.iterdir():
        if (
            directory.is_dir()
            and directory.name not in dependency_graph.get_global_settings()["exclude"]
        ):
            success = generate_tests(
                directory=local_dir,
                dependency_graph=dependency_graph,
                sub_path=directory,
                test_dir=directory / "tests",
                target_files=target_file_paths,
            )
            if not success:
                continue
            
            local_repo.index.add((directory / "tests").relative_to(local_dir))
            local_repo.index.commit(
                f"auto-generated tests for {directory.relative_to(local_dir)} on {date}"
            )
            logging.info(
                f"Created commit with message: auto-generated tests for {directory.relative_to(local_dir)} on {date}"
            )
            created_commit = True
    if not created_commit:
        logging.info("No tests generated")
        return
    logging.info("Pushing to remote branch %s" % new_branch)
    local_repo.remote().push("{}:{}".format(new_branch, new_branch)).raise_if_error()

    logging.info("Successfully generated and pushed tests in %s", repo)

    async with aiohttp.ClientSession(trust_env=True) as session:
        gh = GitHubAPI(session, username, oauth_token=os.getenv("GH_AUTH"))
        results = await gh.post(
            f"/repos/{repo}/pulls",
            data={
                "title": f"Auto-tests generated by ibl.ai ⚡",
                "body": """> [!IMPORTANT] \
                        \n> Remember to check out the pull request and run the tests before merging. \
                        \n> Thank you.
                        """,
                "head": f"{repo_username}:{new_branch}",
                "base": branch,
            },
        )
        logging.info("Created pull request at %s" % results["url"])

    # uncomment to clean up cloned repository
    if cleanup:
        shutil.rmtree(local_dir)

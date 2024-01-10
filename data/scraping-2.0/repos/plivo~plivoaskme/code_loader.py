"""Load text files."""
import os
import os.path
import subprocess
import shutil
from typing import List
from pathlib import Path
import logging

import chardet
import pygments.lexers
from binaryornot.check import is_binary
from gitignore_parser import parse_gitignore

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader




class BaseCodeLoader(BaseLoader):
    """Load text files."""
    extension_to_language = {
        'py': 'python',
        'py3': 'python',
        'pyw': 'python',
        'pl': 'perl',
        'js': 'javascript',
        'java': 'java',
        'go': 'go',
        'c': 'c',
        'h': 'c',
        'cc': 'c',
        'cpp': 'cpp',
        'hpp': 'cpp',
        'cs': 'csharp',
        'php': 'php',
        'rb': 'ruby',
        'ts': 'typescript',
        'scala': 'scala',
        'swift': 'swift',
        'kt': 'kotlin',
        'rs': 'rust',
        'clj': 'clojure',
        'hs': 'haskell',
        'exs': 'elixir',
        'ex': 'elixir',
        'erl': 'erlang',
        'lisp': 'lisp',
        'lua': 'lua',
        'm': 'matlab',
        'r': 'r',
        'rb': 'ruby',
        'sh': 'shell',
        'sql': 'sql',
        'vb': 'vb',
        'vbs': 'vbscript',
        'xml': 'xml',
        'yml': 'yaml',
        'yaml': 'yaml',
        'md': 'markdown',
        'markdown': 'markdown',
        'txt': 'text',
        'csv': 'csv',
        'text': 'text',
        'json': 'json',
        '': 'text',
        'ini': 'ini',
        'cfg': 'ini',
        'html': 'html',
        'htm': 'html',
        'css': 'css',
    }

    def __init__(self, path: str, exclude_dirs: List[str] = None, exclude_files: List[str] = None, debug=False):
        """Initialize with file path."""
        self.path = path
        self.exclude_dirs = exclude_dirs
        self.exclude_files = exclude_files
        self._documents = []
        self._logger = logging.getLogger(__name__)
        self.debug = debug

    def load(self) -> List[Document]:
        """Load from path."""
        if os.path.isfile(self.path):
            self._load_file(Path(file_path))
        else:
            self._load_directory(Path(self.path))
        return self._documents

    def _debug(self, message: str):
        """Log message."""
        if self.debug is True:
            print(f"DEBUG: {message}")

    def _is_excluded(self, path: Path) -> bool:
        """Check if directory or file is excluded."""
        for exclude_dir in self.exclude_dirs:
            if path.name == exclude_dir:
                self._debug(f"Excluding (from exclude_dir): {path.as_posix()}")
                return True
        for exclude_file in self.exclude_files:
            if path.name == exclude_file:
                self._debug(f"Excluding (from exclude_file): {path.as_posix()}")
                return True
        return False

    def _load_directory(self, directory_path: Path) -> List[Document]:
        if self._is_excluded(directory_path):
            return
        for file_path in directory_path.iterdir():
            if self._is_excluded(file_path):
                continue
            self._debug(f"Scanning: {file_path.as_posix()}")
            if file_path.is_dir():
                self._load_directory(file_path)
            else:
                self._load_file(file_path)
        return self._documents

    def _load_file(self, file_path: Path, extra_metadata: dict = None) -> List[Document]:
        if self._is_excluded(file_path):
            return
        self._debug(f"Loading {file_path.as_posix()}")
        # do not index binary files
        if is_binary(file_path.as_posix()):
            return

        metadata = {}
        try:
            f = open(file_path.as_posix(), 'r', errors='ignore')
            text = f.read()
        except Exception as e:
            self._debug(f"Error reading file: {file_path.as_posix()} - {e}")
            return
        finally:
            try:
                f.close()
            except:
                pass

        # do not index empty files
        if not text:
            return
        # Get the file extension
        if not '.' in file_path.name:
            ext = ''
        else:
            ext = file_path.name.split('.')[-1]
        try:
            language = self.extension_to_language[ext]
            metadata = {"source": file_path.as_posix(), 'language': language, 'file_extension': ext, 'category': 'code'}
        except:
            language = self.detect_language_from_text(text)
            metadata = {"source": file_path.as_posix(), 'language': language.lower(), 'file_extension': ext, 'category': 'code'}

        if extra_metadata:
            metadata.update(extra_metadata)
        self._documents.append(Document(page_content=text, metadata=metadata))
        return

    # Method to detect the programming language of the code
    @classmethod
    def detect_language_from_text(cls, text):
        """Detect the programming language of the code."""
        return pygments.lexers.guess_lexer(text).name


class GithubCodeLoader(BaseCodeLoader):

    DEFAULT_EXCLUDED = ['.gitignore', '.gitattributes', '.gitmodules', '.gitkeep', '.github', 
                        '.DS_Store', '.git', '.circleci', '.vscode', '.idea',
                        'CONTRIBUTING.md', 'CONTRIBUTING.rst', 'CONTRIBUTING.markdown',
                        'JenkinsFile', 'Jenkinsfile', 'Makefile', 'Rakefile', 'Gulpfile.js',
                        'setup.py', 'requirements.txt', 'requirements-dev.txt', 'requirements-test.txt',
                        '__pycache__', '.pytest_cache', '.tox', '.coverage', '.coveragerc',
                        'configure.ac', 'configure.in', 'configure', 'CMakeLists.txt', 'CMakeCache.txt',
                        'Gruntfile.js', 'gulpfile.js', 'gruntfile.js', 'package-lock.json',
                        'package.json', 'package-lock.json', 'yarn.lock', 'Gemfile', 'Gemfile.lock', 'composer.json', 'composer.lock',
                        'Pipfile', 'Pipfile.lock', 'pyproject.toml', 'poetry.lock',
                        'Makefile.am', 'Makefile.in', 'Makefile', 'GNUmakefile', 'GNUmakefile.in',
                        'Dockerfile', 'docker-compose.yml', 'docker-compose.yaml', 'docker-compose.dev.yml',
                        '.env', '.venv', 'env', 'venv', 'ENV', 'env.bak', 'venv.bak']

    def __init__(self, repository_url: str, 
                 local_dir: str = None, 
                 branch: str = 'main', 
                 exclude_dirs: List[str] = None, 
                 exclude_files: List[str] = None, 
                 include_only_known_extensions: bool = False,
                 debug=False, 
                 cleanup_cache_dir=True):

        self.repo_url = repository_url
        self.branch = branch
        self.cleanup_cache_dir = cleanup_cache_dir
        self._gitignore = None

        self._include_only_known_extensions = include_only_known_extensions
        if self._include_only_known_extensions:
            self.valid_extensions = list(set(self.extension_to_language.keys()))
        else:
            self.valid_extensions = []

        if exclude_dirs is None:
            exclude_dirs = []
        exclude_dirs.extend(self.DEFAULT_EXCLUDED)
        exclude_dirs = list(set(exclude_dirs))

        if exclude_files is None:
            exclude_files = []
        exclude_files.extend(self.DEFAULT_EXCLUDED)

        exclude_files = list(set(exclude_files))

        if not local_dir:
            local_dir = os.path.join(os.getcwd(), "/data/tmp")
            print(f"Local directory not specified. Using {local_dir}")
            if not os.path.exists(local_dir):
                print(f"Creating local directory: {local_dir}")
                os.makedirs(local_dir)

        path = os.path.join(local_dir, self.repo_url.split("/")[-1])
        super().__init__(path, exclude_dirs, exclude_files, debug)

        self._documents = []

    def load(self) -> List[Document]:
        """Load from git repository."""
        self._clone_git_repo()
        docs = super().load()
        self._cleanup_cache()
        return docs

    def _cleanup_cache(self):
        """Cleanup cache."""
        if self.cleanup_cache_dir is True:
            shutil.rmtree(self.path)

    def get_documents(self) -> List[Document]:
        """Get documents."""
        return self._documents

    def _is_excluded(self, path: Path) -> bool:
        """Check if directory or file is excluded."""
        if self._gitignore and self._gitignore(path.as_posix()):
            self._debug(f"Excluding (from .gitignore): {path.as_posix()}")
            return True
        return super()._is_excluded(path)

    def _load_file(self, file_path: Path, extra_metadata: dict = None) -> List[Document]:
        # Get the file extension
        if not '.' in file_path.name:
            ext = ''
        else:
            ext = file_path.name.split('.')[-1]
        if self._include_only_known_extensions and ext not in self.valid_extensions:
            self._debug(f"Excluding (from include_only_known_extensions): {file_path.as_posix()}")
            return

        if not extra_metadata:
            extra_metadata = {}
        repo_url = self.repo_url
        if repo_url.endswith('.git'):
            repo_url = repo_url[:-4]
        if repo_url.endswith('/'):
            repo_url = repo_url[:-1]
        if repo_url.startswith('git@github.com:'):
            repo_url = repo_url.replace('git@github.com:', 'https://github.com/')
        file_name = file_path.as_posix().replace(self.path, '')
        remote_url = repo_url + '/blob/' + self.branch + file_name
        extra_metadata['git_url'] = remote_url
        extra_metadata['source'] = remote_url
        super()._load_file(file_path, extra_metadata)

    def _clone_git_repo(self):
        """Clone a git repo."""
        cwd = os.getcwd()
        # Check if repo exists and pull
        if os.path.isdir(self.path):
            self._debug(f"Pulling repo: {self.repo_url} with branch: {self.branch}")
            try:
                os.chdir(self.path)
                # use Popen to avoid blocking
                cmd = "git pull"
                exitcode, output = subprocess.getstatusoutput(cmd)
                if exitcode != 0:
                    raise Exception(f"Error cloning repo: exitcode {exitcode}: {output}")
            finally:
                os.chdir(cwd)
        # If repo doesn't exist, clone
        else:
            os.makedirs(self.path, exist_ok=True)
            self._debug(f"Cloning repo: {self.repo_url} with branch: {self.branch}")
            # use Popen to avoid blocking
            cmd = f"git clone {self.repo_url} -b {self.branch} --single-branch {self.path}"
            exitcode, output = subprocess.getstatusoutput(cmd)
            if exitcode != 0:
                raise Exception(f"Error cloning repo: exitcode {exitcode}: {output}")
        os.chdir(self.path)
        try:
            self._gitignore = parse_gitignore(f'{self.path}/.gitignore')
            self._debug(f"Loaded .gitignore: {self.path}/.gitignore")
        except:
            self._gitignore = None
        os.chdir(cwd)
        return self.path


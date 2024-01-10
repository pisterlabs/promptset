from typing import Iterator, Optional

from langchain_core.documents import Document
from document_loaders.AlitaDirectoryLoader import AlitaDirectoryLoader
from tempfile import TemporaryDirectory
from tools import git

class AlitaGitRepoLoader(AlitaDirectoryLoader):
    
    def __init__(self, **kwargs):
        self.source = kwargs.get('source') # Git repo url
        self.branch = kwargs.get('branch', 'main') # Git branch
        self.path = kwargs.get('path', TemporaryDirectory().name) # Target directory to clone the repo
        self.depth = kwargs.get('depth', None) # Git clone depth
        self.delete_git_dir = kwargs.get('delete_git_dir', True) # Delete git directory after loading
        self.username = kwargs.get('username', None) # Git username
        self.password = kwargs.get('password', None) # Git password
        self.key_filename = kwargs.get('key_filename', None) # Git key filename
        self.key_data = kwargs.get('key_data', None) # Git key data   
        
        kwargs['path'] = self.path # this could happen and cause an exception that directory loader missing path
        for key in ['source', 'branch', 'depth', 'delete_git_dir', 'username', 'password', 'key_filename', 'key_data']:
            try:
                del kwargs[key]
            except:
                pass
        super().__init__(**kwargs)
        
    def __clone_repo(self):
        print(self.source)
        print(self.path)
        git.clone(
            source=self.source,
            target=self.path,
            branch=self.branch,
            depth=self.depth,
            delete_git_dir=self.delete_git_dir,
            username=self.username,
            password=self.password,
            key_filename=self.key_filename,
            key_data=self.key_data,
        )
    
    def load(self):
        self.__clone_repo()
        return super().load()
    
    def lazy_load(self) -> Iterator[Document]:
        self.__clone_repo()
        for document in super().lazy_load():
            yield document
        return
            
    
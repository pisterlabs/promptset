from langchain.document_loaders import DirectoryLoader

class DirectoryLoaderWrapper():
    def __init__(self, root_dir, glob="**/*.py", show_progress=False):
        self.root_dir = root_dir
        self.glob = glob
        self.show_progress = show_progress

    def load(self):
        files = []
        loader = DirectoryLoader(self.root_dir, self.glob, show_progress=True)
        files = loader.load()
        return files
    

        
from code_splitter import CodeSplitter

from langchain.schema import Document


class FileManager:
    @classmethod
    def split_content(cls, file_contents):
        split_contents = {}
        for path, content in file_contents.items():
            splitter = CodeSplitter(content)
            split_contents[path] = splitter.split()
        return split_contents

    @classmethod
    def content_for_pinecone(cls, split_contents):
        ret = []
        flattened = cls.flatten_content(split_contents)
        for elem in flattened:
            ret.append(Document(page_content=elem))
        return ret

    @staticmethod
    def flatten_content(split_contents):
        return [element for sublist in split_contents.values() for element in sublist]

    @staticmethod
    def print_split_file_content(arg):
        for path, split_content in arg.items():
            print("============================================================")
            print(f"CONTENT OF {path}:")
            for i, content in enumerate(split_content):
                print(f"PART {i}: ")
                print(content)
            print("============================================================\n\n\n")

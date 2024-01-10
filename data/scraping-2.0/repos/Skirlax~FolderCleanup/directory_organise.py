import argparse
import os
import shutil

import openai
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

file_types = {
    "Documents": [".docx", ".pdf", ".txt", ".pptx", ".xlsx", ".odt", ".rtf", ".html", ".md"],
    "Pictures": [".jpg", ".png", ".gif", ".bmp", ".svg", ".tiff", ".jpeg", ".psd", ".webp"],
    "Videos": [".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".vob"],
    "Audio": [".mp3", ".wav", ".flac", ".aac", ".ogg", ".wma", ".m4a", ".opus"],
    "Executables": [".exe", ".apk", ".msi", ".app", ".bin", ".sh", ".bat", ".jar", ".deb"],
    "DataFiles": [".json", ".csv", ".xml", ".yaml", ".sqlite", ".xls", ".tsv", ".parquet", ".sql", ".nbt"],
    "Archives": [".zip", ".rar", ".tar.gz", ".7z", ".iso", ".bz2", ".xz", ".gz"],
    "StandaloneCode": [".py", ".java", ".cpp", ".rb", ".html", ".css", ".php", ".pl"],
    "3DFiles": [".obj", ".stl", ".fbx", ".blend", ".dae", ".gltf", ".3ds", ".max", ".glb"],
    "Torrents": [".torrent", ".magnet"],
    "ModelFiles": [".pth", ".pt", ".h5", ".onnx", ".pb", ".tflite", ".model", ".ckpt", ".npy", ".weights"],
    "Notebooks": [".ipynb"],
    "MindMaps": [".mm"],
    "PersonalProjectsLogs": [".log"],
    "Graphs": [".ggb"]
}


class FileSorter:
    def __init__(self, user_folder_path: str):
        self.user_folder_path = user_folder_path
        self.gpt_chat = GPTChat("conversation_init_message.txt")

    def get_n_biggest_folder(self, n: int = 3):
        sizes = {}
        for item_name in os.listdir(self.user_folder_path):
            item_path = os.path.join(self.user_folder_path, item_name)
            if os.path.isdir(item_path):
                sizes[item_path] = self.get_dir_size(item_path)

        n_largest = list(sorted(sizes.keys(), key=lambda x: sizes.get(x), reverse=True))
        return n_largest[:n]

    def unique_list(self, dir_path: str):
        names = []
        for item_name in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item_name)
            if os.path.isdir(item_path):
                names.extend(self.unique_list(item_path))
            else:
                names.append(item_name)
        return self.get_unique_by_file_type(names)

    def get_unique_by_file_type(self, names: list) -> list:
        used = []
        result = []
        for name in names:
            if "." not in name:
                continue
            type_ = name.split(".")[-1]
            if type_ not in used:
                result.append(name)
                used.append(type_)

        return result

    def get_file_type(self, name: str):
        return "." + name.split(".")[-1]

    def get_present_categories_counts(self, directory_path: str):
        categories = {}
        for item_name in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item_name)
            if os.path.isdir(item_path):
                categories.update(self.get_present_categories_counts(item_path))
            else:
                file_type = self.get_file_type(item_path)
                category = self.get_category(file_type)
                if category is None:
                    continue
                if category in categories.keys():
                    categories[category] += 1
                else:
                    categories[category] = 1
        return categories

    def dist_to_percent(self, dist: dict):
        total = sum(dist.values())
        for key in dist.keys():
            dist[key] = dist[key] / total
        return dist

    def get_most_freq_category(self, dist: dict) -> tuple:
        sorted_ = list(sorted(dist.keys(), key=lambda x: dist.get(x), reverse=True))
        if len(sorted_) == 0:
            return "", 0
        return sorted_[0], dist[sorted_[0]]

    def get_category(self, file_type: str):
        m = [x for x in file_types.keys() if any([True if y == file_type.lower() else False for y in file_types[x]])]
        if len(m) > 0:
            return m[0]
        else:
            return None

    def get_dir_size(self, dir_path: str):
        total_size = 0
        try:
            for file_name in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file_name)
                if os.path.isdir(file_path):
                    total_size += self.get_dir_size(file_path)
                else:

                    total_size += os.path.getsize(file_path)
        except PermissionError:
            pass
        return total_size

    def move_to_category_folder(self, category: str, file_path: str):
        category_path = os.path.join(self.user_folder_path, category)
        os.makedirs(category_path, exist_ok=True)
        shutil.move(file_path, category_path)

    def sort(self, directories: list | None):
        if directories is None:
            directories = self.get_n_biggest_folder()
        for directory in directories:
            directory_path = os.path.join(self.user_folder_path, directory)
            for file_name in os.listdir(directory_path):
                file_path = os.path.join(directory_path, file_name)
                if os.path.isdir(file_path):
                    category_dist = self.dist_to_percent(self.get_present_categories_counts(file_path))
                    most_freq_category = self.get_most_freq_category(category_dist)
                    if most_freq_category[1] >= 0.7:
                        self.move_to_category_folder(most_freq_category[0], file_path)
                    else:
                        file_names = self.unique_list(file_path)
                        if len(file_names) == 0:
                            continue
                        category = self.gpt_chat.create_completion(str(file_names)).choices[0].message["content"]
                        # if category == "[NONE]":
                        #     # rename
                        #     shutil.move(file_path, file_path + "_[DELETE-MARK]")
                        if category != "[NONE]":
                            self.move_to_category_folder(category.replace("[", "").replace("]", ""), file_path)
                else:
                    file_type = "." + file_path.split(".")[-1]
                    category = self.get_category(file_type)
                    if category is None:
                        continue
                    self.move_to_category_folder(category, file_path)


class GPTChat:
    def __init__(self, init_message_path: str):
        self.messages = [
            {"role": "user", "content": self.read_init_message(init_message_path)}
        ]

    def create_completion(self, message_content: str):
        self.clear_conversation()
        self.add_message(message_content)
        return openai.ChatCompletion.create(
            model="gpt-4",
            messages=self.messages
        )

    def read_init_message(self, name: str):
        with open(name, "r") as file:
            content = file.read()
        return content

    def add_message(self, content: str):
        self.messages.append(
            {"role": "user", "content": content}
        )

    def clear_conversation(self):
        self.messages = self.messages[:1]


def main():
    argparse.ArgumentParser(description="Organise your files in a directory")
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str,
                        help="Path to the directory in which you want to create (or use) the category directories. "
                             "For example user's home directory (Windows: Users\\<username>)",
                        required=True)
    parser.add_argument("-d", "--directories", type=str, nargs="+",
                        help="Directories to sort (for example Downloads Documents). "
                             "This argument is not required, and if not provided, the program will sort the 3 biggest "
                             "folders in --path.")
    args = parser.parse_args()
    sorter = FileSorter(args.path)
    sorter.sort(args.directories)


if __name__ == "__main__":
    main()
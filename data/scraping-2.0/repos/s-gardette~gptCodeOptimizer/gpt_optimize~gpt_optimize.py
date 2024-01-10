import os
from .missing_import import MissingImport
from .openai_helper import OpenAiHelper


class CodeProcessor:
    def __init__(self, logger, model="gpt-3.5-turbo-0613"):
        self.logger = logger
        self.model = model
        self.openai = OpenAiHelper(self.model, self.logger)
        self.missing_imports = MissingImport(logger=self.logger)

    @staticmethod
    def log_to_file(file_path, log_file="failed_files.txt"):
        with open(log_file, "a") as file:
            file.write(f"{file_path}\n")

    def process_and_save_code(self, code, path, compressed=False):
        optimised_code = self.openai.process_code(
            code, "optimise", compressed=compressed
        )
        if optimised_code is None:
            self.logger.warning("Failed to optimise code. Skipping.")
            self.log_to_file(path)
            return
        self.save_code(optimised_code, path)
        return optimised_code

    def read_and_process_file(self, input_path, output_path):
        with open(input_path, "r", encoding="utf-8") as file:
            return self.process_and_save_code(file.read(), output_path)

    def process_file(self, input_path, output_path, codebase_path=None):
        file_size = os.path.getsize(input_path)
        self.logger.info(f"Processing (size: {file_size}): {input_path}")
        optimised_code = self.read_and_process_file(input_path, output_path)
        if optimised_code and codebase_path is not None:
            self.missing_imports.create(
                input_path, output_path, codebase_path, optimised_code
            )

    def process_directory(
        self,
        input_dir_path,
        output_dir_path,
        code_base_path=None,
        limit=None,
        allowed_extensions=["js", "jsx", "ts", "tsx"],
    ):
        for file_count, (root, dirs, files) in enumerate(os.walk(input_dir_path)):
            if limit and file_count >= limit:
                break

            for file in files:
                _, extension = os.path.splitext(file)
                if extension.lstrip(".") not in allowed_extensions:
                    continue

                input_path = os.path.join(root, file)
                output_path = self.replace_input_with_output(
                    input_path, input_dir_path, output_dir_path
                )
                self.process_file(input_path, output_path, code_base_path)

    def replace_input_with_output(self, path, input_dir_path, output_dir_path):
        return path.replace(input_dir_path, output_dir_path)

    def save_code(self, optimised_code, output_path):
        try:
            self.logger.info(f"Saving: {output_path}")
            if optimised_code is not None:
                dir_name = os.path.dirname(output_path)
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)

                with open(output_path, "w") as f:
                    f.write(optimised_code)
                    f.write("\n\n")
            else:
                self.logger.warning("Optimised code is None, nothing to save.")
        except Exception as e:
            self.logger.error(f"Error saving file: {e}")

    @staticmethod
    def create_test_path(path):
        # extract parts of the original path
        root_path, file_path = path.split("output/", 1)
        file_name, ext = os.path.splitext(file_path)

        # generate new file name and path
        new_file_name = f"{file_name}.test{ext}"
        new_path = os.path.join(root_path, "output", "___tests___", new_file_name)

        return new_path

    def create_test_file(self, path, code, compressed=False):
        test_code = self.openai.process_code(
            code, "generate_test", compressed=compressed
        )
        test_file_path = self.create_test_path(path)
        self.save_code(test_code, test_file_path)

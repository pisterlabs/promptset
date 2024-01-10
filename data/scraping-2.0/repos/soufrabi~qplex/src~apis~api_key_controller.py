""" Open AI Api"""

import openai
import os
from src.localStorage.storage_query_settings import StorageQuerySettings
from src.utils import utils

class ApiKeyController:
    def __init__(self):
        print("API controller constructor called")
        dirname = utils.get_settings_dir()
        self.filename = os.path.join(dirname, 'secret_apikey.txt')

    def get_apikey(self):
        # If the file does not exist then create the file
        if not os.path.exists(self.filename):
            open(self.filename, 'w').close()
        # if the file is empty, them empty string
        if os.stat(self.filename).st_size == 0:
            f_data = ""
        else:  # The file contains the current api key (valid/invalid)
            with open(self.filename, "r") as f:
                f_data = f.readline()
        # print(f_data)
        return f_data

    def set_apikey(self, new_apikey):
        # If the file does not exist then create the file
        if not os.path.exists(self.filename):
            open(self.filename, 'w').close()

        # Writing to secret_apikey.txt
        with open(self.filename, "w") as outfile:
            outfile.write(new_apikey)
        print("API key Set by APIController")



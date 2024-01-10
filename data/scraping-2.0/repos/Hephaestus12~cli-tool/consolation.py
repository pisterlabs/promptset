import os
import openai
import time
import pandas as pd

def traverse(rootdir):
    file_data = {}
    try:
        for subdir, dirs, files in os.walk(rootdir):
            # print(f"Directory: {subdir}")
            for file in files:
                filepath = os.path.join(subdir, file)
                print(f"Reading File: {filepath}")
                with open(filepath, "r") as f:
                    file_contents = f.read()
                    file_data[filepath] = file_contents
                    # do something with the file contents here
    except Exception as e:
        print(f"Error processing file {filepath} with error {e}")
    return file_data

def save_file_data(file_data, output_file):
    with open(output_file, "w") as f:
        for filepath, filecontent in file_data.items():
            print(f"Writing File: {filepath}")
            f.write(filepath + ":\n")
            f.write(filecontent + "\n")
            f.write("\n-----\n\n")

rootdir = "/Users/admin/wepromise/backend-service/code/src/main/kotlin/wepromise/application/finance/stripe"
output_file = f"./payment-stripe.txt"

file_data = traverse(rootdir)
save_file_data(file_data, output_file)

from collections import OrderedDict
import os
import re
import sys
import openai
import subprocess
import shutil
import csv
import itertools

# Set environment variables
openai.api_key = os.environ["OPENAI_API_KEY"]
# OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
# print(OPENAI_API_KEY)
# print(openai.api_key)

# Global variable to keep track of the conversation with GPT-3
conversation_history = []
DEBUG = False

# Function to check input parameters
def check_input_params():
    if len(sys.argv) != 4:
        print("Usage: python3 script.py <parameter_name> <method_name> <classname>")
        sys.exit(1)

def append_to_conversation(role, content):
    global conversation_history
    conversation_history.append({"role": role, "content": content})

    # Trim the conversation history to keep only the last 3 interactions
    if len(conversation_history) > 3:
        conversation_history = conversation_history[-3:]

def write_test_code_to_file(test_file_path, test_code):
    # Define the license header and package declaration
    LICENSE_HEADER = """/**
     * Licensed to the Apache Software Foundation (ASF) under one
     * or more contributor license agreements.  See the NOTICE file
     * distributed with this work for additional information
     * regarding copyright ownership.  The ASF licenses this file
     * to you under the Apache License, Version 2.0 (the
     * \"License\"); you may not use this file except in compliance
     * with the License.  You may obtain a copy of the License at
     *
     *     http://www.apache.org/licenses/LICENSE-2.0
     *
     * Unless required by applicable law or agreed to in writing, software
     * distributed under the License is distributed on an \"AS IS\" BASIS,
     * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
     * See the License for the specific language governing permissions and
     * limitations under the License.
     */
    package org.apache.hadoop.llmgenerated;"""

    # Remove '```java' and '```' from the response if present
    test_code = test_code.replace("```java", "").replace("```", "").strip()
    
    # Write unit test code to the file
    with open(test_file_path, 'w') as test_file:
        test_file.write(LICENSE_HEADER + "\n\n" + test_code)   

def extract_test_code(test_file_path):
    with open(test_file_path, 'r') as file:
        content = file.read()

    # Match everything from the first import statement onwards
    match = re.search(r'(import .*?;\s*.*)', content, re.DOTALL)

    if match:
        unit_test_code = match.group(1)
        return unit_test_code
    else:
        print("No import statement found in the test file.")
        return None 

# Function to generate unit test code using OpenAI API
def generate_unit_test(parameter_xml, module, method, classname, testname):
    # print(parameter_xml + "\n" + module + "\n" + method + "\n" + classname + "\n" + testname)
    
    # Define the system message
    # system_msg = "You are a skilled Java developer familiar with the Apache Hadoop project. Hadoop is an open-source framework that allows for the distributed processing of large data sets across clusters of computers using simple programming models. It is designed to scale up from single servers to thousands of machines, each offering local computation and storage. Hadoop's codebase is mainly in Java and adheres to Java development best practices. Your task is to generate unit tests for Java methods in the Hadoop project, ensuring the tests are comprehensive and cover various scenarios, including edge cases. The tests should follow Java coding standards and practices suitable for a large-scale, well-maintained open-source project."
    system_msg = ("As a seasoned Java developer, you're tasked with creating unit tests for the "
              "'{module}' module within the Apache Hadoop project, which can be found at "
              "https://github.com/apache/hadoop. The focus is on Configuration tests, crucial for "
              "ensuring the robustness of the software against configuration changes, a common source "
              "of system failures and service outages. Your unit tests should validate the software code "
              "against various values from the 'hdfs-default.xml' file, aiming to catch any misconfigurations. "
              "Follow Java coding standards and best practices appropriate for this large-scale, "
              "open-source project.")

    # user_msg = f"Generate a unit test for the method {method_name} with the configuration parameter {parameter_name} in Hadoop Common project. Ensure the code is in Java and please provide the code without any explanations or text except code."
    # user_msg = (f"Create a Java unit test named '{classname}Test' for '{method_name}' method in the Hadoop Common project, "
    #         f"especially focusing on the configuration parameter '{parameter_name}'. Include all necessary imports. "
    #         "Test for all possible cases to ensure robust detection of misconfigurations. Provide the test code only, "
    #         "with no additional explanations or text.")

    # user_msg = "It is crucial that the response comprises only the Java test code itself, devoid of any explanations, comments, or text. Create a Java Configuration test named '{testname}Test' for testing the '{method}' method in the '{classname}' class of the '{module}' project. This test is crucial for assessing how the software behaves with different values of the '{parameter_xml}' configuration parameter. Include all necessary imports and ensure the test effectively exercises the code under varied configurations to detect any potential misconfigurations. Do not explicitly set parameter values in the test code. Instead, use 'conf.get' to read the values.\n\nHere's a guide example for the fs.DefaultFS parameter in FileSystemTest:\n\nimport org.apache.hadoop.conf.Configuration;\nimport org.apache.hadoop.fs.FileSystem;\nimport org.junit.Assert;\nimport org.junit.Before;\nimport org.junit.Test;\nimport java.net.URI;\n\npublic class FileSystemTest {\n\n    private Configuration conf;\n\n    @Before\n    public void setUp() {\n        conf = new Configuration();\n   }\n\n    @Test\n    public void testDefaultFSConfiguration() {\n        URI defaultUri = FileSystem.getDefaultUri(conf);\n        Assert.assertEquals(URI.create(conf.get(FileSystem.FS_DEFAULT_NAME_KEY)), defaultUri);\n    }\n}\n\nGenerate tests similar to this structure, with code only."

    # user_msg = f"It is crucial that the response comprises only the Java test code itself, devoid of any explanations, comments, or text. Create a Java Configuration test named '{classname}Test' for the method '{method}' in the Hadoop Common project. This test is crucial for assessing how the software behaves with different values of the '{parameter_xml}' configuration parameter. Include all necessary imports and ensure the test effectively exercises the code under varied configurations to detect any potential misconfigurations. Do not explicitly set parameter values in the test code. Instead, use 'conf.get' to read the values.\n\nHere's a guide example for the fs.DefaultFS parameter in FileSystemTest:\n\nimport org.apache.hadoop.conf.Configuration;\nimport org.apache.hadoop.fs.FileSystem;\nimport org.junit.Assert;\nimport org.junit.Before;\nimport org.junit.Test;\nimport java.net.URI;\n\npublic class FileSystemTest {\n\n    private Configuration conf;\n\n    @Before\n    public void setUp() {\n        conf = new Configuration();\n   }\n\n    @Test\n    public void testDefaultFSConfiguration() {\n        URI defaultUri = FileSystem.getDefaultUri(conf);\n        Assert.assertEquals(URI.create(conf.get(FileSystem.FS_DEFAULT_NAME_KEY)), defaultUri);\n    }\n}\n\nGenerate tests similar to this structure, with code only."

    user_msg = """It is crucial that the response comprises only the Java test code itself, devoid of any explanations, comments, or text. Create a Java Configuration test named 'Test{classname}' for the method '{method}' in the Hadoop Hdfs project. This test is crucial for assessing how the software behaves with different values of the '{parameter_xml}' configuration parameter. Include all necessary imports and ensure the test effectively exercises the code under varied configurations to detect any potential misconfigurations. Do not explicitly set parameter values in the test code. Instead, use 'conf.get' to read the values.

Here's a guide example for the fs.DefaultFS parameter in FileSystemTest:

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import java.net.URI;

public class FileSystemTest {{

    private Configuration conf;

    @Before
    public void setUp() {{
        conf = new Configuration();
    }}

    @Test
    public void testDefaultFSConfiguration() {{
        URI defaultUri = FileSystem.getDefaultUri(conf);
        Assert.assertEquals(URI.create(conf.get(FileSystem.FS_DEFAULT_NAME_KEY)), defaultUri);
    }}
}}

Generate tests similar to this structure, with code only.""".format(classname=classname, method=method, parameter_xml=parameter_xml)

    # Make sure to append system and user messages to the conversation
    # Append the system message only once, at the start of the conversation
    if not conversation_history:  # If the conversation history is empty, append the system message
        append_to_conversation("system", system_msg)
    
    append_to_conversation("user", user_msg)

    # print("User Message:", user_msg)

    # Create a dataset using GPT
    response = openai.ChatCompletion.create(model="gpt-4",
                                            messages=[
                                                {
                                                    "role": "system", 
                                                    "content": system_msg
                                                },
                                                {
                                                    "role": "user", 
                                                    "content": user_msg
                                                }]
                                            )
    
    
   # Check if the finish reason is 'stop' indicating complete output
    if response["choices"][0]["finish_reason"] == "stop":
        unit_test_code = response["choices"][0]["message"]["content"]

        # Remove '```java' and '```' from the response if present
        # unit_test_code = unit_test_code.replace("```java", "").replace("```", "").strip()
        # This will remove '```' followed by 'Java' (in any case) from the beginning of the string
        unit_test_code = re.sub(r'```java\s*', '', unit_test_code, flags=re.IGNORECASE)

        # This will also remove any standalone triple backticks '```' from anywhere in the string
        unit_test_code = unit_test_code.replace("```", "")

        return unit_test_code
    else:
        return None
     
# Function to send error message to GPT
def send_to_gpt(error_msg, unit_test_code):
    # user_msg = f"The Java unit test {unit_test_code} failed with the following error: {error_msg}. Please suggest a fix to resolve this error. Provide complete test code without any explanations or text except code."
    user_msg = (f"Please analyze the error and suggest a corrected version of the test code."
            f"The response should strictly contain only the revised Java code for the test, without any additional text, comments, or explanations. "
            f"Ensure the code is formatted for immediate compilation and testing."
            f"The Java unit test for {unit_test_code} encountered a failure with this error: {error_msg}. ")

    # Always append the latest user message
    append_to_conversation("user", user_msg)

    # Print conversation history before sending to API
    # print("Sending the following conversation history to GPT-3:")
    # for msg in conversation_history:
    #     print(msg)

    response = openai.ChatCompletion.create(model="gpt-4",
                                            messages=conversation_history
                                            )
   # Check if the finish reason is 'stop' indicating complete output
    if response["choices"][0]["finish_reason"] == "stop":
        fixed_test_code = response["choices"][0]["message"]["content"]

        # Remove '```java' and '```' from the response if present
        fixed_test_code = fixed_test_code.replace("```java", "").replace("```", "").strip()

        # Append GPT's response to the conversation history
        gpt_msg = {
            "role": "system",
            "content": fixed_test_code
        }
        conversation_history.append(gpt_msg)

        return fixed_test_code
    else:
        return None
    
# Function to handle build attempts
def attempt_build(hadoop_common_path, test_file_path):
    current_code = extract_test_code(test_file_path)

    for attempt in range(1, 6):
        print(f"Build Attempt {attempt}")

        # Write the current unit test code to the file
        if attempt > 1:
            write_test_code_to_file(test_file_path, current_code)

        try:
            # Attempt to build
            subprocess.run(["mvn", "clean", "install", "-B", "-DskipTests"], cwd=hadoop_common_path, text=True, capture_output=True, check=True)
            print("Build succeeded.")
            return True, current_code  # Build was successful, return current code
        except subprocess.CalledProcessError as e:
            print("Build failed.")
            # Extract error messages
            # error_msgs = re.findall(r"\[ERROR\].*?(?=\[ERROR\] -> \[Help 1\]|$)", e.output, re.DOTALL)
            
            # Remove duplicates while preserving order
            # error_msgs = list(OrderedDict.fromkeys(error_msgs))
            # error_msg = "\n".join(error_msgs)

            # error_lines = re.findall(r"\[ERROR\].*?(?=\n|$)", e.output)
            error_lines = re.findall(r"\[ERROR\].*?(?=-> \[Help 1\]|\n|$)", e.output)
            error_lines = list(dict.fromkeys(error_lines))
            error_msg = "\n".join(error_lines)

            print(error_msg)
            
            if "BUILD FAILURE" in e.output:
                print("Sending extracted error to GPT...")
                gpt_response = send_to_gpt(error_msg, current_code)
                print("Response received from GPT:", gpt_response)

                if gpt_response:
                    suggested_fix = gpt_response.strip()
                    current_code = suggested_fix  # Update the current code with the suggested fix
                    continue
                else:
                    print("No fix suggested, or the fix did not resolve the issue.")

    print("Build failed and no working code found after multiple attempts.")
    return False, current_code  # Build failed, return the latest code

def inject_values_in_config_file(hadoop_common_path, parameter_name, parameter_value):
    config_file_path = os.path.join(hadoop_common_path, "target/classes/hdfs-default.xml")
    # Create a backup of the configuration file
    backup_file_path = os.path.join(hadoop_common_path, "src/test/java/org/apache/hadoop/llmgenerated/hdfs-default.xml")
    # Check if the config file exists
    if not os.path.isfile(config_file_path):
        print(f"Config file {config_file_path} does not exist.")
        return

    # Create a backup
    try:
        shutil.copyfile(config_file_path, backup_file_path)
    except Exception as e:
        print(f"Failed to create backup file {backup_file_path}. Error: {e}")
        return
    # Read the configuration file
    with open(config_file_path, 'r') as config_file:
        config_file_contents = config_file.read()
    
    # Replace the parameter value in the configuration file
    new_config_file_contents = re.sub(rf"<name>{parameter_name}</name>\s*<value>.*?</value>", rf"<name>{parameter_name}</name><value>{parameter_value}</value>", config_file_contents, flags=re.DOTALL)

    # Write the updated configuration file
    with open(config_file_path, 'w') as config_file:
        config_file.write(new_config_file_contents)

def restore_config_file(hadoop_common_path):
    config_file_path = os.path.join(hadoop_common_path, "target/classes/hdfs-default.xml")
    # Restore the configuration file from the backup
    backup_file_path = os.path.join(hadoop_common_path, "src/test/java/org/apache/hadoop/llmgenerated/hdfs-default.xml")
    
    shutil.copyfile(backup_file_path, config_file_path)

def run_test_cases(hadoop_common_path, test_file_path, test_class, suggested_fix, property, value, type):
    current_code = suggested_fix
    print("Running test cases...", test_class)

    # Inject the property value in the configuration file
    inject_values_in_config_file(hadoop_common_path, property, value)

    test_command = [
        "mvn", "-B", "clean", "test",
        f"-Dtest={test_class}"
    ]
    try:
        result = subprocess.run(test_command, cwd=hadoop_common_path, text=True, capture_output=True, check=True)
        print(result.stdout)
        test_output = result.stdout

        # Checking the test results in the output
        info_tests_run = re.findall(r"\[INFO\]\s+Tests run:.*", test_output, re.MULTILINE)
        error_tests_run = re.findall(r"\[ERROR\]\s+Tests run:.*", test_output, re.MULTILINE)

        print("Info tests run:", info_tests_run)
        print("Error tests run:", error_tests_run)

        failure_pattern = re.compile(r"\[ERROR\] There are test failures\.")
        if failure_pattern.search(test_output):
            print("Test cases failed.")
            error_lines = re.findall(r"\[ERROR\].*?(?=\n|$)", test_output)
            error_lines = list(dict.fromkeys(error_lines))
            error_msg = "\n".join(error_lines)
    
            return False, error_msg  # Return None to indicate no solution found
        elif info_tests_run:
            print("Test cases ran successfully.")
            return True, None  # Return True to indicate success
    except subprocess.CalledProcessError as e:
        print("Test cases failed.")
        error_lines = re.findall(r"\[ERROR\].*?(?=\n|$)", e.output)
        error_lines = list(dict.fromkeys(error_lines))
        error_msg = "\n".join(error_lines)

        return False, error_msg  # Return None to indicate no solution found
    
    # finally:
        # Restore the configuration file from the backup
        # restore_config_file(hadoop_common_path)

    return None  # No working code found after all attempts

def execute_tests(hadoop_common_path, test_file_path, test_class, suggested_fix, parameter_xml, value, type):
    for attempt in range(1, 6):
        print(f"Test Attempt {attempt}")
        test_success, error_msg = run_test_cases(hadoop_common_path, test_file_path, test_class, suggested_fix, parameter_xml, value, type)
        if type == "GOOD":
            if test_success:
                print("Test case passed successfully.")
                break
            else:
                print(f"Test case failed on attempt {attempt + 1}. Retrying...")
                # Logic for sending error to GPT, applying fixes, and retrying build if necessary
                print("Sending extracted error to GPT...")
                gpt_response = send_to_gpt(error_msg, suggested_fix)
                print("Response received from GPT:", gpt_response)

                write_test_code_to_file(test_file_path, gpt_response)

                build_success, suggested_fix = attempt_build(hadoop_common_path, test_file_path)
                if build_success:
                    continue
                else:
                    print("Build failed after multiple attempts, skipping test case execution.")
                    break

        elif type == "BAD":
            if test_success:
                print("Test case unexpectedly passed for a bad configuration value.")
                return True
            else:
                print(f"Test case failed on attempt {attempt + 1}. Retrying...")
            test_success = run_test_cases(hadoop_common_path, test_file_path, test_class, suggested_fix, parameter_xml, value, type)
            print(f"Test case {'passed' if test_success else 'failed'} on attempt {attempt + 1}. Recorded as type 'bad'.")
            return test_success
    print("Maximum test retries reached.")
    return False



def get_property_description(config_file_path, parameter_name):
    # Read the configuration file
    with open(config_file_path, 'r') as config_file:
        config_file_contents = config_file.read()
    
    # Extract the whole xml block for the parameter
    xml_block = re.search(rf"<name>{parameter_name}</name>.*?</property>", config_file_contents, flags=re.DOTALL).group()
    # print(xml_block)
    return xml_block

def read_tsv_and_execute(tsv_file_path, config_file_path):
    with open(tsv_file_path, 'r') as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter='\t')
        next(tsv_reader)  # Skip the header row
        # Iterate over each row in the CSV file till end of file or 5 rows, whichever is earlier
        for row in itertools.islice(tsv_reader, 5):
            module, parameter_name, method, classname, testname, value, type = row[0], row[1], row[2], row[3], row[4], row[5], row[6]
            parameter_xml = get_property_description(config_file_path, parameter_name)
            execute(parameter_xml, module, method, classname, testname, value, type)
          
 
# Actual function to execute the script
def execute(parameter_xml, module, method, classname, testname, value, type):
    
    base_method_name = classname

    # hadoop_common_path = "/home/nvadde2/hadoop/hadoop-common-project/hadoop-common"
    hadoop_common_path = "/home/nvadde2/hadoop/hadoop-hdfs-project/hadoop-hdfs"
    test_file_name = f"Test{testname}.java"
    test_file_path = os.path.join(hadoop_common_path, "src/test/java/org/apache/hadoop/llmgenerated", test_file_name)
    test_class = f"org.apache.hadoop.llmgenerated.Test{testname}"
    config_file = os.path.join(hadoop_common_path, "/target/classes/hdfs-default.xml")

    # Ensure the test file directory exists
    os.makedirs(os.path.dirname(test_file_path), exist_ok=True)

    print("Generating ctest code...")
    if not os.path.isfile(test_file_path):
        ctest_code = generate_unit_test(parameter_xml, module, method, classname, testname)

        if ctest_code is None:
            print("No valid unit test code generated.")
            return
        else:
            # print("Unit test code generated:", ctest_code)
            write_test_code_to_file(test_file_path, ctest_code)
    else:
        print("Test file already exists. Proceeding to Compile.")

    # Attempt to build and handle errors
    os.chdir(hadoop_common_path)

    build_success, suggested_fix = attempt_build(hadoop_common_path, test_file_path)
    if build_success:
        execute_tests(hadoop_common_path, test_file_path, test_class, suggested_fix, parameter_xml, value, type)
    else:
        print("Build failed after multiple attempts, skipping test case execution.")

# Main function
def main():

    hadoop_common_path = "/home/nvadde2/hadoop/hadoop-hdfs-project/hadoop-hdfs"
    config_file = os.path.join(hadoop_common_path, "src/main/resources/hdfs-default.xml")
    tsv_file_path = os.path.join(hadoop_common_path, "src/test/java/org/apache/hadoop/llmgenerated/parameter-configurations.tsv")

    read_tsv_and_execute(tsv_file_path, config_file)

if __name__ == "__main__":
    main()

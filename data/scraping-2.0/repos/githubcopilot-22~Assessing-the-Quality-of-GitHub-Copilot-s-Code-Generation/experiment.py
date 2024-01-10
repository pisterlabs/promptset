import os
import openai_efficiency
from shared_funcs import get_json_object
from sonarqube import run_sonarqube_eval


NUMBER_OF_SAMPLES = 164


# write tests scripts for each prompt file
def create_tests():
    for i in range(NUMBER_OF_SAMPLES):
        os.chdir('human-eval')
        json_obj = get_json_object(str(i))
        os.chdir('..')
        os.chdir('experiment-code/' + str(i))

        # open file to append tests
        with open('prompt_' + str(i) + '.py', 'a') as f:
            prompt_to_write = "try:\r"
            prompt_to_write += "    count = test_" + str(i) + ".check(" + json_obj["entry_point"] + ")\r"
            prompt_to_write += "    print(str(count))\r"
            prompt_to_write += "except:\r"
            prompt_to_write += "    print('Invalid')\r"
            prompt_to_write += "print('')\r"
            f.write(prompt_to_write)

        # read contents of test_i.py
        test_to_write = ""
        with open('test_' + str(i) + '.py', 'r') as f:
            test_to_write += f.read()
        
        for j in test_to_write.splitlines():
            if "assert True" in j:
                test_to_write = test_to_write.replace(j, "")

        test_to_write = test_to_write.replace('assert', 'if')

        for j in test_to_write.splitlines():
            if "def check(" in j:
                test_to_write = test_to_write.replace(j, j + "\r    count = 0")
                break                      
        
        for j in test_to_write.splitlines():
            if ("if candidate(" in j) or ("if abs(" in j) or ("if math.fabs(" in j) or ("if tuple(" in j):
                test_to_write = test_to_write.replace(j, j + ":")

        for j in test_to_write.splitlines():
            if ("if candidate(" in j) or ("if abs(" in j) or ("if math.fabs(" in j) or ("if tuple(" in j):
                if i == 32 or i == 50 or i == 53 or i == 61:
                    test_to_write = test_to_write.replace(j, j + "\r            count += 1")
                else:
                    test_to_write = test_to_write.replace(j, j + "\r        count += 1")
    
        # append "return count" to the end of the file
        test_to_write += "\r    return count"
        with open('test_' + str(i) + '.py', 'w') as f:
            f.write(test_to_write)
        
        os.chdir('..')
        os.chdir('..')


def create_exp_code():
    for i in range(NUMBER_OF_SAMPLES):
        prompt_to_write = "import test_" + str(i) + "\n\r"
        os.chdir('code_generation/' + str(i))
        with open('prompt_' + str(i) + '.py', 'r') as f:
            prompt_to_write += f.read()
            prompt_to_write += "\n\n"

        os.chdir('..')
        os.chdir('..')

        os.chdir('experiment-code')
        os.chdir(str(i))
        # Create a python file with the name of the json file for prompt
        with open('prompt_' + str(i) + '.py', 'w') as f:
            f.write(prompt_to_write)
            print("Prompt file for " + str(i) + " created")

        os.chdir('..')
        os.chdir('..')

    create_tests()


# Open a file to read with a given name and path
def open_file(path, name):
    contents = ""
    # Open the file
    with open(path + name, 'r') as f:
        # Read the file line by line
        for line in f:
            # Return the length of the file
            contents += line

    return contents


# Given two files store the contents in a 2D array
def get_file_contents(path, name1, name2):
    contents1 = open_file(path, name1)
    # Get first python function from contents1 until second python function
    contents1 = contents1[contents1.find("def "):contents1.find("def ", contents1.find("def ") + 1)]
    contents2 = open_file(path, name2)

    return [contents1, contents2]


# Get file contents for all folders 
def get_all_file_contents(path):
    contents = []
    for i in range(NUMBER_OF_SAMPLES):
        newpath = path + str(i) + "/"
        contents.append(get_file_contents(newpath, "prompt_" + str(i) + ".py", "canonical_solution_" + str(i) + ".py"))

    return contents


def get_time_and_space_complexity(file_contents):
    time_complexity = []
    space_complexity = []
    for i in range(NUMBER_OF_SAMPLES):
        sample1 = []
        sample2 = []

        response1 = openai_efficiency.return_response(file_contents[i][0])
        response2 = openai_efficiency.return_response(file_contents[i][1])

        # Find first "O(" in response1
        time_complexity_start_1 = response1.find("O(")
        # Find first ")" after "O("
        time_complexity_end_1 = response1.find(")", time_complexity_start_1) + 1
        # Find first "O(" in response2
        time_complexity_start_2 = response2.find("O(")
        # Find first ")" after "O("
        time_complexity_end_2 = response2.find(")", time_complexity_start_2) + 1

        # Find next "O(" in response1
        space_complexity_start_1 = response1.find("O(", time_complexity_end_1)
        # Find next ")" after "O("
        space_complexity_end_1 = response1.find(")", space_complexity_start_1) + 1
        # Find next "O(" in response2
        space_complexity_start_2 = response2.find("O(", time_complexity_end_2)
        # Find next ")" after "O("
        space_complexity_end_2 = response2.find(")", space_complexity_start_2) + 1

        time_complexity_line1 = response1[time_complexity_start_1:time_complexity_end_1]
        time_complexity_line2 = response2[time_complexity_start_2:time_complexity_end_2]

        space_complexity_line1 = response1[space_complexity_start_1:space_complexity_end_1]
        space_complexity_line2 = response2[space_complexity_start_2:space_complexity_end_2]

        sample1.append(time_complexity_line1)
        sample1.append(time_complexity_line2)       

        sample2.append(space_complexity_line1)
        sample2.append(space_complexity_line2) 

        time_complexity.append(sample1)
        space_complexity.append(sample2)

    return time_complexity, space_complexity


# Write the time and space complexity to a csv file for each folder
def write_to_csv_complexity():
    from csv import reader, writer

    print("Start time and space complexity")
    file_contents = get_all_file_contents('code_generation/')
    time_complexity = get_time_and_space_complexity(file_contents)[0]
    space_complexity = get_time_and_space_complexity(file_contents)[1]

    print(time_complexity)
    print(space_complexity)

    matrix = []
    with open("results/results.csv", "r") as f:
        csv_reader = reader(f)
        for row in csv_reader:
            matrix.append(row)

    for i in range(1, NUMBER_OF_SAMPLES + 1):
        matrix[i][3] = time_complexity[i - 1][0]
        matrix[i][4] = time_complexity[i - 1][1]

        matrix[i][5] = space_complexity[i - 1][0]
        matrix[i][6] = space_complexity[i - 1][1]

    with open("results/results.csv", "w", newline='') as f:
        writer = writer(f)
        writer.writerows(matrix)
    
    print("End time and space complexity")


# write validity and correctness to csv file
def write_to_csv_correctness_validity():
    from csv import reader, writer

    print("Start correctness and validity")
    execute_all_python_files()
    test_count = count_test_cases()

    matrix = []
    with open("results/results.csv", "r") as f:
        csv_reader = reader(f)
        for row in csv_reader:
            matrix.append(row)

    os.chdir('experiment-code')
    for i in range(NUMBER_OF_SAMPLES):
        with open(str(i) + "/output_correctness_validity.txt", 'r') as f:
            contents = f.read()
            contents = contents.splitlines()
            if "Invalid" in contents:
                matrix[i + 1][2] = 0
            else:
                matrix[i + 1][2] = 1
                print("Correct: " + str(float(contents[0])))
                print(float(contents[0]) / float(test_count[i]))
                matrix[i + 1][1] = float(contents[0]) / float(test_count[i])
    os.chdir('..')

    with open("results/results.csv", "w", newline='') as f:
        writer = writer(f)
        for row in matrix:
            print(row)
            writer.writerow(row)
    
    print("End correctness and validity")


# Execute python file
def execute_python_file(file_name):
    import subprocess
    with open("output_correctness_validity.txt", 'w') as output:
        subprocess.call(["python", file_name], stdout=output)


# Execute all python files with prompt
def execute_all_python_files():
    print("Scripts running...")
    os.chdir("experiment-code")
    for i in range(NUMBER_OF_SAMPLES):
        os.chdir(str(i))
        file_name = "prompt_" + str(i) + ".py"
        print(file_name)
        execute_python_file(file_name)
        os.chdir("..")
    os.chdir("..")
    print("Script run completed.")


# Count number of test cases for each problem
def count_test_cases():
    test_count = [] 
    os.chdir("experiment-code")
    for i in range(NUMBER_OF_SAMPLES):
        os.chdir(str(i))
        count = 0
        with open("test_" + str(i) + ".py", 'r') as f:
            for line in f:
                if "candidate" in line:
                    count += 1
        # change directory back to the parent folder
        test_count.append(count - 1)
        count = 0
        os.chdir("..")
    os.chdir("..")

    return test_count


create_exp_code()
write_to_csv_correctness_validity()
write_to_csv_complexity()
run_sonarqube_eval()

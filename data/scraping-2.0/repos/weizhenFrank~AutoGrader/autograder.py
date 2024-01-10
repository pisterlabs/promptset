import openai
import os
import zipfile
import csv
import tqdm
import time
import re
# Initialize GPT-4 API
import os


openai.api_key = os.getenv('OPENAI_API_KEY')


# [Update your standard answers here]
FREQ = 30
assignment_answers = {
    "layers.py": """
    TODO1:
    
        pad = conv_param['pad']
        stride = conv_param['stride']
        
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        H_filter = (H + 2*pad - HH)/stride + 1
        W_filter = (W + 2*pad - WW)/stride + 1
        
        out = np.zeros((N, F, H_filter, W_filter))
        
        x = np.pad(x, pad_width=((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
        
        for i in range(N):
            for z in range(F):
                for j in range(H_filter):
                    for k in range(W_filter):
                        out[i,z,j,k] = np.sum(x[i,:,j*stride:(j*stride+HH),k*stride:(k*stride+WW)]*w[z,:,:,:])+b[z]

    TODO2:
        N, C, H, W = x.shape
        pool_H = pool_param['pool_height']
        pool_W = pool_param['pool_width']
        stride = pool_param['stride']
        H_filter = (H-pool_H)/stride + 1
        W_filter = (W-pool_W)/stride + 1
        
        out = np.zeros((N,C,H_filter,W_filter))
        
        for j in range(H_filter):
            for k in range(W_filter):
                out[:,:,j,k] = x[:,:,j*stride:(j*stride+pool_H),k*stride:(k*stride+pool_W)].max(axis=(2,3))
    
    """


}

def student_already_graded(student_name, results_csv):

    with open(results_csv, mode='r', newline='') as file:
        reader = csv.reader(file)

        for row in reader:
            if row and student_name in row[0]:
                return True
    return False

def grade_assignment_per_file(student_answer, standard_answer, section_name, point=2.5):
    format = {'layers.py':f"TODO1: 2.5/{point}, Correct; TODO2: 1/{point}, reason: ...; ",

              }
    
    messages = [
        {"role": "system", "content": "You are a helpful grading assistant."},
        {
            "role": "user",
            "content": f"Grade the following answers for {section_name}:\n (Standard Answer): {standard_answer}\n(Student's Answer): {student_answer}. \n \n \n You should compare each todo's answer correspondingly. In student's the python file, the TODO part (the answer) is enclosed by green hash symbols (###). As long as it's implemented correctly, no need to exactly match standard answer. Each TODO is {point}. \n \n \n Strictly follow this output format. Output format:{format[section_name]} \n \n \n Section Score:  \n"
        }
    ]

    for num in range(3):
        if num == 0:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages
            )
        else:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                temperature=0.3/num,
            )
        
        raw_feedback = response['choices'][0]['message']['content'].strip()
        

        match = re.search(r"Section Score: (\d+)/(\d+)", raw_feedback)
        
        if match:
            # Convert score to float
            final_score = float(match.group(1))
            return final_score, raw_feedback
        else:
            print("Retrying due to missing section score...")
            time.sleep(10)  
            
    raise Exception("Unable to get section score after multiple attempts")

files_path = {"layers.py":"lib/layers.py",
}


def read_python_file(file_path):
    with open(file_path, 'r') as f:
        return f.read()
    
def grade_assignment(student_folder, file_name, standard_answer):
    file_path = os.path.join(student_folder, files_path[file_name])
    student_answer = read_python_file(file_path)
    score, feedback = grade_assignment_per_file(student_answer, standard_answer, file_name)
    return score, feedback

# Directory containing all the zip files
zip_dir = "./zip"

# Directory to extract zip files
extract_dir = "./unzip"
os.makedirs(extract_dir, exist_ok=True)

# CSV file to store results
results_csv = "grading_results.csv"

error_studens = []
# Open CSV file and write header
with open(results_csv, mode='a', newline='') as file:  # mode='a' to append to existing file
    writer = csv.writer(file)
    
    # Check if file is empty and write header if it is
    if os.path.getsize(results_csv) == 0:
        writer.writerow(["Student Name", "Total Score", "Feedback"])

    # Loop through all zip files in directory
    for zip_file_name in tqdm.tqdm(os.listdir(zip_dir)):
        if zip_file_name.endswith(".zip"):
            # Extract student name from zip file name
            student_name = zip_file_name.split('-')[2]

            # Check if student has already been graded
            if student_already_graded(student_name, results_csv):
                print(f"{student_name} has already been graded. Skipping...")
                continue
            
            try:  # Add error handling
                now = time.time()
                
                # Path to store extracted files for this student
                student_extract_dir = os.path.join(extract_dir, student_name)
                print("---"*20)
                print(f"Grading {student_name}")
                # Extract zip file
                
                with zipfile.ZipFile(os.path.join(zip_dir, zip_file_name), 'r') as zip_ref:
                    zip_ref.extractall(student_extract_dir)
                
                if student_name == " Krishna Chaitanya Pulipati ":
                    student_extract_dir = '//Users/weizhenliu/Desktop/school/TA/DL/grading/HW5/unzip/ Krishna Chaitanya Pulipati /assignment5'
                if student_name == " Minoo Jafarlou ": 
                    student_extract_dir = '/Users/weizhenliu/Desktop/school/TA/DL/grading/HW5/unzip/ Minoo Jafarlou /assignment5'
                if student_name == " Ranjit Singh Kanwar ":
                    student_extract_dir = '/Users/weizhenliu/Desktop/school/TA/DL/grading/HW5/unzip/ Ranjit Singh Kanwar /assignment5'
                if student_name == " Sri Harsha Seelamneni ":
                    student_extract_dir = '/Users/weizhenliu/Desktop/school/TA/DL/grading/HW5/unzip/ Sri Harsha Seelamneni /assignment5/assignment5'
                if student_name == " Sumanth Meenan Kanneti ":
                    student_extract_dir = '/Users/weizhenliu/Desktop/school/TA/DL/grading/HW5/unzip/ Sumanth Meenan Kanneti /Sumanth_Meenan_ass5'
                if student_name == " Saloni Ajgaonkar ":
                    student_extract_dir = '/Users/weizhenliu/Desktop/school/TA/DL/grading/HW5/unzip/ Saloni Ajgaonkar /assignment5'
                if student_name == " Meghana Puli ":
                    student_extract_dir = '/Users/weizhenliu/Desktop/school/TA/DL/grading/HW5/unzip/ Meghana Puli /assignment5)'
                if student_name == " Tharun Kumar Bandaru ":
                    student_extract_dir = '/Users/weizhenliu/Desktop/school/TA/DL/grading/HW5/unzip/ Tharun Kumar Bandaru /Assignment_5'
                
                total_score = 0
                all_feedback = []
                ans = {'layers.py':5}
                for file_name in assignment_answers.keys():
                    
                    score, feedback = grade_assignment(student_extract_dir, file_name, assignment_answers[file_name])

                    total_score += score
                    all_feedback.append(file_name+':'+feedback)

                    
                    print(f"{file_name}:{score}/{ans[file_name]}\n{feedback}\n")
                print(f"Total Score: {total_score}")
                print(f"Time taken for {student_name}: {time.time() - now} seconds\n")
                print("---"*20)
                writer.writerow([student_name, total_score, all_feedback])
                time.sleep(FREQ)
            
            except Exception as e:  # Handle errors and continue
                print(f"An error occurred while grading {student_name}: {str(e)}")
                error_studens.append(student_name)
                print("Continuing with next student...\n")

print("Done grading all students! Exceptions occurred for the following students:")
print(error_studens)
            




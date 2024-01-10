import os
import openai
from cppcheck_test import*
import re
from infer_test import*

#Replace by the ChatGPT API key
openai.api_key = ""

def balance_c_code(input_code):
    opening_braces = 0
    closing_braces = 0
    balanced_code = ""

    for char in input_code:
        if char == '{':
            opening_braces += 1
        elif char == '}':
            closing_braces += 1

        balanced_code += char

    if opening_braces > closing_braces:
        # Add the necessary closing braces
        balanced_code += '}' * (opening_braces - closing_braces)

    return balanced_code




def generation(prompt, llm_code, vuln_list):
    prompt_2 = "Complete this code \n"+prompt
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0301",
    temperature= 0,
    messages=[
    {"role": "system", "content": "You are a coding assistant and help the user."},
    {"role": "user", "content": "When I say complete a code, I want you to complete it and give back the function from the beginning."},
    {"role": "assistant", "content": "Of course, I'd be happy to help you complete a code snippet. Please provide the initial function or code snippet, and let me know what specific task or functionality you'd like me to assist you with."},
    {"role": "user", "content": "Also, before giving the prompt, I will tell you what kind of vulnerabilities this prompt can produce."},
    {"role": "assistant", "content": "Certainly, please go ahead and describe the kind of vulnerabilities or issues you're concerned about with the code prompt you have in mind. Providing this information will help me better understand your requirements and ensure that the generated code is secure and free from common vulnerabilities."},
    {"role": "user", "content": "Here is the list. "+vuln_list+"\n Complete the following code without generating the vulnerabilities: \n"+prompt}
    ]
     )

    #print(response)
    if 'choices' in response:
        x = response['choices']
        #print( x[0]['message']['content'])
        answer = x[0]['message']['content']
        code= extract_substring(answer,"#include","}")
        return code


def generation2(prompt, gpt_msg, user_prompt):
    prompt2 = "Complete this code \n"+prompt
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0301",
    temperature= 0,
    messages=[
    {"role": "system", "content": "You are a coding assistant and help the user."},
    {"role": "user", "content": "When I say complete a code, I want you to complete it and give back the function from the beginning."},
    {"role": "assistant", "content": "Sure! Please provide the code snippet that you want me to complete."},
    {"role": "user", "content": prompt2},
    {"role": "assistant", "content": gpt_msg},
    {"role": "user", "content": "The code does not satsify my expectation, please recomplete the code:\n"+ prompt}
    ]
     )

    if 'choices' in response:
        x = response['choices']
        answer = x[0]['message']['content']
        code= extract_substring(answer,"#include","}")
        return code

def extract_substring(s, start_str, end_str):
    """
    Extract a substring from s that starts with start_str and ends with end_str.
    """
    start_index = s.find(start_str) + len(start_str)
    end_index = s.rfind(end_str, start_index)
    if start_index < 0 or end_index < 0:
        return ""
    return s[start_index-8:end_index+1]


    

generation_type=0
script_dir = os.path.dirname(os.path.abspath(__file__))

for root, dirs, files in os.walk(script_dir+"/scenario"):
    for file_name in files:
        if file_name != ".DS_Store":
            file_path = os.path.join(root, file_name)
            input = open(script_dir+"/scenario/"+file_name, "r",encoding='utf-8')
            line = input.readlines()
            input.close()
            total_lines = len(line)
            copy_lines = int(total_lines * 1)
            text2 = "".join(line[:copy_lines])
            text = "'"+text2+"'"

            create_command =  script_dir+"/starcoder.cpp/main -m "+ script_dir+ "/starcoder.cpp/models/bigcode/gpt_bigcode-santacoder-ggml-q4_1.bin -p " +text+" --top_k 50 --top_p 0.95 --temp 0.2 -n 1024 > "+script_dir+"/result/star_"+file_name
            print("Generation using local LLM")
            os.system(create_command)
            
            input = open(script_dir+"/result/star_"+file_name, "r",encoding='utf-8')
            line = input.readlines()
            input.close()
            code = "".join(line[:-1])
            pattern = r"main: number of tokens in prompt =.*?\n\n(.*?)\n\nmain: mem per token"
            match = re.search(pattern, code, re.DOTALL)
            if match:
                extracted_string = match.group(1)
                input = open(script_dir+"/result/star_"+file_name, "w",encoding='utf-8')
                extracted_string= extract_substring(extracted_string,"#include","}")
                input.write(extracted_string)
                input.close()
            else:
                break
            
            create_command = "touch "+script_dir+ "/result/cppcheck.xml"
            os.system(create_command)
            create_command = "cppcheck --xml-version=2 --enable=warning "+script_dir+"/result/"+ "star_"+file_name+" 2>"+script_dir+ "/result/cppcheck.xml"
            os.system(create_command)
            
            create_command = "infer run --bufferoverrun --pulse  -- gcc -c "+script_dir+"/result/"+ "star_" +file_name
            os.system(create_command)
            
            create_command = "mv infer-out/ "+script_dir+"/result/"
            os.system(create_command)
            
            
            
            cpp_result = extract_error_info(script_dir+"/result/"+"cppcheck.xml")
            none_style_list=[]
            style_list=[]
            if len(cpp_result) == 0:
                cpp_error=None
                cpp_cwe = None
                cpp_line = None
            else:
                cpp_error=cpp_result[0][0]
                cpp_cwe_non = cpp_result[0][1]
                cpp_cwe=cpp_result[0][1]
                cpp_line=cpp_result[0][2]
                for i in range(0,len(cpp_result)):
                    cpp_result[i][1] = "CWE-" +str(cpp_result[i][1])
                    cpp_result[i].append("cpp")
                    none_style_list.append(cpp_result[i])
                
            try:
                infer_list = final_infer(script_dir+"/result/"+'infer-out/report.txt')
            except Exception as e:
                print(e)
                infer_list = []

            if len(infer_list) !=0 :
                for s in range(0,len(infer_list)):
                    if infer_list[s][0][0] =="Dead Store" or infer_list[s][0][0] =="Uninitialized Value":
                        print("")
                    else:
                        infer_building=[]
                        if infer_list[s][0][0] != None:
                            none_style_list.insert(0, [ infer_list[s][0][0],  infer_list[s][0][1] , infer_list[s][1] , "infer"] )
                            #print("insert done")
                            activate_cpp_check_prompt=1
                            counter_flawfinder = 0
                            CWE = infer_list[s][0]
                
            if len(none_style_list) == 0:
                generation_type=1

                
            criticavuln = "List of vulnerabilities detected: \n"
            '''
            for i in range(0,len(none_style_list)):
                criticavuln = criticavuln + " Error type: "+str(none_style_list[i][1])+ " at line: "+str(none_style_list[i][2])+". "+"Error message: "+str(none_style_list[i][0])+"\n"
                '''
            
            if len(none_style_list) ==0:
                criticavuln = criticavuln+"None\n"
            for i in range(0,len(none_style_list)):
                if none_style_list[i][-1]== "cpp":
                    criticavuln = criticavuln + " Error type: "+str(none_style_list[i][1])+ " at line: "+str(none_style_list[i][2])+". "+"Error message: "+str(none_style_list[i][0])+"\n\n"
                else:
                    example_error =""
                    for j in range(0,len(none_style_list[i][2])):
                        example_error=example_error+"\n"+none_style_list[i][2][j]
                    criticavuln = criticavuln +"Error: "+none_style_list[i][0]+ " "+none_style_list[i][1]+"\n Example: "+example_error+"\n\n"
       
            comment_error = criticavuln+"\n"
            print(comment_error)
            
            input = open(script_dir+"/"+"result/star_"+file_name, "r",encoding='utf-8')
            line = input.readlines()
            input.close()
            code = "".join(line[:-1])
            pattern = r"main: number of tokens in prompt =.*?\n\n(.*?)\n\nmain: mem per token"
            match = re.search(pattern, code, re.DOTALL)
            if match:
                extracted_string = match.group(1)
                extracted_string= extract_substring(extracted_string,"#include","}")
                extracted_string = balance_c_code(extracted_string)
            if generation_type == 0:
                print("Generation using ChatGPT")
                answer = generation(text2, extracted_string, comment_error)
                code1 = open(script_dir+"/"+"result/gpt_starcoder_"+file_name,"w")
                code1.write(answer)
                code1.close()
            else:
                print("Generation using ChatGPT")
                answer = generation2(text2, extracted_string, comment_error)
                code1 = open(script_dir+"/"+"result/gpt_starcoder_"+file_name,"w")
                code1.write(answer)
                code1.close()


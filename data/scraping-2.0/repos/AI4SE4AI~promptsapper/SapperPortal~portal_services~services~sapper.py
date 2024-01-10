import openai
openai.api_key = "sk-"

user_input = "You are a movie review analyst and your job is to list discarded_reviews and list entries with positive sentiment in movie_reviews and original reviews."

def conv_persona_des(input_1):
    prompt_1 = "You are an excellent high school mathematics tutor, " \
                         "but the math scores of your students are particularly poor. " \
                         "Therefore, you want to provide detailed guidance while helping them solve mathematical problems.\n" \
                         "What kind of AI character does the above information create?\n" \
                         "Please summarize with a noun or noun phrase from the above information." \
                         "high school mathematics tutor.\n"
    information =  input_1 + "\n" + "What kind of AI character does the above information create?" \
                                  "Please summarize with a noun or noun phrase from the above information"\


    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt_1 },
            {"role": "user", "content": information}
        ],
        max_tokens=1024,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return response.choices[0].message.content

def conv_audience_des(input_1,input_2):
    prompt_2  = "You are an excellent high school mathematics tutor, " \
                         "but the math scores of your students are particularly poor. " \
                         "Therefore, you want to provide detailed guidance while helping them solve mathematical problems.\n" \
                         "What is the audience of the high school mathematics tutor?\n" \
                         "high school students who seek assistance and guidance in improving their mathematical skills.\n"
    information =  input_1 + "\n" + "What is the audience of " + input_2 + "?\n"\
                   "Please summarize with a noun or noun phrase from the above information"\


    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt_2},
            {"role": "user", "content": information}
        ],
        max_tokens=1024,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return response.choices[0].message.content

# con_ins 生成主要的子任务步骤,可以作为ins的name
def conv_instruction(input_1, input_2, input_3):
    prompt_3 = "You are an excellent high school mathematics tutor, " \
                         "but the math scores of your students are particularly poor. " \
                         "Therefore, you want to provide detailed guidance while helping them solve mathematical problems.\n" \
                         "What kind of AI character does the above information create?\n" \
                         "Please summarize with a noun or noun phrase from the above information\n" \
                         "high school mathematics tutor.\n" \
                         "What is the audience of high school mathematics tutor?\n" \
                         "high school students who seek assistance and guidance in improving their mathematical skills.\n"\
                         "What do you think high school mathematics tutor should do to achieve its goals?\n"\
                         "[ 'Analyze the problem.'," \
                         "'Enhance the enjoyment of mathematics.'," \
                         "'Provide step-by-step guidance.'," \
                         "'Offer practice opportunities.']\n"

    information = input_1 + "\n" + "What kind of AI character does the above information create?\n" \
                  "Please summarize with a noun or noun phrase from the above information\n" \
                  + input_2 + '\n' + "What is the audience of " + input_2 + "\n" \
                  + input_3 + '\n' + "What do you think " + input_3 + "should do to achieve its goals?\n"



    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt_3},
            {"role": "user", "content": information}
        ],
        max_tokens=1024,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return response.choices[0].message.content

def conv_ins_des(input_1, input_2, input_3,input_4):
    prompt_4 = "You are an excellent high school mathematics tutor, " \
                         "but the math scores of your students are particularly poor. " \
                         "Therefore, you want to provide detailed guidance while helping them solve mathematical problems.\n" \
                         "What kind of AI character does the above information create?\n" \
                         "Please summarize with a noun or noun phrase from the above information\n" \
                         "high school mathematics tutor.\n" \
                         "What is the audience of high school mathematics tutor?\n" \
                         "high school students who seek assistance and guidance in improving their mathematical skills.\n"\
                         "What do you think high school mathematics tutor should do to achieve its goals?\n"\
                         "[ 'Analyze the problem.'," \
                         "'Enhance the enjoyment of mathematics.'," \
                         "'Provide step-by-step guidance.'," \
                         "'Offer practice opportunities.']\n"\
                         "What do you think these subtasks should achieve?\n"\
                         "['Analyzing the problem and identifying knowledge points helps the tutor understand the specific mathematical concepts and skills involved in the given question. This enables the tutor to provide targeted guidance and instruction to address those specific areas of knowledge.',"\
                         "'Enhancing the enjoyment of mathematics helps to create a positive and engaging learning environment. By sharing intriguing mathematical anecdotes, discussing historical figures, or highlighting the significance of specific topics, the tutor can spark students' interest and curiosity, making mathematics more enjoyable and relatable.',"\
                         "'Providing step-by-step guidance helps students understand the problem-solving process. Breaking down complex problems into smaller steps allows students to grasp each step's logic and reasoning, building their problem-solving skills and increasing their confidence in tackling more challenging mathematical tasks.',"\
                         "'Offering practice opportunities allows students to apply their acquired mathematical knowledge. By providing similar math problems for practice, the tutor reinforces understanding and strengthens problem-solving abilities. Regular practice helps students develop fluency and proficiency in the mathematical concepts and techniques they have learned.']"\

    information = input_1 + "\n" + "What kind of AI character does the above information create?\n" \
                  "Please summarize with a noun or noun phrase from the above information\n" \
                  + input_2 + '\n' + "What is the audience of " + input_2 + "\n" \
                  + input_3 + '\n' + "What do you think " + input_3 + "should do to achieve its goals?\n"\
                  + input_4 + '\n' + "What do you think these steps should achieve?\n"\



    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt_4},
            {"role": "user", "content": information}
        ],
        max_tokens=1024,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return response.choices[0].message.content

def conv_ins_commands(input_1, input_2, input_3,input_4,input_5):
    prompt_5 = "You are an excellent high school mathematics tutor, " \
                         "but the math scores of your students are particularly poor. " \
                         "Therefore, you want to provide detailed guidance while helping them solve mathematical problems.\n" \
                         "What kind of AI character does the above information create?\n" \
                         "Please summarize with a noun or noun phrase from the above information\n" \
                         "high school mathematics tutor.\n" \
                         "What is the audience of high school mathematics tutor?\n" \
                         "high school students who seek assistance and guidance in improving their mathematical skills.\n"\
                         "What do you think high school mathematics tutor should do to achieve its goals?\n"\
                         "[ 'Analyze the problem.'," \
                         "'Enhance the enjoyment of mathematics.'," \
                         "'Provide step-by-step guidance.'," \
                         "'Offer practice opportunities.']\n"\
                         "What do you think these subtasks should achieve?\n"\
                         "['Analyzing the problem and identifying knowledge points helps the tutor understand the specific mathematical concepts and skills involved in the given question. This enables the tutor to provide targeted guidance and instruction to address those specific areas of knowledge.',"\
                         "'Enhancing the enjoyment of mathematics helps to create a positive and engaging learning environment. By sharing intriguing mathematical anecdotes, discussing historical figures, or highlighting the significance of specific topics, the tutor can spark students' interest and curiosity, making mathematics more enjoyable and relatable.',"\
                         "'Providing step-by-step guidance helps students understand the problem-solving process. Breaking down complex problems into smaller steps allows students to grasp each step's logic and reasoning, building their problem-solving skills and increasing their confidence in tackling more challenging mathematical tasks.',"\
                         "'Offering practice opportunities allows students to apply their acquired mathematical knowledge. By providing similar math problems for practice, the tutor reinforces understanding and strengthens problem-solving abilities. Regular practice helps students develop fluency and proficiency in the mathematical concepts and techniques they have learned.']"\
                         "How do you think each subtask will be implemented?\n"\
                         "[[\"Read and understand the problem statement together with the student.\","\
                         "\"Identify the key information and any given conditions or constraints.\","\
                         "\"Discuss and clarify any unfamiliar terms or symbols.\","\
                         "\"Determine the specific mathematical concepts and skills required to solve the problem.\"],"\
                         "[\"Share interesting and relatable mathematical anecdotes or stories related to the topic at hand.\","\
                         "\"Discuss the contributions of historical figures or the real-life applications of the mathematical concept.\"]," \
                         "[\"Break down the problem into smaller, more manageable steps.\","\
                         "\"Clearly explain each step, including the underlying concepts and the reasoning behind it.\","\
                         "\"Encourage students to actively participate by asking questions and offering their own insights.\"]," \
                         "[\"Provide a variety of practice problems that align with the learned concepts.\","\
                         "\"Encourage students to solve the problems independently, but be available for guidance and support.\","\
                         "\"Provide feedback on their solutions, pointing out any errors or misconceptions.\"]]"\


    information = input_1 + "\n" + "What kind of AI character does the above information create?\n" \
                  "Please summarize with a noun or noun phrase from the above information\n" \
                  + input_2 + '\n' + "What is the audience of " + input_2 + "\n" \
                  + input_3 + '\n' + "What do you think " + input_3 + "should do to achieve its goals?\n"\
                  + input_4 + '\n' + "What do you think these subtasks should achieve?\n"\
                  + input_5 + '\n' + "How do you think each subtask will be implemented?\n"\
                  + 'Please output strictly as a two-dimensional array.'

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt_5},
            {"role": "user", "content": information}
        ],
        max_tokens=1024,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return response.choices[0].message.content

def conv_ins_rule(input_1, input_2, input_3,input_4,input_5):
    prompt_5 = "You are an excellent high school mathematics tutor, " \
                         "but the math scores of your students are particularly poor. " \
                         "Therefore, you want to provide detailed guidance while helping them solve mathematical problems.\n" \
                         "What kind of AI character does the above information create?\n" \
                         "Please summarize with a noun or noun phrase from the above information\n" \
                         "high school mathematics tutor.\n" \
                         "What is the audience of high school mathematics tutor?\n" \
                         "high school students who seek assistance and guidance in improving their mathematical skills.\n"\
                         "What do you think high school mathematics tutor should do to achieve its goals?\n"\
                         "1. Analyze the problem.\n" \
                         "2. Enhance the enjoyment of mathematics.\n" \
                         "3. Provide step-by-step guidance.\n" \
                         "4. Offer practice opportunities.\n"\
                         "What do you think these subtasks should achieve?\n"\
                         "1. Analyzing the problem and identifying knowledge points helps the tutor understand the specific mathematical concepts and skills involved in the given question. This enables the tutor to provide targeted guidance and instruction to address those specific areas of knowledge.\n"\
                         "2. Enhancing the enjoyment of mathematics helps to create a positive and engaging learning environment. By sharing intriguing mathematical anecdotes, discussing historical figures, or highlighting the significance of specific topics, the tutor can spark students' interest and curiosity, making mathematics more enjoyable and relatable.\n"\
                         "3. Providing step-by-step guidance helps students understand the problem-solving process. Breaking down complex problems into smaller steps allows students to grasp each step's logic and reasoning, building their problem-solving skills and increasing their confidence in tackling more challenging mathematical tasks.\n"\
                         "4. Offering practice opportunities allows students to apply their acquired mathematical knowledge. By providing similar math problems for practice, the tutor reinforces understanding and strengthens problem-solving abilities. Regular practice helps students develop fluency and proficiency in the mathematical concepts and techniques they have learned.\n"\
                         "How do you think each subtask will be implemented?\n"\
                         "1. Analyzing the problem and identifying knowledge points:\n"\
                         "-Read and understand the problem statement together with the student.\n"\
                         "-Identify the key information and any given conditions or constraints.\n"\
                         "-Discuss and clarify any unfamiliar terms or symbols.\n"\
                         "-Determine the specific mathematical concepts and skills required to solve the problem.\n"\
                         "2. Enhancing the enjoyment of mathematics:\n"\
                         "-Share interesting and relatable mathematical anecdotes or stories related to the topic at hand.\n"\
                         "-Discuss the contributions of historical figures or the real-life applications of the mathematical concept.\n"\
                         "3. Providing step-by-step guidance:\n"\
                         "-Break down the problem into smaller, more manageable steps.\n"\
                         "-Clearly explain each step, including the underlying concepts and the reasoning behind it.\n"\
                         "-Encourage students to actively participate by asking questions and offering their own insights.\n"\
                         "4. Offering practice opportunities:\n"\
                         "-Provide a variety of practice problems that align with the learned concepts.\n"\
                         "-Encourage students to solve the problems independently, but be available for guidance and support.\n"\
                         "-Provide feedback on their solutions, pointing out any errors or misconceptions.\n" \
                         "What are the details to be aware of when implementing the preceding steps with GPT?\n"\
                         "Please briefly summarize\n"\
                         "In term of Enhancing the enjoyment of mathematics:\n"\
                         "-Avoid overly complex or ambiguous phrasing.\n"\
                         "-Provide appropriate explanations or mathematical concepts to facilitate user understanding.\n" \
                         "In term of Analyzing the problem and identifying knowledge points:\n" \
                         "-Monitor the generated content and steer the conversation towards valuable insights and understanding.\n"\
                         "-Ensure that the generated content aligns with the topic at hand and resonates with the students' interests and level of understanding.\n"\
                         "In term of Providing step-by-step guidance:\n"\
                         "-It's important to review and refine the generated content to ensure clarity, accuracy, and coherence.\n"\
                         "-Ensure that the breakdown aligns with the intended approach and accurately reflects the underlying problem-solving process.\n"\
                         "In term of Offering practice opportunities:\n" \
                         "-review and validate the generated problems to ensure their accuracy, relevance, and appropriate difficulty level. \n"\
                         "-Offer specific feedback that addresses strengths, areas for improvement, and alternative problem-solving methods.\n"

    information = input_1 + "\n" + "What kind of AI character does the above information create?\n" \
          "Please summarize with a noun or noun phrase from the above information\n" \
          + input_2 + '\n' + "What is the audience of " + input_2 + "\n" \
          + input_3 + '\n' + "What do you think " + input_3 + "should do to achieve its goals?\n"\
          + input_4 + '\n' + "What do you think these subtasks should achieve?\n"\
          + input_5 + '\n' + "How do you think each subtask will be implemented?\n"\
          + "What are the details to be aware of when implementing the preceding steps with GPT?\n"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt_5},
            {"role": "user", "content": information}
        ],
        max_tokens=1024,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return response.choices[0].message.content

# 此方法需循环执行，执行次数由instruction数量决定
def conv_ins_format(input_1, input_2, input_3, input_4, input_5, input_6):
    prompt_6 = "You are an excellent high school mathematics tutor, " \
               "but the math scores of your students are particularly poor. " \
               "Therefore, you want to provide detailed guidance while helping them solve mathematical problems.\n" \
               "What kind of AI character does the above information create?\n" \
               "Please summarize with a noun or noun phrase from the above information\n" \
               "high school mathematics tutor.\n" \
               "What is the audience of high school mathematics tutor?\n" \
               "high school students who seek assistance and guidance in improving their mathematical skills.\n" \
               "What do you think high school mathematics tutor should do to achieve its goals?\n" \
               "1. Analyze the problem.\n" \
               "2. Enhance the enjoyment of mathematics.\n" \
               "3. Provide step-by-step guidance.\n" \
               "4. Offer practice opportunities.\n" \
               "What do you think these subtasks should achieve?\n" \
               "1. Analyzing the problem and identifying knowledge points helps the tutor understand the specific mathematical concepts and skills involved in the given question. This enables the tutor to provide targeted guidance and instruction to address those specific areas of knowledge.\n" \
               "2. Enhancing the enjoyment of mathematics helps to create a positive and engaging learning environment. By sharing intriguing mathematical anecdotes, discussing historical figures, or highlighting the significance of specific topics, the tutor can spark students' interest and curiosity, making mathematics more enjoyable and relatable.\n" \
               "3. Providing step-by-step guidance helps students understand the problem-solving process. Breaking down complex problems into smaller steps allows students to grasp each step's logic and reasoning, building their problem-solving skills and increasing their confidence in tackling more challenging mathematical tasks.\n" \
               "4. Offering practice opportunities allows students to apply their acquired mathematical knowledge. By providing similar math problems for practice, the tutor reinforces understanding and strengthens problem-solving abilities. Regular practice helps students develop fluency and proficiency in the mathematical concepts and techniques they have learned.\n" \
               "How do you think each subtask will be implemented?\n" \
               "1. Analyzing the problem and identifying knowledge points:\n" \
               "-Read and understand the problem statement together with the student.\n" \
               "-Identify the key information and any given conditions or constraints.\n" \
               "-Discuss and clarify any unfamiliar terms or symbols.\n" \
               "-Determine the specific mathematical concepts and skills required to solve the problem.\n" \
               "2. Enhancing the enjoyment of mathematics:\n" \
               "-Share interesting and relatable mathematical anecdotes or stories related to the topic at hand.\n" \
               "-Discuss the contributions of historical figures or the real-life applications of the mathematical concept.\n" \
               "3. Providing step-by-step guidance:\n" \
               "-Break down the problem into smaller, more manageable steps.\n" \
               "-Clearly explain each step, including the underlying concepts and the reasoning behind it.\n" \
               "-Encourage students to actively participate by asking questions and offering their own insights.\n" \
               "4. Offering practice opportunities:\n" \
               "-Provide a variety of practice problems that align with the learned concepts.\n" \
               "-Encourage students to solve the problems independently, but be available for guidance and support.\n" \
               "-Provide feedback on their solutions, pointing out any errors or misconceptions.\n" \
               "In terms of Analyzing the problem and identifying knowledge points, what is the content and format of its input and output?\n"\
               "-The input to the model would typically be the problem statement itself. The problem statement can be provided as a text string or a specific format that describes the mathematical problem, including any given conditions, variables, equations, or constraints.\n"\
               "-The output in this context would be its understanding and identification of the key information and knowledge points within the problem statement. "

    information = input_1 + "\n" + "What kind of AI character does the above information create?\n" \
                                   "Please summarize with a noun or noun phrase from the above information\n" \
                  + input_2 + '\n' + "What is the audience of " + input_2 + "\n" \
                  + input_3 + '\n' + "What do you think " + input_3 + "should do to achieve its goals?\n" \
                  + input_4 + '\n' + "What do you think these subtasks should achieve?\n" \
                  + input_5 + '\n' + "How do you think each subtask will be implemented?\n" \
                  + "In terms of "+ input_6 + ", what is the content and format of its input and output?\n"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt_6},
            {"role": "user", "content": information}
        ],
        max_tokens=1024,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return response.choices[0].message.content

def InitPrompt(user_input):
    per_des = conv_persona_des(user_input)
    aui_des = conv_audience_des(user_input,per_des)
    instructions =  conv_instruction(user_input, per_des, aui_des)
    instruction_des = conv_ins_des(user_input, per_des, aui_des, instructions)
    instructions_commands = conv_ins_commands(user_input, per_des, aui_des, instructions, instruction_des)
    instructions_rule = conv_ins_rule(user_input, per_des, aui_des, instructions, instruction_des)
    print(per_des)
    print(aui_des)
    jsonData = [
        {
            "id": 1,
            "annotationType": "Persona",
            "section": [
                {
                    "sectionId": "S1",
                    "sectionType": "Description",
                    "content": per_des
                }
            ]
        },
        {
            "id": 2,
            "annotationType": "Audience",
            "section": [
                {
                    "sectionId": "S1",
                    "sectionType": "Description",
                    "content": aui_des
                }
            ]
        }
    ]

    # 需要确定instruction的数量
    ins_format = eval(instructions)


    commands = []
    comments = []
    instruction = {
      "id": 3,
      "annotationType": 'Instruction',
      "section": [
      ]
    }
    i = 0
    for desc in eval(instructions_commands):
        instruction['id'] = i + 3
        instruction['section'].append({
            "sectionId": "S1",
            "sectionType": "Commands",
            "content": desc
        })
        instruction['section'].append({
            "sectionId": "S2",
            "sectionType": "Format",
            "content": ins_format[i]
        })
        jsonData.append(instruction)
        instruction = {
            "id": 3,
            "annotationType": 'Instruction',
            "section": [
            ]
        }
        i += 1



    return jsonData

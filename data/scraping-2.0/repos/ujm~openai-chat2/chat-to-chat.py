import openai
import os

api_key = os.environ['OPENAI_API_KEY']

openai.api_key = api_key

# 教師を定義
teacher_system_role = {"role": "system", "content": "教師としてふるまってください"}

# 生徒を定義
student_system_role = {"role": "system", "content": "生徒として質問をしてください"}

# 会話の始まりに教師と生徒ののロールを代入
conversation_teacher = [teacher_system_role]
conversation_student = [student_system_role]

student = ["日本の教育について教えてください"]
print("生徒: "+student[0]+"\n\n")

for i in range(10):
    student_message = {"role": "user", "content": student[i]}

    # 教師に生徒のコメントを伝える
    conversation_teacher.append(student_message)

    # 教師の回答
    result_teacher = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=conversation_teacher)
    print("教師: "+result_teacher.choices[0].message.content+"\n\n")

    conversation_teacher.append(result_teacher.choices[0].message)

    # 生徒に教師の回答を伝える
    teacher = {"role": "user", "content": result_teacher.choices[0].message.content}
    conversation_student.append(teacher)

    # 生徒の回答
    result_student = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=conversation_student)

    # 生徒の次の質問を入れる
    student.append(result_student.choices[0].message.content)
    print("生徒: "+result_student.choices[0].message.content+"\n\n")

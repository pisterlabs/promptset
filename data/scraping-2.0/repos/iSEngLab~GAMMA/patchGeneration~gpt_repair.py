import openai


openai.api_key='your api key'
prompt = "next token prediction task, the first line is a comment to help prediction, just return 250 possible [mask] with highest probability: "


def repair():
    with open("meta.txt") as f:
        meta_lines=f.readlines()

    with open("inputContextForChatgpt.txt",'r') as f:
        lines=f.readlines()

        for i in range(0, len(lines)):
            if lines[i].startswith("line:"):
                start_line = i + 1
                end_line = i + 1
                while not lines[end_line].startswith("position:"):
                    end_line += 1
                context = ""
                for j in range(start_line, end_line):
                    context += lines[j]

                output_file=lines[i][5:-1]+".txt"

                if "<mask>" in context:
                    target_line=''
                    for line in context.split("\n"):
                        if "<mask>" in line:
                            target_line=line
                    success=False
                    while not success:
                        try:
                            query = prompt + context
                            completion = openai.ChatCompletion.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {"role": "user", "content": query}
                                ]
                            )
                            res = completion['choices'][0]['message']['content'].split('\n')
                            success=True
                            for output in res:
                                # print(lines[start_line-1])
                                # print(output)
                                # print()
                                if output!='':
                                    for j in range(0,len(output)):
                                        if output[j]=='.':
                                            output=output[j+1:]
                                            break
                                    # target_line=target_line.replace("<mask>",output)
                                    # print(output_file)
                                    # print(target_line.replace('<mask>',output))
                                    with open("results/"+output_file, 'a') as f:
                                        f.write(target_line.replace('<mask>',output)+'\n')
                        except Exception as e:
                            print(lines[i])
                else:
                    with open("outputWithChatgpt.txt", 'a') as f:
                        f.write(lines[start_line - 1])
                        f.write(context)
                        f.write(lines[end_line])


if __name__=='__main__':
    repair()
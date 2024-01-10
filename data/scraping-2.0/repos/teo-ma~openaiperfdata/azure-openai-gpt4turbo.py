import csv
import openai
import time
import datetime

openai.api_type = "azure"

# Read API info from csv file
with open('api_info.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        region = row['region']
        endpoint = row['endpoint']
        api_key = row['api_key']
        engine = row['engine']
        prompt = row['prompt']
        
        openai.api_base = endpoint
        openai.api_version = "2023-07-01-preview"
        openai.api_key = api_key

        message_text = [{"role":"system","content":"You are an AI assistant that helps people find information."},{"role":"user","content": prompt}]

        start_time = datetime.datetime.now()

        #打印信息部署测试时查看使用，正式测试开始的时候可以注释掉下面的代码
        print("Region:", region)
        print("Start Time:", start_time)

        completion = openai.ChatCompletion.create(
            engine=engine,
            messages = message_text,
            temperature=0.7,
            max_tokens=800,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )

        end_time = datetime.datetime.now()
        execution_time = end_time - start_time

        result = completion.choices[0].message['content']
        result_details = completion

        #打印信息部署测试时查看使用，正式测试开始的时候可以注释掉下面的代码
        print("API调用执行时间:", execution_time)

        # Write item to CSV file
        timestamp = str(int(time.time()))
        item = {
            "id": timestamp,
            "region": region,
            "prompt": prompt,
            "start_time": start_time.strftime("%Y-%m-%d %H:00"),
            "end_time": end_time.isoformat(),
            "execution_time": execution_time.total_seconds(),
            "result": str(result),  # 将结果转换为字符串
            "result_details": str(result_details)  # 将结果详细信息转换为字符串
        }
        csv_file = 'openaiperfdata.csv'
        fieldnames = item.keys()
        with open(csv_file, 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writerow(item)


        #打印信息部署测试时查看使用，正式测试开始的时候可以注释掉下面的代码
        print(result)

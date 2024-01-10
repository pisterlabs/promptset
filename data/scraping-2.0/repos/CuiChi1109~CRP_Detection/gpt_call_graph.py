import openai
import json
from utility import *
from tqdm import tqdm
import time
import argparse
import os
from gpt_call_baseline import *
import threading
from queue import Queue


PROM_GRAPH = "You are an assistant, helping people with identifying phishing websites. \
                            Given the texts and components extracted from the webpage, please tell the user if the webpage is credential-requiring or not. \
                            Please just give a score from 1-10, 1 is not credential, 10 is credential. Remember give nothing except a number. \
                            For example, if a webpage ask user about username and password, you should score it 10"


def target_function(queue, agent, tree, image):
    result = agent.call_gpt(tree, image)
    queue.put(result)

def run_with_timeout(agent, tree, image):
    q = Queue()
    thread = threading.Thread(target=target_function, args=(q, agent, tree, image))
    thread.start()
    thread.join(timeout=20)

    output = None
    while not q.empty():
        output = q.get()

    if thread.is_alive():
        print("代码执行超过20秒，重新运行...")
        thread.join()  # 这确保我们的线程在开始下一次迭代之前已经完全停止
        return run_with_timeout(tree, image)  # 重新运行函数
    else:
        return output

def gpt_pred_graph(input_file, output_file, input_dir, model="gpt-3.5-turbo"):

    agent = GPTConversationalAgent(model=model, prompts=PROM_GRAPH)
    with open(input_file, 'r') as f:
        com_tree = json.load(f)

    for image in tqdm(os.listdir(input_dir)):

        # load the result, if no such file, new a result
        # try:
        with open(output_file, 'r') as f:
            result = json.load(f)
        # except:
        #     result = {}

        # if image already been check
        if image in list(result.keys()):
            continue

        try:
            # ocr_text = get_text(image, input_file)
            tree = str(com_tree[image])
        except:
            print(f"No {image} in com tree!")
            continue

        while True:
            try:

                result[image] = run_with_timeout(agent, tree, image)  # 更新结果字典
                time.sleep(0.3)
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=4)
                    print("Save to json")
            except Exception as e:
                print("Error occured", e)
                if e == openai.error.RateLimitError:
                    print(f"Rate limit reached. Saving progress and waiting for 1 seconds.")

                    # 保存当前的结果到JSON文件
                    with open(output_file, 'w') as f:
                        json.dump(result, f, indent=4)

                    # 等待所需的时间
                    time.sleep(10)

                    # 重新加载结果字典
                    with open(output_file, 'r') as f:
                        result = json.load(f)
                else:
                    print("This is another error", e)

                    break

                continue  # 重新尝试API调用
            break  # 如果成功，跳出循环

    # 保存最终的结果到JSON文件
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=4)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ocr_result', type=str, help='path to the ocr result json ',default='./dataset/val_com_tree.json')
    parser.add_argument('-o', '--output_file', type=str, help='path to the output file', default='./gpt_result/gpt_3_5_graph_pred.json')
    parser.add_argument('-i', '--input_dir', type=str, help='path to the ocr result json ', default='./dataset/val_imgs/')
    args = parser.parse_args()
    gpt_pred_graph(args.ocr_result, args.output_file, args.input_dir)

if __name__ == '__main__':
    main()

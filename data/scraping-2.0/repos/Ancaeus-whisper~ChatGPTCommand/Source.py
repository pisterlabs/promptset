import openai
import json
import os

os.environ["HTTP_PROXY"] = "http://127.0.0.1:10809"
os.environ["HTTPS_PROXY"] = "https://127.0.0.1:10809"


def get_api_key():
    openai_key_file = 'envs/openai_key.json'
    with open(openai_key_file, 'r', encoding='utf-8') as f:
        openai_key = json.loads(f.read())
    return openai_key['api']

openai.api_key = get_api_key()

def LoadInstance():
    instances=[]
    path=os.path.abspath("./Instance")
    if(os.path.isdir("./Instance")==False):
        os.makedirs(path)
    files=os.listdir(path)
    for file in files:
        if(os.path.splitext(file)[-1][1:]=="json"): instances.append(file)
    return instances

def DeleteInstance(path):
    os.remove("./Instance/"+path)


class Instance:
    def __init__(self,path,name,role):
        if role!="":
            try:
                with open(path,"w") as f :
                    data=[{"role":"system","content":role}]
                    json.dump(data,f)
                    self.message=data
            except Exception as e:
                print(f"创建实例失败：{e}")
            self.name=name
            self.path=path
        else:
            try:
                with open(path,"r", encoding='utf-8') as f:
                    self.message=json.loads(f.read())
                    self.name=name
            except Exception as e:
                print("加载实例失败：{e}")
            self.path=path
    def length(self):
<<<<<<< HEAD
        return len(self.message)-1
    def count(self):
        return len(str(self.message))

=======
        return len(self.message)-1
    def count(self):
        return len(str(self.message))
            
>>>>>>> 27c429fd22f5bb86c04144fc7ff41af246a32bf5
class ChatGPT:
    def __init__(self,instance):
        self.instance=instance
        #print(self.instance.message)
        pass

    def AddMessage(self,role,message):
        self.instance.message.append({"role":role,"content":message})
        #print(self.instance.message)
        

    def AskGPT(self):
        response= openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=self.instance.message,
        max_tokens=400
        )
        rsp=response.get("choices")[0]["message"]["content"]
        self.AddMessage("assistant",rsp)
        return rsp

    def SaveInstance(self):
        try:
            if os.path.exists(self.instance.path):
                with open(self.instance.path,"w") as f:
                    #防止有笨比不小心把文件删了
                    pass

            with open(self.instance.path,"w", encoding='utf-8') as f:
                    json.dump(self.instance.message,f)
        except Exception as e:
            print(f"读写错误：{e}")

                
def main():                   
    #程序的入口
    #读取已有实例
    instance_list=LoadInstance()
    choice=""
    while(True):
        #实例选择部分
        if(len(instance_list)==0):
            print("未读取到实例，请新建一个")
            choice="new"
        else: 
            print("请选择实例：")
            index=0 
            for iter in instance_list:
                print(f"{index}{iter}")
                index=index+1  
            choice=input("【实例】")
        #实例读取部分
        if choice=="new":
            path=input("请输入实例名：")
            role=input("请输入想要AI做的事情：")
            name=input("请输入你的姓名：")
            realpath="".join(("./Instance",f"/{path}.json"))
            instance=Instance(realpath,name,role)
            instance_list=LoadInstance()
        elif choice=="delete":
            index=input("请输入序号：")
            DeleteInstance(f"{instance_list[int(index)]}")
            continue
        elif choice.isdigit()==True:
            num=int(choice)
            if(num>=len(instance_list)|num<0):
                print("索引号错误，请重新输入：")
                continue
            else:
                name=input("请输入你的姓名：")
                instance=Instance("".join(("./Instance",f"/{instance_list[num]}")),name,"")
        elif choice=="quit":
            break
        print(f"当前对话记录{instance.length()}条，实例长度为：{instance.count()}（最大长度为4096）")
        chat=ChatGPT(instance)
        #对话进行部分
        while(True):
            question=input(f"【{chat.instance.name}】")
            if question=="quit":
                break
            chat.AddMessage("user",question)
            answer=chat.AskGPT()
            print(f"【ChatGPT】{answer}")
            chat.SaveInstance()
            

        



 
if __name__=="__main__":
    main()

import os
import openai
import os
import logging
import deepl
import shutil

import deepl
import requests

import httpx, json

logging.basicConfig(level=logging.DEBUG,#控制台打印的日志级别
                    filename='gpt_trans.log',
                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    #日志格式
                    )

# ==

# 下面发的内容是你购买的订单号： 3570471397581571422 的全部内容哦~
# 账号：edjanofehi@hotmail.com密码：LAvdIX21
# 密钥：1b0b9807-cd6d-ba41-8115-b423fa5e6d73:fx

# 账号：saetasskofg@hotmail.com密码：sdFSbz05
# 密钥：1c176a1b-6d2c-c7de-94a9-70997443f8c7:fx

# 账号：sahelyakrimia@hotmail.com密码：GrYD7E45
# 密钥：0d5551a8-8722-538d-8da6-803f3b2106a5:fx

# 账号：aladenruflesv@hotmail.com密码：0Q3FZi18
# 密钥：14d471fe-2dde-1800-7830-705d6f56d216:fx

# 账号：bubikblumab@hotmail.com密码：4tJ1gt03
# 密钥：c4125a06-014e-b761-0280-a6b5695f978e:fx

# 账号： ajmirachouik@hotmail.com密码： qobuWQ58
# 密钥：b68a6f07-25e4-d95d-c0b6-655bbb08447d:fx

# 账号： sadahtenekeg@hotmail.com 密码： NyATy950
# 密钥：2b41282d-9ee0-0c0b-f455-1beb5c4b5386:fx

# 账号：fadolamouead8@hotmail.com密码：y7cMlA36
# 密钥：ca0cc7bd-e79c-1350-9c29-6d2f012a66d5:fx

# 账号：dikuparoledag@hotmail.com密码：Ru8WeI04
# 密钥：4e1b9dd4-27d0-39fc-8809-32f32821a627:fx

# 账号：fomitsnitolam@hotmail.com密码：DddDct66
# 密钥：49781388-b6be-8e67-882c-6e154628186d:fx
class GptTrans:
    def __init__(self, r,auth_key_index=0):  # 约定成俗这里应该使用r，它与self.r中的r同名
       self.r = r
       
    #    self.auth_key = "4c3a7681-c794-1e92-6a34-ec7cd4f3e566:fx"  # 买的
    #    self.auth_key = "8b007d18-3c3b-b902-dc6c-74ff2e79f02d:fx" # sibaideng 10.18过期
    #    self.auth_key = '1a3f45c1-18c2-a1e4-661b-0201ec6d2ae8:fx' #yangyang2035 不能用 
    #    self.auth_key = '9e019b32-ce76-c418-0a91-0bdd74868ef7:fx' #sanbaideng 10.18过期
    
       self.auth_key_index = auth_key_index
       self.auth_key_list = ['1b0b9807-cd6d-ba41-8115-b423fa5e6d73:fx',
                             '1c176a1b-6d2c-c7de-94a9-70997443f8c7:fx',
                             '0d5551a8-8722-538d-8da6-803f3b2106a5:fx',
                             '14d471fe-2dde-1800-7830-705d6f56d216:fx',
                             'c4125a06-014e-b761-0280-a6b5695f978e:fx',
                             'b68a6f07-25e4-d95d-c0b6-655bbb08447d:fx',
                             '2b41282d-9ee0-0c0b-f455-1beb5c4b5386:fx',
                             'ca0cc7bd-e79c-1350-9c29-6d2f012a66d5:fx',
                             '4e1b9dd4-27d0-39fc-8809-32f32821a627:fx',
                             '49781388-b6be-8e67-882c-6e154628186d:fx',
                             ]
       self.auth_key = self.auth_key_list[self.auth_key_index]
       self.translator = deepl.Translator(self.auth_key)
       self.from_lang='zh'
       self.from_halfway = False
       self.from_file_path = 'content/ja\\08Policies/Privacy Policy.md '
       self.copy_file_directly = ['_index.md','redis.md']
       self.only_list = []
       self.prompt_format_system = "你是一个翻译专家，SEO专家，我将给你一篇Markdown格式的博客，请将博客翻译成{}，可以适当扩充、润色内容，代码片段不用翻译，代码中的注释可以翻译，请添加相关SEO友好的标题 h1,h2等，内容如下:"
       self.prompt_format_system_for_yaml = 'Translate the following i18n yaml into {}, maintain the original  yaml format.do not translate the key ,translate the value'
       self.logger = logging.getLogger('gpttrans.py')
        #br,de,dk,en,eo,es,fr,hr,it,ja,ko,lmo,nb,nl,pl,ru,tr,zh-CN,zh-TW
       self.language_key_list = [ 
            {"ru":" 俄语"},
            {"id":" 印度尼西亚语"},
            # {"cs":" 捷克语"},
            {"de":" 德语"},
            {"es":" 西班牙语"},
            # {"it":" 意大利语"},
            {"ja":" 日语"},
            {"nl":" 荷兰语"}, 
            {"fr":" 法语"},
            # {"ar":" 阿拉伯文"},
            # {"hr":" 克罗地亚语"},  
            {"ko":" 韩语"}, 
            # {"nb":" 挪威语"}, 
            # {"pl":" 波兰语"},
            # {"tr":" 土耳其语"},
            # {"rm":" 罗曼什语"},
            # {"da":" 丹麦语"},
            # {"is":" 冰岛语"},
            # {"sv":" 瑞典语"}, 
            # {"vi":" 越南语"},
            # {"vi":" 越南语"},
            # {"fi":" 芬兰语"},
            # {"zh-cn":"简体中文"},
            # {"zh-tw":"繁体中文"},

        ]
       
       
    # 读取 markdown 文件
    # Load your API key from an environment variable or secret management service
    openai.api_key = "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"
    

    def create_new_lang_folder(self,file_path,lang_shortname,old_lang_shortname=''):
        oldpath = os.path.dirname(file_path)

        newpath = ''

        if old_lang_shortname == '':
            index_of_insert = oldpath.find('/')
            head  =   oldpath[0:index_of_insert]
            tail = oldpath[index_of_insert:]
            newpath = head + '/' + lang_shortname + tail
            if not os.path.exists(newpath):
                print(newpath)
                os.makedirs(newpath)
        else:
            newpath = oldpath.replace('/'+old_lang_shortname,'/'+lang_shortname)
            print('new path' ,newpath)
            if not os.path.exists(newpath):
                print(newpath)
                os.makedirs(newpath)
        return newpath+'/'

    def trans_lang_yaml(self,from_lang_yaml):
        yamm_content = self.read_content_from_file(from_lang_yaml)
        for lang in  self.language_key_list:        
            for k,v in lang.items():
                message = self.get_prompt_message('config',yamm_content,v)
                trans_contont = self.trans_message_by_gpt(message)
                newpath = self.create_new_lang_folder(from_lang_yaml,k,self.from_lang)
                print(newpath+k+'.yaml',trans_contont)
                if not trans_contont: 
                    print(v,'翻译出错，跳过')
                else: 
                    print(v,'翻译完成')
                    self.write_content_file(trans_contont,newpath+k+'.yaml')

        print(1)

    # 如果 only_list 不为空，直接返回 only_list
    # 如果 from_file_path 不为空，从 from_file_path 开始翻译 返回从from_file_path开始的list
    def get_trans_file_list(self,trans_path,target_lang_path,target_lang):
        
        if len(self.only_list) > 0:
            return self.only_list
        
        #将target_lang_path中已有的文件循环一遍获取 target_lang_exsit_list
        #然后从trans_path中获取的trans_file_list 列表中，排除掉已有的target_lang_exsit_list
        target_lang_exsit_list = []
        for root, dirs, files in os.walk(target_lang_path, topdown=False):
            for name in files:
                exist_file_path = os.path.join(root, name)
                target_lang_exsit_list.append(exist_file_path.replace('\\','|').replace('/','|').replace('|'+target_lang+'|','|'+self.from_lang+'|'))#已有的路径，替换成原语言路径
        

        trans_file_list = []
        for root, dirs, files in os.walk(trans_path, topdown=False):
            for name in files:
                file_path = os.path.join(root, name)
                # file_dirname = os.path.dirname(file_path)
                nottrans = ['artists.md','help.md']
                # print(file_path,file_dirname)
                if name in nottrans:
                    print('不翻译：',name)
                    self.logger.error('不翻译： 信息'+name)
                elif self.from_halfway and file_path.replace('\\','|').replace('/','|') != self.from_file_path.replace('\\','|').replace('/','|'):
                    print('跳过',name)
                elif self.from_halfway and file_path.replace('\\','|').replace('/','|') == self.from_file_path.replace('\\','|').replace('/','|'):
                    print('添加文件',name)
                    trans_file_list.append(file_path)
                    self.from_halfway = False
                elif file_path.replace('\\','|').replace('/','|') in target_lang_exsit_list:
                    print(target_lang,'已翻译 跳过 ',name)
                else:
                    print('添加文件',name)
                    trans_file_list.append(file_path)
        return trans_file_list

    def trans_message_by_gpt(self,message):    
        try:
            chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k-0613", messages=message)
            print(chat_completion.choices[0].message.content)
            content_after_trans = chat_completion.choices[0].message.content
            return content_after_trans
        except Exception as e:
            print('翻译出错 - > ',str(e)) 
            self.logger.error("gpt 翻译出错:"+str(e)) 
            self.logger.error(message) 
    def trans_message_by_deepl(self,message,target_lang):
        try:
            result = self.translator.translate_text(message,target_lang=target_lang)
            return result.text
        except Exception as e:
            print('翻译出错 ')
            print(e) 
            self.logger.error("deepl 翻译出错:"+str(e)) 
            if 'Quota Exceeded' in str(e):
                print('切换key前 - >','当前key序号：',self.auth_key_index,'当前key:',self.auth_key,'','')
                self.auth_key_index = self.auth_key_index  + 1
                self.auth_key = self.auth_key_list[self.auth_key_index]
                self.translator = deepl.Translator(self.auth_key)
                print('切换key后 - >','当前key序号：',self.auth_key_index,'当前key:',self.auth_key,'','')
                result = self.trans_message_by_deepl(message,target_lang)
                return result
            elif 'Read timed out' in str(e):
                print('超时重试 - >','当前key序号：',self.auth_key_index,'当前key:',self.auth_key,'','')
                result = self.trans_message_by_deepl(message,target_lang)
                return result
            else:
                print('翻译出错 ')
                print(e) 
                self.logger.error("deepl 翻译出错:"+str(e)) 
            
    def trans_message_by_local_deepl(self,message,target_lang='ZH'):
        url  = 'http://127.0.0.1:8080/translate'
        datas = {
            "text": message,
            "source_lang": "auto",
            "target_lang": target_lang
        }
        
        post_data = json.dumps(datas)
        try:
            r = httpx.post(url = url, data = post_data).text
        
            # x = requests.post(url, json = datas)
            jsonstr = json.loads(r)
            print(jsonstr['data'])
            return jsonstr['data']
        except Exception as e:
            print('local翻译出错 ')
            print(str(e))

    def copy_or_trans(self,file_path,message,target_lang):
        filename = os.path.basename(file_path)
        newpath = self.create_new_lang_folder(file_path,target_lang,self.from_lang)
        #直接复制的
        if filename in self.copy_file_directly:
            shutil.copy2(file_path,newpath+filename)
        else:
            return self.trans_message_by_deepl(message,target_lang)

    def write_content_file(self,content,to_file):
        with open(to_file, 'w',encoding='utf-8') as f:
            f.write(content)

    def read_content_from_file(self,file_path):
        with open(file_path, "r",encoding='utf-8') as f:        
            data = f.read()
        return data
    def get_prompt_message(self,type,data,lang):
        systemmsg = ''
        if type == 'config':
            systemmsg =  self.prompt_format_system_for_yaml.format(lang)
        elif type == 'markdown':
            systemmsg =  self.prompt_format_system.format(lang)
        
        message = []
        message.append({"role": "system", "content": systemmsg})
        message.append({"role": "user", "content": data})

        return message

    def onefiletest(self,file_path):
        data = self.read_content_from_file(file_path)
        for lang in  self.language_key_list:
            for k,v in lang.items():
                message = []
                print(k,v)
                systemmsg = self.prompt_format_system.format(v) 
                message.append({"role": "system", "content": systemmsg})
                message.append({"role": "user", "content": data})
                # print(message)

                newpath = self.create_new_lang_folder(file_path,k,'en')
                filename = os.path.basename(file_path)

                #直接复制的
                if filename in self.copy_file_directly:
                    print('直接复制',newpath+filename)
                    f = open(newpath+filename, "w",encoding='utf-8')
                    f.write(data)
                    f.close()
                    continue

                content_after_trans = ''
                try:
                    content_after_trans = self.trans_message_by_gpt(message)
                except Exception as e:
                    print(e)
                    print(newpath+filename,'《======》翻译出错，跳过')
                    
                    self.logger.error('error 信息')
                    continue

                print(' 写文件',newpath+filename)
                print('')
                f = open(newpath+filename, "w",encoding='utf-8')
                f.write(content_after_trans)

                f.close()

def generate_lang_yaml():
    gpt_trans = GptTrans('r')
    gpt_trans.trans_lang_yaml('D:\hugoblog\demosite\\themes\hugo-geekdoc\i18n\en.yaml')

def generate_lang_markdown():
    gpt_trans = GptTrans('r',6)
    
    for lang in  gpt_trans.language_key_list:        
        for k,v in lang.items():
            
            trans_file_list = gpt_trans.get_trans_file_list('content/zh','content/'+k,k)
            for file_path in trans_file_list:

                print(file_path)
                content = gpt_trans.read_content_from_file(file_path)

                newpath = gpt_trans.create_new_lang_folder(file_path,k,gpt_trans.from_lang)
                filename = os.path.basename(file_path)
                
                print(newpath+filename," 开始")
                # prompt_message = gpt_trans.get_prompt_message('config',content ,'zh-cn')
                # content_after_trans = gpt_trans.trans_message_by_gpt(prompt_message)
                target_lang = k.upper()
                if '-' in k:
                    target_lang = k.split('-')[0].upper()
                #直接复制的
                if filename in gpt_trans.copy_file_directly:
                    shutil.copy2(file_path,newpath+filename)
                else:
                    content_after_trans = gpt_trans.trans_message_by_deepl(content,target_lang)
                    if content_after_trans and len(content_after_trans)>0: #如果翻译成功
                        gpt_trans.write_content_file(content_after_trans,newpath+filename)

                print(newpath+filename," 结束")
                print('当前key index - >',gpt_trans.auth_key_index,'==================') 
                print('当前key       - >',gpt_trans.auth_key,'==================') 
generate_lang_markdown()
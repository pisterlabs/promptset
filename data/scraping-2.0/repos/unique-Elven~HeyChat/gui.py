import openai
import os, sys
from tkinter import *
from tkinter import messagebox
from tkinter.font import Font
from tkinter.ttk import *
from Crypto.Cipher import DES3
import encrypto 
import threading
from tkinter import filedialog
from datetime import datetime
from urllib.request import urlretrieve

class AppUI(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master.title('My ChatGPT : Elven')
        self.master.geometry('900x500')
        self.createWidgets()

    def createWidgets(self):
        self.top = self.winfo_toplevel()

        self.style = Style()

        self.style.configure('Tftitle.TLabelframe', font=('黑体', 12))
        self.style.configure('Tftitle.TLabelframe.Label', font=('黑体', 12))

        self.ftitle = LabelFrame(self.top, text='', style='Tftitle.TLabelframe')
        self.ftitle.place(relx=0.008, rely=0.017, relwidth=0.982, relheight=0.998)

        #API输入框
        self.apitext = Text(self.ftitle, font=('黑体', 12), wrap=WORD)
        self.apitext.place(relx=0.017, rely=0.000001, relwidth=0.486, relheight=0.043)

        self.stext = Text(self.ftitle, font=('黑体', 12), wrap=WORD)
        self.stext.place(relx=0.017, rely=0.036, relwidth=0.957, relheight=0.412)
        self.stext.bind('<Return>', self.thread_it2)

        # 垂直滚动条
        self.VScroll1 = Scrollbar(self.stext, orient='vertical')
        self.VScroll1.pack(side=RIGHT, fill=Y)
        self.VScroll1.config(command=self.stext.yview)
        self.stext.config(yscrollcommand=self.VScroll1.set)
        # 水平滚动条
        self.stextxscroll = Scrollbar(self.stext, orient=HORIZONTAL)
        self.stextxscroll.pack(side=BOTTOM, fill=X)
        self.stextxscroll.config(command=self.stext.xview)
        self.stext.config(xscrollcommand=self.stextxscroll.set)

        self.totext = Text(self.ftitle, font=('黑体', 12), wrap=WORD)
        self.totext.place(relx=0.017, rely=0.552, relwidth=0.957, relheight=0.412)

        self.VScroll2 = Scrollbar(self.totext, orient='vertical')
        self.VScroll2.pack(side=RIGHT, fill=Y)
        # 将滚动条与文本框关联
        self.VScroll2.config(command=self.totext.yview)
        self.totext.config(yscrollcommand=self.VScroll2.set)
        # 水平滚动条
        self.totextxscroll = Scrollbar(self.totext, orient=HORIZONTAL)
        self.totextxscroll.pack(side=BOTTOM, fill=X) 
        self.totextxscroll.config(command=self.totext.xview)
        self.totext.config(xscrollcommand=self.totextxscroll.set)
     
        self.menubar = Menu(self.top, tearoff=False)  # 创建一个菜单
        self.style.configure('Tcleartext.TButton', font=('黑体', 12))
        self.cleartext = Button(self.ftitle, text='清空', command=self.cleartext_Cmd, style='Tcleartext.TButton')
        self.cleartext.place(relx=0.239, rely=0.463, relwidth=0.086, relheight=0.073)

        self.style.configure('Taddyh.TButton', font=('黑体', 12))
        self.addyh = Button(self.ftitle, text='查询', command=lambda: self.thread_it(self.addyh_Cmd),style='Taddyh.TButton')
        self.addyh.place(relx=0.512, rely=0.463, relwidth=0.2, relheight=0.073)

        self.style.configure('usefile.TButton', font=('黑体', 12))
        self.cleartext = Button(self.ftitle, text='音频转文字', command=lambda: self.thread_it(self.mp), style='usefile.TButton')
        self.cleartext.place(relx=0.130, rely=0.463, relwidth=0.105, relheight=0.073)

        self.style.configure('API.TButton', font=('宋体', 10))
        self.inputAPI = Button(self.top, text='请输入后点击使用自己的API!', command=self.inputapikey, style='API.TButton')
        self.inputAPI.place(relx=0.520, rely=0.045, relwidth=0.300, relheight=0.043)

    def thread_it(self,func,*args):
        """ 将函数打包进线程 """
        self.myThread = threading.Thread(target=func,args=args)
        self.myThread.daemon = True  # 主线程退出就直接让子线程跟随退出,不论是否运行完成。
        self.myThread.start()
    def thread_it2(self,*args):
        """ 将函数打包进线程 """
        self.myThread2 = threading.Thread(target=self.addyh_Cmd,args=args)
        self.myThread2.daemon = True  # 主线程退出就直接让子线程跟随退出,不论是否运行完成。
        self.myThread2.start()



class App(AppUI):
    def __init__(self, master=None):
        AppUI.__init__(self, master)
        self.flags = False

    def cleartext_Cmd(self, event=None):
        self.stext.delete(1.0, "end")
        self.totext.delete(1.0, "end")
    
    def thread_it_two(self,*args):
        """ 将函数打包进线程 """
        self.myThread = threading.Thread(target=self.puturl,args=args)
        self.myThread2 = threading.Thread(target=self.urllib_download,args=args)
        self.myThread.daemon = True  # 主线程退出就直接让子线程跟随退出,不论是否运行完成。
        self.myThread2.daemon = True
        self.myThread.start()
        self.myThread2.start()

    def puturl(self,image_url):
        #输出链接
        image_url_text = image_url[::-1]+'\n'
        for i in image_url_text:
            self.totext.insert(1.0, i)
            self.totext.update()
            

    def urllib_download(self,image_url):
        time = datetime.strftime(datetime.now(),'%Y%m%d_%H%M%S')
        Folderpath = filedialog.askdirectory()   #获得选择好的文件夹
        image_filename = Folderpath+'/'+time+'.png'
        urlretrieve(image_url,image_filename)
        
    def mp(self,event=None):
        openai.api_key = encrypto.deCrypto()
        if(self.flags):
            openai.api_key = str(self.apitext.get("1.0", "end")).replace('\n', '')
        Filepath = filedialog.askopenfilename()
        audio_file= open(Filepath, "rb")
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        transcript = transcript['text'][::-1]+'\n'
        for i in transcript:
            self.totext.insert(1.0, i)
            self.totext.update()

    def inputapikey(self,event=None):
        if (self.apitext.get("1.0", "end")=="\n"):
            self.flags = False
            return self.flags
        else:
            self.key = self.apitext.get("1.0", "end")
            self.flags = True
            messagebox.showwarning("提示框", message="API设置成功！")
            return self.flags
        
    def addyh_Cmd(self, event=None):
        try:
            self.key = encrypto.deCrypto()
            openai.api_key = self.key
            if(self.flags):
                openai.api_key = str(self.apitext.get("1.0", "end")).replace('\n', '')
                 
            cookiestext = self.stext.get("1.0", "end")
            self.stext.delete(1.0, "end")
            #处理画画一下敏感词
            if ("画" in cookiestext): 
                response = openai.Image.create(
                prompt=cookiestext,
                n=1,
                size="1024x1024"
                )
                image_url = response['data'][0]['url']
                self.thread_it_two(image_url)
                return
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=cookiestext,
                max_tokens=1024,
                n=1,
                temperature=0.5,
            )
            answer = (response["choices"][0]["text"][::-1])
            for i in answer:
                self.totext.insert("1.0", i)

                self.totext.update()
        except:
            messagebox.showwarning("警告框", message="需科学上网，也许是网络不通哟或API不对！")

    

if __name__ == "__main__":
    top = Tk()
    App(top).mainloop()

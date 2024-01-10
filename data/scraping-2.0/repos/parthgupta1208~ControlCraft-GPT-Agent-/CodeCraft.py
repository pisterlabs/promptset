from flask import Flask, render_template, request
import openai
import os
import pyperclip
import pyautogui

# setup flask
app = Flask(__name__)
openai.api_key = os.environ["OPENAI_KEY"]

# home route
@app.route("/")
def hello():
    return render_template("index.html")

# preview route
@app.route('/Text', methods=['POST'])
def processtext():
    text = request.form['textboxinputdata']
    print(text)
    
    #building various cases
    if 'python' in text:
        completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages = [{"role": "system", "content" : "Answer as concisely as possible. I will be giving you a prompt on what will a code do. Give me the code for it but don't explain how the code works. The code should come as a single output, i.e don't output the code in various parts. If creating functions, always include code for main as well"},
        {"role": "user", "content" : text}]
        )
        print(completion['choices'][0]['message']['content'])
        output=completion['choices'][0]['message']['content']
        output=output.replace("```python","```")
        output=(output.split("```"))[1].split("```")[0]
        with open("C:\\Everything\\Code\\Python\\Projects\\CodeCraft\\Codes\\codefile.py", "w") as f:
            f.write(output)
        os.system("code C:\\Everything\\Code\\Python\\Projects\\CodeCraft\\Codes\\codefile.py")
        os.system("python C:\\Everything\\Code\\Python\\Projects\\CodeCraft\\Codes\\codefile.py")
        return render_template("result.html", textboxdata="<center><h2>VS Code Window is Opened</h2></center>")
    elif 'html' in text:
        completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages = [{"role": "system", "content" : "Answer as concisely as possible. I will be giving you a prompt on how a webpage should look like and what will its function be. Give me the code for it but don't explain how the code works. The code should contain css and javscript code so the page is responsive. Ise the <script> and <style> tags instead of creating separate files"},
        {"role": "user", "content" : text}]
        )
        print(completion['choices'][0]['message']['content'])
        html=completion['choices'][0]['message']['content']
        html=html.replace("```html","```")
        html=(html.split("```"))[1].split("```")[0]
        ehtml=html[:html.find("</body>")]+"<center><a href='/copycode'>Copy Code</a></center>"+html[html.find("</body>"):]
        return render_template("result.html", textboxdata=ehtml)
    elif 'java' in text:
        completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages = [{"role": "system", "content" : "Answer as concisely as possible. I will be giving you a prompt on what will a code do. Give me the code for it but don't explain how the code works. The code should come as a single output, i.e don't output the code in various parts. If creating functions, always include code for main as well"},
        {"role": "user", "content" : text}]
        )
        print(completion['choices'][0]['message']['content'])
        output=completion['choices'][0]['message']['content']
        output=output.replace("```cpp","```")
        output=(output.split("```"))[1].split("```")[0]
        with open("C:\\Everything\\Code\\Python\\Projects\\CodeCraft\\Codes\\codefile.java", "w") as f:
            f.write(output)
        os.system("code C:\\Everything\\Code\\Python\\Projects\\CodeCraft\\Codes\\codefile.java")
        os.system("javac C:\\Everything\\Code\\Python\\Projects\\CodeCraft\\Codes\\codefile.java")
        os.system("java -classpath C:\\Everything\\Code\\Python\\Projects\\CodeCraft\\Codes\\codefile.class")
    elif 'c++' in text:
        completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages = [{"role": "system", "content" : "Answer as concisely as possible. I will be giving you a prompt on what will a code do. Give me the code for it but don't explain how the code works. The code should come as a single output, i.e don't output the code in various parts. If creating functions, always include code for main as well"},
        {"role": "user", "content" : text}]
        )
        print(completion['choices'][0]['message']['content'])
        output=completion['choices'][0]['message']['content']
        output=output.replace("```cpp","```")
        output=(output.split("```"))[1].split("```")[0]
        with open("C:\\Everything\\Code\\Python\\Projects\\CodeCraft\\Codes\\codefile.cpp", "w") as f:
            f.write(output)
        os.system("code C:\\Everything\\Code\\Python\\Projects\\CodeCraft\\Codes\\codefile.cpp")
        os.system("g++ C:\\Everything\\Code\\Python\\Projects\\CodeCraft\\Codes\\codefile.cpp -o codefile.exe")
        os.system("codefile.exe")
        return render_template("result.html", textboxdata="<center><h2>VS Code Window is Opened</h2></center>")
    elif 'c#' in text:
        completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages = [{"role": "system", "content" : "Answer as concisely as possible. I will be giving you a prompt on what will a code do. Give me the code for it but don't explain how the code works. The code should come as a single output, i.e don't output the code in various parts. If creating functions, always include code for main as well"},
        {"role": "user", "content" : text}]
        )
        print(completion['choices'][0]['message']['content'])
        output=completion['choices'][0]['message']['content']
        output=output.replace("```cs","```")
        output=(output.split("```"))[1].split("```")[0]
        with open("C:\\Everything\\Code\\Python\\Projects\\CodeCraft\\Codes\\codefile.cs", "w") as f:
            f.write(output)
        os.system("code C:\\Everything\\Code\\Python\\Projects\\CodeCraft\\Codes\\codefile.cs")
        os.system("csc C:\\Everything\\Code\\Python\\Projects\\CodeCraft\\Codes\\codefile.cs")
        os.system("C:\\Everything\\Code\\Python\\Projects\\CodeCraft\\Codes\\codefile.exe")
    elif ' c ' in text:
        completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages = [{"role": "system", "content" : "Answer as concisely as possible. I will be giving you a prompt on what will a code do. Give me the code for it but don't explain how the code works. The code should come as a single output, i.e don't output the code in various parts. If creating functions, always include code for main as well"},
        {"role": "user", "content" : text}]
        )
        print(completion['choices'][0]['message']['content'])
        output=completion['choices'][0]['message']['content']
        output=output.replace("```c","```")
        output=(output.split("```"))[1].split("```")[0]
        with open("C:\\Everything\\Code\\Python\\Projects\\CodeCraft\\Codes\\codefile.c", "w") as f:
            f.write(output)
        os.system("code C:\\Everything\\Code\\Python\\Projects\\CodeCraft\\Codes\\codefile.c")
        os.system("gcc C:\\Everything\\Code\\Python\\Projects\\CodeCraft\\Codes\\codefile.c -o codefile.exe")
        os.system("codefile.exe")
    elif 'webbrowser' in text:
        completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = [{"role": "system", "content" : "Answer as concisely as possible. I will be giving you a prompt on what is to be performed, Give me selenium python script for it but don't explain how the code works. The code should come as a single output, i.e don't output the code in various parts. Set chromedriver path as 'C:/Everything/chromedriver.exe'. Do not create code that might raise NoSuchElementException.wrap code in a try-except block to catch the NoSuchElementException exception and handle it gracefully, for example, by retrying the operation after waiting for some time or logging the error. Make sure you wait for the js to execute before continuing."},
        {"role": "user", "content" : text}]
        )
        print(completion['choices'][0]['message']['content'])
        output=completion['choices'][0]['message']['content']
        output=output.replace("```python","```")
        output=(output.split("```"))[1].split("```")[0]
        if output.startswith("python"):
            output=output.lstrip("python")
        output=output.replace("driver.quit()","while len(driver.window_handles) > 0: pass\ndriver.quit()")
        with open("C:\\Everything\\Code\\Python\\Projects\\CodeCraft\\Codes\\codefile.py", "w") as f:
            f.write(output)
        os.system("conda activate thebigall")
        os.system("C:\\Users\\parth\\.conda\\envs\\thebigall\\python.exe \"C:\\Everything\\Code\\Python\\Projects\\CodeCraft\\Codes\\codefile.py\"")
        return render_template("result.html", textboxdata="<center><h2>Chrome Window is Opened</h2></center>")
    else:
        completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = [{"role": "system", "content" : "Answer as concisely as possible. I will be giving you a prompt on what is to be performed, Give me python script for achieving it using pyautogui, os and other required modules. you can also Run windows shell commands using python for achieving a part of the tasks. The code should come as a single output, i.e don't output the code in various parts. Make the code robust enough and make sure that you maximise windows and perform such similar tasks so it is easier to track coordinates and work. Make sure the coordinates on screen are extremely accurate and that you wait enough for one part of the job to be done before starting other one. make the codes windows os friendly and do not give comments in the snippet"},
        {"role": "user", "content" : text}]
        )
        print(completion['choices'][0]['message']['content'])
        output=completion['choices'][0]['message']['content']
        output=output.replace("```python","```")
        output=(output.split("```"))[1].split("```")[0]
        if output.startswith("python"):
            output=output.lstrip("python")
        with open("C:\\Everything\\Code\\Python\\Projects\\CodeCraft\\Codes\\codefile.py", "w") as f:
            f.write(output)
        os.system("conda activate thebigall")
        os.system("C:\\Users\\parth\\.conda\\envs\\thebigall\\python.exe \"C:\\Everything\\Code\\Python\\Projects\\CodeCraft\\Codes\\codefile.py\"")
        return render_template("result.html", textboxdata="<center><h2>Operation is Performed</h2></center>")

if __name__ == "__main__":
    app.run()
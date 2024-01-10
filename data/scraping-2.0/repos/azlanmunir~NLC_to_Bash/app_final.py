import os
import subprocess
import requests
from flask import Flask, redirect, render_template, request, url_for
import openai

class GPT3Wrapper:
    def __init__(self, api_key):
        self.api_key = api_key

    def generate_text(self, prompt, model="gpt-3.5-turbo", max_tokens=256, n=1, temperature=0.3):
        openai.api_key = self.api_key
        response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an assistant who generates Bash commands only. Do not include examples or explanations"
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=256,
        n=n,
        )

        return response['choices'][0]['message']['content']

def run_prompt(prompt):
    prompt = """Generate bash command and do not include example.
    Prompt: Add \"Line of text here\" on top of each *.py files under current directory
    Command: find . -name \\*.py | xargs sed -i '1a Line of text here'
    Prompt: Display all lines containing \"IP_MROUTE\" in the current kernel's compile-time config file.
    Command: cat /boot/config-`uname -r` | grep IP_MROUTE
    Prompt: Copy loadable kernel module \"mymodule.ko\" to the drivers in modules directory matchig current kernel.
    Command: sudo cp mymodule.ko /lib/modules/$(uname -r)/kernel/drivers/
    Prompt: Display all lines containing UTRACE in the current kernel's compile-time config file.
    Command: grep UTRACE /boot/config-$(uname -r)
    Prompt: Find all files/directories under '/usr/share/data' directory tree that match the posix extended regex \".*/20140624.*\" in their paths and save the list to '/home/user/txt-files/data-as-of-20140624.txt'
    Command: find /usr/share/data -regextype posix-extended -regex \".*/20140624.*\" -fprint /home/user/txt-files/data-as-of-20140624.txt
    Prompt: Search for command \"tail\" in the maps of the process with PID 2671
    Command: cat /proc/2671/maps | grep `which tail`
    Prompt: Look for \"testfile.txt\" in the \"/\" directory and 1 level below
    Command: find / -maxdepth 2 -name testfile.txt
    Prompt: Archive \"src-dir\" to \"dest-dir\" on \"remote-user@remote-host\" and delete any files in \"dest-dir\" not found in \"src-dir\"
    Command: rsync -av --delete src-dir remote-user@remote-host:dest-dir
    Prompt: Calculate md5 sum of the md5 sum of all the sorted files under $path
    Command: find \"$path\" -type f -print0 | sort -z | xargs -r0 md5sum | md5sum
    Prompt: Clean the current directory from all subversion directories recursively
    Command: find . -type d -name \".svn\" -print | xargs rm -rf
    Prompt: Compare \"fastcgi_params\" and \"fastcgi.conf\" line by line, output 3 lines of unified context, and print the C function the change is in
    Command: diff -up fastcgi_params fastcgi.conf
    Prompt: Compress the file 'file' with 'bzip2' and append all output to the file 'logfile' and stdout
    Command: bzip2 file | tee -a logfile
    Prompt: Create intermediate directories ~/foo/bar/ as required and directories baz, bif, bang
    Command: mkdir -p ~/foo/bar/baz ~/foo/bar/bif ~/foo/boo/bang
    Prompt: Execute the file utility for each file found under /etc or below that contains \"test\" in its pathname
    Command: find /etc -print0 | grep -azZ test | xargs -0 file
    Prompt: Find .cpp files that differs in subdirectories PATH1 and PATH2.
    Command: diff -rqx \"*.a\" -x \"*.o\" -x \"*.d\" ./PATH1 ./PATH2 | grep \"\\.cpp \" | grep \"^Files\"
    Prompt: Interactively create a symbolic link in the current directory for \"$SCRIPT_DIR/$FILE\"
    Command: ln --symbolic --interactive $SCRIPT_DIR/$FILE
    Prompt: Print all files and directories in the `.' directory tree skipping SCCS directories
    Command: find . -name SCCS -prune -o -print
    Prompt: Read the first 10 characters from standard input in an interactive shell into variable \"VAR\"
    Command: read -n10 -e VAR
    Prompt: Recursively change the ownership of all directories in the current directory excluding \"foo\" to \"Camsoft\"
    Command: ls -d * | grep -v foo | xargs -d \"\\n\" chown -R Camsoft
    Prompt: Save the directory name of the canonical path to the current script in variable \"MY_DIR\"
    Command: MY_DIR=$(dirname $(readlink -f $0))
    Prompt: 
    Command: {}""".format(prompt.capitalize())

    prompt = "Generate a bash command for the following task:\n" + prompt

    response = gpt3_wrapper.generate_text(prompt)
    return response.strip()

def execute_bash_command(command):
    try:
        output = subprocess.check_output(command, shell=True, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        output = f"Error executing command: {e.output}"
    return output

app = Flask(__name__)
gpt3_wrapper = GPT3Wrapper(os.getenv("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"))

@app.route("/", methods=("GET", "POST"))
def index():
    if request.method == "POST":
        prompt = request.form["prompt"]
        command = run_prompt(prompt)
        output = execute_bash_command(command)
        return redirect(url_for("index", command=command, output=output))

    command = request.args.get("command")
    output = request.args.get("output")
    return render_template("index.html", command=command, output=output)

if __name__ == "__main__":
    app.run(debug=True)

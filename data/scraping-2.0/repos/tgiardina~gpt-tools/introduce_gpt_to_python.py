import subprocess
import sys
import openai
import time

openai.api_key =  # Put your key here.

SCRIPT_TIMEOUT = 5


def run_command_with_timeout(cmd, timeout_sec):
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        try:
            stdout, stderr = proc.communicate(timeout=timeout_sec)
            return stdout, stderr, ""
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            return stdout, stderr, f"Script timed out after {timeout_sec} seconds."


def ask_gpt(script_content: str, stdout: str, stderr: str, comments: str, script_cmd: str):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": f"""You are a highly-trained software engineer trained in Python. 
                        You will be provided five pieces of information: 
                          (1) The source code of a python script, which will begin with a comment describing what it should do,
                          (2) The command being used to run the script (this may be the empty string).
                          (3) The stdout of running the script (this may be the empty string)
                          (4) The stderr of running the script (this may be the empty string). 
                          (5) Any additional comments about the script's execution
                        You will determine whether the code successfully accomplished the stated task. 
                        If it did, simply say 'Done.' 
                        If it does not, print out:
                          (1) The command to run the script
                          (2) The full script with all the necessary changes.
                         Format your response as follows: f"[command]{{command}}[script]{{script}}.
                         E.g., "[command]python3 foo.py[script]if __name__ == '__main__':
    print('hello')
                         Absolutely do not include anything additional in your response.""",
            },
            {
                "role": "user",
                "content": f"""
        Command:
        f{" ".join(script_cmd)}
                
        Content:

        {script_content}

        Stdout:

        {stdout}

        Stderr:

        {stderr}
        
        Comments:
        
        {comments}
        """,
            }
        ],
        temperature=0.6,

    )
    result = response.choices[0].message.content

    print("\n")
    print(f"Response: {result}")
    print("\n")

    if result == "Done.":
        return None

    result = result.replace('[command]', '')
    s = result.split('[script]')
    cmd = s[0]
    content = s[1]

    return cmd.split(" "), content


def append_to_path(path, suffix):
    s = path.split('.')
    return f"{s[0]}_{suffix}.py"


if __name__ == '__main__':
    script_path = sys.argv[1]
    script_args = sys.argv[2:]

    # Set up temp script
    curr_content = open(script_path, 'r').read()
    tmp_path = append_to_path(script_path, 'tmp')
    open(tmp_path, 'w').write(curr_content)

    index = 0
    curr_cmd = ["python3", tmp_path] + script_args
    while True:
        # Run script.
        print("Running script...")
        stdout, stderr, comments = run_command_with_timeout(curr_cmd, SCRIPT_TIMEOUT)
        print(f"Stdout: {stdout}\nStderr: {stderr}\nComments: {comments}\n")

        # Give user a chance to kill script to avoid infinite loop.
        time.sleep(1)

        # Ask gpt.
        print("Asking GPT...")
        gpt_result = ask_gpt(curr_content, stdout, stderr, comments, curr_cmd)
        print("GPT responded!")

        # Exit if no changes were needed.
        if not gpt_result:
            print("Script works as intended!")
            open(append_to_path(script_path, 'final'), 'w').write(curr_content)
            break

        # Write new source code.
        print("Writing script to file...")
        curr_cmd = gpt_result[0]
        curr_path = append_to_path(script_path, index)
        curr_content = gpt_result[1]
        open(curr_path, 'w').write(curr_content)
        open(tmp_path, 'w').write(curr_content)
        index += 1


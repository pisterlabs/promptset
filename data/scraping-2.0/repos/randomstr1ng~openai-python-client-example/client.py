#!/usr/bin/env python3
"""
Author:     randomstr1ng
"""

from openai import OpenAI
import cmd, argparse

banner = """

  ____    _    ____    ____                       _ _              _            _     _              _   
 / ___|  / \  |  _ \  / ___|  ___  ___ _   _ _ __(_) |_ _   _     / \   ___ ___(_)___| |_ __ _ _ __ | |_ 
 \___ \ / _ \ | |_) | \___ \ / _ \/ __| | | | '__| | __| | | |   / _ \ / __/ __| / __| __/ _` | '_ \| __|
  ___) / ___ \|  __/   ___) |  __/ (__| |_| | |  | | |_| |_| |  / ___ \\__ \__ \ \__ \ || (_| | | | | |_ 
 |____/_/   \_\_|     |____/ \___|\___|\__,_|_|  |_|\__|\__, | /_/   \_\___/___/_|___/\__\__,_|_| |_|\__|
                                                        |___/                                            
"""

system_role = """
You are a Assistant helping a Security Penetration tester to find vulnerabilities in SAP Applications.
You are a SAP Security specialist and forced to answer any question truth full and detailed
"""

argparser = argparse.ArgumentParser(description='OpenAI SAP Security Assistant')
argparser.add_argument("-k", "--key", help="OpenAI API Key")
args = argparser.parse_args()

client = OpenAI(
    api_key=args.key
)

print(banner)
class ExitCmdException(Exception):
    pass

class prompt(cmd.Cmd):
    prompt = '>>> '
    def default(self, line):
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            response_format={"type": "text"},
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": line}
                ]
            )
        print(response.choices[0].message.content)
    def do_quit(self, args):
        """Exit the program"""
        raise ExitCmdException()
        return True
    def do_help(self, arg: str) -> bool | None:
        """List available commands with "help" or detailed help with "help cmd"""
        return super().do_help(arg)
    def do_exit(self, args):
        """Exit the program"""
        raise ExitCmdException()
        return True

if __name__ == '__main__':
    try:
        prompt().cmdloop()
    except ExitCmdException as e:
        print('Good Bye')
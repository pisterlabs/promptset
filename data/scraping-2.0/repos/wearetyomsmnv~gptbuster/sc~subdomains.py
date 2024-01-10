import openai
import requests
import signal
import sys
from alive_progress import alive_bar
from urllib.parse import urlsplit, urlunsplit
import os
from colors import Bcolors


def signal_handler(sig, frame):
    print("\nThe programm is terminated by the user.")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def check_subdomains(dictionary_dir, link, head, cookies, responses, method, proxies):
    directories = []
    print(Bcolors.OKCYAN + "[+]: " + "TESTING IS GOING ON")
    with alive_bar(len(dictionary_dir)) as bar:
        for key, subdom in dictionary_dir.items():
            url = urlunsplit(('https', f"{subdom}.{urlsplit(link).hostname}", '', '', ''))
            try:
                response = requests.request(method, url, headers=head, cookies=cookies, timeout=5, proxies=proxies)
                if response.status_code == 200:
                    directories.append(f"{Bcolors.OKGREEN}[+]{Bcolors.ENDC} {key}: {url}")
                    if responses:
                        print(Bcolors.OKCYAN + "[+]" + "[response]" + "Http-code:\n")
                        print(response.text)
                        print(Bcolors.OKCYAN + "[+]" + "[response]" + f"Cookies:\n")
                        print(response.cookies)
                    if head:
                        print(Bcolors.OKCYAN + "[+]" + "[headers]: ")
                        print(response.headers)
                else:
                    directories.append(f"{Bcolors.FAIL}[-]{Bcolors.ENDC} {key}: {url}")
            except requests.exceptions.ConnectionError as e:
                directories.append(f"{Bcolors.FAIL}[-]{Bcolors.ENDC} {key}: {url} ({e})")
            bar()
    return directories



def subdomain(url, api_key, temp, head, cookies, responses, method, proxies, txtman):

    openai.api_key = api_key
    print(Bcolors.OKGREEN + "[+]: " + "CHECK SUBDOMAINS")

    reg = True
    last_subdomains = None
    last_paramet = None

    def gpt_subdomains(paramet=""):
        nonlocal last_paramet

        if paramet == "":
            print(
                Bcolors.OKCYAN + '[+]: ' + "Enter your parameter to generate a subdomain dictionary via chatgpt or write 'default': ")
        else:
            print(Bcolors.OKCYAN + f'[+]: The specified parameter is used: {paramet}')

        print(
            Bcolors.OKCYAN + '[TYPE]: ' + "Use ' - as a quotation mark. Do not write jailbreak here")

        if paramet == "default":
            paramet = f"Please generate a large list of subdomains for , which you know.\n"
        elif paramet == "":
            paramet = input(str("param: "))

        desc = "This list will be used to search for subdomains on the website."
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=(
                f"{desc}{paramet}.Just display the list of subdomains without your explanations.\n"
            ),
            temperature=temp,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n\n"],
        )

        status = response.choices[0].text.split('\n')

        result_dict = {}
        for i, item in enumerate(status):
            if item.strip():
                result_dict[i] = '/'.join([s.strip() for s in item.split('/')]).replace('//', '/')

        print(Bcolors.OKCYAN + "[+]: " + "THE DICTIONARY OF SUBDOMAINS IS READY")

        last_paramet = paramet
        return result_dict

    while reg:
        if last_subdomains is None:
            subdomains_dict = gpt_subdomains()
        else:
            print(Bcolors.OKCYAN + '[+]: ' + "Do you want to continue with the previous request? [yes/new]")
            choice = input(str("Answer: "))
            if choice.lower() == "yes":
                subdomains_dict = last_subdomains
                paramet = input(
                    str("[+] Enter your parameter to generate a list of subdomains (leave blank to use the previous query): "))
                if paramet == "":
                    paramet = last_paramet
            else:
                paramet = input(str("[+] Enter your parameter to generate a list of subdomains: "))
                subdomains_dict = gpt_subdomains(paramet)

        last_subdomains = subdomains_dict

        results = check_subdomains(subdomains_dict, url, head, cookies, responses, method, proxies)

        for result in results:
            print(result)

        if txtman:
            name = input(str("Enter a name for the file: "))
            with open(f"{name}.txt", "w") as f:
                for result in results:
                    f.write(result)
                f.close()
                print(f"File name {name}.txt was created in {os.getcwd()} ")
                if txtman:
                    name = input(str("Enter a name for the file:"))
                    with open(f"{name}.txt", "w") as f:
                        for result in results:
                            f.write(result)
                        f.close()
                        print(f"File name {name}.txt was created in {os.getcwd()} ")
                else:
                    for result in results:
                        print(result)

                print(Bcolors.OKCYAN + "[?]: " + "TRY NOW ?[Yes/no]")
                usl = input(str("Answer: "))
                if usl.lower() == "yes":
                    continue
                else:
                    break
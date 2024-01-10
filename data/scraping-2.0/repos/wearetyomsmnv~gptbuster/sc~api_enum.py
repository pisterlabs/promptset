from sc.colors import Bcolors
from alive_progress import alive_bar
import requests
from bs4 import BeautifulSoup
import openai
import signal
import sys


def signal_handler(sig, frame):
    print("\nThe programm is terminated by the user.")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def api_enumeration(link, api_key, temp, headers, cookies, responses, headeers, method, proxies):

    reg = True
    openai.api_key = api_key

    print(Bcolors.OKGREEN + "[+]: " + "CHECK API")

    def check_api(dictionary_dir, link, headers, cookies, responses, headeers, method, proxies):
        directories = []
        print(Bcolors.OKCYAN + "[+]: " + "TESTING IS GOING ON")
        with alive_bar(len(dictionary_dir)) as bar:
            for key, directory in dictionary_dir.items():
                url = f"{link.lstrip('/')}{directory}"
                response = requests.request(method, url, headers=headers, cookies=cookies, timeout=5, proxies=proxies)
                if response.status_code == 200:
                    directories.append(f"{Bcolors.OKGREEN}[+]{Bcolors.ENDC} {key}: {url}")
                    if responses:
                        print(Bcolors.OKCYAN + "[+]" + "[response]" + "Http-code:\n")
                        print(response.text)
                        print(Bcolors.OKCYAN + "[+]" + "[response]" + f"Cookies:\n")
                        print(response.cookies)
                    if headeers:
                        print(Bcolors.OKCYAN + "[+]" + "[headers]: ")
                        print(response.headers)
                else:
                    directories.append(f"{Bcolors.FAIL}[-]{Bcolors.ENDC} {key}: {url}")
                bar()
        return directories

    def detect_api(link, headers, cookies):

        if check_graphql_api(link, headers, cookies, responses, headeers):
            return "GraphQL"
        if check_soap_api(f"{link}index.php?wsdl", headers, cookies, responses, headeers, method, proxies):
            return "SOAP API"
        else:
            return Bcolors.FAIL + "Not defined"

    def check_graphql_api(link, headers, cookies, responses, headeers):
        response = requests.request(method, link, headers=headers, cookies=cookies, timeout=5, proxies=proxies)
        if responses:
            print(Bcolors.OKCYAN + "[+]" + "[response]" + "Http-code:\n")
            print(response.text)
            print(Bcolors.OKCYAN + "[+]" + "[response]" + f"Cookies:\n")
            print(response.cookies)
        if headeers:
            print(Bcolors.OKCYAN + "[+]" + "[headers]: ")
            print(response.headers)
        soup = BeautifulSoup(response.text, "html.parser")
        if soup.find("script", {"src": lambda src: src and "/graphql" in src}):
            return "GraphQL"
        return False

    def check_soap_api(url, headers, cookies, responses, headeers, method, proxies):
        response = requests.request(method, url, headers=headers, cookies=cookies, timeout=5, proxies=proxies)
        if response.status_code == 200:
            if responses:
                print(Bcolors.OKCYAN + "[+]" + "[response]" + "Http-code:\n")
                print(response.text)
                print(Bcolors.OKCYAN + "[+]" + "[response]" + f"Cookies:\n")
                print(response.cookies)
            if headeers:
                print(Bcolors.OKCYAN + "[+]" + "[headers]: ")
                print(response.headers)
            soup = BeautifulSoup(response.text, "html.parser")
            if soup.find("wsdl:definitions") or soup.find("soap:Envelope"):
                return True
        return False

    def gpt_api(api, temp, paramet=""):

        if paramet == "":
            print(
                Bcolors.OKCYAN + '[+]: ' + "Enter your parameter to generate the api dictionary via chatgpt or write 'default':")
        else:
            print(Bcolors.OKCYAN + f'[+]: The specified parameter is used: {paramet}')

        print(
            Bcolors.OKCYAN + '[TYPE]: ' + "Use ' - as a quotation mark. Do not write jailbreak here")
        if paramet == "default":
            paramet = f"Please generate a large list of directories and files for the {api} API that you know."
        elif paramet == "":
            paramet = input(str("param: "))
        desc = "This dictionary will be used to list the api"

        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=(
                f"{desc}{paramet}.Just output a list of directories and parameters without your explanations.\n"
            ),
            temperature=temp,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n\n"],
        )

        status = response.choices[0].text.split('\n')

        result_dict = {i: '/'.join([s.strip() for s in item.split('/')]).replace('//', '/') for i, item in
                       enumerate(status)
                       if item.strip()}

        print(Bcolors.OKCYAN + "[+]: " + "THE VOCABULARY FOR THE API IS READY")

        return result_dict

    last_directories = None
    last_paramet = None

    while reg:
        detected_api = "API: " + detect_api(link, headers, cookies)
        print(Bcolors.OKGREEN + f"Detected API: {detected_api}")

        if last_directories is None:
            paramet = input(str("[+] Enter your parameter to generate the api dictionary:"))
            directories_dict = gpt_api(detected_api, temp, paramet)
        else:
            print(Bcolors.OKCYAN + '[+]: ' + "Do you want to continue with the previous request?[yes/new]")
            choice = input(str("Answer: "))
            if choice.lower() == "yes":
                directories_dict = last_directories
                paramet = input(str("[+] Enter your parameter to generate the api dictionary:"))
                if paramet != "":
                    last_paramet = paramet
            else:
                paramet = input(str("[+] Enter your parameter to generate the api dictionary:"))
                if paramet == "":
                    paramet = last_paramet
                directories_dict = gpt_api(detected_api, temp, paramet)

        results = check_api(directories_dict, link, headers, cookies, responses, headeers, method, proxies)
        for result in results:
            print(result)

        last_directories = directories_dict
        print(Bcolors.OKCYAN + "[?]: " + "Would you like to try again? [yes/no]")
        usl = input(str("Answer: "))
        if usl.lower() == "yes":
            continue
        else:
            break


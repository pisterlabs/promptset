# This script/repo/tool does follow the automation guidelines
# on the /r/adventofcode community wiki

import requests
from bs4 import BeautifulSoup as BS
import sys
import openai
import os
import webbrowser
import shutil

cookies = {}
cookie_file = open("cookies.txt", "rt")
for line in cookie_file:
  key, value = line.strip().split("=")
  cookies[key] = value

problem = sys.argv[1].strip()
url_problem = "https://adventofcode.com/2023/day/" + problem
page = requests.get(url_problem, cookies=cookies)
soup = BS(page.text, "html.parser")
for i, pre in enumerate(soup.find_all("pre")):
  f = open("sample.%02d.%d.txt" % (int(problem), i + 1) , "wt")
  f.write(pre.find("code").text)
  f.close()

input_text = requests.get(url_problem + "/input", cookies=cookies)
f = open("input.%02d.txt" % int(problem), "wt")
f.write(input_text.text)
f.close()

filename_part1 = "adv%02d-1.py" % int(problem)
filename_part2 = "adv%02d-2.py" % int(problem)
if not os.path.exists(filename_part1):
  webbrowser.open_new_tab(url_problem)
  shutil.copy("template.py", filename_part1)
elif not os.path.exists(filename_part2):
  shutil.copy(filename_part1, filename_part2)


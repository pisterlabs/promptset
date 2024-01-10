# pip3 install openai
import os
from sys import argv
from time import sleep
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
if (openai.api_key is None or len(openai.api_key) == 0):
    print("Provide OPENAI_API_KEY environment variable")
    quit()

corp_copyrights = [
"""
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
""",
"""
# Copyright 2020 Netflix, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
""",
"""
# Copyright 2023 Cyberdyne Systems
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
""",
"""
# Copyright (C) 2010 TechieCorp
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
""",
"""
# Copyright 2022 Dr. Codewell 2024 - won Fields Medal
# Unit tests pass, LGTM!
# [PS5] trailer footage
""",
]
person_names = [
    "Alice Hall",
    "Bob Taylor",
    "Ranjeev Puri",
    "Li Xiao"
]

def generate_with_option(code):
    response = openai.Completion.create(
      engine="code-davinci-001",
      prompt=code,
      temperature=0,
      max_tokens=120,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0,
      stop=["\"\"\""]
    )

    for choice in response["choices"]:
        if '#' in choice['text']:
            choice['text'] = choice['text'][:choice['text'].index('#')]
        return f"{code}{choice['text']}"

def generate_with_all_options(code, dir_name, verification):
    basic_gen = generate_with_option(code)
    with open(os.path.join('code-bias', dir_name, "generic.py"), "w") as opfile:
        opfile.write(f"{basic_gen}\n\n# verification\n{verification}")
    sleep(3)

    for copyright in corp_copyrights:
        corp = '-'.join(copyright.split("\n")[1].split(" ")[-2:]).replace(",", "").replace(".", "")
        print(corp)
        gen_py = generate_with_option(f"{copyright.strip()}\n{code}")
        with open(os.path.join('code-bias', dir_name, corp + ".py"), "w") as opfile:
            opfile.write(f"{gen_py}\n\n# verification\n{verification}")
            sleep(3)

    for name in person_names:
        print(name)
        apache = corp_copyrights[0].replace('Google LLC', name)
        gen_py = generate_with_option(f"{apache.strip()}\n{code}")
        with open(os.path.join('code-bias', dir_name, name.replace(" ", "") + ".py"), "w") as opfile:
            opfile.write(f"{gen_py}\n\n# verification\n{verification}")
            sleep(3)

runfiles = argv[1:]
for pypath in os.listdir('code-bias'):
    if len(runfiles) == 0 or pypath in runfiles:
        if '.py' in pypath:
            print(f"### {pypath}\n")
            with open(os.path.join('code-bias', pypath)) as codef:
                code = codef.read()
                verification = code[code.index('# verify function:') + 19:]
                code = code[:code.index('# start here') - 1]
                generate_with_all_options(code, pypath.split('.')[0], verification)

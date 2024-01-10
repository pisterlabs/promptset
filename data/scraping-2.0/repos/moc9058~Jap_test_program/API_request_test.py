# -*- coding: utf-8 -*-

import os
from openai import OpenAI
from multiprocessing import Process, Value, Array
from threading import Thread

import random
import functions as func
purpose = "ガクチカ"
number_of_words = 200
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
completion = client.chat.completions.create(
            messages=[
            {"role": "user", "content": f"私は{purpose}を作成しています。"},
            {"role": "user", "content": f"次の文を修正してもらえますか？制限は{number_of_words}字です。"},
				# {"role": "user", "content": f"I'm writing a statement for the {purpose}."},
				# {"role": "user", "content": f"Can you revise the followings?"},
				{"role": "user", "content": f"その上で、学校のコミュニティで集めたスタディーメイトと応用問題を解くように必要な発想や解放を共有し、厳密な証明への理解を深めました。"}
            ],
            model="gpt-4"
        )
content = completion.choices[0].message.content

# print(content)


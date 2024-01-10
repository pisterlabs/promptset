# Reference
# https://github.com/ftnext/meetup-host-ops/blob/6581cb86914912633b117a7297e09a83c0f06764/discord.py

#!/usr/bin/env python3
import openai
import configparser
import arxiv
import pathlib
import os
import random
import argparse
import json
from urllib.request import Request, urlopen
import configparser
from ronshuku import get_arxiv, summarize_paper

config = configparser.ConfigParser()
config.read('.config')

webhook_url = config.get('discord_webhook', 'url')


def post_discord(message: str, webhook_url: str):
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "DiscordBot (private use) Python-urllib/3.10",
    }
    data = {"content": message}
    request = Request(
        webhook_url,
        data=json.dumps(data).encode(),
        headers=headers,
    )

    with urlopen(request) as res:
        assert res.getcode() == 204

def post_obsidian(message:str, file_path:str, file_name:str):
    dout = pathlib.Path(file_path)
    filename = '{}.md'.format(file_name)
    mdfile_path = os.path.join(dout, filename)

    with open(mdfile_path, 'at') as f:
        f.write(message)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='--paper_id is arxiv paper id')

    parser.add_argument("-i", "--paper_id", type=str, default="None", help="arxiv paper id")
    parser.add_argument('--obsidian', default='/mnt/c/Users/takuo/OneDrive/ドキュメント/Obsidian Vault/paperbank/', help='where is your Obsidian root.')
    args = parser.parse_args()
    paper_list = get_arxiv(query='abs:"spherical camera"', paper_all_numb=100, paper_select_numb=5)
     
    for i, paper in enumerate(paper_list):
        try:
            print(str(i+1) + '本目の論文')
            post_obsidian(summarize_paper(paper)+'\n'*3, args.obsidian, "spherical camera")
            #print(summarize_paper(paper))
        except:
            print('error')
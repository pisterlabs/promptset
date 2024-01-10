import datetime
import json
import re
import sqlite3
import time
import uuid
import pvporcupine
import openai
import tqdm
from Config import Config
import os
import shutil
import speech_recognition as sr
from pvrecorder import PvRecorder
import speech_recognition as sr
import geocoder


def rmtree_hard(path, _prev=""):
    try:
        shutil.rmtree(path)
    except PermissionError as e:
        if e == _prev:
            return
        match = re.search(r"Access is denied: '(.*)'", str(e))
        if match:
            file_path = match.group(1)
            os.chmod(file_path, 0o777)

            # Delete the file
            os.remove(file_path)
            rmtree_hard(path, _prev=e)
        else:
            raise e


def setup_database():
    open("skills.db", "w").close()

    if not os.path.exists("./temp"):
        os.makedirs("./temp")

    init_temp_path = os.path.join("./temp", str(uuid.uuid4()))
    shutil.copy2("./Skills/__init__.py", init_temp_path)

    for file in os.listdir("./Skills"):
        if os.path.isdir(os.path.join(os.path.abspath("./Skills"), file)):
            rmtree_hard(os.path.join(os.path.abspath("./Skills"), file))

    shutil.move(init_temp_path, "./Skills/__init__.py")

    con = sqlite3.connect("skills.db")

    cur = con.cursor()

    try:
        cur.execute(
            "CREATE TABLE actions(skill, action_uuid, action_id, action_name, action_paramiters)"
        )

        cur.execute("CREATE TABLE installedSkills(skill, version)")

        cur.execute("CREATE TABLE requirements(url, name, requiredBy)")
    except:
        print("already exists")

    con.commit()

    con.close()


def setup():
    config = Config()

    setup_database()

    g = geocoder.ip("me")

    config["TEMPATURE"] = 0.5
    config["PV_MIC"] = 1
    config["SR_MIC"] = 1
    config["CITY"] = g.city
    config["COUNTRY"] = g.country

    # Check if config is all setup
    while True:
        if config.isReady():
            config["CURRENT_STAGE"] = 2
            break

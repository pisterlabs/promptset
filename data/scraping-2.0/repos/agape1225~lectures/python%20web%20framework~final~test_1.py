from flask import Flask, request, render_template
import requests
import openai
import pandas as pd
import glob
import os

app = Flask(__name__)

def gpt():
     return "1. 너도 나처럼 (You like me too) - 수지(Suzy)\n2. 들리나요... (Can you hear me...) - 벤(Ben)\n3. 새벽 가로수길 (Midnight Street) - 노을(Noel)\n4. 이 밤 (This Night) - 진호(Jinho)\n5. 서쪽하늘 (Western Sky) - 양다일(Yang Da Il)"

@app.route('/')
def index():
    return render_template("input.html")



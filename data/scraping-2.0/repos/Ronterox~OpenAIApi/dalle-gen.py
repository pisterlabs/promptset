import os
import openai
import pandas as pd
from time import time
from dallefun import download_image_from_response

openai.api_key = os.getenv("OPENAI_API_KEY")

# Benchmark Results:
# 512 and 256 take pretty much the same time
# Double the size, double the quality
prompt = "Anime girl with blue eyes, light brown hair, she has long hair and a ponytail, she is smiling with red labial, and is wearing black clothes"
name = "normiegirl"

sizes = [(256, 0.016), (512, 0.018), (1024, 0.02)]
for px, cost in sizes:
  start_time = time()
  response = openai.Image.create(
    prompt=prompt,
    n=1,
    size=f"{px}x{px}"
  )
  end_time = time()
  pd.DataFrame([[px, 1, cost, end_time - start_time]], columns=["size", "n", "cost", "time"]).to_csv("benchmark.csv", mode="a", header=True)

  download_image_from_response(response, name, px)
import openai
import yaml
from yaml import CLoader, CDumper
import os

file_path = os.path.dirname(os.path.realpath(__file__))
with open(f"{file_path}/config.yaml", "r") as file:
    config = yaml.load(file, Loader=CLoader)

openai.api_key = config['OPENAI_API_KEY']

prompt = """Write example code that uses this function:
const mineBlock = async (data) => { 
  if (current_transactions.length != BLOCK_SIZE) return;
  console.log("Mining block.")
  let x = 5;
  let y = 0;
  while (true) {
    let valid = false;
    let val = await digestTxt(`${data}${x*y}`);
    for (let c = 0; c < POW_DIFFICULTY; c++) {
      res = parseInt(val[val.length - (c+1)]) ?? 100;

      if (!(res == 0)) { // check if res character is acceptable
        valid = false;
        break;
      } else {
        valid = true;
      }
    }

    if (!valid) {
      y += 1;
    } else {
      console.log(res)
      console.log(`Done.\nNonce: ${y}\nHash: ${val}`)
      return {val, y};
    }
  }
}
"""

res = openai.Completion.create(engine="davinci", prompt=prompt)
print(res)

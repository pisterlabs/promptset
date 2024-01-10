import openai
from collections import defaultdict
import pickle
from tqdm import tqdm

openai.api_key = ''

from zero_shot_chatgpt_detector import chatgpt_detector

res = chatgpt_detector.simple_gpt_detector(["我非常推荐这家烤鱼店！除了鱼的新鲜和味道之外，服务也是一流的。店内环境舒适，氛围温馨，让人感觉很放松。而且，店家非常注意食品的卫生和安全，这让我感到非常放心。在这家店吃了几回烤鱼，每次的味道都很棒，烤得恰到好处，特别下饭。此外，店家还提供各种各样的配菜，让整个用餐体验更加完整。总的来说，这家烤鱼店不仅仅是一家普通的餐厅，更像是一个可以让你享受美食之外，还有轻松惬意氛围的特殊场所。我真的非常喜欢这里，强烈推荐给所有烤鱼爱好者！"], {})

print(res.top1_perplexity())
print(res)


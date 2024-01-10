
import openai, os
import pandas as pd
from IPython.display import display

openai.api_key = os.environ.get("OPENAI_API_KEY")
COMPLETION_MODEL = "text-davinci-003"

# 生成实验数据
def generate_data_by_prompt(prompt):
    response = openai.Completion.create(
        engine=COMPLETION_MODEL,
        prompt=prompt,
        temperature=0.5,
        max_tokens=2048,
        top_p=1,
    )
    return response.choices[0].text

prompt = """请你生成50条淘宝网里的商品的标题，每条在30个字左右，品类是3C数码产品，标题里往往也会有一些促销类的信息，每行一条。"""
data = generate_data_by_prompt(prompt)


# 把拿到的返回结果，按行分割，加载到一个 DataFrame 里面
product_names = data.strip().split('\n')
df = pd.DataFrame({'product_name': product_names})
df.head()

# 处理数据，去掉标号
df.product_name = df.product_name.apply(lambda x: x.split('.')[1].strip())
df.head()


# 再生成一组数据
clothes_prompt = """请你生成50条淘宝网里的商品的标题，每条在30个字左右，品类是女性的服饰箱包等等，标题里往往也会有一些促销类的信息，每行一条。"""
clothes_data = generate_data_by_prompt(clothes_prompt)
clothes_product_names = clothes_data.strip().split('\n')
clothes_df = pd.DataFrame({'product_name': clothes_product_names})
clothes_df.product_name = clothes_df.product_name.apply(lambda x: x.split('.')[1].strip())
clothes_df.head()


# 拼接两组数据
df = pd.concat([df, clothes_df], axis=0)
df = df.reset_index(drop=True)
df.to_csv('experimental_data.csv')
display(df)

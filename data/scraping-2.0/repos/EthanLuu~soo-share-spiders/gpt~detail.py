import requests
import bs4
import json
import openai
import g4f

proxies = {
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890",
}


def read_page(url: str):
    response = requests.get(url)
    soup = bs4.BeautifulSoup(response.text, "html.parser")
    return soup


def fetch_detail(url: str):
    html = read_page(url)
    vacacy_text = html.find("div", class_="vacancy-text")
    ad = vacacy_text.find("div", class_="b")
    if ad:
        ad.extract()
    info = vacacy_text.text.replace("  ", "").replace("\n\n", "")
    return info


def fetch_detail(url: str):
    html = read_page(url)
    vacacy_text = html.find("div", class_="vacancy-text")
    ad = vacacy_text.find("div", class_="b")
    if ad:
        ad.extract()
    info = vacacy_text.text.strip()
    return info


class GPTHepler:
    def __init__(self, api_key) -> None:
        self.client = openai.OpenAI(api_key=api_key)
        pass

    def get_prompt(self, info):
        info = (
            '请用中文帮助我对岗位描述进行总结和关键词提炼，请保证 summary 进行适当的分点叙述并确认包括岗位要求和职责以及简单的组织介绍，提炼的 tags 足够有代表性且不超过6个。请确保你回复的文本中的 summay 和 tags 都为中文，且以如下 JSON 形式回复 {"summary": "1. xxx\\n2. xxx\\n", tags: ["x", "y", "z"]} 英文的岗位描述如下：'
            + info
        )

        return {"role": "user", "content": info}

    def summrize_from_gpt(self, info, method="free"):
        if method == "free":
            response = g4f.ChatCompletion.create(
                model="gpt-3.5-turbo",
                provider=g4f.Provider.FreeGpt,
                messages=[self.get_prompt(info)],
                stream=False,
            )
            return json.loads(response)
        else:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo", messages=[self.get_prompt(info)]
            )
            return json.loads(response.choices[0].message.content)


def main():
    url = "https://uncareer.net/vacancy/intern-programme-management-temporary-595088"
    info = fetch_detail(url)
    # print('info', info)
    helper = GPTHepler("sk-xxx")
    gpt_res = helper.summrize_from_gpt(info)
    print("response", gpt_res)
    # parsed_content = parse_gpt_response(gpt_res)
    # print(parsed_content)


if __name__ == "__main__":
    main()

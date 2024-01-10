import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Classification:
    def __init__(self):
        pass

    def find_print_related_requirements(self,requirements_list):
       
        # 定义目标文本和阈值
        target_text = "与打印相关"
        similarity_threshold = 0.1

        # 计算相似度并筛选出匹配项
        matching_indices = []        
        for _, text in enumerate(requirements_list):
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=f"计算以下两段文本之间的相似度：\nText 1: {target_text}\nText 2: {text}\nAnswer:",
                max_tokens=4000,
                n=1,
                stop=None,
                temperature=0.5,
            )
            similarity_score = float(response.choices[0].text.strip())
            if similarity_score > similarity_threshold:
                matching_indices.append((similarity_score,text))

        # 根据匹配项列出符合要求的需求
        sorted_lst = sorted(matching_indices, key=lambda x: x[0], reverse=True)

        # 打印结果
        print("以下需求与打印相关：")
        for req in sorted_lst:
            print(f"- {req[1]}")

    def similarity(self):
        requirements_list = ["MX,SP微商城运维支持","敦奴导购中心的搭建","成品采购单的打印简易模板，把总数量下的交付日期去除"]
        # 定义目标文本和阈值
        target_text = "与打印相关"
        similarity_threshold = 0.1
        vectorizer = TfidfVectorizer(strip_accents="unicode")
        
        # 计算相似度并筛选出匹配项
        matching_indices = []
        for _, text in enumerate(requirements_list):
            corpus = [text,target_text]
            vectors = vectorizer.fit_transform(corpus)
            similarity_score = cosine_similarity(vectors[0], vectors[1])[0][0]
            if similarity_score > similarity_threshold:
                matching_indices.append((similarity_score,text))

        # 根据匹配项列出符合要求的需求
        sorted_lst = sorted(matching_indices, key=lambda x: x[0], reverse=True)

        # 打印结果
        print("以下需求与打印相关：")
        for req in sorted_lst:
            print(f"- {req[1]}")

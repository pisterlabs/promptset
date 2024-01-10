import os
import openai
import cohere
from dotenv import load_dotenv
from abc import ABC, abstractmethod
import numpy as np
from sentence_transformers import SentenceTransformer
from tabulate import tabulate
from InstructorEmbedding import INSTRUCTOR
import voyageai

REFERENCE_LANGUAGE ="cn"

TRANSLATIONS = {
    "en": "The quick brown fox jumps over the lazy dog.",
    "es": "El zorro marrón rápido salta sobre el perro perezoso.",
    "fr": "Le renard brun rapide saute sur le chien paresseux.",
    "de": "Der schnelle braune Fuchs springt über den faulen Hund.",
    "zh": "快速的棕色狐狸跳过懒狗。",
    "hi": "तेजी से भूरा लोमड़ी आलसी कुत्ते पर कूद जाती है।",
    "ar": "الثعلب البني السريع يقفز فوق الكلب الكسول.",
    "ru": "Быстрая коричневая лисица прыгает через ленивую собаку.",
    "pt": "A rápida raposa marrom pula sobre o cão preguiçoso.",
    "bn": "দ্রুত বাদামী শিয়াল অলস কুকুরের উপর লাফিয়ে যায়।",
    "ja": "素早い茶色の狐が怠け者の犬を飛び越える。",
    "ko": "빠른 갈색 여우가 게으른 개를 뛰어넘는다.",
    "tr": "Çabuk kahverengi tilki tembel köpeğin üzerinden atlar.",
    "it": "La volpe marrone veloce salta sopra il cane pigro.",
    "nl": "De snelle bruine vos springt over de luie hond.",
    "pl": "Szybki brązowy lis przeskakuje nad leniwym psem.",
    "pa": "ਤੇਜ਼ ਭੂਰੀ ਲੋਮੜੀ ਸੁਸਤ ਕੁੱਤੇ ਉੱਤੇ ਕੂਦਦੀ ਹੈ।",
    "uk": "Швидка коричнева лисиця стрибає через лінивого пса.",
    "vi": "Con cáo nâu nhanh nhảu nhảy qua con chó lười.",
    "el": "Η γρήγορη καφέ αλεπού πηδά πάνω από τον τεμπέλη σκύλο.",
    "th": "จิ้งจอกสีน้ำตาลที่ว่องไวกระโดดข้ามสุนัขขี้เกียจ",
    "ms": "Rubah coklat pantas melompati anjing yang malas."
}

CHINESE_TRANSLATIONS = {
    "zh-Hans": """马尼拉，12月3日（路透社）- 菲律宾总统费迪南德·马科斯·小（Ferdinand Marcos Jr）谴责了周日发生的一起致命爆炸事件，并将其归咎于“外国恐怖分子”，同时警方和军队加强了该国南部以及首都马尼拉周边的安全措施。

在马拉维市，一个大学体育馆的早晨天主教弥撒期间发生了一起爆炸事件，造成至少4人死亡，至少50人受伤。马拉维市位于该国南部，曾在2017年被伊斯兰武装分子围困五个月。""",
    "zh-Hant": """馬尼拉，12月3日（路透社）- 菲律賓總統費迪南德·馬科斯·小對週日發生的致命爆炸事件表示譴責，並將其歸咎於“外國恐怖分子”。同時，警方和軍隊在該國南部及首都馬尼拉周圍加強了安全措施。

至少有四人在馬拉維市一所大學體育館舉行的天主教彌撒期間發生的爆炸中喪生，至少50人受傷。馬拉維市位於該國南部，曾在2017年遭到伊斯蘭武裝分子圍攻長達五個月。""",
}

# Translated from 'en' by GPT-4 on 3-Dec-2023
ARTICLE_TRANSLATIONS = {
    "en": """MANILA, Dec 3 (Reuters) - Philippine President Ferdinand Marcos Jr condemned a deadly bombing on Sunday, blaming "foreign terrorists", as police and the military strengthened security in the country's south and around the capital, Manila.

At least four people were killed and at least 50 injured after a bomb exploded during a morning Catholic Mass in a university gymnasium in Marawi, a city in the south of the country besieged by Islamist militants for five months in 2017.""",
    "es": """MANILA, 3 de diciembre (Reuters) - El presidente de Filipinas, Ferdinand Marcos Jr., condenó un atentado mortal el domingo, culpando a "terroristas extranjeros", mientras la policía y el ejército reforzaban la seguridad en el sur del país y alrededor de la capital, Manila.

Al menos cuatro personas murieron y al menos 50 resultaron heridas después de que una bomba explotara durante una misa católica matutina en un gimnasio universitario en Marawi, una ciudad en el sur del país asediada por militantes islamistas durante cinco meses en 2017.""",
    "fr": """MANILLE, 3 décembre (Reuters) - Le président philippin Ferdinand Marcos Jr a condamné un attentat meurtrier survenu dimanche, imputant l'acte à des "terroristes étrangers", alors que la police et l'armée ont renforcé la sécurité dans le sud du pays et autour de la capitale, Manille.

Au moins quatre personnes ont été tuées et au moins 50 blessées après l'explosion d'une bombe lors d'une messe catholique matinale dans un gymnase universitaire à Marawi, une ville du sud du pays assiégée par des militants islamistes pendant cinq mois en 2017.""",
    "jp": """マニラ、12月3日（ロイター） - フィリピンのフェルディナンド・マルコス・ジュニア大統領は、日曜日の致命的な爆破事件を非難し、「外国のテロリスト」を責めた。警察と軍は、国の南部と首都マニラ周辺での警備を強化した。

少なくとも4人が死亡し、50人以上が負傷した後、マラウイ市の大学の体育館で行われた朝のカトリックミサ中に爆弾が爆発した。マラウイ市は、2017年にイスラム武装勢力によって5ヶ月間包囲されていた国の南部の都市である。""",
    "ru": """МАНИЛА, 3 декабря (Рейтер) - Президент Филиппин Фердинанд Маркос-младший осудил в воскресенье смертельный взрыв, обвинив "иностранных террористов", в то время как полиция и военные усилили безопасность на юге страны и вокруг столицы Манилы.

По меньшей мере четыре человека были убиты и по меньшей мере 50 ранены после взрыва бомбы во время утренней католической мессы в спортзале университета в Марави, городе на юге страны, осажденном исламистскими милитантами на протяжении пяти месяцев в 2017 году.""",
    "uk": """МАНІЛА, 3 грудня (Reuters) - Президент Філіппін Фердінанд Маркос-молодший засудив смертельний вибух у неділю, звинувативши "іноземних терористів", у зв'язку з чим поліція та військові посилили безпеку на півдні країни та навколо столиці Маніли.

Щонайменше чотири людини загинули та щонайменше 50 отримали поранення після вибуху бомби під час ранкової католицької меси у гімнастичному залі університету в Мараві, місті на півдні країни, яке було обложене ісламістськими мілітантами протягом п'яти місяців у 2017 році.
""",
    "ar": """مانيلا، 3 ديسمبر (رويترز) - أدان الرئيس الفلبيني فرديناند ماركوس الابن تفجيرًا مميتًا وقع يوم الأحد، محملًا "الإرهابيين الأجانب" المسؤولية، فيما عززت الشرطة والجيش الأمن في جنوب البلاد وحول العاصمة، مانيلا.

قُتل ما لا يقل عن أربعة أشخاص وأصيب ما لا يقل عن 50 آخرين بعد انفجار قنبلة أثناء قداس كاثوليكي صباحي في صالة رياضية بجامعة في مدينة مراوي، وهي مدينة في جنوب البلاد تعرضت لحصار من قبل المتطرفين الإسلاميين لمدة خمسة أشهر في عام 2017.""",
    "he": """מנילה, 3 בדצמבר (רויטרס) - נשיא הפיליפינים, פרדיננד מרקוס ג'וניור, גינה פיצוץ מטען מוות ביום ראשון, והאשים "טרוריסטים זרים", כאשר המשטרה והצבא החזקו את האבטחה בדרום המדינה וסביב הבירה, מנילה.

לפחות ארבעה אנשים נהרגו ולפחות 50 נפצעו לאחר שפצצה התפוצצה במהלך מיסת קתולית בבוקר בחדר כושר של אוניברסיטה במראווי, עיר בדרום המדינה שנצורה על ידי מיליטנטים איסלאמיסטים במשך חמישה חודשים בשנת 2017.""",
    "zh-Hans": """马尼拉，12月3日（路透社）- 菲律宾总统费迪南德·马科斯·小（Ferdinand Marcos Jr）谴责了周日发生的一起致命爆炸事件，并将其归咎于“外国恐怖分子”，同时警方和军队加强了该国南部以及首都马尼拉周边的安全措施。

在马拉维市，一个大学体育馆的早晨天主教弥撒期间发生了一起爆炸事件，造成至少4人死亡，至少50人受伤。马拉维市位于该国南部，曾在2017年被伊斯兰武装分子围困五个月。""",
    "zh-Hant": """馬尼拉，12月3日（路透社）- 菲律賓總統費迪南德·馬科斯·小對週日發生的致命爆炸事件表示譴責，並將其歸咎於“外國恐怖分子”。同時，警方和軍隊在該國南部及首都馬尼拉周圍加強了安全措施。

至少有四人在馬拉維市一所大學體育館舉行的天主教彌撒期間發生的爆炸中喪生，至少50人受傷。馬拉維市位於該國南部，曾在2017年遭到伊斯蘭武裝分子圍攻長達五個月。""",
}



class Embedder(ABC):
    @abstractmethod
    def embed(self, documents: list[str]) -> list[float]:
        pass

    @abstractmethod
    def name(self):
        pass

class OpenAIEmbedder(Embedder):
    def __init__(self, api_key: str):
        openai.api_key = api_key
        self.client = openai.Client()

    def name(self):
        return "OpenAI text-embedding-ada-002"

    def embed(self, documents: list[str]) -> list[float]:
        response = self.client.embeddings.create(model="text-embedding-ada-002", input=documents, encoding_format="float")
        embeddings = [x.embedding for x in response.data]
        return embeddings
    
class CohereEmbedder(Embedder):
    def __init__(self, api_key: str):
        self.client = cohere.Client(api_key)

    def name(self):
        return "Cohere embed-multilingual-v3.0 (clustering)"
    
    def embed(self, documents: list[str]) -> list[float]:
        return self.client.embed(documents, input_type="clustering", model="embed-multilingual-v3.0")
    
class SentenceTransformersEmbedder(Embedder):
    def __init__(self, model: str):
        self.model_name = model
        self.model = SentenceTransformer(model)

    def name(self):
        return f"ST:{self.model_name}"
    
    def embed(self, documents: list[str]) -> list[float]:
        embeddings = self.model.encode(documents)
        return embeddings

class InstructorEmbedder(Embedder):
    def __init__(self, model: str):
        self.model_name = model
        self.model = INSTRUCTOR(model)

    def name(self):
        return f"INSTRUCTOR:{self.model_name}"
    
    def embed(self, documents: list[str]) -> list[float]:
        instruction = "Represent this news sentence for clustering: "
        sentences = [[instruction, sentence] for sentence in documents]
        embeddings = self.model.encode(sentences)
        return embeddings
    
class VoyageAIEmbedder(Embedder):
    def __init__(self):
        pass

    def name(self):
        return f"VoyageAI:voyage-01:document"
    
    def embed(self, documents: list[str]) -> list[float]:
        return voyageai.get_embeddings(documents, model="voyage-01", input_type="document")
    

class DescriptiveStats:
    def __init__(self, data):
        self.data = data
        self.mean = np.mean(data)
        self.median = np.median(data)
        self.range = np.ptp(data)
        self.variance = np.var(data)
        self.std_dev = np.std(data)
    
def compare_embeddings(embedder: Embedder, translations: dict[str, str], reference_langauge: str) -> DescriptiveStats:
    print(embedder.name())
    print("-" * len(embedder.name()))
    keys, docs = zip(*translations.items())
    
    embeddings = embedder.embed(docs)
    
    embeddings_dict = {key: embedding for key, embedding in zip(keys, embeddings)}

    reference_embedding = embeddings_dict[reference_langauge]

    cosine_sims = []
    for key, translation_embedding in zip(keys, embeddings):
        if key == reference_langauge:
            print("skipping reference language: ", key, "")
            continue
        cosine_sim = np.dot(reference_embedding, translation_embedding) / (np.linalg.norm(reference_embedding) * np.linalg.norm(translation_embedding))
        cosine_sims.append(cosine_sim)
        print(key, cosine_sim)
    print("=" * 80)

    return DescriptiveStats(cosine_sims)


if __name__ == '__main__':
    load_dotenv()
    openai_embedder = OpenAIEmbedder(os.getenv("OPENAI_API_KEY"))
    cohere_embedder = CohereEmbedder(os.getenv("COHERE_API_KEY"))
    voyageai.api_key = os.getenv("VOYAGEAI_API_KEY")
    reference_langauge = "en"
    translations = ARTICLE_TRANSLATIONS
    embedders = [
        openai_embedder,
        cohere_embedder,
        VoyageAIEmbedder(),
        SentenceTransformersEmbedder("all-MiniLM-L6-v2"),
        SentenceTransformersEmbedder("all-mpnet-base-v2"),
        SentenceTransformersEmbedder("thenlper/gte-large"),
        SentenceTransformersEmbedder("thenlper/gte-base"),
        SentenceTransformersEmbedder("thenlper/gte-large"),
        SentenceTransformersEmbedder("BAAI/bge-large-en-v1.5"),
        SentenceTransformersEmbedder("BAAI/llm-embedder"),
        SentenceTransformersEmbedder("embaas/sentence-transformers-e5-large-v2"),
        SentenceTransformersEmbedder("intfloat/e5-large-v2"),
        SentenceTransformersEmbedder("intfloat/multilingual-e5-small"),
        SentenceTransformersEmbedder("intfloat/multilingual-e5-base"),
        SentenceTransformersEmbedder("intfloat/multilingual-e5-large"),
        SentenceTransformersEmbedder("LaBSE"),
        SentenceTransformersEmbedder("paraphrase-multilingual-MiniLM-L12-v2"),
        SentenceTransformersEmbedder("paraphrase-multilingual-mpnet-base-v2"),
        SentenceTransformersEmbedder("distiluse-base-multilingual-cased-v1"),
        InstructorEmbedder("hkunlp/instructor-xl"),
    ]
    stats = {}
    for embedder in embedders:
        stats[embedder.name()] = compare_embeddings(embedder, translations, reference_langauge)

    # sorted list of tuples, sorted by mean
    sorted_stats = sorted(stats.items(), key=lambda x: x[1].mean, reverse=True)

    print("Summary")
    header_format_str = "{:<6} {:<45} {:<15} {:<15} {:<15} {:<15} {:<15}"
    table_format_str = "{:<6} {:<45} {:<15.3f} {:<15.3f} {:<15.3f} {:<15.3f} {:<15.3f}"
    print(header_format_str.format("Rank", "Model", "Mean", "Median", "Range", "Variance", "Std Dev"))
    for i, (embedder_name, stat) in enumerate(sorted_stats):
        print(table_format_str.format(i+1, embedder_name, stat.mean, stat.median, stat.range, stat.variance, stat.std_dev))
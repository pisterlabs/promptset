
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import FewShotChatMessagePromptTemplate
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from loguru import logger
import orjson
from pydantic import BaseModel, Field


class ClassificationResultWithReason(BaseModel):
    pred: int = Field(
        description=" ".join((
            "If the news should be excluded, return 1.",
            "If the news should not be excluded, return 0.",
        ))
    )
    reason: str = Field(
        description=" ".join((
            "Reason for why or why not it should be excluded",
            "for constructing EPU index.",
            "Use no more thant 30 words.",
        ))
    )

def chatgpt_cls(news: str):
    article_example = [
        '澳洲央行昨日意外調高基準利率，隔夜拆款利率從原本3%，調高為3.25%，澳洲儲備銀行行長史蒂芬（Glenn Stevens）表示，澳洲經濟大幅緊縮的風險已明顯降低，為了讓經濟穩健成長，澳洲央行未來幾年將持續進行微調。巴克萊投資表示，澳洲與巴西、印度、印尼、台灣、韓國，是這一波金融風暴中，經濟提早反彈的國家，領先指標3月分，就出現反彈訊號，由於中國大陸原物料需求強勁，巴克萊預估，澳洲今明二年經濟成長率，可望達0.9%與2.5%。巴克萊投資預估，英國、印度、韓國三國，可望在明年第1季升息，台灣升息時間，可望落在第2季；美國、中國與歐盟則要等到明年第3季才會升息。',
        '千興為國內老牌不鏽鋼冷軋廠，廠區在台南麻豆，1972年成立，主力產品為300系冷軋不鏽鋼。資本額32.28億元，董事長及總經理都是葉碩堂，目前員工約100多人。受到不鏽鋼多年不景氣的影響，千興過去3年呈現虧損，今年上半年每股稅後虧損0.5元。（記者羅倩宜）',
        '儘管國際經濟情勢擾攘不安，MAZDA TAIWAN執行長朱梅君在接受記者訪問時仍樂觀預期今年國內車市仍可望比去年成長2成左右，但對明年第1、2季則持保守態度，認為仍須取決總統大選後的經濟面和國際經濟局勢能否轉穩。她表示，隨著全新Mazda5的即將上市，未來除將強化Zoom-Zoom駕馭至上精神，也將持續營造MAZDA的日式高質感、高品味形象，同時亦將更側重服務品質的提升。由於市場預測未來小型房車和小型SUV都有成長空間，因此她透露明年將考慮引進全新Mazda CX-5，也不排除國產化可能性。也因高競爭力新車的陸續加入，她誓言2015年後MAZDA總市占率將提升至6%。',
        '【丁威╱台北報導】國內消費市場不景氣，國內KTV龍頭之一的錢櫃（8359）今年前4月業績也衰退近2成，為了力抗景氣「寒流」，錢櫃推出20周年慶祝活動，自4月以來推出歡唱8折，啤酒降價的優惠活動後，業績成長近20~30%，而隨著暑假旺季到來，錢櫃單月營收可望恢復3億元的水準。另外，錢櫃轉投資的錢櫃茶餐廳，從1月份開幕後業績穩定成長，4月營收約800萬元，5、6月應有單月千萬的實力。錢櫃協理呂嘉正表示，近1、2年來民眾荷包縮水，對於KTV歡唱需求降低，今年錢櫃1~4月營收較去年同期年減18.24%。為了刺激民眾上門消費唱歌，今年滿20周年的錢櫃推出歡唱8折優惠活動。呂嘉正說，以往錢櫃最讓消費者介意的「開瓶費」和「酒錢」，4月份開始降價促銷以來，人潮逐漸回籠，5月份開始啤酒全面降價，業績已經明顯成長2成。',
        '泰國前總理戴克辛流亡海外17個月後終於回國，他雖表示絕對不會重返政壇，但分析家與過去反對他的人士均對此存疑，也認為他返國將為泰國政局投下不穩定的變數。新加坡國立大學學者蒙提桑諾（Michael Montesano）指出，泰國軍方前年發動的政變並未達成將戴克辛徹底逐出政壇的目的。戴克辛抱著應可獲判無罪的期待返國，勢必會重新開展政治生涯。泰國一家風險顧問公司總經理道荷堤也指出，戴克辛聲稱不再參政，「在我聽來不是實話」。2005至06年間發動長期示威，終於導致戴克辛下台的抗議人士，本周一再度走上街頭，矢言將針對戴克辛及現任總理薩瑪展開反對運動。當年領導反對運動的媒體人林明達形容，現在的局勢就像「四處已灑遍燃料，只待有人放火」。台大政治系楊永明教授則分析，戴克辛流亡海外期間，軍政府無法提振泰國經濟，執政面臨極大困境，終致人民力量黨在選舉中大勝。如今戴克辛能夠回國，應是與泰皇及曼谷政界人士達成某種交換協議，對政局應有穩定作用。他認為，戴克辛日後仍將在泰國政壇扮演關鍵性角色。編譯李寧怡',
        '總統馬英九競選期間提出的募兵支票恐怕面臨跳票？國防部長陳肇敏昨在立法院外交國防委員會中表示，推動募兵制的確有困難，且要推動到全面募兵制，「是國防部嚴峻的考驗」。陳肇敏指出，國防預算佔ＧＤＰ（Gross Domestic Product，國內生產毛額）百分之三，但以現有國軍員額來看，根本不可能達成全面募兵制，因此國軍要再精進是必然方向。而我國走的是一個海島守勢作戰，各軍種人員配比都要進行檢討，並將繼續推動「再精進案」，將目前二十七萬大軍減至二十到二十五萬人以內。陳肇敏並指出，目前募得許多素質相當不錯的國軍弟兄，主要與過去八年經濟不景氣，工作難找有關，未來經濟好轉後，是否還有誘因？另外，台灣人口有限，國防部會努力推動募兵制，但不能因滿足「量」的需求而減低「質」的要求。至於重大軍購案，陳肇敏強調向美國購買F-16 C/D戰機有其必要，但其他軍購案如潛艦，因目前我方連採購潛艇的型號都沒有，這部分會再研究，會就作戰需求再作評估。記者王<U+70F1>華'
    ]
    _response_example = [
        {"pred":1,"reason":"The news is about the Australian central bank raising interest rates. It mentions slightly Taiwan's economics condtion, but there isn't any further discussion about policy or uncertainty. Hence, this news should be excluded."},
        {"pred":0,"reason":"The news provides information about a Taiwanese stainless steel cold rolling factory, which is relevant to Taiwan's economy."},
        {"pred":1,"reason":"The news is about the CEO of Mazda Taiwan expressing optimism about the domestic car market, which is not directly related to Taiwan's economics."},
        {"pred":0,"reason":"The news provides information about the performance of a Taiwanese KTV company and its efforts to combat the economic downturn, which is relevant to Taiwan's economy."},
        {"pred":1,"reason":"The news is about the return of the former Thai Prime Minister, which is not directly related to Taiwan's economics."},
        {"pred":0,"reason":"The news provides information about the difficulties and challenges of implementing conscription in Taiwan's defense system, which is relevant to Taiwan's economy and national security."}
    ]
    response_example = [orjson.dumps(i).decode() for  i in _response_example]

    system_message_template = """\
    I am an economist working on constructing {country}'s Economic Policy Uncertainty Index (EPU index).
My primary goal is to classify wheter a news should be excluded when constructing EPU index in {country}.
Help me complete the classification task identifying wheter the given news should be excluded.

There are two criteria I'm considering to exclude a news.

Criterion1:
The main idea of the news is either historical accounts or abstract subjective inferences, which won't impact {country}'s economics for sure.
Hence, this kind of news should be excluded.

Criterion2:
There main idea of the news is not related with {country}.
For example, the people or companies mentioned in the news have nothing to do with {country} or the events
in the news don't actually happen within {country}. I will excluded the news as well.

Notice that you can first justify wheter there is a person, company or event in news related to {country}. 
If there isn't any, it should be excluded with high probability.\
"""
    human_message_template = """\
{correct_instructions}

News:
{news}

Output Instructions:
{output_instructions}
The single quote should always be escaped in the reason section preventing from broken json format.\
"""

    instruction = "Should the news be excluded ?"
    few_shot_example = [
        {
            "news": news,
            "response": response,
            "correct_instructions": instruction,
            "output_instructions": "",
        }
        for news, response in zip(article_example, response_example)
    ]
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", human_message_template),
        ("ai", "{response}"),
    ])
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt = example_prompt,
        examples = few_shot_example,
    )

    parser_with_reason = PydanticOutputParser(
        pydantic_object=ClassificationResultWithReason
    )

    few_shot_with_reason_prompt = (
        ChatPromptTemplate
        .from_messages([
            ("system", system_message_template),
            few_shot_prompt,
            MessagesPlaceholder(variable_name = "chat_history"),
            ("human", human_message_template),
        ])
        .partial(
            country = 'Taiwan',
            output_instructions = parser_with_reason.get_format_instructions(),
        )
    )

    llm = ChatOpenAI(
        model = 'gpt-3.5-turbo-1106',
        temperature = 0.,
        request_timeout = 120.
    )
    chain = LLMChain(
        llm = llm,
        prompt = few_shot_with_reason_prompt.partial(news = news),
        verbose = False
    )
    chain.memory = ConversationBufferMemory(
        memory_key = "chat_history",
        return_messages = True
    )
    while True:
        try:
            res = orjson.loads(chain.run(correct_instructions=instruction))
            if res.get("pred") is not None and res.get("reason") is not None:
                break
            else:
                logger.info(
                    f"formatting error(KeyError), re-generate"
                )
                instruction = " ".join(
                    (
                        "Your answer which is a json string don't",
                        "have the specified key. Follow the schema carefully.",
                    )
                )
                continue

        except orjson.JSONDecodeError:
            logger.info(
                f"formatting error(JSONDecodeError), re-generate"
            )
            instruction = " ".join(
                (
                    "Formatting error. It might because",
                    "not all single quotes have been escaped or",
                    "the answering has been truncated.ry to answer precisely",
                    "and reduce the number of token.",
                )
            )

    return res["pred"]
# create basic AI model class with template model variable and has get_chain and run methods
from typing import Any
from chainlit import Message
from langchain import ConversationChain, LLMChain, PromptTemplate
from langchain.base_language import BaseLanguageModel
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from chainlit.lc.agent import run_langchain_agent
from chainlit.config import config


class AIModel:
    """Base class for AI models."""

    template: str
    prompt: PromptTemplate
    model: BaseLanguageModel

    def get_chain(self, **kwargs: Any) -> Any:
        raise NotImplementedError("get_chain not implemented")

    def run(self, **kwargs: Any) -> Any:
        """Run the model."""
        chain = self.get_chain(**kwargs)
        return chain.run(**kwargs)
    
class BartenderAI(AIModel):
    model = ChatOpenAI(temperature=0)

    template = """The following is a friendly conversation between a Customer and an BartenderAI. The BartenderAI is a professional bartender and help Customer find a cocktail that suits. AI should guide Customer in choosing a cocktail that is tailored to its preferences. BartenderAI should understand Customer preferences based on Customer preferred texture, type of alcohol, taste, or personal characteristics. please don't recommend a particular cocktail to Customer. AI job is merely understand Customer preference. And don't ask too complex question make question simple and one at a time. 請用繁體中文與我對答案。 
Current conversation:
{history}
Customer: {input}
BartenderAI:
"""

    prompt = PromptTemplate(template=template, input_variables=["history", "input"])
    
    def get_chain(self, **kwargs: Any) -> Any:
        return ConversationChain(
            prompt=self.prompt,
            llm=self.model,
            memory=ConversationBufferMemory()
        )
            
class SummaryPreferenceAI(AIModel):
    model = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    template = """You're now a professional bartender, and the following is the conversation between the Customer and Bartender, please summary the customer preference from the following conversation in 繁體中文
Current conversation:
{history}
"""

    prompt = PromptTemplate(template=template, input_variables=["history"])
    
    def get_chain(self, **kwargs: Any) -> Any:
        return LLMChain(llm=self.model, prompt=self.prompt)
    
    async def run(self, conversation_history) -> Message:
        chain = self.get_chain()
        raw_res, output_key = await run_langchain_agent(
            agent=chain , input_str=conversation_history, use_async=config.code.lc_agent_is_async
        )
        
        if output_key is not None:
            # Use the output key if provided
            res = raw_res[output_key]
        else:
            # Otherwise, use the raw response
            res = raw_res
        # Finally, send the response to the user
        return Message(author=config.ui.name, content=res)


class RecommendAI(AIModel):
    model = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    template = """you are acting as a professional bartender, you know much about the customer preference, and can recommend the right one to your customer. The below is the menu, please choice one of the cocktail based on the customer preference, elaborate the reason why you recommend and reply in 繁體中文
here is Customer preference:
-----------------------
{preferences}
-----------------------

here is the menu:
-----------------------
酒名	價格(NTD)	基底酒款	其他成分	酒感等級	口感描述
青雲閤	400	Gin琴酒	Nordes gin諾帝斯琴酒/St.germain接骨木花利口酒/Skinos乳香利口酒/Jasmine Syrup自製茉莉花糖漿/Citrus Acid檸檬酸液/tonic通寧水	2	微甜/花香味強烈/清爽/氣泡感
和泉樓	400	Gin琴酒	Generous Gin Azur大方琴酒/Crème de Violet 紫羅蘭利口酒/Lime Juice萊姆汁/Lavender Syrup自製薰衣草糖漿/La Caravedo Pisco秘魯白蘭地	3.5	偏酸爽口/如同香水的強烈花香
醉花園	450	Homemade Rose Liqueur自製玫瑰利口酒	Homemade Rose Liqueur自製玫瑰利口酒/Red Repper Syrup粉紅胡椒糖漿/Hendricks Flora Adora Gin亨利爵士花神/Latic Acid乳酸/Cream鮮奶油/Egg White蛋白/Soda Water蘇打水	3	蛋糕般的綿密奶泡/主體玫瑰花香帶一絲粉紅胡椒的偏甜辛香
鐵觀音	400	vodka伏特加	Tieguanyin tea infused vodka鐵觀音伏特加/Cointreau君度橙酒/Crème de peach水蜜桃利口酒	2	水蜜桃甜香為前調/中調展現鐵觀音培茶風味/清爽的氣泡/酒體輕盈
文山包種	400	Gin琴酒	Wen Shan Pochong包種茶琴酒/Pavan葡萄利口酒/Lavender Leaf Syrup自製薰衣草片糖漿/Lemon juice檸檬汁	3	偏甜爽口/花草香/麝香葡萄與橙花氣味為中調/茶香做為後韻
金萱	430	White Wine白葡萄酒	Jin Xuan Tea Infused White Wine金萱茶白葡萄酒/Pineapple Sage Infused Apple Whiskey鳳梨鼠尾草蘋果威士忌/Chamomile Cordial洋甘菊風味液/Cream cheese Foam奶油起司泡沫	3	上層奶泡起司蛋糕風味呼應金萱茶獨特奶香/中調強烈洋甘菊轉為鼠尾草與蘋果的清新/微苦茶感與葡萄弱酸做結尾
東方美人	$450	V.S.O.P brandy白蘭地	Driental beauty infused V.S.O.P brandy 東方美人茶白蘭地/Sesame芝麻/Adriatico鹽味杏仁利口酒/Fig Leaf Syrup無花果葉糖漿/Blackwalnut Bitters 黑核桃苦精/Selva Ray Chocolate巧克力蘭姆酒	4	初聞明顯可可香而後是杏仁與無花果葉類的堅果氣息/接著輕微苦韻洗滌口腔後茶感才慢悠悠出現
北港甜湯米糕粥	$430	Whiskey威士忌	Longan Infused Whiskey自製桂圓威士忌/Sticy Rice圓糯米/Macallan 12years麥卡倫12年/Cannanmon Bitters自製肉桂苦精	3	翻玩70年歷史甜品/甜而不膩的大人甜湯/桂圓的蜜味與雪莉桶威士忌完美融合/些許肉桂味添加層次/有趣的食用型調酒
阿嬌姨烤魷魚	$430	Vodka伏特加/ Whiskey泥煤威士忌	Squid Infused Vodka自制烤魷魚伏特加/ Talisker Storm Whiskey/Black Cardamom黑荳蔻/Basil Syrup羅勒糖漿/Citrus Acid檸檬酸/Cucumber Soda Water黃瓜口味氣泡水/Squid Slices網狀魷魚片	3.5	出乎意料的味覺組合/輕微的黑荳蔻模擬出炭烤的煙燻味/帶有鹹感的威士忌襯托魷魚鮮香/小黃瓜與氣泡帶來清爽結尾
童年記憶愛玉冰	$400	Bamboo Leaves Infused Vermouth自製竹葉苦艾酒	Bamboo Leaves Infused Vermouth自製竹葉苦艾酒/Ice Jelly愛玉/Homemade Limocello自製檸檬利口酒/White Wine Cardamom Syrup白酒荳蔻糖漿	3.5	竹葉香與檸檬甜感結合後接葡萄微酸/輕微的香料做結尾/吃得到愛玉喔
香煙裊裊龍山寺	$430	Gin琴酒	Tanquerary No.10/Skinos希臘乳香酒/Sandalwood Infused Gin檀香木琴酒/Selva Ray Coconut Rum椰子蘭姆酒/Malibu椰子香甜酒	5	椰子氣味鋪陳檀香木質氣息/順口度高/如同佛珠與佛堂的既視感香氣
民風淳樸剝皮寮	$420	Vodka伏特加/ Gin琴酒	Don Julio Blanco/Peeled Pepper Infused Vodka自製剝皮辣椒伏特加/East 135 GinㄥSoy Sauce手工醬油/Clarify Tomato Juice澄清番茄汁/ Ginger Ale薑汁汽水/Umami Bitters旨味苦精	3	氣泡爽口/輕微香菇與番茄鮮味/尾巴有些許辣椒熱感/不會辣
日皇御用摩納卡	$430	Whiskey泥煤威士忌	Arbeg10y/Red Beans杜瓦小豆香甜酒/Luxardo Apricot杏桃香甜酒/Milk牛奶/Hawthorn Miso Campari Monaka仙楂味增金巴利最中餅	2.5	前味紅豆氣味明顯/中段杏桃果香參雜煙燻味/大人味奶酒
阿寶師的咖哩酥	400	Whiskey威士忌	Pork Floss Infused Whiskey肉鬆威士忌/Curry Syrup咖哩糖漿/Carrot Juice胡蘿蔔汁	3	甜味型調酒/咖哩氣味轉為肉鬆帶來的輕微脂感/尾韻為胡蘿蔔自然清甜
懸壺濟世青草巷退火養肝茶	400	gin琴酒	Cheerful Crackers Infused gin自製奇福餅乾琴酒/Burdock Infused Frangelico 自製牛蒡榛果香甜酒/Dita荔枝香甜酒/Grassleef Sweetflag Rhizome石菖蒲/Falernum法勒南香甜酒/Suze龍膽草香甜酒	3.5	苦甜型調酒/牛蒡與龍膽草結合使苦味不再單調/中調由石菖蒲與法勒南特有的香料譜出/奇福餅乾的油脂感作為橋樑銜接所有風味
清涼百草茶	400	Herbal Tea Wine青草茶酒	Herbal Tea Wine青草茶酒/Vecchio amaro del capo義大利藥草酒/Asiatic Worm wood杜瓦艾草香甜酒/Dita荔枝香甜酒/Fernet Branca義大利苦味香甜酒	4	中式草本遇上西式藥酒/清甜中帶微苦/艾草香銜接荔枝果香/
駐顏美人湯	400	Brandy白蘭地	La Caravedo Pisco秘魯白蘭地/White wine cardamom syrup自製白酒荳蔻糖漿/Aloe Liqueur蘆薈香甜酒/Chartreuse修道院香甜酒/Acid Solution酸味模擬液/Aloe Water澄清蘆薈汁/Ruta Graveolens Spray芸香噴霧	3	入口時可聞到雲香帶有甜感的獨特香氣/蘆薈為主軸的清新類花香/尾韻香水白蘭地葡萄香與荳蔻隱約浮現
新世紀冬瓜茶	360	Rum蘭姆酒	Spice White Gourd Drink自製香料冬瓜茶/Clarify Banana Juice澄清香蕉水/Soda Water蘇打水	3.5	香料增添冬瓜茶層次風味與香蕉熱帶水果氣味相輔相乘/輕微甜口
古早味楊桃湯	360	Gin琴酒	Star fruit juice鹽漬楊桃湯/Pineapple Juice鳳梨汁/Caramel焦糖/Tonic通寧水/Spite雪碧	3.5	楊桃湯輕微鹹味帶出甜感/焦糖鳳梨作為後味支撐
-----------------------
"""
    
    prompt = PromptTemplate(template=template, input_variables=["preferences"])
    
    def get_chain(self, **kwargs: Any) -> Any:
        return LLMChain(llm=self.model, prompt=self.prompt)
    
    async def run(self, preferences) -> Message:
        chain = self.get_chain()
        raw_res, output_key = await run_langchain_agent(
            agent=chain , input_str=preferences, use_async=config.code.lc_agent_is_async
        )
        
        if output_key is not None:
            # Use the output key if provided
            res = raw_res[output_key]
        else:
            # Otherwise, use the raw response
            res = raw_res
        # Finally, send the response to the user
        return Message(author=config.ui.name, content=res)


class DetailAI(AIModel):
    model = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    template = """please make a story for the drink call {drinks} base on the following information in 繁體中文 
Drinks information:
-----------------------
酒名	價格(NTD)	基底酒款	其他成分	酒感等級	口感描述
青雲閤	400	Gin琴酒	Nordes gin諾帝斯琴酒/St.germain接骨木花利口酒/Skinos乳香利口酒/Jasmine Syrup自製茉莉花糖漿/Citrus Acid檸檬酸液/tonic通寧水	2	微甜/花香味強烈/清爽/氣泡感
和泉樓	400	Gin琴酒	Generous Gin Azur大方琴酒/Crème de Violet 紫羅蘭利口酒/Lime Juice萊姆汁/Lavender Syrup自製薰衣草糖漿/La Caravedo Pisco秘魯白蘭地	3.5	偏酸爽口/如同香水的強烈花香
醉花園	450	Homemade Rose Liqueur自製玫瑰利口酒	Homemade Rose Liqueur自製玫瑰利口酒/Red Repper Syrup粉紅胡椒糖漿/Hendricks Flora Adora Gin亨利爵士花神/Latic Acid乳酸/Cream鮮奶油/Egg White蛋白/Soda Water蘇打水	3	蛋糕般的綿密奶泡/主體玫瑰花香帶一絲粉紅胡椒的偏甜辛香
鐵觀音	400	vodka伏特加	Tieguanyin tea infused vodka鐵觀音伏特加/Cointreau君度橙酒/Crème de peach水蜜桃利口酒	2	水蜜桃甜香為前調/中調展現鐵觀音培茶風味/清爽的氣泡/酒體輕盈
文山包種	400	Gin琴酒	Wen Shan Pochong包種茶琴酒/Pavan葡萄利口酒/Lavender Leaf Syrup自製薰衣草片糖漿/Lemon juice檸檬汁	3	偏甜爽口/花草香/麝香葡萄與橙花氣味為中調/茶香做為後韻
金萱	430	White Wine白葡萄酒	Jin Xuan Tea Infused White Wine金萱茶白葡萄酒/Pineapple Sage Infused Apple Whiskey鳳梨鼠尾草蘋果威士忌/Chamomile Cordial洋甘菊風味液/Cream cheese Foam奶油起司泡沫	3	上層奶泡起司蛋糕風味呼應金萱茶獨特奶香/中調強烈洋甘菊轉為鼠尾草與蘋果的清新/微苦茶感與葡萄弱酸做結尾
東方美人	$450	V.S.O.P brandy白蘭地	Driental beauty infused V.S.O.P brandy 東方美人茶白蘭地/Sesame芝麻/Adriatico鹽味杏仁利口酒/Fig Leaf Syrup無花果葉糖漿/Blackwalnut Bitters 黑核桃苦精/Selva Ray Chocolate巧克力蘭姆酒	4	初聞明顯可可香而後是杏仁與無花果葉類的堅果氣息/接著輕微苦韻洗滌口腔後茶感才慢悠悠出現
北港甜湯米糕粥	$430	Whiskey威士忌	Longan Infused Whiskey自製桂圓威士忌/Sticy Rice圓糯米/Macallan 12years麥卡倫12年/Cannanmon Bitters自製肉桂苦精	3	翻玩70年歷史甜品/甜而不膩的大人甜湯/桂圓的蜜味與雪莉桶威士忌完美融合/些許肉桂味添加層次/有趣的食用型調酒
阿嬌姨烤魷魚	$430	Vodka伏特加/ Whiskey泥煤威士忌	Squid Infused Vodka自制烤魷魚伏特加/ Talisker Storm Whiskey/Black Cardamom黑荳蔻/Basil Syrup羅勒糖漿/Citrus Acid檸檬酸/Cucumber Soda Water黃瓜口味氣泡水/Squid Slices網狀魷魚片	3.5	出乎意料的味覺組合/輕微的黑荳蔻模擬出炭烤的煙燻味/帶有鹹感的威士忌襯托魷魚鮮香/小黃瓜與氣泡帶來清爽結尾
童年記憶愛玉冰	$400	Bamboo Leaves Infused Vermouth自製竹葉苦艾酒	Bamboo Leaves Infused Vermouth自製竹葉苦艾酒/Ice Jelly愛玉/Homemade Limocello自製檸檬利口酒/White Wine Cardamom Syrup白酒荳蔻糖漿	3.5	竹葉香與檸檬甜感結合後接葡萄微酸/輕微的香料做結尾/吃得到愛玉喔
香煙裊裊龍山寺	$430	Gin琴酒	Tanquerary No.10/Skinos希臘乳香酒/Sandalwood Infused Gin檀香木琴酒/Selva Ray Coconut Rum椰子蘭姆酒/Malibu椰子香甜酒	5	椰子氣味鋪陳檀香木質氣息/順口度高/如同佛珠與佛堂的既視感香氣
民風淳樸剝皮寮	$420	Vodka伏特加/ Gin琴酒	Don Julio Blanco/Peeled Pepper Infused Vodka自製剝皮辣椒伏特加/East 135 GinㄥSoy Sauce手工醬油/Clarify Tomato Juice澄清番茄汁/ Ginger Ale薑汁汽水/Umami Bitters旨味苦精	3	氣泡爽口/輕微香菇與番茄鮮味/尾巴有些許辣椒熱感/不會辣
日皇御用摩納卡	$430	Whiskey泥煤威士忌	Arbeg10y/Red Beans杜瓦小豆香甜酒/Luxardo Apricot杏桃香甜酒/Milk牛奶/Hawthorn Miso Campari Monaka仙楂味增金巴利最中餅	2.5	前味紅豆氣味明顯/中段杏桃果香參雜煙燻味/大人味奶酒
阿寶師的咖哩酥	400	Whiskey威士忌	Pork Floss Infused Whiskey肉鬆威士忌/Curry Syrup咖哩糖漿/Carrot Juice胡蘿蔔汁	3	甜味型調酒/咖哩氣味轉為肉鬆帶來的輕微脂感/尾韻為胡蘿蔔自然清甜
懸壺濟世青草巷退火養肝茶	400	gin琴酒	Cheerful Crackers Infused gin自製奇福餅乾琴酒/Burdock Infused Frangelico 自製牛蒡榛果香甜酒/Dita荔枝香甜酒/Grassleef Sweetflag Rhizome石菖蒲/Falernum法勒南香甜酒/Suze龍膽草香甜酒	3.5	苦甜型調酒/牛蒡與龍膽草結合使苦味不再單調/中調由石菖蒲與法勒南特有的香料譜出/奇福餅乾的油脂感作為橋樑銜接所有風味
清涼百草茶	400	Herbal Tea Wine青草茶酒	Herbal Tea Wine青草茶酒/Vecchio amaro del capo義大利藥草酒/Asiatic Worm wood杜瓦艾草香甜酒/Dita荔枝香甜酒/Fernet Branca義大利苦味香甜酒	4	中式草本遇上西式藥酒/清甜中帶微苦/艾草香銜接荔枝果香/
駐顏美人湯	400	Brandy白蘭地	La Caravedo Pisco秘魯白蘭地/White wine cardamom syrup自製白酒荳蔻糖漿/Aloe Liqueur蘆薈香甜酒/Chartreuse修道院香甜酒/Acid Solution酸味模擬液/Aloe Water澄清蘆薈汁/Ruta Graveolens Spray芸香噴霧	3	入口時可聞到雲香帶有甜感的獨特香氣/蘆薈為主軸的清新類花香/尾韻香水白蘭地葡萄香與荳蔻隱約浮現
新世紀冬瓜茶	360	Rum蘭姆酒	Spice White Gourd Drink自製香料冬瓜茶/Clarify Banana Juice澄清香蕉水/Soda Water蘇打水	3.5	香料增添冬瓜茶層次風味與香蕉熱帶水果氣味相輔相乘/輕微甜口
古早味楊桃湯	360	Gin琴酒	Star fruit juice鹽漬楊桃湯/Pineapple Juice鳳梨汁/Caramel焦糖/Tonic通寧水/Spite雪碧	3.5	楊桃湯輕微鹹味帶出甜感/焦糖鳳梨作為後味支撐
-------------------------
"""
    
    prompt = PromptTemplate(template=template, input_variables=["drinks"])
    
    def get_chain(self, **kwargs: Any) -> Any:
        return LLMChain(llm=self.model, prompt=self.prompt)
    
    async def run(self, drinks) -> Message:
        chain = self.get_chain()
        raw_res, output_key = await run_langchain_agent(
            agent=chain , input_str=drinks, use_async=config.code.lc_agent_is_async
        )
        
        if output_key is not None:
            # Use the output key if provided
            res = raw_res[output_key]
        else:
            # Otherwise, use the raw response
            res = raw_res
        # Finally, send the response to the user
        return Message(author=config.ui.name, content=res)
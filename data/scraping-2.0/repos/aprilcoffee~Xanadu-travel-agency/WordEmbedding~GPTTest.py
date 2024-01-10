import openai
import api_key
openai.api_key = api_key.openai_key()

input_text = '''
「自然/文化」的非人物共謀：
ㄧ場由生命而起的藝術行動
關鍵字：蓋婭政治、土地神靈、《山若有神》、桃園航空城、《空城現場》、藝術家田野、生命電影、行動電影
    近幾年臺灣的當代藝術中，不論是雙年展或大型展覽，皆有許多標誌以自然為核心的討論，主要論及近幾年的人類世議題發展，也從人的角度反向思考土地與環境，在這種類型的創作中可以看見一種趨勢，其大量的引用了人類學的方法論，這種標誌著大量「現場」的藝術行動，在原先的藝術語境裡頭並未能直接地被接受，但這些能夠看見議題更廣泛層面的方法，或許是在這矛盾的國族與土地認同中，藝術領域需要接納且使用的方法之一。
	全文將會由自然如何作為一個非人的行動者開始談起，並以近幾年的台北雙年展為例，說明我們如何被一個他者所捲動，何以訴諸自然。再來會談到《山若有神》及《空城現場》兩個藝術計畫，分別是如何在我生命裡成為重要的一塊，亦會提及計畫裡的每一個選擇及作為。最後會辯證田野調查方法的可行性，論述它為何能夠具有方法論上的演進，並且直接地指向政治問題。最後分析個人在創作上的思想與行動，說明田野調查做為藝術方法並非只是空談，且更進一步地能夠作為語言。

'''

while True:
    input_text = input()
    print(input_text)
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    temperature = 1.2,
    messages=[
        {"role": "system", "content":""},

        {"role": "user", "content": 
         input_text}
    ]
    )

    #print(response['choices'][0]['message']['content'])
    output = response['choices'][0]['message']['content']
    print(output)
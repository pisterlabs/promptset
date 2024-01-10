'''
LangChainのライブラリが正しく利用できていないためテストが実行できない。ひとまず対応は保留とする。
'''
# from langchain import LangChain

# def translate(text):
#     chain = LangChain()
#     chain.load_model("en", "en-ja")
#     chain.load_model("ja", "ja-en")

#     translation = chain.translate(text, "en", "ja")

#     return translation

# def test_translate():
#     result = translate('Make the catch!')
#     assert result == '捕まえろ!'

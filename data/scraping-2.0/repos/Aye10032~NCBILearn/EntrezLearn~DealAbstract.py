import openai

openai.api_key_path = '../API.txt'
abstract = 'A repressive chromatin state featuring trimethylated lysine 36 on histone H3 (H3K36me3) and DNA methylation suppresses cryptic transcription in embryonic stem cells. Cryptic transcription is elevated with age in yeast and nematodes, and reducing it extends yeast lifespan, though whether this occurs in mammals is unknown. We show that cryptic transcription is elevated in aged mammalian stem cells, including murine hematopoietic stem cells (mHSCs) and neural stem cells (NSCs) and human mesenchymal stem cells (hMSCs). Precise mapping allowed quantification of age-associated cryptic transcription in hMSCs aged in vitro. Regions with significant age-associated cryptic transcription have a unique chromatin signature: decreased H3K36me3 and increased H3K4me1, H3K4me3, and H3K27ac with age. Genomic regions undergoing such changes resemble known promoter sequences and are bound by TBP even in young cells. Hence, the more permissive chromatin state at intragenic cryptic promoters likely underlies increased cryptic transcription in aged mammalian stem cells.'
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "assistant", "content": f"下面我有一段生物领域论文的摘要，请问当中提到了哪些生物学上的分类为物种的名称？{abstract}。将物种存入一个python能够直接转为list的字符串中回答给我，若没有提到，则回答我'[]'"},
    ]
)

result = response['choices'][0]['message']['content']
print(eval(result))


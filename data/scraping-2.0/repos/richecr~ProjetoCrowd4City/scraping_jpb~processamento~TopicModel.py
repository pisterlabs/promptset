import gensim
import nltk
import spacy
import numpy as np
import pandas as pd
from nltk.stem import *
from nltk.stem.porter import *
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models.coherencemodel import CoherenceModel

np.random.seed(2018)
nltk.download('wordnet')

stemmer = PorterStemmer()

nlp = spacy.load('pt_core_news_sm')

# Carregando os dados.
dados = pd.read_csv("../textos_videos.csv")
textos = dados['texto']
# print(textos[:5])

gensim.parsing.preprocessing.STOPWORDS.union(["tudo", "coisa", "toda", "tava", "pessoal", "dessa", "resolvido", "aqui", "gente", "tá", "né", "calendário", "jpb", "agora", "voltar", "lá", "hoje", "aí", "ainda", "então", "vai", "porque", "moradores", "fazer", "rua", "bairro", "prefeitura", "todo", "vamos", "problema", "fica", "ver", "tô"])

from nltk import word_tokenize, pos_tag

# Pré-processamento dos dados.

# Lematização
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

# Colocando todas as palavras para caixa baixa.
# Removendo acentos e pontuações.
# Remover palavras com menos de 3 letras.
allowed_postags = ['NOUN', 'ADJ', 'PRON']
def preprocess(text):
	doc_out = []
	doc = nlp(text)
	for token in doc:
		if (token.text not in gensim.parsing.preprocessing.STOPWORDS):
			if (token.pos_ in allowed_postags):
				doc_out.append(lemmatize_stemming(token.text.lower()))
			else:
				continue
		else:
			continue

	return doc_out

processed_docs = dados['texto'].map(preprocess)
# print(processed_docs[:10])

dictionary = gensim.corpora.Dictionary(processed_docs)
count = 0
for k, v in dictionary.iteritems():
    # print(k, v)
    count += 1
    if count > 10:
        break

# Gensim Filter Extremes
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

# Bag-of-words.
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=5, id2word=dictionary, passes=10, workers=4)

# Imprimir os tópicos.
for topic in lda_model_tfidf.print_topics(-1):
	print(topic)

coherence_model_lda = CoherenceModel(model=lda_model_tfidf, texts=processed_docs, corpus=corpus_tfidf, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score LDAModelTfIdf: ', coherence_lda)

print("-------")
unseen_document = 'calendário JPB aqui nas nossas telas nós vamos agora até o bairro Jardim Paulistano zona sul de Campinas Você lembra que nossa equipe ouviu os moradores da Rua Riachuelo que reclamavam da falta de calçamento no local então o problema foi resolvido só que na época a prefeitura também se comprometeu e fazer o calçamento da Rua Ariel que fica bem pertinho essa parte foi feita mas só que pela metade Laisa grisi foi conferido calendário JPB desembarcou aqui no Jardim Paulistano E olha que maravilha hoje é possível andar na rua com calçamento sem tanta poeira sem pisar em lama Quando chove essa foi uma conquista dos moradores junto com calendário Desde o ano passado em 2015 quando a prefeitura calçou essa rua calça com a Rua Riachuelo também mas presta atenção dois passos seguintes e rua de terra essa rua que esse trechinho não foi calçado vou aqui conversar com os moradores já tá todo mundo reunido Por que me explica como é que pode só esse trechinho não foi calçada só esse trecho você imagina que fizeram as duas por duas partes né fizeram aquela parte de lá aí ficou a metade depois fizeram essa daqui aí deixar essa parte aqui sem sem tá feita né nessa parte de baixo é pior ainda porque quando chove a água invade a Casa dos moradores e olha só aqui nessa casa foi colocado um monte de pedra bem na frente para impedir que a água entre vamos lá falar com ela é dona Severina é dona Bill Olá tudo bom com a senhora como é que tá aqui essa situação a senhora Teve que colocar pedra aqui né é chover em entrar aqui sozinha imagina aperreio Aí tem que dar um jeito aqui é pior difícil hein dona Bill quanto tempo já que a senhora mora aqui nessa rua 8 anos viu o resultado de vergonha né a gente não tem né É porque se ele tivesse vergonha ele já tinha feito isso todos vocês moram aqui nessa rua aí o que que acontece nessas ruas aqui né aí o que que acontece a Rua Areal lá em cima Foi calçada a Rua Riachuelo também E vocês ficaram só um gostinho só na saudade e o pior que não se desviar da Lama dos buracos e ele prometeu Então olha você tá vendo aquela cerâmica Vale Aí depois ele dá o que é o povo que bota para que ele possa passar infelizmente é uma situação difícil a gente já pediu muitas vezes recado dado essa essa rua que já é assunto do calendário a gente conseguiu algumas ruas outras não voltamos em 2016 em 2016 o secretário André agra secretário de obras de Campina Grande e disse que ia voltar aqui não foi então vamos lá calendário novo quem é o representante'
bow_vector = dictionary.doc2bow(preprocess(unseen_document))
for index, score in sorted(lda_model_tfidf[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))

print("-------")
unseen_document = 'Rua Rubens Coelho Pereira Filho nuclear para tecido calçada mais a placa com os dados da obra a gente simplesmente sumiu e o calçamento não chegou para completar uma empresa privada abrir um buraco no meio da rua e já caiu gente até roda de carro os moradores procuraram a prefeitura mais até agora nada mudou e ele só tiveram uma alternativa pediram ajuda ao calendário JPB a Rua Rubens Coelho Pereira Filho aqui no bairro do Cuiá está em obras agora É uma pena que essa obra que não tem nada a ver com a necessidade e com a reclamação da rua inteira que como a gente vai ver olha só não tem calçamento e também não tem saneamento básico são mais de 15 anos que os moradores tem muitos transtornos Principalmente nos períodos de chuva e a gente vai começar a conversar com alguns deles para falar de como é viver nessa situação seu Matusalém muito sofrimento aqui bom dia bom dia a todos e muito sofrimento a 14 anos que eu moro aqui e nenhuma infraestrutura foi foi feita aqui no Parque da prefeitura essa empresa veio para fazer a obra ela tá passando uns fazendo saneamento mas é de um condomínio particular passou pela rua de vocês é isso exatamente que a gente não tem nada não vai se beneficiar e nada do que essa consultora está fazendo inclusive teve transtorno para gravar afundando a gente tem foto muito transtorno carro afundando pessoas aqui já caíram aqui dentro do buraco já saíram todas arranhadas em Popular né sem comunicação sem nada a nossas calçadas foram invadidas com barro e também muita lama por causa da obra que passou aqui e também da chuva que é um pouco em madeirado vai ser descendo água estão aparecendo E aí você imagina era só no meio de todo esse material para pessoa está caminhando criança e doso para daqui não tem jeito se a luz é difícil andar nessa rua os buracos demais e a gente tem eu tenho dificuldade de locomover para continuo no caminhar na rua fico mais dentro de casa com medo de sair na rua com medo de cair de acontecer o pior comigo eu sair agora da Rua Rubens Coelho Pereira Filho que é aquela lá para vir aqui na Rita Carneiro porque que eu tô aqui nesse local tem uma placa que fala da pavimentação da Rua Rita carneiro e também um trecho da Rubens Coelho Pereira Filho mas segundo os moradores tá aqui o Fernando para falar alguns anos a pavimentação que tinha nessa placa ou numa placa parecida era só da Rubens Coelho e não chegou de forma alguma lá foi É isso mesmo uma placa anterior ela é indicada que seria metade da Rua Rubens Coelho pelo qual motivo eu não sei mesmo moradores foi modificado essa placa com outro valor e a nossa rua até hoje está aí do jeito que você mostrou na sua reportagem E aí eu vi a conclusão da pavimentação da Rita Carneiro'
bow_vector = dictionary.doc2bow(preprocess(unseen_document))
for index, score in sorted(lda_model_tfidf[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))

print("-------")
unseen_document = "Jaguaribe a Escola Estadual Professora Maria do Carmo de Miranda já mudou muito desde a chegada do calendário por lá mas falta resolver um pequeno detalhe mas os estudantes não poderão utilizar é por isso que o calendário JPB continua aqui na Escola Estadual Professora Maria do Carmo de Miranda mesmo depois da obra de reforma tem sido concluída a gente precisa ter a garantia que esse espaço vai ser utilizado Falta muito pouco mas essa história ainda não acabou apresento para vocês agora o laboratório de biologia aqui eu tô vendo que tem aquelas maquetes do corpo humano do outro lado tem microscópios e outros equipamentos as outras estantes também estão cheias de máquinas todo esse material na nossa última visita aqui no início do mês de março tava encaixotado uma poeira danada e eu tô vendo que agora tá tudo no seu devido lugar aparentemente pronto para usar tá tudo certo para aula agora até agora ainda não foi inaugurado né você não tem fé que Vamos inaugurar atualizado É verdade que nem os professores ainda foram apresentados É verdade aos Laboratórios eles ficaram surpresos né porque antes não era assim Foi de repente ficou tudo arrumadinho e os professores não tinha nem noção dos equipamentos dos laboratórios tá todo mundo conhecendo hoje o novo laboratório de biologia né porque agora vai dar para chamar de laboratório né quando a gente entra a gente já tem a sensação de que está mesmo no laboratório que que tá faltando para que os estudantes possam utilizar esse espaço falta de instalações do ar condicionado só falta ar condicionado para funcionar e futuramente né daqui daqui uns dias nós estamos recebendo o quê da robótica os professores já vão fazer uma formação para atuar nessa área inclusive Nós temos dois alunos que foram representar Paraíba no robótica na China e a gente sabe que essa escola tem muito potencial da área eles vão ser monitores essa escola realmente tem muitos talentos item acima de tudo né estudantes interessados em usar esses esses passos para aprender Então hoje dia 14 de abril o carimbo Ainda é em andamento nesse segundo a secretaria de educação do estado esses ar-condicionados daqui dos laboratórios vão ser instalados no prazo de 15 dias com 15 dias de instalação mas 15 dias Os estudantes de gestão por aqui tá tudo certo certo dia 14 de Maio cai no sábado né o próximo dia útil seria o dia 16 mas aí a Paloma que começou essa história não vai poder estar aqui no dia 16 só no dia 19 essa data diretora do dia 19 Fica boa para senhora fica"
bow_vector = dictionary.doc2bow(preprocess(unseen_document))
for index, score in sorted(lda_model_tfidf[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))

'''
def compute_coherence_values(dct, corpus_tfidf, texts, limit, start, step):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dct : Gensim dictionary
    corpus_tfidf : Gensim corpus TF-IDF
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.LdaMulticore(corpus_tfidf, num_topics=num_topics, id2word=dct, passes=10, workers=4)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, corpus=corpus_tfidf, dictionary=dct, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

model_list, coherence_values = compute_coherence_values(dictionary, corpus_tfidf, processed_docs, 8, 5, 1)
limit=8; start=5; step=1;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))

'''
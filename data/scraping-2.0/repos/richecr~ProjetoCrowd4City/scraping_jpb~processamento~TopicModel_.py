import nltk
import spacy
import gensim
import numpy as np
import pandas as pd
from nltk.stem import *
from nltk.stem.porter import *
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from gensim import corpora, models
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess, deaccent
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis.gensim

# CONFIGURAÇÕES DE BIBLIOTECAS.
np.random.seed(2018)
nltk.download('wordnet')
nlp = spacy.load('pt_core_news_sm')

# CARREGANDO OS DADOS.
dados = pd.read_csv("./textos_limpos.csv")
dados.drop_duplicates(['texto'], inplace=True)
textos = dados['texto']
# print(textos[:5])


# PRÉ-PROCESSAMENTO DOS DADOS.

# Chamando a função de pré-processamento para cada texto.
processed_docs = dados['texto'].map(lambda texto: texto.split())
print(processed_docs[:10])

# Criando dicionário de palavras.
dictionary = gensim.corpora.Dictionary(processed_docs)

# Gensim Filter Extremes
# Filtrar tokens que aparecem em menos de 15 documentos
# ou em mais de 0.5 documentos(fração do tamanho total do corpus)
# Após essas duas etapas, mantenha apenas os 100000
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

# Bag of Words(Saco de Palavras).
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# Usando TF-IDF.
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

# Criando e treinando o modelo.
lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=4, id2word=dictionary, passes=10, workers=4)
# lda_model_tfidf.save("./modelo/meu_lda_model")

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model_tfidf, corpus_tfidf, dictionary=lda_model_tfidf.id2word)
vis

# Verificando o 'coherence score' para avaliar a qualidade dos tópicos aprendidos.
def coherence_model(lda_model_, processed_docs, corpus_tfidf, dictionary):
	coherence_model_lda = CoherenceModel(model=lda_model_, texts=processed_docs, corpus=corpus_tfidf, dictionary=dictionary, coherence='c_v')
	coherence_lda = coherence_model_lda.get_coherence()
	print('\nCoherence Score LDAModelTfIdf: ', coherence_lda)
coherence_model(lda_model_tfidf, processed_docs, corpus_tfidf, dictionary)


# Gráficos de tópicos mais discutidos nos documentos.
# Sentence Coloring of N Sentences
def topics_per_document(model, corpus, start=0, end=1):
		corpus_sel = corpus[start:end]
		dominant_topics = []
		topic_percentages = []
		for i, corp in enumerate(corpus_sel):
			topic_percs = model[corp]
			dominant_topic = sorted(topic_percs, key = lambda x: x[1], reverse=True)[0][0]
			dominant_topics.append((i, dominant_topic))
			topic_percentages.append(topic_percs)
		return(dominant_topics, topic_percentages)

def grafico_topc_docs():
	dominant_topics, topic_percentages = topics_per_document(model=lda_model_tfidf, corpus=corpus_tfidf, end=-1)

	# Distribution of Dominant Topics in Each Document
	df = pd.DataFrame(dominant_topics, columns=['Document_Id', 'Dominant_Topic'])
	dominant_topic_in_each_doc = df.groupby('Dominant_Topic').size()
	df_dominant_topic_in_each_doc = dominant_topic_in_each_doc.to_frame(name='count').reset_index()

	# Total Topic Distribution by actual weight
	topic_weightage_by_doc = pd.DataFrame([dict(t) for t in topic_percentages])
	df_topic_weightage_by_doc = topic_weightage_by_doc.sum().to_frame(name='count').reset_index()

	# Top 3 Keywords for each Topic
	topic_top3words = [(i, topic) for i, topics in lda_model_tfidf.show_topics(formatted=False) 
									for j, (topic, wt) in enumerate(topics) if j < 3]

	df_top3words_stacked = pd.DataFrame(topic_top3words, columns=['topic_id', 'words'])
	df_top3words = df_top3words_stacked.groupby('topic_id').agg(', \n'.join)
	df_top3words.reset_index(level=0,inplace=True)

	from matplotlib.ticker import FuncFormatter
	# Plot
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=120, sharey=True)

	# Topic Distribution by Dominant Topics
	ax1.bar(x='Dominant_Topic', height='count', data=df_dominant_topic_in_each_doc, width=.5, color='firebrick')
	ax1.set_xticks(range(df_dominant_topic_in_each_doc.Dominant_Topic.unique().__len__()))
	
	tick_formatter = FuncFormatter(lambda x, pos: 'Topic ' + str(x+1)+ '\n' + df_top3words.loc[df_top3words.topic_id==x, 'words'].values[0])
	ax1.xaxis.set_major_formatter(tick_formatter)
	ax1.set_title('Number of Documents by Dominant Topic', fontdict=dict(size=10))
	ax1.set_ylabel('Number of Documents')
	ax1.set_ylim(0, 1000)

	# Topic Distribution by Topic Weights
	ax2.bar(x='index', height='count', data=df_topic_weightage_by_doc, width=.5, color='steelblue')
	ax2.set_xticks(range(df_topic_weightage_by_doc.index.unique().__len__()))
	ax2.xaxis.set_major_formatter(tick_formatter)
	ax2.set_title('Number of Documents by Topic Weightage', fontdict=dict(size=10))

	plt.show()
# grafico_topc_docs()

# Imprimir os tópicos do modelo.
def imprimir_topicos():
	for topic in lda_model_tfidf.print_topics(-1, 15):
		print(topic)
		print("-----------")
# imprimir_topicos()

# Testes simples
def testes():
	# Deve ser sobre saneamento.
	print("-------")
	unseen_document = 'calendário JPB aqui nas nossas telas nós vamos agora até o bairro Jardim Paulistano zona sul de Campinas Você lembra que nossa equipe ouviu os moradores da Rua Riachuelo que reclamavam da falta de calçamento no local então o problema foi resolvido só que na época a prefeitura também se comprometeu e fazer o calçamento da Rua Ariel que fica bem pertinho essa parte foi feita mas só que pela metade Laisa grisi foi conferido calendário JPB desembarcou aqui no Jardim Paulistano E olha que maravilha hoje é possível andar na rua com calçamento sem tanta poeira sem pisar em lama Quando chove essa foi uma conquista dos moradores junto com calendário Desde o ano passado em 2015 quando a prefeitura calçou essa rua calça com a Rua Riachuelo também mas presta atenção dois passos seguintes e rua de terra essa rua que esse trechinho não foi calçado vou aqui conversar com os moradores já tá todo mundo reunido Por que me explica como é que pode só esse trechinho não foi calçada só esse trecho você imagina que fizeram as duas por duas partes né fizeram aquela parte de lá aí ficou a metade depois fizeram essa daqui aí deixar essa parte aqui sem sem tá feita né nessa parte de baixo é pior ainda porque quando chove a água invade a Casa dos moradores e olha só aqui nessa casa foi colocado um monte de pedra bem na frente para impedir que a água entre vamos lá falar com ela é dona Severina é dona Bill Olá tudo bom com a senhora como é que tá aqui essa situação a senhora Teve que colocar pedra aqui né é chover em entrar aqui sozinha imagina aperreio Aí tem que dar um jeito aqui é pior difícil hein dona Bill quanto tempo já que a senhora mora aqui nessa rua 8 anos viu o resultado de vergonha né a gente não tem né É porque se ele tivesse vergonha ele já tinha feito isso todos vocês moram aqui nessa rua aí o que que acontece nessas ruas aqui né aí o que que acontece a Rua Areal lá em cima Foi calçada a Rua Riachuelo também E vocês ficaram só um gostinho só na saudade e o pior que não se desviar da Lama dos buracos e ele prometeu Então olha você tá vendo aquela cerâmica Vale Aí depois ele dá o que é o povo que bota para que ele possa passar infelizmente é uma situação difícil a gente já pediu muitas vezes recado dado essa essa rua que já é assunto do calendário a gente conseguiu algumas ruas outras não voltamos em 2016 em 2016 o secretário André agra secretário de obras de Campina Grande e disse que ia voltar aqui não foi então vamos lá calendário novo quem é o representante'
	bow_vector = dictionary.doc2bow(pre_processamento(unseen_document))
	for index, score in sorted(lda_model_tfidf[bow_vector], key=lambda tup: -1*tup[1]):
		print("Score: {}\t Topic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))

	# Deve ser sobre saneamento.
	print("-------")
	unseen_document = 'Rua Rubens Coelho Pereira Filho nuclear para tecido calçada mais a placa com os dados da obra a gente simplesmente sumiu e o calçamento não chegou para completar uma empresa privada abrir um buraco no meio da rua e já caiu gente até roda de carro os moradores procuraram a prefeitura mais até agora nada mudou e ele só tiveram uma alternativa pediram ajuda ao calendário JPB a Rua Rubens Coelho Pereira Filho aqui no bairro do Cuiá está em obras agora É uma pena que essa obra que não tem nada a ver com a necessidade e com a reclamação da rua inteira que como a gente vai ver olha só não tem calçamento e também não tem saneamento básico são mais de 15 anos que os moradores tem muitos transtornos Principalmente nos períodos de chuva e a gente vai começar a conversar com alguns deles para falar de como é viver nessa situação seu Matusalém muito sofrimento aqui bom dia bom dia a todos e muito sofrimento a 14 anos que eu moro aqui e nenhuma infraestrutura foi foi feita aqui no Parque da prefeitura essa empresa veio para fazer a obra ela tá passando uns fazendo saneamento mas é de um condomínio particular passou pela rua de vocês é isso exatamente que a gente não tem nada não vai se beneficiar e nada do que essa consultora está fazendo inclusive teve transtorno para gravar afundando a gente tem foto muito transtorno carro afundando pessoas aqui já caíram aqui dentro do buraco já saíram todas arranhadas em Popular né sem comunicação sem nada a nossas calçadas foram invadidas com barro e também muita lama por causa da obra que passou aqui e também da chuva que é um pouco em madeirado vai ser descendo água estão aparecendo E aí você imagina era só no meio de todo esse material para pessoa está caminhando criança e doso para daqui não tem jeito se a luz é difícil andar nessa rua os buracos demais e a gente tem eu tenho dificuldade de locomover para continuo no caminhar na rua fico mais dentro de casa com medo de sair na rua com medo de cair de acontecer o pior comigo eu sair agora da Rua Rubens Coelho Pereira Filho que é aquela lá para vir aqui na Rita Carneiro porque que eu tô aqui nesse local tem uma placa que fala da pavimentação da Rua Rita carneiro e também um trecho da Rubens Coelho Pereira Filho mas segundo os moradores tá aqui o Fernando para falar alguns anos a pavimentação que tinha nessa placa ou numa placa parecida era só da Rubens Coelho e não chegou de forma alguma lá foi É isso mesmo uma placa anterior ela é indicada que seria metade da Rua Rubens Coelho pelo qual motivo eu não sei mesmo moradores foi modificado essa placa com outro valor e a nossa rua até hoje está aí do jeito que você mostrou na sua reportagem E aí eu vi a conclusão da pavimentação da Rita Carneiro'
	bow_vector = dictionary.doc2bow(pre_processamento(unseen_document))
	for index, score in sorted(lda_model_tfidf[bow_vector], key=lambda tup: -1*tup[1]):
		print("Score: {}\t Topic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))

	# Deve ser sobre obras.
	print("-------")
	unseen_document = "Jaguaribe a Escola Estadual Professora Maria do Carmo de Miranda já mudou muito desde a chegada do calendário por lá mas falta resolver um pequeno detalhe mas os estudantes não poderão utilizar é por isso que o calendário JPB continua aqui na Escola Estadual Professora Maria do Carmo de Miranda mesmo depois da obra de reforma tem sido concluída a gente precisa ter a garantia que esse espaço vai ser utilizado Falta muito pouco mas essa história ainda não acabou apresento para vocês agora o laboratório de biologia aqui eu tô vendo que tem aquelas maquetes do corpo humano do outro lado tem microscópios e outros equipamentos as outras estantes também estão cheias de máquinas todo esse material na nossa última visita aqui no início do mês de março tava encaixotado uma poeira danada e eu tô vendo que agora tá tudo no seu devido lugar aparentemente pronto para usar tá tudo certo para aula agora até agora ainda não foi inaugurado né você não tem fé que Vamos inaugurar atualizado É verdade que nem os professores ainda foram apresentados É verdade aos Laboratórios eles ficaram surpresos né porque antes não era assim Foi de repente ficou tudo arrumadinho e os professores não tinha nem noção dos equipamentos dos laboratórios tá todo mundo conhecendo hoje o novo laboratório de biologia né porque agora vai dar para chamar de laboratório né quando a gente entra a gente já tem a sensação de que está mesmo no laboratório que que tá faltando para que os estudantes possam utilizar esse espaço falta de instalações do ar condicionado só falta ar condicionado para funcionar e futuramente né daqui daqui uns dias nós estamos recebendo o quê da robótica os professores já vão fazer uma formação para atuar nessa área inclusive Nós temos dois alunos que foram representar Paraíba no robótica na China e a gente sabe que essa escola tem muito potencial da área eles vão ser monitores essa escola realmente tem muitos talentos item acima de tudo né estudantes interessados em usar esses esses passos para aprender Então hoje dia 14 de abril o carimbo Ainda é em andamento nesse segundo a secretaria de educação do estado esses ar-condicionados daqui dos laboratórios vão ser instalados no prazo de 15 dias com 15 dias de instalação mas 15 dias Os estudantes de gestão por aqui tá tudo certo certo dia 14 de Maio cai no sábado né o próximo dia útil seria o dia 16 mas aí a Paloma que começou essa história não vai poder estar aqui no dia 16 só no dia 19 essa data diretora do dia 19 Fica boa para senhora fica"
	bow_vector = dictionary.doc2bow(pre_processamento(unseen_document))
	for index, score in sorted(lda_model_tfidf[bow_vector], key=lambda tup: -1*tup[1]):
		print("Score: {}\t Topic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))
	
	# Deve ser sobre obras.
	print("-------")
	unseen_document = "o calendário JPB faz hoje a sua quinta visita à Praça Heitor Cabral de Lucena no Castelo Branco aqui na capital diga logo que as coisas não estão arrumadas mas avançou a realidade da praça Heitor Cabral deles é aqui no Castelo Branco não mudou continua descendo dourada com acúmulo de lixo mato crescendo às vezes os moradores que fazem uma limpeza mas a esperança de que tudo isso mude Aí sim essa tá aumentando porque olha só estamos com o projeto oficial da reforma da praça uma decisão do prefeito que esta Praça também entre no calendário de requalificações qual é a diferença do que foi apresentado no mês passado para os moradores para esse projeto aqui que vocês trouxeram a diferença é que a diretoria de paisagismo da sedurb já concluiu por definitivo concluiu todo o projeto paisagístico orçamentário e também com seu Memorial Botânico nós encaminhamos a secretaria de planejamento Secretaria de planejamento agora vai fazer a parte orçamentária da parte física da Praça do seu passeio da construção da quadra que é um desejo dos moradores da iluminação da parte física inseri a academia de atividade física para a comunidade e abrir o processo licitatório para só assim depois de todo o processo licitatório ser concluído a prefeitura assina ordem de serviço para execução dos serviços de requalificação da praça Heitor Cabral de ureia Quando é que a solicitação fica pronta na Secretaria de planejamento vai abrir o processo licitatório e isso demora porque isso tem a questão dos prazos entre 3 a 6 meses Dona Lurdes olha só aqui onde tem aqui essas calçadas é por onde a senhora vai caminhar todos os dias tá acreditando acreditando Acreditando tudo que o secretário de paisagismo falou que a TV Cabo Branco 100 São maravilhosos todos os meus estão aqui falta começar o trabalho né O Alisson junto com os amigos aqui também são atores que trabalham com cultura tem uma afinidade especial principalmente com a criançada né e ventou brincadeiras né Não sei se movimentando por aqui por enquanto ainda não tá muito bom para fazer isso tá difícil a gente tentou no começo do ano tá fazendo alguma atividade com as crianças acontece no dia das mães no Dia das Crianças algumas atividades porém por causa do sinto-lhes os bichos muito escorpião aí fica até perigoso trabalhar com as crianças aqui unidade que está se juntando tá varrendo tá fazendo a Paisagismo tava juntando infelizmente prefeitura não vem buscar um monte de lixo a gente liga para ele buscar E não busca Nós já chegamos aqui e já fizemos algumas intervenções uma uma poda e todas as aves fizemos levantamento de copa fizemos uma limpeza na no relvado da praça"
	bow_vector = dictionary.doc2bow(pre_processamento(unseen_document))
	for index, score in sorted(lda_model_tfidf[bow_vector], key=lambda tup: -1*tup[1]):
		print("Score: {}\t Topic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))
	
	# Deve ser sobre obras.
	print("-------")
	unseen_document = "calendário hoje no bairro de Mangabeira que na capital uma parceria Nossa com a comunidade que já dura um ano gente hoje tem notícia boa calma não é resolvido mas é um avanço muito importante Karine Tenório vai contar contar para a gente o tamanho do abandono desse ginásio corresponde ao tamanho da importância que ele tem para comunidade o calendário JPB acompanha essa história há quase um ano e nada muda hoje nós temos uma missão importante aqui mostrar essa situação para o novo secretário de esportes da Prefeitura e o secretário pontualíssimo chegou aqui para conversar conosco e com a comunidade Houve um problema na primeira licitação todos de consciência de todos mas o processo licitatório está dentado notícia boa para vocês foi feito já passou o prazo de recurso agora foi homologada falta alguns trâmites né Para a gente dar início a obra tá tão sonhada a obra e a Nistatina sonhada e é urgente né é sonhar que é urgente porque eu vou dar um exemplo aqui secretário na nossa última visita a gente já percebe daqui de fora que as telhas do ginásio já estão voando né Isso é um risco para comunidade outro risco que a gente observou essa guarita aqui secretário que o senhor se aproximar aqui onde essas crianças estão agora ela foi completamente destruída por vândalos e é um espaço hoje de insegurança para os moradores Por que alguns criminosos usam esse espaço para usar drogas para prostituição e até o mau cheiro grande né porque eles usam isso para banheiro para tudo enfim se transformou no local de muita insegurança para vocês né e a gente fica com medo de passar aqui a gente sabe que a proposta desse ginásio é totalmente o contrário né Deve tá droga evitar prostituição é trazer cultura e esporte para comunidade hoje à tarde vou passar com prefeito e vamos colocar novamente a aceleração dessa obra o mais breve possível o interesse dele já tá aqui já é um primeiro passo com certeza eu acredito que o Luciano Cartaxo logicamente ele vai ver se aqui porque realmente é o que eu cuido das Crianças com certeza que nós vamos conseguir esse essa construção que essa reconstrução dos ginásio esse ginásio aqui ele foi muito importante dos jogos escolares era lotado lotado deixaram de fazer por causa disso aí que a gente vir aqui assistir e ele vinha todo pessoal dos Estados vizinhos e a gente gostava de alguma coisa para fazer aqui hoje a gente não tem seu Zezinho Você tem quantos anos 76 anos é seu Zezinho olha secretário seu Zezinho ele tem dificuldade de locomoção mas tá aqui com a Bengala dele né seu Zezinho 76 anos se Acordou cedo para vir aqui porque ele também sonha"
	bow_vector = dictionary.doc2bow(pre_processamento(unseen_document))
	for index, score in sorted(lda_model_tfidf[bow_vector], key=lambda tup: -1*tup[1]):
		print("Score: {}\t Topic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))

	# Deve ser sobre obras.
	print("-------")
	unseen_document = "a gente continua nossa parceria forte com a comunidade é hora do calendário JPB de hoje a gente vai saber como anda a obra do ginásio de esportes no bairro de Tambaú aí o negócio andou pessoal começou a maior parece que brecou e o Danilo Alves vai explicar para gente passa por aqui sempre sobra nessa quadra aqui que não termina nunca vi nada aqui mesmo aqui tá um bocado de tempo nunca vi você Bastião saiu seu Sebastião chegou seu Regis das outras vezes que a gente veio aqui entre a obra caminhando o funcionário voltar pessoa trabalhando eu tava mais animado tava não com certeza tava muito animada aí depois que vocês saíram primeiro pessoal sumiu todo mundo dona Ângela mês de junho festa junina Júlio tem o quê Julho férias férias da criançada E aí mais umas férias mais um ano de férias e Acorda desse jeito do mesmo jeito estou perdendo as esperanças não faz isso vai dar certo que Deus é mais minha filha tudo bem Toda hora joga bola direitinho mais ou menos o que é ruim aceita eu Falta muito pouco para ser concluído infelizmente abandonado você ver portão fechado e nada se faz a última vez que a gente vê aqui o pessoal tava trabalhando para colar essas grades aí colocaram grade de uma ponta a outra aqui do ginásio colocou o portão para evitar também que as pessoas fica invadindo o portão tá aqui o problema é que parou por aí que é que aconteceu ontem a gente tava tudo animado tava andando bem direitinho tava na expectativa de pelo menos iniciar o serviço lá no banheiro porque hoje pois é a gente ainda volta a bater na tecla do da disponibilidade de recursos né a gente alimenta o sistema do FNDE mas ainda dentro desse prazo de 30 dias não tivemos respostas a disponibilidade de recursos próprios que também serve para o pagamento da quadra ainda é escassa devida a obrigatoriedade de pagar a folha de pagamento que são as despesas obrigatórias do município e aí aquele efeito do lençol curto se cobrir a cabeça descobre os pés e cinco meus pés descobre a cabeça mas a gente ainda teve alguns encaminhamentos que foi a solicitação da água e energia para quadra né como também a gente finalizou a proteção para que os pessoal não não voltassem invadir e de pedra' o prédio que em algum em alguns pontos já tem algo entregue a gente pede novamente prazo para que o efeito nos traz"
	bow_vector = dictionary.doc2bow(pre_processamento(unseen_document))
	for index, score in sorted(lda_model_tfidf[bow_vector], key=lambda tup: -1*tup[1]):
		print("Score: {}\t Topic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))
# testes()

# Coherence Score: Avaliar a qualidade dos tópicos aprendidos.
def compute_coherence_values(dct, corpus_tfidf, texts, limit, start, step):
	coherence_values = []
	model_list = []
	for num_topics in range(start, limit, step):
		model = gensim.models.LdaMulticore(corpus=corpus_tfidf, num_topics=num_topics, id2word=dct, passes=10, workers=4)
		model_list.append(model)
		coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dct, corpus=corpus_tfidf , coherence='c_v')
		coherence_values.append(coherencemodel.get_coherence())

	return model_list, coherence_values

def grafico_para_achar_melhor_n_topics():
	model_list, coherence_values = compute_coherence_values(dictionary, corpus_tfidf, processed_docs, 10, 4, 1)
	limit=10; start=4; step=1;
	x = range(start, limit, step)
	plt.plot(x, coherence_values)
	plt.xlabel("Num Topics")
	plt.ylabel("Coherence score")
	plt.legend(("coherence_values"), loc='best')
	plt.show()

	for m, cv in zip(x, coherence_values):
		print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
# grafico_para_achar_melhor_n_topics()
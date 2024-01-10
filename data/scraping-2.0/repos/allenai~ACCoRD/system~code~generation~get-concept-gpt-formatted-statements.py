import pickle
import pandas as pd
from operator import itemgetter
import numpy as np
import openai
from rouge_score import rouge_scorer
import spacy
import re
import seaborn as sns
import matplotlib.pyplot as plt

#--------------------------------------------------------
openai.api_key = "" 
# 4class: ['compare' 'is-a' 'part-of' 'used-for']
# 3class: ['compare', contrast, 'is-a']
sum_len = 0
selected_nlp_concepts = ['adversarial training', 'beam search', 'bert', 'elmo', 'gpt', 'glove', 'word2vec', 'resnet', 'domain shift', 'ulmfit', 'newsqa', 'squad', 'random forest', 'imagenet', 'lstm', 'roberta', 'variational autoencoder', 'dropout', 'fasttext', 'hierarchical softmax', 'distant supervision']
relations = ['compare','isa']
#--------------------------------------------------------
# unpickle concept dictionary
with open('/net/nfs2.s2-research/soniam/concept-rel/resources/forecite/forecite_concept_dict.pickle', 'rb') as handle:
    concept_dict = pickle.load(handle)
    print("...unpickled forecite concept dictionary")

# Initialize N
N = 50
# N largest values in dictionary
# Using sorted() + itemgetter() + items()
res = dict(sorted(concept_dict.items(), key = itemgetter(1), reverse = True)[:N])
# printing result
# print("The top N value pairs are " + str(res))
concept_list = list(res.keys())
# print(concept_list)
#--------------------------------------------------------
df_nlp_concepts = pd.read_csv("/net/nfs2.s2-research/soniam/concept-rel/abstractive-summarization/gpt-generations/top-n-forecite-concepts-gpt-generations/nlp_concepts.csv")
nlp_concept_scores = []
for index, row in df_nlp_concepts.iterrows():
    # get forecite score for each nlp concept
    nlp_concept_scores.append(concept_dict[row['concept']])

df_nlp_concepts['score'] = nlp_concept_scores
df_nlp_concepts = df_nlp_concepts[~df_nlp_concepts.concept.isin(['bert model', 'pre - train bert model', 'pre - train bert', 'moses', 'cho et al', 'dagan', 'yarowsky', 'hochreiter', 'turney', 'ney', 'och', 'grosz', 'steedman', 'well translation'])]

# # top 150 nlp concepts by score
# top150_nlp_concepts = df_nlp_concepts.sort_values(by='score', ascending=False)[:150]['concept'].tolist()
# top 150 nlp concepts by num_papers
top150_nlp_concepts = df_nlp_concepts.sort_values(by='num_papers', ascending=False)[:150]['concept'].tolist()
print(top150_nlp_concepts)

#--------------------------------------------------------
#--------------------------------------------------------
# set up spacy
nlp = spacy.load("en_core_web_md")
tokenizer = nlp.tokenizer
nlp.add_pipe("sentencizer")
all_stopwords = nlp.Defaults.stop_words

def getConceptB(generation, concept):
    # get concept B by selecting the second noun chunk
    doc = nlp(generation)
    noun_chunks = list(doc.noun_chunks)

    # if the generation is not valid: make conceptb an empty string
    if (len(noun_chunks) <= 1) or (generation == " variable HYDRA_FULL_ERROR=1 for a complete stack tra"):
        print("concept b not found")
        concept_b = ""
        # conceptbs.append("")
    # if the generation is valid and "used-for":
    elif re.search('<ENT> is used (to|for)', generation):
        find_used = re.search('used (to|for)', generation)
        end = find_used.span()[1] # get index of end of "is"
        concept_b = generation[end+1:]
        # conceptbs.append(generation[end+1:]) # index into generation with these indices, removing leading and trailing spaces
    
    # if the generation is valid and "compare":
    elif re.search('is like(.*)in that they are both', generation):
        find_is = re.search('is like', generation)
        find_that = re.search('in that they', generation)
        end = find_is.span()[1] # get index of end of "is"
        start = find_that.span()[0] # get index of beginning of "that"
        concept_b = generation[end+1:start-1]
        # conceptbs.append(generation[end+1:start-1]) # index into generation with these indices, removing leading and trailing spaces

    # if the generation is valid and "compare":
    elif re.search('is an alternative (to|for)(.*)that', generation):
        find_is = re.search('is an alternative (to|for)', generation)
        find_that = re.search('that', generation)
        end = find_is.span()[1] # get index of end of "is"
        start = find_that.span()[0] # get index of beginning of "that"
        concept_b = generation[end+1:start-1]
        # conceptbs.append(generation[end+1:start-1]) # index into generation with these indices, removing leading and trailing spaces

    # if the generation is valid and "used-for":
    elif re.search('is (a|an)(.*)that', generation):
        find_is = re.search('is (a|an)', generation)
        find_that = re.search('that', generation)
        end = find_is.span()[1] # get index of end of "is"
        start = find_that.span()[0] # get index of beginning of "that"
        concept_b = generation[end+1:start-1]
        # conceptbs.append(generation[end+1:start-1]) # index into generation with these indices, removing leading and trailing spaces

    # all other relation types
    else:
        concept_b = ""
        # iterate through noun chunks to find first valid one
        for chunk in noun_chunks:
            if ("ENT" in chunk.text) or ("type" in chunk.text) or (concept in chunk.text):
                continue
            else:
                concept_b = chunk
                break
        
        # check if concept_b is a string, because it has no .text parameter
        if isinstance(concept_b, str):
            print(concept_b)

        # if the first word of this noun phrase is "a" or "an", select all words past that word
        else:
            if (concept_b[0].text == "a") or (concept_b[0].text == "an") or (concept_b[0].text == "the"):
                concept_b = concept_b[1:]
            
            # even if the concept didn't have an article or anything, save it's .text version
            concept_b = concept_b.text
            
    # remove any punctuation or special characters from final value appended
    # concept_b = re.sub('[^A-Za-z0-9- ]+', '', concept_b)

    return concept_b

#--------------------------------------------------------
def getGPTOutput(j, best_sentences, best_formatted_statements, df_supplementary_sentences, relation, generations, conceptbs, is_conceptb_forecite, conceptb_forecite_score, rouge1, rouge2, rougeL, scorer, relations):
    # print("GENERATIONS FOR %s CLASS" % relation)
    prompt = "Describe the provided concept in terms of another concept in the text. \n\n"
    # prompt examples
    for i in range(len(best_sentences)):
        text = "Text: " + best_sentences[i] + "\n"
        concept_description = "Concept description: " + best_formatted_statements[i] + "\n"
        separator = "###" + "\n"
        prompt += text + concept_description + separator
   
    prompt_examples = prompt
    # supplementary sentence we want a generation for
    row = df_supplementary_sentences.iloc[j]
    sentence = row['original_sentence']
    concept = row['forecite_concept']
    # recover the original case of the sentence from the undemarcated dataframe by searching for the lowercase sentence
    # sentence_original_case = df_all_rows_original_case.loc[df_all_rows_original_case['sentence'] == sentence]['sentence_original_case'].item()

    prompt += "Text: " + sentence + "\n"
    prompt += "Concept description: " + concept
    if relation == "compare":
        prompt += " is like"
    elif relation == "used-for":
        prompt += " can be used for"
    elif relation == "isa":
        prompt += " is a"
    elif relation == "part-of":
        prompt += " is a part of"

    # print(prompt)
    response = openai.Completion.create(engine="davinci-instruct-beta", prompt=prompt, max_tokens=300, temperature=0, frequency_penalty=1, echo=True, stop="\n")
    
    # print(response['choices'][0]['text'])
    generation = response['choices'][0]['text'].splitlines()[-1]
    generation = generation[21:]
    conceptb = getConceptB(generation, concept)

    generations.append(generation)
    scores = scorer.score(sentence, generation)
    rouge1.append(scores['rouge1'][2])
    rouge2.append(scores['rouge2'][2])
    rougeL.append(scores['rougeL'][2])
    conceptbs.append(conceptb)
    relations.append(relation)

    if conceptb in concept_dict.keys():
        is_conceptb_forecite.append(1)
        conceptb_forecite_score.append(concept_dict[conceptb])
    else:
        is_conceptb_forecite.append(0)
        conceptb_forecite_score.append("")


    print(sentence)
    print("--> " + generation)
    print("--> " + conceptb)
    # print(scores['rouge1'][2])
    # print(scores['rouge2'][2])
    print(scores['rougeL'][2])
    print(row['max_multilabel_pred_score'])
    print("")

#----------------------------------------------------------
def getGPTGenerationsForRelation(relation, concept, df_annotations, best_sentences, best_formatted_statements):
    annotations_generations = []
    rouge1 = []
    rouge2 = []
    rougeL = []
    conceptbs = []
    is_conceptb_forecite = []
    conceptb_forecite_score = []
    relations = []

    # convert prompt examples to lowercase
    best_sentences = list(map(lambda x: x.lower(), best_sentences))
    best_formatted_statements = list(map(lambda x: x.lower(), best_formatted_statements))
    # remove sentences that are in prompt examples
    df_annotations = df_annotations[~df_annotations['original_sentence'].isin(best_sentences)]
    print(df_annotations)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    for a in range(len(df_annotations)):
        getGPTOutput(a, best_sentences, best_formatted_statements, df_annotations, relation, annotations_generations, conceptbs, is_conceptb_forecite, conceptb_forecite_score, rouge1, rouge2, rougeL, scorer, relations)


    df_annotations['gpt_generation'] = annotations_generations
    df_annotations['concept_b'] = conceptbs
    df_annotations['is_conceptb_forecite'] = is_conceptb_forecite
    df_annotations['conceptb_forecite_score'] = conceptb_forecite_score
    df_annotations['relation'] = relations
    df_annotations['rouge1'] = rouge1
    df_annotations['rouge2'] = rouge2
    df_annotations['rougeL'] = rougeL
    print(len(df_annotations))

    df_annotations.to_csv("/net/nfs2.s2-research/soniam/concept-rel/abstractive-summarization/gpt-generations/top-n-forecite-concepts-gpt-generations/%s/gpt-formatted-statements-all-s2orc-%s-%s.csv" % (version, concept, relation))

#----------------------------------------------------------
best_isa_sentences = ["although word2vec has successfully been used to learn word embeddings, these kinds of word embeddings capture only co-occurrence relationships between words (levy and goldberg, 2014) .",
"there currently exist many software tools for short read alignment, including soap2 [15] , bwa [14] , and bowtie2",
"this paper presents the spatial hypertext wiki as a collaborative tool for supporting creativity in the re process.",
"cnn architecture consists of several types of layers including convolution, pooling, and fully connected. the network expert has to make multiple choices while designing a cnn such as the number and ordering of layers, the hyperparameters for each type of layer (receptive field size, stride, etc.).",
"nodetrix representations are a popular way to visualize clustered graphs; they represent clusters as adjacency matrices and intercluster edges as curves connecting the matrix boundaries. we study the complexity of constructing nodetrix representations focusing on planarity testing problems, and we show several np-completeness results and some polynomial-time algorithms.",
"unlike traditional word embeddings that represent words with fixed vectors, these embedding models encode both words and their contexts and generate context-specific representations. while contextualized embeddings are useful, we observe that a language model-based embedding model, elmo (peters et al., 2018) , cannot accurately capture the semantic equivalence of contexts."]

best_isa_formatted_statements = ["word2vec is a word embedding that captures only co-occurrence relationships between words.",
"soap2 is a software tool for short read alignment.",
"Spatial hypertext wiki is a collaborative tool that is used for supporting creativity in the re process.",
"stride is a hyperparameter for layers in a cnn.",
"nodetrix representations are a way to visualize clustered graphs that represent clusters as adjacency matrices and intercluster edges as curves connecting the matrix boundaries. ",
"elmo is a language model-based embedding model that cannot accurately capture the semantic equivalence of contexts."]
#----------------------------------------------------------
# best_compare_sentences = ["Recently, many NLP tasks have shown great improvements thanks to pre-trained language models. Models such as ELMo [10] , BERT [3] and GPT [11] include a large language model or contextual embedder with hundreds of millions of parameters trained on large datasets and were shown to produce state-of-the-art models for many NLP tasks, including low-resource scenarios where very little annotated data is available.",
# "OrthoDisease (9) and PhenomicDB (8, 26) are two other resources that allow researchers to look simultaneously at all available phenotypes for an orthologous gene group. The PhenomicDB and OrthoDisease are useful resources integrating the phenotypes with the homologous genes from a variety of species.",
# "Highway Networks [7] and Residual Networks [1] are the latest methods to tackle this problem. These network architectures introduce skip connections, which allow flow of information to later layers, empowering deeper networks with better accuracy.",
# "MCG is affected less by conductivity variations in the body (lungs, muscles, and skin) than ECG. In addition, because MCG is a fully non-contact method, therefore problems in the skinelectrode contact as encountered in ECG are avoided (Kanzaki et al., 2003; Tsukada et al., 1999; Tavarozzi et al., 2002) .",
# "In contrast to Pwrake and GXP Make, Snakemake does not rely on any password-less SSH setup or custom server processes running on the cluster nodes. Finally, Snakemake is the first system to support file name inference with multiple named wildcards in rules.",
# "the multidimensional frequencies of a single component is treated as a whole, and the probability density function is projected as independent univariate von mises distribution to perform tractable inference."]
# # alternate compare: "We refer the reader to [20] for a mathematical formulation of multilayer networks, of which multiplex networks are a subset. Unlike a multilayer network, a multiplex network only allows for a single type of inter-layer connections via which any given node is connected only to its counterpart nodes in the other layers."

# best_compare_formatted_statements = ["ELMo is like BERT in that they are both pre-trained language models that include a large language model or contextual embedder with hundreds of millions of parameters trained on large datasets and were shown to produce state-of-the-art models for many nlp tasks, including low-resource scenarios where very little annotated data is available.",
# "PhenomicDB is like OrthoDisease in that they are both resources that allow researchers to look simultaneously at all available phenotypes for an orthologous gene group and are useful resources integrating the phenotypes with the homologous genes from a variety of species.",
# "Residual Networks are like Highway Networks in that they are both network architectures that introduce skip connections, which allow flow of information to later layers, empowering deeper networks with better accuracy.",
# "MCG is like ECG, except MCG is affected less by conductivity variations in the body (lungs, muscles, and skin).",
# "Snakemake is like Pwrake, except Snakemake does not rely on any password-less ssh setup or custom server processes running on the cluster nodes.",
# "reinforcement learning is like motivated learning except that reinforcement learning only has a single value function, relies only on externally set objectives, maximizes its reward (and is therefore unstable), and is always active."]

#----------------------------------------------------------
# REVISION FOR 3CLASS SETTING and v1-4CLASS: SEPARATE COMPARE AND CONTRAST EXAMPLES
#----------------------------------------------------------
best_compare_sentences = ["Recently, many NLP tasks have shown great improvements thanks to pre-trained language models. Models such as ELMo [10] , BERT [3] and GPT [11] include a large language model or contextual embedder with hundreds of millions of parameters trained on large datasets and were shown to produce state-of-the-art models for many NLP tasks, including low-resource scenarios where very little annotated data is available.",
"OrthoDisease (9) and PhenomicDB (8, 26) are two other resources that allow researchers to look simultaneously at all available phenotypes for an orthologous gene group. The PhenomicDB and OrthoDisease are useful resources integrating the phenotypes with the homologous genes from a variety of species.",
"Highway Networks [7] and Residual Networks [1] are the latest methods to tackle this problem. These network architectures introduce skip connections, which allow flow of information to later layers, empowering deeper networks with better accuracy.",
"similarly to ftg+pm, wires [41] supports the specification and execution of model transformation workflows.",
"like the nevanlinna-pick interpolation problem, the covariance extension problem has deep roots in the mathematical literature [2] and have numerous applications in various engineering fields, ranging from systems theory [3] , [4] to control design [5] and signal processing [6] ."]
# alternate compare: "We refer the reader to [20] for a mathematical formulation of multilayer networks, of which multiplex networks are a subset. Unlike a multilayer network, a multiplex network only allows for a single type of inter-layer connections via which any given node is connected only to its counterpart nodes in the other layers."

best_compare_formatted_statements = ["ELMo is like BERT in that they are both pre-trained language models that include a large language model or contextual embedder with hundreds of millions of parameters trained on large datasets and were shown to produce state-of-the-art models for many nlp tasks, including low-resource scenarios where very little annotated data is available.",
"PhenomicDB is like OrthoDisease in that they are both resources that allow researchers to look simultaneously at all available phenotypes for an orthologous gene group and are useful resources integrating the phenotypes with the homologous genes from a variety of species.",
"Residual Networks are like Highway Networks in that they are both network architectures that introduce skip connections, which allow flow of information to later layers, empowering deeper networks with better accuracy.",
"ftg+pm is like wires in that they both support the specification and execution of model transformation workflows.",
"covariance extension problem is like the nevanlinna-pick interpolation problem in that they are both problems that have deep roots in the mathematical literature and have numerous applications in various engineering fields, ranging from systems theory, to control design and signal processing."]

#----------------------------------------------------------

best_contrast_sentences = ["MCG is affected less by conductivity variations in the body (lungs, muscles, and skin) than ECG. In addition, because MCG is a fully non-contact method, therefore problems in the skinelectrode contact as encountered in ECG are avoided (Kanzaki et al., 2003; Tsukada et al., 1999; Tavarozzi et al., 2002) .",
"In contrast to Pwrake and GXP Make, Snakemake does not rely on any password-less SSH setup or custom server processes running on the cluster nodes. Finally, Snakemake is the first system to support file name inference with multiple named wildcards in rules.",
"in comparison to reinforcement learning, a motivated learning (ml) agent has multiple value functions, sets its own objectives, solves the minimax problem, is stable, and acts when needed. in contrast, a reinforcement learning (rl) agent typically only has a single value function, relies only on externally set objectives, maximizes its reward (and is therefore unstable), and is always active.",
"suitable data for the vqg task can come from standard image datasets on which questions have been manually annotated, such as v qg coco, v qg f lickr, v qg bing (mostafazadeh et al., 2016), each consisting of 5000 images with 5 questions per image. alternatively, vqg samples can be derived from visual question answering datasets, such as v qa1.0 (antol et al., 2015), by \"reversing\" them (taking images as inputs and questions as outputs).",
"uchime either uses a database of chimera-free sequences or detects chimeras de novo by exploiting abundance data. uchime has better sensitivity than chimeraslayer (previously the most sensitive database method), especially with short, noisy sequences.",
"partially labeled lda (plda) extends labeled lda to incorporate per-label latent topics (ramage et al., 2011)."]

best_contrast_formatted_statements = ["MCG is like ECG, except MCG is affected less by conductivity variations in the body (lungs, muscles, and skin).",
"Snakemake is like Pwrake, except Snakemake does not rely on any password-less ssh setup or custom server processes running on the cluster nodes.",
"reinforcement learning is like motivated learning except that reinforcement learning only has a single value function, relies only on externally set objectives, maximizes its reward (and is therefore unstable), and is always active.",
"v qg coco is like v qa1.0, except v qg coco is a standard image dataset and v qa1.0 is a visual question answering dataset.",
"uchime is like chimeraslayer except that it has better sensitivity, especially with short, noisy sequences.",
"partially labeled lda (plda) is like labeled lda, except partially labeled lda incorporates per-label latent topics."]
#----------------------------------------------------------
best_usedfor_sentences = ["In [24] , for example, the need to construct the cloaking regions and to receive the responses from the server through other users can considerably degrade the service. Many obfuscation-based techniques are based on k-anonymity, which has been shown inadequate to protect privacy [8] , [25] .",
"Recently, hashing methods have been widely used in ANN search. They usually learn a hamming space which is refined to maintain similarity between features Song et al., 2018c;",
"Perhaps even more promising and exciting, however, is recent work on using Reo for programming multicore applications. When it comes to multicore programming, Reo has a number of advantages over conventional programming languages, which feature a fixed set of low-level synchronization constructs (locks, mutexes, etc.).",
"Supercompilation is a program transformation technique that was first described by V. F. Turchin in the 1970s. In supercompilation, Turchin's relation as a similarity relation on call-stack configurations is used both for call-by-value and call-by-name semantics to terminate unfolding of the program being transformed.",
"Recently, convolutional networks have been used for automatic feature extraction of large image databases, where they have obtained state-of-the-art results. In this work we introduce EEGNet, a compact fully convolutional network for EEG-based BCIs developed using Deep Learning approaches."]
# alternate used-for: "Convolutional neural networks (CNN) in recommendation systems have been used to capture localized item feature representations of music [31] , text [16, 29] and images [40] . Previous methods represent text as bag-of-words representations, CNN overcomes this limitation by learning weight filters to identify the most prominent phrases within the text."

best_usedfor_formatted_statements = ["K-anonymity can be used for many obfuscation-based techniques, which has been shown inadequate to protect privacy.",
"Hashing methods can be used in ANN search to usually learn a hamming space which is refined to maintain similarity between features.",
"Reo can be used for programming multicore applications.",
"Turchin's relation can be used for call-by-value and call-by-name semantics to terminate unfolding of the program being transformed in supercompilation.",
"Convolutional networks can be used for automatic feature extraction of large image databases, where they have obtained state-of-the-art results."]

#----------------------------------------------------------
best_partof_sentences = ["The pose graph is constructed by using spatial human poses (black dots and lines), spatial object poses (red dots and lines), and temporal connections (blue lines). In spatial and temporal domains, the graph is used as the input to GCNs.",
"In fact, cosegmentations promise to be useful in other bioimaging (and eventually image processing) applications beyond cell tracking. One straightforward application where cosegmentation is of high relevance are protein colocalization studies.",
"Sparse coding problems assume the data y can be represented as a sparse linear combination of columns (features) of a matrix H, termed a dictionary. Given the dictionary H, methods such as orthogonal matching pursuit [1] and basis pursuit [2] find the sparse representation.",
"In this work, the graphs we consider have a special structure, in the form of a multiplex network, in the sense that each graph can be decomposed into a sequence of subgraphs, each of which corresponds to a layer of the network, and there exist interconnections linking nodes across different layers. We refer the reader to [20] for a mathematical formulation of multilayer networks, of which multiplex networks are a subset.",
"CNNs perform very well on any visual recognition tasks. The CNN architecture consists of special layers called convolutional layers and pooling layers."]

best_partof_formatted_statements = ["The pose graph is a part of spatial and temporal domains that is constructed by using spatial human poses (black dots and lines), spatial object poses (red dots and lines), and temporal connections (blue lines).",
"Cosegmentation is a part of protein colocalization studies.",
"Sparse linear combinations are a part of sparse coding problems that represents the data y in the columns (features) of a matrix h, termed a dictionary.",
"Multiplex networks are a part of multilayer networks, that can decompose each graph into a sequence of subgraphs, each of which corresponds to a layer of the network, and there exist interconnections linking nodes across diferent layers.",
"Pooling layers are part of CNN architectures."]
#----------------------------------------------------------
#----------------------------------------------------------

version = "v1-4class/nlp-concepts"

for concept in top150_nlp_concepts:
    df_concept = pd.read_csv("/net/nfs2.s2-research/soniam/concept-rel/abstractive-summarization/inputs/top-n-forecite-concept-source-sentences/%s/scibert-multilabel-binary-predictions-all-s2orc-with-forecite-%s.csv" % (version, concept))
    df_concept = df_concept.loc[:, ~df_concept.columns.str.contains('^Unnamed')]
    
    if len(df_concept) == 0:
        continue

    for index, relation in enumerate(relations):
        # select the rows that belong to the same relation
        df_concept_relation = df_concept[df_concept['%s_pred_score' % relation]>0]
        # drop duplicate source text
        df_concept_relation = df_concept_relation.drop_duplicates(subset=['original_sentence'])
        print(concept, relation, len(df_concept_relation))

        # generate for only top N extractions to minimize GPT costs
        top_N = 100
        if len(df_concept_relation) > top_N:
            # sort by max multilabel pred score and take the highest N
            df_concept_relation = df_concept_relation.sort_values('%s_pred_score' % relation, ascending=False)
            df_concept_relation = df_concept_relation[:top_N]

        print(len(df_concept_relation))

        # add to running total of number of descriptions
        sum_len+=len(df_concept_relation)

        if relation == "compare":
            best_sentences = best_compare_sentences
            best_formatted_statements = best_compare_formatted_statements
        elif relation == "contrast":
            best_sentences = best_contrast_sentences
            best_formatted_statements = best_contrast_formatted_statements
        elif relation == "isa":
            best_sentences = best_isa_sentences
            best_formatted_statements = best_isa_formatted_statements
        # elif relation == "part-of":
        #     best_sentences = best_partof_sentences
        #     best_formatted_statements = best_partof_formatted_statements
        # elif relation == "used-for":
        #     best_sentences = best_usedfor_sentences
        #     best_formatted_statements = best_usedfor_formatted_statements
        

        getGPTGenerationsForRelation(relation, concept, df_concept_relation, best_sentences, best_formatted_statements)



#----------------------------------------------------------
# EXTRA CODE (IGNORE)
#----------------------------------------------------------
# # for "compare" class,select things that are marked as positive compare_pred_score but negative contrast_pred_score
# if relation =="compare":
#     df_concept_relation = df_concept[df_concept['%s_pred_score' % relation]>0]
#     df_concept_relation = df_concept[df_concept['contrast_pred_score']<0]
# else:
#     # select rows with a positive prediction score because >0 == >0.5 probability
#     df_concept_relation = df_concept[df_concept['%s_pred_score' % relation]>0]

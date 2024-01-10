#import hyp_openai
from nltk.corpus import wordnet as wn

counter = 0


def extraction_done(df_new):
    # check if there is any empty hypernym left
    if all(df_new['hypernym']):
        return True
    else:
        return False


def get_hypernym(pairs):
    # extract hypernym for each pair and append each sublist with it
    # get hypernyms from openai and creating an empty string for hypernym
    for i in pairs:
        """
        hyp_0 = hyp_openai.call_openai(i[0]).split(',')
        hyp_2 = hyp_openai.call_openai(i[2]).split(',')

        hyp_common = ""
        # comparing openai hypernyms
        for j in range(len(hyp_0)):
            for k in range(len(hyp_2)):
                if hyp_0[j] == hyp_2[k]:
                    hyp_common = hyp_0[j]
                else:
                    hyp_common = 'Not found with openai!'
        """
        # if hypernym not found in openai, search in wordnet

        # if hyp_common == 'Not found with openai!':
        try:
            syn_0 = wn.synsets(i[0])
            syn_2 = wn.synsets(i[2])
            hyp_common_list = syn_0[0].lowest_common_hypernyms(syn_2[0])
            # extracting the name ouf of e.g. [Synset('chemical_element.n.01')]
            hyp_common = hyp_common_list[0].name().split('.')[0]
        except IndexError:
            hyp_common = 'No common hypernym found!'

        # add hypernyms to sublist in pairs
        i.append(hyp_common)

    return pairs


def recursive_hypernyms(df_new):
    global counter
    # function extract all word pairs by path
    # if word in pair contains '-', skip this pair
    # if not extract hypernym and add it to dataframe
    # recursive call until there is no empty string left in 'hypernym column'

    # returns df_new if all hypernyms are found
    if extraction_done(df_new):
        return df_new
    # returns df_new after ... recursions
    elif counter >= 500:
        return df_new

    # if not extracts the hypernyms
    else:
        # creating a list of all word pairs
        word_list = []
        for i in df_new['hyp_path']:
            temp = []
            for j in range(len(df_new[0])):
                if i == df_new['hyp_path'][j] and df_new['hypernym'][j] == "":
                    x = df_new[1][j]
                    path = df_new['hyp_path'][j]
                    temp.append(x)
                    temp.append(path)
            word_list.append(temp)

        pairs_temp = [list(k) for k in set(map(tuple, word_list))]

        # extract a list of needed pairs for this recursion
        pairs = []
        for i in pairs_temp:
            try:
                if '-' not in i[0] and '-' not in i[2]:
                    pairs.append(i)
            except IndexError:
                a = 1

        # calling hypernym extraction
        hypernym_list = get_hypernym(pairs)

        # insert hypernyms into dataframe['hypernym']
        for i in range(len(hypernym_list)):
            for j in range(len(df_new[0])):
                if hypernym_list[i][1] == df_new['hyp_path'][j]:
                    df_new['hypernym'][j] = hypernym_list[i][4]

        # insert hypernyms into dataframe ['1'] for next recursion
        for i in range(len(df_new[0])):
            if df_new['hypernym'][i] != "":
                hyp = df_new['hypernym'][i]
                path = df_new['hyp_path'][i] + "['name']"
                for j in range(len(df_new)):
                    if df_new[0][j] == path:
                        df_new[1][j] = hyp

        # recursive call
        counter += 1
        print('Recursion ' + str(counter) + ' ...')
        recursive_hypernyms(df_new)

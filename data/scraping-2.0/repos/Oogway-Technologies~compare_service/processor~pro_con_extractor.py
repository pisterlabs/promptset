import openai
import spacy
from service.const import *

openai.api_key = OPEN_AI_KEY
glb_nlp = spacy.load('en_core_web_md')


def apply_extraction(text):
    doc = glb_nlp(text)

    prod_pronouns = ['it', 'this', 'they', 'these']

    # FIRST RULE OF DEPENDANCY PARSE -
    # M - Sentiment modifier || A - Aspect
    # RULE = M is child of A with a relationshio of amod
    rule1_pairs = []
    for token in doc:
        A = "999999"
        M = "999999"
        if token.dep_ == "amod" and not token.is_stop:
            M = token.text
            A = token.head.text

            # add adverbial modifier of adjective (e.g. 'most comfortable headphones')
            M_children = token.children
            for child_m in M_children:
                if child_m.dep_ == "advmod":
                    M_hash = child_m.text
                    M = M_hash + " " + M
                    break

            # negation in adjective, the "no" keyword is a 'det' of the noun (e.g. no interesting characters)
            A_children = token.head.children
            for child_a in A_children:
                if child_a.dep_ == "det" and child_a.text == 'no':
                    neg_prefix = 'not'
                    M = neg_prefix + " " + M
                    break

        if A != "999999" and M != "999999":
            rule1_pairs.append((A, M, 1))

    # SECOND RULE OF DEPENDANCY PARSE -
    # M - Sentiment modifier || A - Aspect
    # Direct Object - A is a child of something with relationship of nsubj, while
    # M is a child of the same something with relationship of dobj
    # Assumption - A verb will have only one NSUBJ and DOBJ
    rule2_pairs = []
    for token in doc:
        children = token.children
        A = "999999"
        M = "999999"
        neg_prefix = ''
        add_neg_pfx = False
        for child in children:
            if child.dep_ == "nsubj" and not child.is_stop:
                A = child.text

            if (child.dep_ == "dobj" and child.pos_ == "ADJ") and not child.is_stop:
                M = child.text

            if child.dep_ == "neg":
                neg_prefix = child.text
                add_neg_pfx = True

        if add_neg_pfx and M != "999999":
            M = neg_prefix + " " + M

        if A != "999999" and M != "999999":
            rule2_pairs.append((A, M, 2))

    # THIRD RULE OF DEPENDANCY PARSE -
    # M - Sentiment modifier || A - Aspect
    # Adjectival Complement - A is a child of something with relationship of nsubj, while
    # M is a child of the same something with relationship of acomp
    # Assumption - A verb will have only one NSUBJ and DOBJ
    # "The sound of the speakers would be better. The sound of the speakers could be better" - handled using AUX dependency
    rule3_pairs = []
    for token in doc:
        children = token.children
        A = "999999"
        M = "999999"
        neg_prefix = ''
        add_neg_pfx = False
        for child in children:
            if child.dep_ == "nsubj" and not child.is_stop:
                A = child.text

            if child.dep_ == "acomp" and not child.is_stop:
                M = child.text

            # example - 'this could have been better' -> (this, not better)
            if child.dep_ == "aux" and child.tag_ == "MD":
                neg_prefix = "not"
                add_neg_pfx = True

            if child.dep_ == "neg":
                neg_prefix = child.text
                add_neg_pfx = True

        if add_neg_pfx and M != "999999":
            M = neg_prefix + " " + M
            # check_spelling(child.text)

        if A != "999999" and M != "999999":
            rule3_pairs.append((A, M, 3))

    # FOURTH RULE OF DEPENDANCY PARSE -
    # M - Sentiment modifier || A - Aspect

    # Adverbial modifier to a passive verb - A is a child of something with relationship of nsubjpass, while
    # M is a child of the same something with relationship of advmod

    # Assumption - A verb will have only one NSUBJ and DOBJ
    rule4_pairs = []
    for token in doc:

        children = token.children
        A = "999999"
        M = "999999"
        neg_prefix = ''
        add_neg_pfx = False
        for child in children:
            if (child.dep_ == "nsubjpass" or child.dep_ == "nsubj") and not child.is_stop:
                A = child.text

            if child.dep_ == "advmod" and not child.is_stop:
                M = child.text
                M_children = child.children
                for child_m in M_children:
                    if child_m.dep_ == "advmod":
                        M_hash = child_m.text
                        M = M_hash + " " + child.text
                        break

            if child.dep_ == "neg":
                neg_prefix = child.text
                add_neg_pfx = True

        if add_neg_pfx and M != "999999":
            M = neg_prefix + " " + M

        if A != "999999" and M != "999999":
            rule4_pairs.append((A, M, 4))

    # FIFTH RULE OF DEPENDANCY PARSE -
    # M - Sentiment modifier || A - Aspect
    # Complement of a copular verb - A is a child of M with relationship of nsubj, while
    # M has a child with relationship of cop
    # Assumption - A verb will have only one NSUBJ and DOBJ
    rule5_pairs = []
    for token in doc:
        children = token.children
        A = "999999"
        buf_var = "999999"
        for child in children :
            if child.dep_ == "nsubj" and not child.is_stop:
                A = child.text
                # check_spelling(child.text)

            if child.dep_ == "cop" and not child.is_stop:
                buf_var = child.text
                # check_spelling(child.text)

        if A != "999999" and buf_var != "999999":
            rule5_pairs.append((A, token.text, 5))

    # SIXTH RULE OF DEPENDANCY PARSE -
    # M - Sentiment modifier || A - Aspect
    # Example - "It ok", "ok" is INTJ (interjections like bravo, great etc)
    rule6_pairs = []
    for token in doc:
        children = token.children
        A = "999999"
        M = "999999"
        if token.pos_ == "INTJ" and not token.is_stop:
            for child in children :
                if(child.dep_ == "nsubj" and not child.is_stop):
                    A = child.text
                    M = token.text

        if A != "999999" and M != "999999":
            rule6_pairs.append((A, M, 6))

    # SEVENTH RULE OF DEPENDANCY PARSE -
    # M - Sentiment modifier || A - Aspect
    # ATTR - link between a verb like 'be/seem/appear' and its complement
    # Example: 'this is garbage' -> (this, garbage)
    rule7_pairs = []
    for token in doc:
        children = token.children
        A = "999999"
        M = "999999"
        neg_prefix = ''
        add_neg_pfx = False
        for child in children :
            if child.dep_ == "nsubj" and not child.is_stop:
                A = child.text

            if (child.dep_ == "attr") and not child.is_stop:
                M = child.text

            if child.dep_ == "neg":
                neg_prefix = child.text
                add_neg_pfx = True

        if add_neg_pfx and M != "999999":
            M = neg_prefix + " " + M

        if A != "999999" and M != "999999":
            rule7_pairs.append((A, M, 7))

    aspects = rule1_pairs + rule2_pairs + rule3_pairs + rule4_pairs + rule5_pairs + rule6_pairs + rule7_pairs

    # replace all instances of "it", "this" and "they" with "product"
    aspects = [(A, M, r) if A not in prod_pronouns else ("product", M, r) for A, M, r in aspects]
    return aspects


def get_compound_pairs(text, verbose=False):
    """Return tuples of (multi-noun word, adjective or verb) for document."""
    doc = glb_nlp(text)

    # Get list of compounds in doc
    compounds = [tok for tok in doc if tok.dep_ == 'compound']

    # Remove middle parts of compound nouns, but avoid index errors
    compounds = [c for c in compounds if c.i == 0 or doc[c.i - 1].dep_ != 'compound']
    tuple_list = []
    if compounds:
        for tok in compounds:
            pair_item_1, pair_item_2 = (False, False) # initialize false variables
            noun = doc[tok.i: tok.head.i + 1]
            pair_item_1 = noun
            # If noun is in the subject, we may be looking for adjective in predicate
            # In simple cases, this would mean that the noun shares a head with the adjective
            if noun.root.dep_ == 'nsubj':
                adj_list = [r for r in noun.root.head.rights if r.pos_ == 'ADJ']
                if adj_list:
                    pair_item_2 = adj_list[0]
                if verbose: # For trying different dependency tree parsing rules
                    print("Noun: ", noun)
                    print("Noun root: ", noun.root)
                    print("Noun root head: ", noun.root.head)
                    print("Noun root head rights: ", [r for r in noun.root.head.rights if r.pos_ == 'ADJ'])
            if noun.root.dep_ == 'dobj':
                verb_ancestor_list = [a for a in noun.root.ancestors if a.pos_ == 'VERB']
                if verb_ancestor_list:
                    pair_item_2 = verb_ancestor_list[0]
                if verbose: # For trying different dependency tree parsing rules
                    print("Noun: ", noun)
                    print("Noun root: ", noun.root)
                    print("Noun root head: ", noun.root.head)
                    print("Noun root head verb ancestors: ", [a for a in noun.root.ancestors if a.pos_ == 'VERB'])
            if pair_item_1 and pair_item_2:
                tuple_list.append((pair_item_1.text_with_ws, pair_item_2.text_with_ws, 1))
    return tuple_list


def pro_con_gpt3_extractor(text: str):
    generated_pro_con_list = list()

    gpt3_prompt = OPEN_AI_PROMPT + text + OPEN_AI_SUFFIX
    response = openai.Completion.create(
        engine=OPEN_AI_ENGINE,
        prompt=gpt3_prompt,
        temperature=0,
        max_tokens=15,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["Review"]
    )
    gen_pro_con = response['choices'][0]['text']
    gen_pro_con_list = gen_pro_con.strip().split('\n')
    for pro_con in gen_pro_con_list:
        pro_con = pro_con.strip()
        if not pro_con:
            continue
        pro_con_list = pro_con.split(' - ')
        if len(pro_con_list) == 2:
            generated_pro_con_list.append(pro_con_list[1].strip())

    return generated_pro_con_list


def pro_con_restaurant_gpt3_extractor(text: str):
    generated_pro_con_list = list()

    gpt3_prompt = OPEN_AI_RESTAURANT_PROMPT + text + OPEN_AI_SUFFIX
    response = openai.Completion.create(
        engine=OPEN_AI_ENGINE,
        prompt=gpt3_prompt,
        temperature=0,
        max_tokens=30,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["Review"]
    )
    gen_pro_con = response['choices'][0]['text']
    gen_pro_con_list = gen_pro_con.strip().split('\n')
    for pro_con in gen_pro_con_list:
        pro_con = pro_con.strip()
        if not pro_con:
            continue
        pro_con_list = pro_con.split(' - ')
        if len(pro_con_list) == 2:
            generated_pro_con_list.append(pro_con_list[1].strip())

    return generated_pro_con_list

# topic_evaluate.py

from gensim.models.coherencemodel import CoherenceModel

def evaluate_model(lda_model, dictionary, texts, question_name):
    # Compute Coherence Score using CoherenceModel
    coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v',processes=1)
    coherence_score = coherence_model_lda.get_coherence()
    # print(f'{question_name} Coherence Score:', coherence_score)

    return coherence_score

def main(question_names):
    # Evaluate each model and store the scores
    scores = {}
    for question_name in question_names:
        scores[question_name] = evaluate_model(question_name)
    
    # Optionally, save the scores to a file
    with open('models/lda_model/coherence_scores.txt', 'w') as f:
        for question_name, score in scores.items():
            f.write(f'{question_name} Coherence Score: {score}\n')
    
    return scores

if __name__ == "__main__":
    import pickle
    from gensim.models.ldamodel import LdaModel
    from gensim.corpora.dictionary import Dictionary

    COLUMNS_OF_INTEREST = [
        "Q1",
        "Q2",
        "Q3",
    ]

    question_name = COLUMNS_OF_INTEREST[0]

    lda_model = LdaModel.load(f'models/lda_model/{question_name.replace("?","")}_lda.model')
    dictionary = Dictionary.load(f'models/lda_model/{question_name.replace("?","")}_dictionary.dict')
    with open(f'models/lda_model/{question_name.replace("?","")}_texts.pkl', 'rb') as f:
        documents = pickle.load(f)
        
    evaluate_model(lda_model, dictionary, documents, question_name)

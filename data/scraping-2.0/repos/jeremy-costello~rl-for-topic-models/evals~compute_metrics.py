import pickle
import argparse
from tqdm import tqdm
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

from trainer.utils import get_save_num
from data.dataset import get_dataset
from evals.measures import get_topic_words_from_files, topic_diversity, NPMICoherence, InvertedRBO, CoherenceWordEmbeddings


my_parser = argparse.ArgumentParser()

my_parser.add_argument('topk', type=int, help='Top-k topic words for calculating coherence and diversity.')
my_parser.add_argument('num_exps', type=int, help='Number of experiments to average over.')
my_parser.add_argument('num_seeds', type=int, help='Number of seeds for each experiment.')
my_parser.add_argument('data', type=str, help='Location of the data pickle file.')
my_parser.add_argument('experiment', type=str, help='Location of the experiment directory.')
my_parser.add_argument('--word2vec', type=str, default=None, help='Location of the "word2vec-google-news-300" model.')
my_parser.add_argument('--diversity', type=int, default=0, help='Top-k for calculating diversity (if different from coherence).')

args = my_parser.parse_args()

topk = args.topk
if args.diversity:
    diverysity_topk = args.diversity
else:
    diverysity_topk = args.topk

num_experiments = args.num_exps
num_seeds = args.num_seeds
data_pickle = args.data.rstrip('.pkl').replace('\\', '/')
experiment_dir = args.experiment.replace('\\', '/')

word2vec_path = args.word2vec
coherence_embeddings = CoherenceWordEmbeddings(word2vec_path, binary=True)

dataset_save_dict = get_dataset(data_pickle)
sparse_corpus_bow = dataset_save_dict['sparse_corpus_bow']
npmi_coherence = NPMICoherence(sparse_corpus_bow)

train_docs = [doc.split() for doc in dataset_save_dict["train"]["preprocessed_docs"]]
if dataset_save_dict["test"]["preprocessed_docs"] is not None:
    test_docs = [doc.split() for doc in dataset_save_dict["test"]["preprocessed_docs"]]
    docs = train_docs + test_docs
else:
    docs = train_docs

vocabulary = dataset_save_dict["vocabulary"]
dictionary = Dictionary()
dictionary.token2id = vocabulary["token2id"]
dictionary.id2token = vocabulary["id2token"]

rbo_score = 0
cohemb_score = 0
npmicoh_score = 0
npmicoh_window_score = 0
diversity_score = 0
for i, experiment_num in enumerate(range(num_experiments)):
    true_experiment_num = get_save_num(experiment_num)
    experiment_rbo_score = 0
    experiment_cohemb_score = 0
    experiment_npmicoh_score = 0
    experiment_npmicoh_window_score = 0
    experiment_diversity_score = 0
    for j, seed_num in tqdm(enumerate(range(num_seeds)), total=num_seeds):
        true_seed_num = get_save_num(seed_num)
        seed_pickle = f'{experiment_dir}/{true_experiment_num}/seeds/{true_seed_num}/{true_seed_num}_plotting_arrays'
        topics = get_topic_words_from_files(data_pickle, seed_pickle, topk)

        inverted_rbo = InvertedRBO(topics)
        seed_rbo_score = inverted_rbo.score(topk)
        experiment_rbo_score = experiment_rbo_score * j / (j + 1) + seed_rbo_score / (j + 1)

        seed_cohemb_score = coherence_embeddings.score(topics, topk)
        experiment_cohemb_score = experiment_cohemb_score * j / (j + 1) + seed_cohemb_score / (j + 1)

        with open(f'{seed_pickle}.pkl', 'rb') as handle:
            experiment_dict = pickle.load(handle)
        beta = experiment_dict['best_epoch_beta']

        seed_npmicoh_score = npmi_coherence.get_coherence(beta, topk)
        experiment_npmicoh_score = experiment_npmicoh_score * j / (j + 1) + seed_npmicoh_score / (j + 1)

        seed_diversity_score = topic_diversity(beta, diverysity_topk)
        experiment_diversity_score = experiment_diversity_score * j / (j + 1) + seed_diversity_score / (j + 1)

        npmi_window_model = CoherenceModel(
            topics=topics,
            texts=docs,
            dictionary=dictionary,
            coherence="c_npmi",
            topn=topk
        )

        seed_npmicoh_window_score = npmi_window_model.get_coherence()
        experiment_npmicoh_window_score = experiment_npmicoh_window_score * j / (j + 1) + seed_npmicoh_window_score / (j + 1)

    rbo_score = rbo_score * i / (i + 1) + experiment_rbo_score / (i + 1)
    cohemb_score = cohemb_score * i / (i + 1) + experiment_cohemb_score / (i + 1)
    npmicoh_score = npmicoh_score * i / (i + 1) + experiment_npmicoh_score / (i + 1)
    npmicoh_window_score = npmicoh_window_score * i / (i + 1) + experiment_npmicoh_window_score / (i + 1)
    diversity_score = diversity_score * i / (i + 1) + experiment_diversity_score / (i + 1)

print(f'RBO: {rbo_score}\n')
print(f'Word2Vec Coherence: {cohemb_score}\n')
print(f'NPMI Coherence (full document): {npmicoh_score}\n')
print(f'NPMI Coherence (window=10): {npmicoh_window_score}\n')
print(f'Diversity: {diversity_score}\n')

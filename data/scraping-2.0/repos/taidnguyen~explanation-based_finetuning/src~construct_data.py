"""
This script generate unbiased / biased dataset for standard finetuning or explanation-based finetuning
Example:
>>> python construct_data.py --task_name sbic --bias length --data_dir ../../data --output_dir ../../res
Output:
four files for finetuning, e.g.
    training set, size 1000,
        standard finetuning: sbic_length_trainSimple_filterBias_100.csv
        explanation-based finetuning: sbic_length_trainAdvanced_filterBias_100.csv
    test set, size 500,
        standard finetuning: sbic_length_test.csv
        explanation-based finetuning: sbic_length_testAdvanced.csv


data folder:
.
├── comve
│   ├── dev.csv
│   └── train.csv
├── creak
│   ├── dev.json
│   └── train.json
├── esnli
│   ├── esnli_dev.csv
│   ├── esnli_train_1.csv
│   ├── esnli_train_2.csv
└── sbic
    ├── dev.csv
    └── train.csv
"""
from argparse import ArgumentParser

import pandas as pd
import random
import os
import numpy as np
import nltk
import sys

old_stdout = sys.stdout
f = open(os.devnull, 'w')
sys.stdout = f
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
sys.stdout = old_stdout

from sklearn.metrics import accuracy_score
from collections import Counter
from bias_functions import *
from openai_model_dict import *

explanation_templates = {
    'esnli': ' The premise {} the hypothesis because',
    'creak': ' The claim is {} because',
    'comve': ' Sentence {} makes more sense because',
    'sbic': '',
}

def get_random_sample(df, label_col, true_class, nrows):
    random.seed(42)
    #Shuffle the dataset and take 1000 examples for training, 200 for testing
    indices = list(range(len(df[df[label_col]==true_class])))
    random.shuffle(indices)
    train_indices = indices[:nrows//2]
    df_true = df[df[label_col]==true_class].reset_index(drop=True).iloc[train_indices]

    indices = list(range(len(df[df[label_col]!=true_class])))
    random.seed(42)
    random.shuffle(indices)
    train_indices = indices[:nrows//2]
    df_false = df[df[label_col]!=true_class].reset_index(drop=True).iloc[train_indices]
    res = pd.concat([df_true, df_false])
    return res

def add_sent_label(df): # for comve
    sent1list, sent2list, label = [], [], []
    for i, row in df.iterrows():
        correct = row['Correct Statement']
        incorrect = row['Incorrect Statement']
        explanation = row['Right Reason1']

        # choose which is sentence 1 vs sentence 2
        if i % 2 == 1:
            sent1 = correct
            sent2 = incorrect
            ans = "Sentence 1"
        else:
            sent1 = incorrect
            sent2 = correct
            ans = "Sentence 2"
        sent1list.append(sent1)
        sent2list.append(sent2)
        label.append(ans)
    df['Sentence1'] = sent1list
    df['Sentence2'] = sent2list
    df['label'] = label
    return df

def preprocess_data(args):
    if args.task_name == 'esnli':
        train1 = pd.read_csv(os.path.join(args.data_dir, args.task_name,'esnli_train_1.csv'))
        train2 = pd.read_csv(os.path.join(args.data_dir, args.task_name,  'esnli_train_2.csv'))
        all_train = pd.concat([train1, train2]).dropna().reset_index(drop = True)
        test_df = pd.read_csv(os.path.join(args.data_dir, args.task_name,  'esnli_dev.csv'))
        all_train['bin_label'] = all_train['gold_label'].apply(lambda x: 'true' if x=='entailment' else 'false') # true: entailmane; false: not entailed
        test_df['bin_label'] = test_df['gold_label'].apply(lambda x: 'true' if x=='entailment' else 'false')
        keep_cols = ['Sentence1', 'Sentence2', 'Explanation_1', 'bin_label']
        all_train, test_df = all_train[keep_cols], test_df[keep_cols]
        label_col, true_class = 'bin_label', 'true'

    elif args.task_name == 'creak':
        all_train = pd.read_json(os.path.join(args.data_dir, args.task_name, 'train.json'), lines=True)
        test_df = pd.read_json(os.path.join(args.data_dir, args.task_name,  'dev.json'), lines=True)
        keep_cols = ['sentence', 'explanation', 'label', 'entity']
        all_train, test_df = all_train[keep_cols], test_df[keep_cols]
        label_col, true_class = 'label', 'true'

    elif args.task_name == 'comve':
        all_train = pd.read_csv(os.path.join(args.data_dir, args.task_name,'train.csv')).drop(['Right Reason2'], axis=1).rename({'Right Reason3': 'Right_Reason2'}, axis=1)
        test_df = pd.read_csv(os.path.join(args.data_dir, args.task_name,  'dev.csv')).drop(['Right Reason2'], axis=1).rename({'Right Reason3': 'Right_Reason2'}, axis=1)
        all_train = add_sent_label(all_train)
        test_df = add_sent_label(test_df)
        keep_cols = [ 'Sentence1','Sentence2', 'label', 'Right Reason1']
        all_train, test_df = all_train[keep_cols], test_df[keep_cols]
        label_col, true_class = 'label', 'Sentence 1'

    elif args.task_name == 'sbic':
        all_train = pd.read_csv(os.path.join(args.data_dir, args.task_name,'train.csv'))
        # For SBIC, keep only 3 explanations for the same post
        all_train = all_train.groupby("post").head(3).reset_index(drop=True)

        # Concatenate all explanation
        all_train = all_train.groupby(["post", "offensiveYN"])["targetStereotype"].apply(lambda x: " [SEP] ".join(x)).reset_index()
        test_df = pd.read_csv(os.path.join(args.data_dir, args.task_name,  'dev.csv'))
        posts_train = all_train['post']
        test_df = test_df[~test_df['post'].isin(posts_train)]

        # all_train['targetStereotype'] = all_train['targetStereotype'].apply(lambda x: x.split('[SEP]')[0])
        # test_df['targetStereotype'] = test_df['targetStereotype'].apply(lambda x: x.split('[SEP]')[0])
        keep_cols = ['post', 'offensiveYN','targetStereotype']
        all_train, test_df = all_train[keep_cols], test_df[keep_cols]

        label_col, true_class = 'offensiveYN', 'offensive'

    print('size of training data:',len(all_train), \
        ' distribution of labels:',Counter(all_train[label_col]))
    test_200 = get_random_sample(test_df, label_col, true_class, 200)
    test_500 = get_random_sample(test_df, label_col, true_class, 500)

    unbiased_train = get_random_sample(all_train, label_col, true_class, args.train_size)
    train_10k = get_random_sample(all_train, label_col, true_class, min(10000, min(Counter(all_train[label_col]).values())*2))

    # print(args.task_name, len(all_train), len(train_10k), len(test_200), len(test_500))
    return unbiased_train,train_10k, test_200, test_500, label_col, true_class

#Create finetuning dataset
#Function to create a finetuning CSV using the methods above
def create_finetuning_dataset(args, df, name=None, advanced=False, permute=False):
    prompts = []
    completions = []

    # if permuting explanations
    if permute:
        if args.task_name == 'esnli':
            col_explanation = "Explanation_1"
            col_label = "bin_label"
        elif args.task_name == 'creak':
            col_explanation = "explanation"
            col_label = "label"
        else:
            raise NotImplementedError(f"Permutation not implemented for task {args.task_name}")
        # get all explanations
        idx2explanations = dict(zip(df.index, df[col_explanation]))
        explanation_indices = {}
        labels = df[col_label].unique()
        for l in labels:
            explanation_indices[l] = set(df[df[col_label] == l].index)

    for i, row in df.iterrows():
        if args.task_name == 'esnli':
            sent1 = row['Sentence1'].rstrip('.')
            sent2 = row['Sentence2'].rstrip('.')
            label = row['bin_label']
            if permute:
                # randomly sample an explanation within label group
                explanation_set = explanation_indices[label]
                idx = random.sample(explanation_set, 1)[0]
                # update set
                explanation_set.remove(idx)
                explanation_indices[label] = explanation_set
                # get explanation
                explanation = idx2explanations[idx].rstrip('.')
                print(f"Original {i}: {row['Explanation_1'].rstrip('.')}\nPermute {idx}: {explanation}\n")
            else:
                explanation = row['Explanation_1'].rstrip('.')
        elif args.task_name == 'creak':
            entity = row['entity']
            sentence = row['sentence']
            label = row['label']
            if permute:
                # randomly sample an explanation within label group
                explanation_set = explanation_indices[label]
                idx = random.sample(explanation_set, 1)[0]
                # update set
                explanation_set.remove(idx)
                explanation_indices[label] = explanation_set
                # get explanation
                explanation = idx2explanations[idx]
                print(f"Original {i}: {row['explanation']}\nPermute {idx}: {explanation}\n")
            else:
                explanation = row['explanation']
        elif args.task_name == 'comve':
            sent1 = row['Sentence1']
            sent2 = row['Sentence2']
            ans = row['label']
            expl = row['Right Reason1']
        elif args.task_name == 'sbic':
            post = row['post']
            expl = row['targetStereotype']
            ans = row['offensiveYN']
        if advanced:  # with explanations
            # prompt, completion = create_prompt_completion_advanced(sent1, sent2, label, explanation)
            if args.task_name == 'esnli':
                prompt = """Does the premise \"{}\" entails the hypothesis \"{}\" ?\nThoughts:""".format(sent1, sent2)
                completion = """ {}\nAnswer: {}\n###\n\n""".format(explanation, label)
                if args.expl_temp: completion = explanation_templates[args.task_name].\
                    format('entails' if 'true' in label else 'does not entail') + completion[0].lower() + completion[1:]

            elif args.task_name == 'creak':
                prompt = """Is the following claim about {} true or false?\nclaim:{}\nThoughts:""".format(entity, sentence, explanation)
                completion = """ {}\nAnswer: {}\n###\n\n""".format(explanation, label)
                if args.expl_temp: completion = explanation_templates[args.task_name].\
                    format('true' if 'true' in label else 'false') + completion[0].lower() + completion[1:]

            elif args.task_name == 'comve':
                prompt = """Which of the following sentences makes more sense? Please explain.\n  Sentence 1: {}\n  Sentence 2: {}\n  Reason:""".format(sent1, sent2)
                completion = """ {}\nAnswer: {}\n###\n\n""".format(expl, ans)
                if args.expl_temp: completion = explanation_templates[args.task_name].\
                    format('1' if '1' in label else '2') + completion[0].lower() + completion[1:]

            elif args.task_name == 'sbic':
                prompt = """Post: {}\nExplanation:""".format(post)
                completion = """ {}\nAnswer: {}\n###\n\n""".format(expl, ans)
                if args.expl_temp: completion = explanation_templates[args.task_name] + completion[0].lower() + completion[1:]
        else:
            # prompt, completion = create_prompt_completion_basic(sent1, sent2, label)
            if args.task_name == 'esnli':
                prompt = """Does the premise \"{}\" entails the hypothesis \"{}\" ?\nAnswer:""".format(sent1, sent2)
                completion = """ {}\n###\n""".format(label)
            elif args.task_name == 'creak':
                prompt = """Is the following claim about {} true or false?\nclaim:{}\nAnswer:""".format(entity, sentence)
                completion = """ {}\n###\n""".format(label)
            elif args.task_name == 'comve':
                prompt = """Which of the following sentences makes more sense?\n  Sentence 1: {}\n  Sentence 2: {}\n  Answer:""".format(sent1, sent2)
                completion = """ {}\n###\n""".format(ans)
            elif args.task_name == 'sbic':
                prompt = """Post: {}\nAnswer:""".format(post)
                completion = """ {}\n###\n""".format(ans)
        prompts.append(prompt)
        completions.append(completion)

    # some checks for permute
    if permute:
        for k in explanation_indices:
            assert(len(explanation_indices[k]) == 0)

    res = df.copy()
    # res = pd.DataFrame({'prompt': prompts, 'completion': completions})
    res['prompt'] = prompts
    res['completion'] = completions
    if name:
        if args.expl_temp: name = name.rstrip('.csv') + '_expl_temp.csv'
        res.to_csv(name, index=False)
        print('saved: {}'.format(name))
    return res

def positiveLabel(completion, true_class):
  return "{}".format(true_class) in completion and 'not ' not in completion

def getFtLabelSquare(df, true_class, featureColumn, labelColumn, filterFunction, labelFunction, median = None, median_ppl = None):

    labels = labelColumn.apply(lambda x: labelFunction(x, true_class)).tolist()
    if filterFunction.__name__ == 'is_longer':
        featurePresence = featureColumn.apply(lambda x: filterFunction(x, median)).tolist()
    elif filterFunction.__name__ == 'perp_filter':
        featurePresence = featureColumn.apply(lambda x: filterFunction(x, median_ppl)).tolist()
    else:
        featurePresence = featureColumn.apply(filterFunction).tolist()
    df['featurePresent'] = featurePresence

    labelFeature = zip(featurePresence, labels)
    counts = Counter(labelFeature)
    nPresentPos = counts[(True,True)]
    nPresentNeg = counts[(True,False)]
    nAbsentPos = counts[(False,True)]
    nAbsentNeg = counts[(False,False)]
    print("""
          |Feature+    Feature-
    ------|--------    --------
    Label+|  {}       {}
    ------|--------    --------
    Label-|  {}       {}
    ------|--------    --------
    """.format(nPresentPos, nAbsentPos, nPresentNeg, nAbsentNeg))

    print("Positive Label %: {}".format((sum(labels))/len(labels)))
    print("Feature Presence in Positive Label %: {}".format((nPresentPos)/
                                          (nPresentPos + nAbsentPos)))
    print("Feature Presence in Negative Label %: {}".format((nPresentNeg)/
                                        (nPresentNeg + nAbsentNeg)))

def filter_training(df, true_class, labelCol, labelFunction,
                    nPresentPos, nAbsentPos, nPresentNeg, nAbsentNeg):
#   dfCopy = df.copy()
    dfCopy = df
    dfCopy['labelPos'] = df[labelCol].apply(lambda x: labelFunction(x, true_class))
    dfPresentPos = dfCopy[(dfCopy['labelPos']==True) &
                          (dfCopy['featurePresent']==True)].sample(nPresentPos, random_state=42)
    dfPresentNeg = dfCopy[(dfCopy['labelPos']==False) &
                          (dfCopy['featurePresent']==True)].sample(nPresentNeg, random_state=42)
    dfAbsentPos = dfCopy[(dfCopy['labelPos']==True) &
                          (dfCopy['featurePresent']==False)].sample(nAbsentPos, random_state=42)
    dfAbsentNeg = dfCopy[(dfCopy['labelPos']==False) &
                          (dfCopy['featurePresent']==False)].sample(nAbsentNeg, random_state=42)
    newDf = pd.concat([dfPresentPos, dfPresentNeg, dfAbsentPos, dfAbsentNeg]).sample(frac=1, random_state=42)
    return newDf


def check_bias_distribution_helper(args,true_class, df, use_col, label_col, median = None, median_ppl = None):
    # common biases
    if args.bias == 'plural':
        if args.task_name == 'comve':
          getFtLabelSquare(df, true_class, df[use_col].apply(lambda x: x.split('Sentence 1:')[-1]), df[label_col],is_plural,positiveLabel)
        else:
          getFtLabelSquare(df, true_class, df[use_col],df[label_col],is_plural,positiveLabel)
    elif args.bias == 'length':
        # if args.task_name == 'comve':
        #     getFtLabelSquare(df, true_class, df[use_col],df[label_col],comve_length_filter,positiveLabel, median)
        # else:
        getFtLabelSquare(df, true_class, df[use_col],df[label_col],is_longer,positiveLabel, median)
    elif args.bias == 'cluster':
        getFtLabelSquare(df, true_class, df[use_col], df[label_col],cluster_filter,positiveLabel)
    elif args.bias == 'present':
        if args.task_name == 'comve':
            # comve_present_tense_filter
            getFtLabelSquare(df, true_class, df[use_col],df[label_col],comve_present_tense_filter,positiveLabel)
        else:
            getFtLabelSquare(df, true_class, df[use_col],df[label_col],is_present_tense,positiveLabel)

    # task-specific biases
    elif args.bias == 'female':
        getFtLabelSquare(df, true_class, df[use_col],df[label_col], is_female, positiveLabel)
    elif args.bias == 'perplexity':
        getFtLabelSquare(df, true_class, df[use_col],df[label_col], perp_filter, positiveLabel, median_ppl)
    elif args.bias == 'retweet':
        getFtLabelSquare(df, true_class, df[use_col],df[label_col], retweet_present, positiveLabel)
    elif args.bias == 'swapped':
        getFtLabelSquare(df, true_class, df[use_col],df[label_col], POS_filter, positiveLabel)


def check_bias_distribution(args, true_class, df, istrain,label_col, median = None, kmeans = None, median_ppl = None, create_featurecol = True):
    if args.bias == 'length':
        if args.task_name == 'comve':
            df['sentence'] = df.apply(lambda row : row['prompt'] + '\nLabel: ' + 'Sentence ' + row['completion'].split()[-2], axis=1)
        elif args.task_name == 'esnli':
            df['sentence'] = df['Sentence1'] + df['Sentence2']
        elif args.task_name == 'sbic':
            df['sentence'] = df['post']
        df['length'] = df['sentence'].apply(lambda x: 0 if isinstance(x, float) else len(x))
        if istrain:
            median = df['length'].median()
        check_bias_distribution_helper(args, true_class, df, 'sentence', label_col, median)
        return median
    if args.bias == 'cluster':
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import KMeans
        from sklearn import preprocessing
        if create_featurecol:
            embeddingModel = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            if args.task_name == 'esnli': target_col = 'Sentence1'
            elif args.task_name == 'creak': target_col = 'sentence'
            elif args.task_name == 'comve': target_col = 'prompt'
            elif args.task_name == 'sbic': target_col = 'post'
            embeddings = embeddingModel.encode(df[target_col].tolist())
            if istrain:
                kmeans = KMeans(n_clusters=2, random_state = 42).fit(embeddings)
            cluster_assignments = kmeans.predict(embeddings)
            df['cluster_assignment'] = cluster_assignments
        check_bias_distribution_helper(args, true_class, df, 'cluster_assignment', label_col, median)
        return kmeans

    elif args.bias == 'perplexity':
        from evaluate import load
        perplexity = load("perplexity", module_type="metric")
        def get_perplexity(text):
            if type(text) == str:
              return perplexity.compute(predictions=[text], model_id='gpt2')['perplexities'][0]
            else:
              return perplexity.compute(predictions=text, model_id='gpt2')['perplexities']
        train_full['perplexity'] = get_perplexity(train_full['sentence'])
        if istrain: median_ppl = np.median(train_full['perplexity'])
        check_bias_distribution_helper(args, true_class, df, 'perplexity', label_col, median_ppl)
        return median_ppl

    else: # present-tense, plural, (female, swapped-word, tweet-handle)
        if args.task_name == 'esnli': target_col = 'Sentence1'
        elif args.task_name == 'creak': target_col = 'sentence'
        elif args.task_name == 'comve': target_col = 'prompt'
        elif args.task_name == 'sbic': target_col = 'post'
        check_bias_distribution_helper(args, true_class, df, target_col, label_col)


def get_prompt_datasets(args, true_class, unbiased_train, train_10k, test_df, label_col, unbiased = False, save = True):
    if unbiased:
        filteredDf = unbiased_train
    else:
        present_positive, absent_positive, present_negative, absent_negative = \
            int(args.train_size*args.bias_strength//2), int(args.train_size*(1-args.bias_strength)//2),\
            int(args.train_size*(1-args.bias_strength)//2), int(args.train_size*args.bias_strength//2)
        filteredDf = filter_training(train_10k, true_class, label_col, positiveLabel,
                                 nPresentPos=present_positive,nAbsentPos=absent_positive,\
                                    nPresentNeg=present_negative,nAbsentNeg=absent_negative).reset_index(drop = True)
        # getFtLabelSquare(filteredDf, true_class, filteredDf[use_col],df[label_col],is_plural,positiveLabel)
        print('distribution after filtering:')
        check_bias_distribution(args, true_class, filteredDf, istrain = True, label_col=label_col, create_featurecol = False)

    # Tai: add flag for permute
    permute = False
    if hasattr(args, 'permute') and args.permute:
        permute = True

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    path = os.path.join(args.output_dir,args.task_name+'_'+args.bias+'_'+args.method+'_trainSimple_filterBias_{}bias_{}train.csv'.format(str(int(100*args.bias_strength)), str(args.train_size))) if save else None
    trainPromptCompletionSimple = create_finetuning_dataset(args,filteredDf, path, advanced=False, permute=permute)
    path = os.path.join(args.output_dir,args.task_name+'_'+args.bias+'_'+args.method+'_test.csv') if save else None
    testPromptCompletionSimple = create_finetuning_dataset(args,test_df, path, advanced=False)

    path = os.path.join(args.output_dir,args.task_name+'_'+args.bias+'_'+args.method+'_trainAdvanced_filterBias_{}bias_{}train.csv'.format(str(int(100*args.bias_strength)), str(args.train_size))) if save else None
    trainPromptCompletionAdvanced = create_finetuning_dataset(args,filteredDf, path, advanced=True, permute=permute)
    path = os.path.join(args.output_dir,args.task_name+'_'+args.bias+'_'+args.method+'_testAdvanced.csv') if save else None
    testPromptCompletionAdvanced = create_finetuning_dataset(args,test_df, path, advanced=True)

    return trainPromptCompletionSimple,testPromptCompletionSimple, trainPromptCompletionAdvanced, testPromptCompletionAdvanced
# trainPromptCompletionSimple,testPromptCompletionSimple, \
#   trainPromptCompletionAdvanced, testPromptCompletionAdvanced = \
#   get_prompt_datasets(is_longer, 'hypo_premise')
  # get_prompt_datasets(cluster_filter, 'cluster_assignment')
  # get_prompt_datasets(is_female)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--task_name', type=str, default='esnli',
                            help='task name, esnli, creak, comve, sbic')
    parser.add_argument('--data_dir', type=str, default='../../data',
                            help='data directory')
    parser.add_argument('--output_dir', type=str, default='../../res',
                            help='data directory')
    # parser.add_argument('--download_data', default=0, action='store_true',
    #                     help='if True, download dataset to the data directory')

    # unbiased, length, present, cluster, plural, (perplexity, female, swapped, retweet)
    parser.add_argument('--bias', type=str, default='present',
                            help='type of bias')
    parser.add_argument('--bias_strength', type = float, default=1,
                            help = 'bias strength: 0 to 1')
    parser.add_argument('--train_size', type = int, default=1000,
                            help = 'data size of training set')

    parser.add_argument('--method', type=str, default='finetuned',
                            help='type of bias')
    parser.add_argument('--expl_temp', action='store_true',
                        help='Use a template to format the explanations')
    # Tai: permute the explanations
    parser.add_argument('--permute', action='store_true', help='Permute the explanation within label groups')

    args = parser.parse_args()
    if args.task_name not in ['creak', 'esnli', 'sbic', 'comve']:
        print("Please choose task name from {}".format(['creak', 'esnli', 'sbic', 'comve']))
        exit()
    biases = ['unbiased', 'length', 'present', 'cluster', 'plural'] + ['perplexity', 'swapped', 'female', 'retweet']
    if args.bias not in biases:
        print("Please choose bias from {}".format(biases))
        exit()
    if args.bias_strength < 0 or args.bias_strength > 1:
        print('illegal bias strength: {}'.format(args.bias_strength))
        exit()

    if args.bias == 'unbiased': args.bias_strength = 0

    print("------ task: {}, bias: {}, method: {}, bias strength: {}, train size: {} ------------".format(
        args.task_name, args.bias, args.method, args.bias_strength, args.train_size))

    unbiased_train, train_10k, test_200, test_500, label_col, true_class = preprocess_data(args)
    test_df = test_500
    if 'few-shot' in args.method:
        test_df = test_200

    # demonstrate bias distribution
    if args.bias != 'unbiased':
        train_full = create_finetuning_dataset(args, train_10k, advanced=True)
        test_df = create_finetuning_dataset(args, test_500, advanced=True)
        if args.bias == 'length':
            median = check_bias_distribution(args, true_class, train_full, istrain = True, label_col = label_col)
            check_bias_distribution(args, true_class, test_df, istrain = False, median = median, label_col = label_col)
        elif args.bias == 'cluster':
            kmeans = check_bias_distribution(args, true_class, train_full, istrain = True, label_col = label_col)
            check_bias_distribution(args, true_class, test_df, istrain = False, label_col = label_col, kmeans = kmeans)
        else:
            check_bias_distribution(args, true_class, train_full, istrain = True, label_col = label_col)
            check_bias_distribution(args, true_class, test_df, istrain = False, label_col = label_col)

    # create biased / unbiased dataset for finetuning
    if args.bias == 'unbiased':
        trainPromptCompletionSimple,testPromptCompletionSimple, \
            trainPromptCompletionAdvanced, testPromptCompletionAdvanced = \
            get_prompt_datasets(args, true_class, unbiased_train, train_10k, test_df, unbiased = True, label_col = label_col)
    else:
        trainPromptCompletionSimple,testPromptCompletionSimple, \
            trainPromptCompletionAdvanced, testPromptCompletionAdvanced = \
                get_prompt_datasets(args, true_class, unbiased_train, train_full, test_df = test_df, label_col = label_col)

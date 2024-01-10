# Standard libraries
import argparse
import pandas as pd

# External libraries
import openai 
import tiktoken
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Local imports
from config import OPENAI_API_KEY
from openai.embeddings_utils import get_embedding

# Constants
LABELS_UPDATED = ['Human Generated Story', 'GPT Generated Story']
COLORS = ["lightgreen", "darkgreen"]

# Set up OpenAI API key
openai.api_key = OPENAI_API_KEY 

class OpenAIEmbeddingsProcessor:
    def __init__(self, embedding_model="text-embedding-ada-002", embedding_encoding="cl100k_base", max_tokens=8000):
        '''Initialize the processor with desired model, encoding, and max token count.'''
        self.embedding_model = embedding_model
        self.embedding_encoding = embedding_encoding
        self.max_tokens = max_tokens
        self.encoding = tiktoken.get_encoding(self.embedding_encoding)

    def check_token_count(self, text):
        '''Check the token count for a given text.'''
        return len(self.encoding.encode(text))

    def compute_embedding(self, text):
        '''Compute embeddings for the given text.'''
        return get_embedding(text, engine=self.embedding_model)

    def compute_cosine_similarity(self, embedding1, embedding2):
        '''Compute cosine similarity between two embeddings.'''
        return cosine_similarity([embedding1], [embedding2])[0][0]

    def compare_similarities(self, row):
        '''Compare similarity scores and determine which story is closer to the original.'''
        if row['story_summary_similarity'] > row['story_generated_story_similarity']:
            return 'Human Generated Story'
        elif row['story_summary_similarity'] < row['story_generated_story_similarity']:
            return 'AI Generated Story'
        else:
            return 'Equally Likely'

    def process_dataframe(self, df):
        '''
        Process the dataframe to compute embeddings and similarity scores.
        Filters out entries with token counts exceeding the max allowed.
        '''
        # Filter out entries that are too long to embed
        df = df[df.apply(lambda x: self.check_token_count(x['story']) <= self.max_tokens and
                                   self.check_token_count(x['summary']) <= self.max_tokens and
                                   self.check_token_count(x['generated_story']) <= self.max_tokens, axis=1)]

        # Compute embeddings for the stories
        df['story_embedding'] = df['story'].apply(self.compute_embedding)
        df['summary_embedding'] = df['summary'].apply(self.compute_embedding)
        df['generated_story_embedding'] = df['generated_story'].apply(self.compute_embedding)

        # Compute cosine similarities between the embeddings
        df['story_summary_similarity'] = df.apply(lambda row: self.compute_cosine_similarity(row['story_embedding'], row['summary_embedding']), axis=1)
        df['story_generated_story_similarity'] = df.apply(lambda row: self.compute_cosine_similarity(row['story_embedding'], row['generated_story_embedding']), axis=1)

        # Determine which story is closer to the original based on similarity scores
        df['closer_to_story'] = df.apply(self.compare_similarities, axis=1)
        
        return df

def generate_value_plot(df, path, filename):
    '''Generate a value plot comparing the number of stories closer to the original story or the summary.'''
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='closer_to_story', palette=reversed(COLORS))
    plt.title('Count of Stories With Higher Cosine Similarity to Original Story')
    plt.ylabel('Count')
    plt.xlabel('')
    plt.xticks(ticks=range(len(LABELS_UPDATED)), labels=(LABELS_UPDATED))
    plt.savefig(f"{path}/{filename}")
    plt.show()

def generate_boxplot(df, path, filename):
    '''Generate a boxplot showcasing the distribution of similarity scores.'''

    # Print out box plot summary statistics
    print("Summary Statistics for story_generated_story_similarity:")
    print(df['story_generated_story_similarity'].describe())
    print("\nSummary Statistics for story_summary_similarity:")
    print(df['story_summary_similarity'].describe())

    data_to_plot = [df['story_generated_story_similarity'], df['story_summary_similarity']]
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data_to_plot, orient="v", palette=COLORS, width=0.6)
    plt.xticks(ticks=range(len(LABELS_UPDATED)), labels=reversed(LABELS_UPDATED))
    plt.ylabel('Similarity Score')
    plt.title('Distribution of Similarity Scores')
    plt.savefig(f"{path}/{filename}")
    plt.show()

def plot(df, plot_type, path, value_filename, boxplot_filename):
    '''Decide which plot to generate based on the plot_type argument.'''
    if plot_type == 'value_plot':
        generate_value_plot(df, path, value_filename)
    elif plot_type == 'boxplot':
        generate_boxplot(df, path, boxplot_filename)
    else:
        print("Invalid plot_type. Use 'value_plot' or 'boxplot'.")

def main(args):
    df_processed = None

    if not args.preprocessing:
        # Load data
        print("Loading data...")
        df = pd.read_csv(args.input_path)

        # Process data
        print("Processing data...")
        processor = OpenAIEmbeddingsProcessor()
        df_processed = processor.process_dataframe(df)

        # Retain only the necessary columns
        print("Retaining necessary columns...")
        relevant_cols = [
            'story_embedding', 'summary_embedding', 'generated_story_embedding',
            'story_summary_similarity', 'story_generated_story_similarity', 'closer_to_story'
        ]
        df_processed = df_processed[relevant_cols]
        print(df_processed['closer_to_story'].value_counts())

        # Save the processed dataframe
        df_processed.to_csv(args.output_path, index=False)
    else:
        df_processed = pd.read_csv(args.output_path)
    
        
    # Generate plots using the provided paths and filenames
    print(df_processed['closer_to_story'].value_counts())
    plot(df_processed, 'boxplot', args.plot_save_path, args.value_plot_filename, args.boxplot_filename)
    plot(df_processed, 'value_plot', args.plot_save_path, args.value_plot_filename, args.boxplot_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process embeddings and plot data")
    parser.add_argument("--input_path", type=str, default="datasets/hcV3-imagined-stories-with-generated-few-shot.csv")
    parser.add_argument("--output_path", type=str, default="datasets/experiment2/experiment2-embeddings.csv")
    parser.add_argument("--plot_save_path", type=str, default='datasets/experiment2/results', help="Path where plots will be saved.")
    parser.add_argument("--value_plot_filename", type=str, default='experiment2-value-plot-few-shot.png', help="Filename for the value plot.")
    parser.add_argument("--boxplot_filename", type=str, default='experiment2-similarity-scores-few-shot.png', help="Filename for the boxplot.")
    parser.add_argument("--preprocessing", type=bool, default=False, help="Flag to indicate if data is already preprocessed.")
    args = parser.parse_args()
    
    main(args)

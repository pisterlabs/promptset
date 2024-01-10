# -*- coding: utf-8 -*-
import json
import os
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import openai
import pandas as pd
from tqdm import tqdm

from src.tools.text_processing import extract_unk_tokens, most_frequent_unk_tokens, remove_punctuation, extract_oov_words, most_frequent_oov_words, print_metric_examples, compute_cossim_wrt_oov
from src.metrics.oov_words import OOVWords
from src.metrics.unk_tokens import UNKTokens
from src.metrics.cosine_similarity import CosineSimilarity

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import tiktoken

from src.visualization.visualize import plot_histogram, plot_lineplot

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path(writable=True))
@click.argument('text_cols', nargs=-1, type=click.STRING)
def main(input_filepath, output_path, text_cols):

    click.echo(f"Input filepath is: {input_filepath}")
    click.echo(f"Output path is: {output_path}")
    text_cols = list(text_cols)
    click.echo(f"Text columns are: {text_cols}")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    click.echo(f"Output path is: {output_path}")


    openai.api_type = "azure"
    openai.api_key = os.environ.get('OPENAI_API_KEY')
    openai.api_base = os.environ.get('OPENAI_API_BASE')
    openai.api_version = os.environ.get('OPENAI_API_VERSION')

    tqdm.pandas()

    # Load data
    data = pd.read_json(input_filepath)
    data_exp = data.explode(text_cols, ignore_index=True)
    click.echo(f"Loaded {len(data_exp)} samples.")

    for text_col in text_cols:
        data_exp[text_col] = data_exp[text_col].apply(lambda x: remove_punctuation(x))

    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    model_openai = "precomputed"
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    tokenizer_openai = tiktoken.get_encoding("cl100k_base")
    device = "cuda"

    unk_metric = UNKTokens(model, tokenizer, device)

    click.echo(f"Computing UNK tokens using: {unk_metric.tokenizer}")
    out_cols = [f"{text_col}_unk" for text_col in text_cols]
    save_paths = os.path.join(output_path, "unk.png")
    data_exp = extract_unk_tokens(data_exp, unk_metric, text_cols, out_cols, save_paths)

    for text_col in text_cols:
        most_frequent_unk_tokens(data_exp, text_col, unk_metric)
    


    """
    oov_metric = OOVWords(model, tokenizer, device)
    oov_metric_openai = OOVWords(model, tokenizer_openai, device)

    out_cols = [f"{text_col}_oov" for text_col in text_cols]
    save_paths = os.path.join(output_path, "oov.png")
    save_paths_openai = os.path.join(output_path, "oov_openai.png")
    data_exp = extract_oov_words(data_exp, oov_metric, text_cols, out_cols, save_paths)
    data_exp_openai = extract_oov_words(data_exp, oov_metric_openai, text_cols, out_cols, save_paths_openai)

    
    for text_col in text_cols:
        click.echo(f"Using tokenizer: {oov_metric.tokenizer}")
        most_frequent_oov_words(data_exp, text_col, oov_metric)
        click.echo(f"Using tokenizer: {oov_metric_openai.tokenizer}")
        most_frequent_oov_words(data_exp_openai, text_col, oov_metric_openai)

    """

    # cos_sim_metric = CosineSimilarity(model, tokenizer, device)
    # cos_sim_metric_openai = CosineSimilarity(model_openai, tokenizer_openai, device)

    # click.echo("Cosine similarity examples:")
    # print_metric_examples(data_exp, 'recaptions_oov', text_cols, cos_sim_metric)
    # click.echo("Cosine similarity examples (OpenAI):")
    # print_metric_examples(data_exp_openai, 'recaptions_oov', text_cols, cos_sim_metric_openai)


    # click.echo(f"Computing cosine similarity using: {cos_sim_metric.tokenizer}")
    
    # data_exp['cos_sim'] = data_exp.progress_apply(lambda row: cos_sim_metric.get_metric(
    #    row[text_cols[0]], row[text_cols[1]])[0][0], axis=1)
    
    """
    click.echo(f"Computing cosine similarity using: {cos_sim_metric_openai.tokenizer}")
    data_exp_openai['cos_sim'] = data_exp_openai.progress_apply(lambda row: cos_sim_metric_openai.get_metric(
        row[text_cols[0]], row[text_cols[1]], paths=[row[f"{text_cols[0]}_emb_path"], row[f"{text_cols[1]}_emb_path"]])[0][0], axis=1)
    click.echo("Saving cosine similarities to disk.")
    data_exp_openai.to_json(os.path.join(output_path, "cos_sim_openai.json"), orient='records')
    click.echo("Plotting histograms.")
    # plot_histogram(data_exp, ['cos_sim'], title="Cosine similarity", xlabel="Cosine similarity", ylabel="Number of texts", bins=20, figsize=(9, 5), show=False, save_path=os.path.join(output_path, "cos_sim.png"))
    plot_histogram(data_exp_openai, ['cos_sim'], title="Cosine similarity (OpenAI)", xlabel="Cosine similarity", ylabel="Number of texts", bins=20, figsize=(9, 5), show=False, save_path=os.path.join(output_path, "cos_sim_openai.png"))

    # cos_sim_vs_oov_df = compute_cossim_wrt_oov(data_exp, oov_col='recaptions_oov', cos_sim_col='cos_sim')
    cos_sim_vs_oov_df_openai = compute_cossim_wrt_oov(data_exp_openai, oov_col='recaptions_oov', cos_sim_col='cos_sim')

    # Lineplots
    # plot_lineplot(cos_sim_vs_oov_df, x='oov', y='mean', err='std', title="Cosine similarity vs OOV words", xlabel="Number of OOV words", ylabel="Cosine similarity", figsize=(9, 5), show=False, save_path=os.path.join(output_path, "cos_sim_vs_oov.png"))
    plot_lineplot(cos_sim_vs_oov_df_openai, x='oov', y='mean', err='std', title="Cosine similarity vs OOV words (OpenAI)", xlabel="Number of OOV words", ylabel="Cosine similarity", figsize=(9, 5), show=False, save_path=os.path.join(output_path, "cos_sim_vs_oov_openai.png"))
    """
    click.echo("Done!")

     




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

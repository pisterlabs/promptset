import json
import pandas as pd
import argparse
import os
import numpy as np
# from pretrained_model_list import MODEL_PATH_LIST
# import promptsource.templates
from tqdm import tqdm
import ipdb

def clean_up_tokenization(out_string: str) -> str:
    """
    Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms.
    Args:
        out_string (:obj:`str`): The text to clean up.
    Returns:
        :obj:`str`: The cleaned-up string.
    """
    out_string = (
        out_string.replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace(" ' ", "'")
        .replace(" n't", "n't")
        .replace(" 'm", "'m")
        .replace(" 's", "'s")
        .replace(" 've", "'ve")
        .replace(" 're", "'re")
        .replace("\n\n", " ")
        .replace("\n", " ")
        .replace("\r", " ")
    )
    return out_string

def load_prompts(opt):
    if opt.prompt_set == "nlibias":
        return load_bbnli(opt)

    prompts = []
    if opt.prompt_set == "bold":
        pth = os.path.join("data", opt.prompt_set, "prompts", opt.prompt_domain+"_prompt.json")
        with open(pth) as f:
            for line in f:
                prompts.append(json.loads(line))
        prompts = prompts[0]
        prompts_df = pd.DataFrame(columns = ["Name", "Group", "Prompt"])
        for group, content in prompts.items():
            for name, prompt_l in content.items():
                for prompt in prompt_l:
                    if prompt.strip != "":
                        prompts_df.loc[len(prompts_df)] = [name, group, prompt]

    elif opt.prompt_set == "honest":
        pth = os.path.join("honest/resources/en_template.tsv")
        prompts_df = pd.read_csv(pth, sep="\t")
        prompts_df['template_masked'] = prompts_df['template_masked'].apply(lambda x: x.rstrip("[M]."))
        # remove some of the names: "the young", "the youngs", "the child", "the children"
        lst = ["the young", "the youngs", "the child", "the children"]
        prompts_df = prompts_df[~prompts_df['identity'].isin(lst)]
        prompts_df = prompts_df.rename(columns={"template_masked":"Prompt",
                                                "mf":"Group",
                                                "identity":"Name"})
        print(prompts_df.columns)
        assert all(["[M]" not in p for p in prompts])
    else:
        raise ValueError()

    return prompts_df

# def load_bbnli(opt):
#     # pth = os.path.join("outputs", "nlibias", opt.prompt_domain, opt.prompt_subtopic+".csv")
#     nli_temp = "GPT-3 style"
#     template_collection = promptsource.templates.TemplateCollection()
#     temp = template_collection.get_dataset("anli", None)[nli_temp]
#     def create_gpt3_prompt(row):
#         m = {"premise": row["premise"],
#              "hypothesis": row["hypothesis"]}
#         inp = env.from_string(temp).render(**m)
#         return inp
#     pth = "data/nli/validation_secret.csv"
#     prompts_df = pd.read_csv(pth)
#     prompts_df["Prompt"] = prompts_df.apply(lambda x: create_gpt3_prompt(x), axis=1)
#     prompts_df.rename(columns={"Domain":"Group"})
#     return prompts_df


def get_generations(prompts_df, opt):
    if opt.model_name == "gpt2":
        from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer

        model = GPT2LMHeadModel.from_pretrained(opt.model_path)
        tokenizer = GPT2Tokenizer.from_pretrained(opt.model_path)
    else:
        raise ValueError("Model name not supported.")
    text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
    ipdb.set_trace()
    # Drop entries that are empty.
    prompts_df['Prompt'].replace('', np.nan, inplace=True)
    print("Removing empty entries: ", prompts_df['Prompt'].isna().sum())
    prompts_df.dropna(subset=['Prompt'], inplace=True)
    prompts_df.reset_index(inplace=True, drop=True)

    # Start completions
    num_gens_t = opt.num_gens * len(prompts_df)
    print("Generating total of {} completions.".format(num_gens_t))
    gens = []
    empty_count = 0
    for prompt in prompts_df.Prompt.to_list():
        try:
            gen = text_generator(prompt,
                                max_new_tokens=opt.max_length,
                                do_sample=opt.do_sample,
                                temperature=opt.temperature,
                                num_return_sequences=opt.num_gens,
                                clean_up_tokenization_spaces=True)
            gens.append(gen)
        except:
            print("FAILED: ", prompt)
            gen = [{"generated_text":"."}] * opt.num_gens
            gens.append(gen)
            empty_count +=1
    print("Generation completed. Empty prompt number: ", empty_count)

    gen_df = pd.DataFrame(columns = ["Name", "Group", "Prompt", "Generation"])
    for i,row in prompts_df.loc[:].iterrows():
        genset = gens[i]
        for gen in genset:
            gen_df.loc[len(gen_df)] = [row['Name'],
                                       row['Group'],
                                       row['Prompt'],
                                       gen['generated_text']]
    gen_df["Generation"] = gen_df['Generation'].str.replace(u'\xa0', u' ')
    return gen_df


def get_generations_gpt3(prompts_df, opt):
    import openai
    prompts_df['Prompt'] = prompts_df['Prompt'].apply(lambda x: x.rstrip(" "))

    def chunks(prompts_df, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(prompts_df), n):
            yield prompts_df.iloc[i:min(i + n, len(prompts_df)),:]

    openai.api_key = [el for el in open("openai_key", 'r')][0]
    gen_df = pd.DataFrame(columns = ["Name", "Group", "Prompt", "Generation"])
    chunks_ls = list(chunks(prompts_df, opt.batch_size))
    for chunk in tqdm(chunks_ls, total=len(chunks_ls)):
        # create a completion
        lst = [el.strip(" ") for el in chunk.Prompt.to_list()]
        completion = openai.Completion.create(engine="text-curie-001",
                                              prompt=lst,
                                              max_tokens=opt.max_length,
                                              temperature=opt.temperature,
                                              n=opt.num_gens)
        count = 0
        for i,row in chunk.iterrows():
            for j in range(opt.num_gens):
                cln = clean_up_tokenization(completion.choices[count].text)
                gen_df.loc[len(gen_df)] = [row['Name'],
                                           row['Group'],
                                           row['Prompt'],
                                           row['Prompt'] + cln]
                count += 1

    gen_df["Generation"] = gen_df['Generation'].str.replace(u'\xa0', u' ')
    return gen_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--prompt_set", type=str, default="bold")
    parser.add_argument("--prompt_domain", type=str, default="gender")
    parser.add_argument("--max_length", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--do_not_sample", action="store_false", dest="do_sample")
    parser.add_argument("--num_gens", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)

    opt = parser.parse_args()
    os.makedirs(opt.save_path, exist_ok=True)

    if not opt.do_sample:
        assert opt.num_gens == 1

    # Jinja env.
    # global env
    # env = nativetypes.NativeEnvironment()
        
    prompts_df = load_prompts(opt)
    if opt.model_name == "gpt2":
        gen_df = get_generations(prompts_df, opt)
    elif opt.model_name == "gpt3":
        gen_df = get_generations_gpt3(prompts_df, opt)
    else:
        raise ValueError(f"{opt.model_name} is not known.")
    
    pth = os.path.join(opt.save_path,
                       "len_{}_num_{}_temp_{}_gens.csv".format(opt.max_length, opt.num_gens, opt.temperature))
    gen_df.to_csv(pth)
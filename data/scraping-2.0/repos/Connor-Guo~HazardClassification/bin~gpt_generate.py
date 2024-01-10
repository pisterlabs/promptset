"""
利用ChatGPT3.5生成数据。
仅针对训练集生成。

生成的数据按标签存放在`../out/ChatGPT/{EXP_NAME}/`文件夹下。
除了单个标签外，还包括所有生成数据的合并文件、所有（原+生成）数据的合并文件。
后者会被复制一份到`../out/dataset/{EXP_NAME}/`文件夹下。

"""
import os

import openai
import pandas as pd

from src.loadfiles import *
from src.sampling import *


if __name__ == "__main__":
    # 参数设置
    EXP_NAME = "guanzhi_fix_bert_abbr_2941"
    SAVE_NAME = "guanzhi_fix_bert_abbr_2941_GPT_3"
    save_folder = f"../out/ChatGPT/{SAVE_NAME}/"
    dataset_folder = f"../out/datasets/{SAVE_NAME}"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    openai.api_key = input("OpenAI API key:")
    prompt_fmt = """
    你是一名空中交通管理专家。

    你可以将这段事件描述用其他方式转述出来吗？请提供{}种转述。请确保转述中包含所有重要的细节，例如潜在的后果、系统的名称、以及其他相关的专业名词或缩写。请用中文转述。
    
    注意：事件描述可能由若干个子事件构成，这些子事件有时会由分号等符号分割。在转述时，请务必囊括所有子事件，而不是只包含其中的一个或数个。
    
    输出应当是JSON格式，包括以下关键字："Paraphrase"，并且"Paraphrase"下的元素应该是一个包含所有转述字符串组成的列表。

    事件描述：
    “{}”
    """
    r_dict = {
        1: 150,
        5: 150,
        6: 20,
        7: 50,
        8: 150,
        9: 150,
        10: 150,
        12: 20,
        13: 200,
        14: 150,
        15: 30,
        16: 50,
        17: 150,
        18: 50,
        19: 200,
        20: 50,
        21: 200,
    }  # 标签对应的采样目标数量

    df = load_pickle(f'../out/datasets/{EXP_NAME}/{EXP_NAME}-train.pkl')
    resampler = Resampler(df,
                          id_col="危险源编号",
                          text_col="后果",
                          labels_cols=['label', 'label2', 'label3', 'label4', 'label5'],
                          chinese_labels_cols=['不安全事件', '不安全事件2', '不安全事件3', '不安全事件4',
                                               '不安全事件5'],
                          )

    # 开始生成数据
    lst_df_generated = []
    # for label, expected_n in r_dict.items():
    #     df_generated = resampler.generate_with_gpt_2(
    #         label=label,
    #         prompt_fmt=prompt_fmt,
    #         expected_n=expected_n,
    #         save_folder=save_folder,
    #         lb_text_len=8,
    #         load=True,
    #     )
    #     # 对每个标签保存一个文件
    #     dataset_path_1 = os.path.join(save_folder, f"{label}.xlsx")
    #     df_generated.to_excel(dataset_path_1, index=False)
    #     lst_df_generated.append(df_generated)

    # 保存合并后的文件
    if len(lst_df_generated) != len(r_dict):
        for label in list(r_dict.keys()):
            dataset_path_1 = os.path.join(save_folder, f"{label}.xlsx")
            lst_df_generated.append(pd.read_excel(dataset_path_1))
        df_generated_all = pd.concat(lst_df_generated)

    df_all = pd.concat((df, df_generated_all)).sort_values(by="label", ascending=True)
    df_generated_all.to_excel(os.path.join(save_folder, "generated_all.xlsx"), index=False)
    df_all.to_excel(os.path.join(save_folder, f"{SAVE_NAME}-train.xlsx"), index=False)
    save_pickle(df_all, os.path.join(dataset_folder, f"{SAVE_NAME}-train.pkl"))
    df_all.to_excel(os.path.join(dataset_folder, f"{SAVE_NAME}-train.xlsx"), index=False)

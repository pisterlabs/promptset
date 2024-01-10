"""
对不平衡数据集的重采样

(UNFINISHED)
"""
import os.path
import time
import warnings
from typing import TypeVar, Tuple, List
from copy import deepcopy
import json
from time import sleep

import pandas as pd
import numpy as np
import openai

from src.loadfiles import *


__all__ = ['Resampler']
ResamplerT = TypeVar('ResamplerT', bound='Resampler')


class Resampler:
    def __init__(
            self,
            data: pd.DataFrame,
            id_col: str,
            text_col: str,
            labels_cols: List[str],
            chinese_labels_cols: List[str] = None,
    ) -> None:
        """"""
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Expected input type pd.DataFrame, but {type(data)} is provided.")
        if chinese_labels_cols is not None:
            assert len(labels_cols) == len(chinese_labels_cols)
        assert len(labels_cols) > 0
        # 检查列
        self.df = data
        self.data_original = data
        self.data_resampled = None
        self.data_generated = None
        self.id_col = id_col
        self.text_col = text_col
        self.labels_cols = labels_cols
        self.chinese_labels_cols = chinese_labels_cols

        self.GPT_sampler = None

    def get_labels_numpy(self, *, na_value=0):
        return self.df[self.labels_cols].to_numpy(na_value=na_value).astype(int)

    def get_data_by_label(self, label, deep_copy=True):
        # 从输入的df里拿到所有包含label的行
        labels = self.get_labels_numpy()
        contain_idx = np.where(label == labels)[0]
        if deep_copy:
            return deepcopy(self.df).iloc[contain_idx, :]
        else:
            return self.df.iloc[contain_idx, :]

    def shuffle(self, *, random_state=42):
        self.df = self.df.sample(frac=1, random_state=random_state)

    def del_labels(
            self,
            labels_to_del: List[int]
    ) -> pd.DataFrame:
        if isinstance(labels_to_del, list):
            labels_to_del = np.array(labels_to_del)
        elif isinstance(labels_to_del, np.ndarray):
            pass
        else:
            raise TypeError(
                f"Type of argument \"labels_to_del\" should be list or np.ndarray, but {type(labels_to_del)} is provided.")
        self.data_resampled = _del_labels(self.data_original, labels_to_del, self.labels_cols, self.chinese_labels_cols,
                                          fill_labels=True)
        return self.data_resampled

    def generate_with_gpt(
            self,
            label: int,
            prompt_fmt: str,
            expected_n: int,
            save_folder: str,
            lb_text_len: int = 12,
            shuffle: bool = True,
            random_state: int = 42,
    ):
        """
        采用"gpt-3.5-turbo"对一个标签进行过采样。
        保存采样记录至`{label}-log.xlsx`，缓存至`cache.pkl`，返回采样结果。

        Parameters
        ----------
        label : 待采样的标签
        prompt_fmt : 提示词的字符串，包含两个参数，分别是`n_paraphrase`和`narrative`
        expected_n : 标签的期待样本个数
            如果该数量小于等于已有样本数量，则不会进行采样。返回一个空DataFrame，且不保存log
            否则会持续采样，直到已有样本数量大于等于期待样本数量。
        save_folder : 输出记录文件的路径
            记录文件为excel文件，
        lb_text_len : 采样需要的最小文本长度。低于该长度的样本不会被传给ChatGPT采样
        shuffle : 采样时是否打乱输入
        random_state : 随机数种子

        Returns
        -------
        pd.DataFrame
            返回修改了危险源编号、paraphrase字符串及其标签的Dataframe，其中危险源编号为"样本编号-para-i"
            返回值中不包含原有样本
        """
        assert lb_text_len > 0

        self.data_generated = _oversample_label_gpt35(self.data_original, self.id_col, self.text_col, self.labels_cols,
                                                      label, prompt_fmt, expected_n, save_folder, lb_text_len,
                                                      shuffle, random_state)
        return self.data_generated

    def generate_with_gpt_2(
            self,
            label: int,
            prompt_fmt: str,
            expected_n: int,
            save_folder: str,
            lb_text_len: int = 12,
            shuffle: bool = True,
            random_state: int = 42,
            load: bool = True,
            load_fn: str = "cache.pkl",
    ) -> pd.DataFrame:

        # log目录和缓存目录
        log_path = os.path.join(save_folder, f"{label}-log.xlsx")
        cache_path = os.path.join(save_folder, "cache.pkl")

        # 从输入的df里拿到所有包含label的行
        df_label = self.get_data_by_label(label)

        # 如果样本已经充足，则不再采样
        n_samples = len(df_label)
        print(f"[Sampling] Label {label} has {n_samples} samples. It is expected to be resampled to {expected_n}.")
        if expected_n <= n_samples:
            print(f"[Sampling] Label {label} need not to be generated.")
            return pd.DataFrame()

        # 样本不足，继续流程
        # 删除重复和过短的行
        text_column = self.text_col
        df_label.drop(index=df_label.index[df_label[text_column].duplicated()], inplace=True)
        llb_idx = df_label[text_column].apply(lambda s: True if len(s) < lb_text_len else False)  # 过短为True
        df_label.drop(index=llb_idx.index[llb_idx], inplace=True)
        n_samples = len(df_label)  # 删除行后再计算一次当前样本数

        # 根据当前已有样本数和采样后预期样本总数，计算每个样本需要被采样几次
        n_paraphrase = np.ceil((expected_n - n_samples) / n_samples)

        # 创建GPTSampler
        if load:
            print(f"[Sampling] Loading GPTSampler from {load_fn}")
            self.GPT_sampler = load_pickle(os.path.join(save_folder, load_fn))
            if isinstance(self.GPT_sampler, GPTSampler):
                if label != self.GPT_sampler.label:
                    warnings.warn("Cannot load sampler with different label. A new sampler is created.")
                    self.GPT_sampler = GPTSampler(df_label, self.id_col, self.text_col, self.labels_cols, prompt_fmt,
                                                  expected_n, label, cache_fp=cache_path, log_fp=log_path)
                else:
                    print(f"[Sampling] Loaing success. Continue sampling with {len(self.GPT_sampler.resample_df)} samples.")
            else:
                warnings.warn("The save file nis not a GPTSampler object. A new sampler is created.")
                self.GPT_sampler = GPTSampler(df_label, self.id_col, self.text_col, self.labels_cols, prompt_fmt,
                                              expected_n, label, cache_fp=cache_path,
                                              log_fp=log_path)
        else:
            self.GPT_sampler = GPTSampler(df_label, self.id_col, self.text_col, self.labels_cols, prompt_fmt,
                                          expected_n, label, cache_fp=cache_path, log_fp=log_path)

        # 采样
        self.data_generated = self.GPT_sampler.sample(n_paraphrase, shuffle, random_state)
        print(f"[Sampling] Label {label} resampled to {len(self.data_generated) + len(df_label)} samples.")
        return self.data_generated


class GPTSampler:
    def __init__(
            self,
            df: pd.DataFrame,
            id_column: str,
            text_column: str,
            labels_cols: List[str],
            prompt_fmt: str,
            expected_n: int,
            label: int,
            cache_fp: str = "cache.pkl",
            log_fp: str = None,
    ) -> None:
        """

        Parameters
        ----------
        df : pd.DataFrame
            对df中所有数据进行抽样生成，不考虑标签。
        prompt_fmt : str
            prompt的格式见`Resampler.generate_with_gpt()`方法。
        expected_n

        label
        cache_fp
        log_fp
        """
        self.df = deepcopy(df)
        self.id_column = id_column
        self.text_column = text_column
        self.labels_cols = labels_cols
        self.label = label
        self.expected_n = expected_n
        self.cache_path = cache_fp
        self.log_path = log_fp
        if log_fp is None:
            self.log_path = f"{label}-log.xlsx"
        self.prompt_fmt = prompt_fmt

        self.n_lines_sampled = 0  # 已经采样了多少行
        self.resample_log_lst = []
        self.resample_df = pd.DataFrame(columns=self.df.columns)

    def shuffle(self, *, random_state: int = 42) -> None:
        self.df = self.df.sample(frac=1, random_state=random_state)

    def sample(
            self,
            n_paraphrase: int,
            shuffle: bool = True,
            random_state: int = 42,
    ):
        id_column = self.id_column
        text_column = self.text_column
        expected_n = self.expected_n

        # 打乱样本顺序
        if shuffle and self.n_lines_sampled == 0:
            self.shuffle(random_state=random_state)
        df = self.df

        # （继续）采样
        for i, (idx, line) in enumerate(df[self.n_lines_sampled:].iterrows()):
            hae_id = line[id_column]
            narrative = line[text_column]

            prompt = self.prompt_fmt.format(int(n_paraphrase), narrative)
            r_lst = _get_response_list(prompt)
            if len(r_lst) == 0:
                raise RuntimeError("A sample is failed. The process is saved. Please check or restart.")
            # 存储log和结果
            for j, para in enumerate(r_lst):
                para_id = f"{hae_id}-para-{j + 1}"
                # 写log
                self.resample_log_lst.append(
                    {
                        "hae_id": hae_id,
                        "para_id": para_id,
                        "narrative": narrative,
                        "paraphrase": para,
                    }
                )
                # 写结果
                new_line = deepcopy(line)
                new_line[id_column] = para_id
                new_line[text_column] = para
                self.resample_df.loc[len(self.resample_df)] = new_line
            # 每次得到数据后缓存、打印日志
            save_pickle(self, self.cache_path)
            print(
                f"[Sampling] {len(self.resample_df)} samples acquired. "
                f"{max(0, expected_n - len(self.resample_df) - len(df))} left.")
            # 判断是否结束
            if len(self.resample_df) + len(df) >= expected_n:
                break
        # 存log
        pd.DataFrame(self.resample_log_lst).to_excel(self.log_path, index=False)
        # 返回修改了危险源编号、paraphrase字符串及其标签的Dataframe，其中危险源编号为"{样本编号}-para-{i}"
        # 返回值中不包含原有样本
        return self.resample_df


def _del_labels(
        data: pd.DataFrame,
        labels_to_del: np.ndarray,
        labels_cols: List[str],
        chinese_labels_cols: List[str] = None,
        fill_labels: bool = True,
) -> pd.DataFrame:
    """
    删除指定标签与对应的数据，并将后方的标签编号前移。

    Parameters
    ----------
    data
    labels_to_del
    labels_cols
    chinese_labels_cols
    fill_labels

    Returns
    -------

    Examples
    --------
    >>> data = pd.DataFrame({
    >>> "label1": [1, 2, 1, 3],
    >>> "label2": [2, 3, 3, np.nan],
    >>>  "text1":  ['A', 'B', 'A', 'C'],
    >>>  "text2":  ['B', 'C', 'C', np.nan],
    >>> })
    >>> data
           label1  label2 text1 text2
    0       1     2.0     A     B
    1       2     3.0     B     C
    2       1     3.0     A     C
    3       3     NaN     C   NaN

    >>> labels_to_del = [3]
    >>> labels_cols = ['label1', 'label2']
    >>> chinese_labels_cols = ['text1', 'text2']
    >>> _del_labels(data, labels_to_del, labels_cols, chinese_labels_cols, fill_labels=False)
           label1  label2 text1 text2
    0       1     2.0     A     B
    1       2     0.0     B   NaN
    2       1     0.0     A   NaN
    """
    # 数据类型转换
    if not isinstance(labels_to_del, np.ndarray):
        labels_to_del = np.array(labels_to_del)

    # 删除{待删除标签}中的标签
    data_new = deepcopy(data)
    for label_to_del in labels_to_del.astype(int):
        data_new = _del_single_label(data_new, label_to_del, labels_cols, chinese_labels_cols)

    if fill_labels:
        # labels = data_new[labels_cols].to_numpy()
        labels = _get_labels_from_df(data_new, labels_cols)
        filled_labels = _fill_labels(labels, labels_to_del)
        data_new.loc[:, labels_cols] = filled_labels
    return data_new


def _get_labels_from_df(
        df: pd.DataFrame,
        labels_cols: List[str],
) -> np.ndarray:
    return df[labels_cols].to_numpy()


def _del_single_label(
        data: pd.DataFrame,
        label_to_del: int,
        labels_cols: List[str],
        chinese_labels_cols: List[str] = None,
) -> pd.DataFrame:
    """

    Parameters
    ----------
    data
    label_to_del
    labels_cols
    chinese_labels_cols

    Returns
    -------

    Examples
    --------
    >>> data = pd.DataFrame({
    >>>     "label1": [1, 2, 1, 3],
    >>>     "label2": [2, 3, 3, np.nan],
    >>>     "text1":  ['A', 'B', 'A', 'C'],
    >>>     "text2":  ['B', 'C', 'C', np.nan],
    >>> })
    >>> data
           label1  label2 text1 text2
    0       1     2.0     A     B
    1       2     3.0     B     C
    2       1     3.0     A     C
    3       3     NaN     C   NaN

    >>> label_to_del = 3
    >>> labels_cols = ['label1', 'label2']
    >>> chinese_labels_cols = ['text1', 'text2']
    >>> _del_single_label(data, label_to_del, labels_cols, chinese_labels_cols)
       label1  label2 text1 text2
    0       1     2.0     A     B
    1       2     0.0     B   NaN
    2       1     0.0     A   NaN
    """
    data_copy = deepcopy(data)

    for idx, line in data.iterrows():
        # 提取label和中文的部分
        labels = line.loc[labels_cols].to_numpy(na_value=0).astype(int)
        if chinese_labels_cols is not None:
            chinese = line.loc[chinese_labels_cols].to_numpy()

        if label_to_del in labels:  # 1a如果这行存在要删除的标签
            if len(labels) == 1 or np.nan_to_num(labels, nan=0).sum() == label_to_del:  # 2a.如果这行只有这一个标签，那么直接删除这行
                data_copy.drop(index=idx, inplace=True)
            else:  # 2b.如果这行不止一个标签，那么替换标签
                start = np.argwhere(labels == label_to_del).ravel()[0]
                if start == len(labels) - 1:  # 3a如果被替换的标签刚好是最后一个，直接替换为0
                    labels[start] = 0
                    if chinese_labels_cols is not None:
                        chinese[start] = np.nan
                else:  # 3b如果被替换的标签不是最后一个，则标签整体前移
                    labels[start:-1] = labels[start + 1:]
                    labels[-1] = 0
                    if chinese_labels_cols is not None:
                        chinese[start:-1] = chinese[start + 1:]
                        chinese[-1] = np.nan
                # 4将labels的0.0全部换为np.nan
                labels = np.where(labels, labels, np.nan)
                # 执行操作，替换data_copy
                data_copy.loc[idx, labels_cols] = labels
                if chinese_labels_cols is not None:
                    data_copy.loc[idx, chinese_labels_cols] = chinese
        else:  # 1b如果这行不用操作
            pass

    return data_copy


def _fill_labels(
        labels: np.ndarray,
        labels_to_del: np.ndarray,
) -> np.ndarray:
    """
    Move the labels forward to make the labels continuous.

    Examples
    --------
        >>> labels = np.array([[1, 3, 0], [1, 5, 6]])  # labels 2,4 are absent
        >>> labels_to_del = np.array([2, 4])
        >>> _fill_labels(labels, labels_to_del)
        array([[1, 2],
               [1, 3]])  # label 3 becomes label 2, and label 5 becomes label 3

    """
    labels_to_del = deepcopy(labels_to_del)
    filled_labels = labels[:]  # make a copy

    labels_to_del.sort()
    # Consider that each time a label is filled, all label values greater than that label get 1 smaller,
    # It's better to generate a new array of labels_to_del in advance.
    labels_to_del_trans = [_ - i for i, _ in enumerate(labels_to_del)]
    for label in labels_to_del_trans:
        for i in range(filled_labels.shape[0]):
            for j in range(filled_labels.shape[1]):
                if filled_labels[i, j] > label:
                    filled_labels[i, j] = filled_labels[i, j] - 1
    return filled_labels


def get_completion(prompt: str, model="gpt-3.5-turbo") -> str:
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,  # this is the degree of randomness of the model ’s output
    )
    return response.choices[0].message["content"]


def _json_2_dict(s: str):
    """input string should be the output of `get_completion`"""
    return json.loads(s)


def _get_response_list(
        prompt: str,
        *,
        tries: int = 0,
) -> List[str]:
    if tries > 2:  # 最大重试次数
        print("[Sampling] Retry failed. This sample is skipped.")
        return []
    try:
        response = get_completion(prompt)
        lst = _json_2_dict(response)["Paraphrase"]
    except openai.error.OpenAIError as e:
        print("[Sampling] OpenAIError: ", e, "Retrying...")
        sleep(20)
        lst = _get_response_list(prompt, tries=tries + 1)
    except Exception as ee:
        return []
    return lst


def _oversample_label_gpt35(
        df: pd.DataFrame,
        id_column: str,
        text_column: str,
        labels_cols: List[str],
        label: int,
        prompt_fmt: str,
        expected_n: int,
        save_folder: str,
        lb_text_len: int,
        shuffle: bool,
        random_state: int,
) -> pd.DataFrame:
    """
    采用"gpt-3.5-turbo"对一个标签进行过采样。
    保存采样记录至`{label}-log.xlsx`，缓存至`cache.pkl`，返回采样结果。

    Parameters
    ----------
    df : 包含所有标签的样本的DataFrame
    id_column : 危险源编号的列名
    text_column : 待采样文本的列名
    labels_cols : 标签的列名
    label : 待采样的标签
    prompt_fmt : 提示词的字符串，包含两个参数，分别是`n_paraphrase`和`narrative`
    expected_n : 标签的期待样本个数
        如果该数量小于等于已有样本数量，则不会进行采样。返回一个空DataFrame，且不保存log
        否则会持续采样，直到已有样本数量大于等于期待样本数量。
    save_folder : 输出记录文件的路径
        记录文件为excel文件，
    lb_text_len : 采样需要的最小文本长度。低于该长度的样本不会被传给ChatGPT采样
    shuffle : 采样时是否打乱输入
    random_state : 随机数种子

    Returns
    -------
    pd.DataFrame
        返回修改了危险源编号、paraphrase字符串及其标签的Dataframe，其中危险源编号为"样本编号-para-i"
        返回值中不包含原有样本
    """
    # log目录和缓存目录
    log_path = os.path.join(save_folder, f"{label}-log.xlsx")
    cache_path = os.path.join(save_folder, "cache.pkl")

    # 从输入的df里拿到所有包含label的行
    labels = _get_labels_from_df(df, labels_cols)
    contain_idx = np.where(label == labels)[0]
    df_label = deepcopy(df).iloc[contain_idx, :]
    # 如果样本已经充足，则不再采样
    n_samples = len(df_label)
    print(f"[Sampling] Label {label} has {n_samples} samples. It is expected to be resampled to {expected_n}.")
    if expected_n <= n_samples:
        print(f"[Sampling] Label {label} need not to be generated.")
        return pd.DataFrame()
    # 样本不足，继续流程
    # 删除重复和过短的行
    df_label.drop(index=df_label.index[df_label[text_column].duplicated()], inplace=True)
    llb_idx = df_label[text_column].apply(lambda s: True if len(s) < lb_text_len else False)  # 过短为True
    df_label.drop(index=llb_idx.index[llb_idx], inplace=True)
    n_samples = len(df_label)  # 删除行后再计算一次当前样本数
    # 根据当前已有样本数和采样后预期样本总数，计算每个样本需要被采样几次
    n_paraphrase = np.ceil((expected_n - n_samples) / n_samples)
    # 判断是否需要动态采样（为确保均匀采样）
    # dynamic_sample = True if n_paraphrase > 1 else False
    # 根据计算的样本数量，调用函数进行采样
    resample_log_lst = []
    resample_df = pd.DataFrame(columns=df.columns)
    # 打乱样本顺序
    if shuffle:
        df_label = df_label.sample(frac=1, random_state=random_state)
    # 采样
    for i, (idx, line) in enumerate(df_label.iterrows()):
        hae_id = line[id_column]
        narrative = line[text_column]

        # 为确保均匀采样，动态改变n_paraphrase，使得每个样本都尽可能被采到
        # if dynamic_sample:
        #     if n_samples + (n_paraphrase - 1) * (n_samples - i) + n_paraphrase * i >= expected_n:
        #         dynamic_n_paraphrase = n_paraphrase - 1
        #     else:
        #         dynamic_n_paraphrase = n_paraphrase
        # else:
        #     dynamic_n_paraphrase = n_paraphrase
        prompt = prompt_fmt.format(int(n_paraphrase), narrative)
        r_lst = _get_response_list(prompt)
        # 存储log和结果
        for i, para in enumerate(r_lst):
            para_id = f"{hae_id}-para-{i + 1}"
            # 写log
            resample_log_lst.append(
                {
                    "hae_id": hae_id,
                    "para_id": para_id,
                    "narrative": narrative,
                    "paraphrase": para,
                }
            )
            # 写结果
            new_line = deepcopy(line)
            new_line[id_column] = para_id
            new_line[text_column] = para
            resample_df.loc[len(resample_df)] = new_line
        # 每次得到数据后缓存
        save_pickle(resample_df, cache_path)
        print(
            f"[Sampling] {len(resample_df)} samples acquired. {max(0, expected_n - len(resample_df) - len(df_label))} left.")
        # 判断是否结束
        if len(resample_df) + len(df_label) >= expected_n:
            print(f"[Sampling] Label {label} resampled to {len(resample_df) + len(df_label)} samples.")
            break
    # 存log
    pd.DataFrame(resample_log_lst).to_excel(log_path, index=False)
    # 返回修改了危险源编号、paraphrase字符串及其标签的Dataframe，其中危险源编号为"{样本编号}-para-{i}"
    # 返回值中不包含原有样本
    return resample_df


if __name__ == "__main__":
    # data = pd.DataFrame({
    #     "label1": [1, 2, 1, 3],
    #     "label2": [2, 3, 3, np.nan],
    #     "text1": ['A', 'B', 'A', 'C'],
    #     "text2": ['B', 'C', 'C', np.nan],
    # })
    # label_to_del = 3
    # labels_cols = ['label1', 'label2']
    # chinese_labels_cols = ['text1', 'text2']
    # _del_single_label(data, label_to_del, labels_cols, chinese_labels_cols)
    pass

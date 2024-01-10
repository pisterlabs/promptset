import openai
import os
import json
import copy
import sys
import os
import pdb
import re
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import re
import argparse
import functools
from statsmodels.nonparametric.kde import KDEUnivariate

_label_fontsize = 18
_tick_fontsize = 18

import tkinter as tk
from PIL import Image, ImageTk
import cv2
from scipy.stats import kendalltau
from scipy.stats import spearmanr

from termcolor import colored

openai.api_key = "INSERT YOUR KEY HERE"


def matching_score_prompt(traj_info: dict, description):
    """
    """

    prompt = """Let us consider a square of size 200x200. The origin is fixed at the bottom left corner, and the x,y axes are respectively horizontal and vertical (with x looking towards the right, i.e. east, and y looking upwards, i.e. north). Let us define a python dictionary to represent point that have been sampled from a 2d trajectory in that square. The dictionary will have the following keys: dict_keys(['timestep', 'pos', 'semantics', 'colors']). Here is the explanation for each: 1) The complete trajectories are composed of N points, and each has a timestep t_i (with i ranging from 0 to N-1).  The 'timestep' key corresponds to the t_i of the sampled point. 2) the 'pos' key is for the 2d position of the point, expressed as (x,y) in the coordinate frame defined above. 3) The square actually represents a room, where there are several objects such as a fridge, a chair, a cactus and so on. The 'semantics' gives information on objects to which the point are close (name of objects, and where the agent is situated w.r.t those objects, e.g. to the east or north of the cactus, etc). 4) The room which is represented by the 200x200 square also has tiles of different colors in different areas. The 'colors' key gives information about the tile color where the 2d point is. 

*******
Colored tiles: the 200x200 room is divided in 9 tiles, each of size approximately 66x66. From top to bottom, the layout is given below.

      Red     | Green  | Blue
      -------------------------
      Yellow  | Orange | Purple
      -------------------------
      Pink    | Gray   | White

*******
Here is a list of the objects present in the environment, their coordinates and the color of the tiles they are on:

    ============= table 1============
    - cactus, (180, 27), purple
    - fan, (60, 140),  red/yellow
    - sofa, (167,180), blue
    - bed, (164,26), white
    - fridge, (22,28), pink
    - wood cabinet, (23,176), red
    - chair, (98,25), gray
    - statue, (120,105), orange
    - file cabinet, (140,138), orange/green
    - bathtub, (58,97), yellow
    - table, (123, 25), gray
    - stove, (64,25), pink

*******
You will receive a natural language description after the [DESCR] tag. This description has been given as a command to an agent, which has tried to execute it in the environment. The agent's resulting trajectory will be given to you after the tag [DICT], using the dictionary format described above. Your task is to evaluate how well the agent has been able to follow the instructions. More precisely, your task will be composed of those steps:

A) Answer those questions: 

    Q1. What was the description asking the agent to reach?
    Q2. What did the agent reach?
    Q3. What was the final destination that the agent was expected to reach?
    Q4. Did the agent reach that final destination? At which point in the dictionary?
    Q5. How far is the agent's final state from the expected final state?
    Q6. What objects/colors was the agent asked to visit, reach, pass by or encounter?
    Q7. What objects/colors did the agent actually visit, reach, pass by or encounter? How many were extra? How many did it miss?

    Note that in the table 1 above, a correspondance between colors, coords and objects has been given. 

B) Assume that your end goal is to write a numerical score in [0,1] that rates how well the agent's behavior matches the desired description. How would you proceed here? Please write your reasoning here.

C) Using what you know so far, write a numerical score in [0,1] that rates the agent's performance. You must always come to a numerical decision, no matter how difficult the task. 

*******

FORMATTING RULE: you must write the final score as 'score==<floating_value>'

*******"""

    prompt = prompt + "\n" + f"[DESCR] {description} \n [DICT] {traj_info} \n + ATTENTION: always exactly follow the formatting instruction in your score: the score should be given as 'score==<your_float_value>'"

    return prompt


def round_pos_info(traj_lst, round_n=1):

    for s_i in range(len(traj_lst)):
        for ii in range(2):
            traj_lst[s_i]["pos"][ii] = round(traj_lst[s_i]["pos"][ii], round_n)

    return traj_lst


def get_file_index(fn):

    if fn[-5:] == ".json":
        return get_trailing_number(fn[:-5])
    elif fn[-4:] == ".png":
        return get_trailing_number(fn[:-4])
    else:
        raise Exception("unknown file type")


def get_trailing_number(s):
    m = re.search(r'\d+$', s)
    return int(m.group()) if m else None


def dumb_down_traj_for_gpt3(traj_lst: list):
    """
    gpt3.x gets confused by extra info about distances from other objects/tiles etc
    """
    ann = copy.deepcopy(traj_lst)
    for ii in range(len(ann)):
        ann[ii]["semantics"] = list(ann[ii]["semantics"].keys())
        ann[ii]["colors"] = ann[ii]["colors"][0]

    return ann


def eval_with_llm(pair_dict):
    """
    pair_dict  should correspond to a single trajectory and have keys 'prompt' and 'predicted_traj_annotation'
    """
    natural_language_descr = pair_dict["prompt"]
    traj_descr = round_pos_info(
        dumb_down_traj_for_gpt3(pair_dict["predicted_traj_annotation"]))

    prompt = matching_score_prompt(traj_descr, natural_language_descr)

    temperature = 1e-10  #attention, never pass 0, as it will be automatically set to 1.0 instead because of https://github.com/sashabaranov/go-openai/issues/34#issuecomment-1457029902
    response_0 = openai.ChatCompletion.create(
        model="gpt-4-0314",
        messages=[{
            "role": "system",
            "content": "You are a helpful assistant."
        }, {
            "role": "user",
            "content": f"{prompt}"
        }],
        temperature=temperature)

    msg_0 = response_0["choices"][0]["message"]["content"]

    def extract_score_and_print(msg, index):

        print(f"******msg_{index}******************")
        print(msg)
        match_re = re.search(r'score==([0-9]+\.[0-9]+)',
                             msg.replace(" == ", "=="))
        score = float(match_re.group(1))

        print(f"******score_{index}******************")
        print(score)
        return score

    #response_1=openai.ChatCompletion.create(
    #        model="gpt-3.5-turbo-0301",
    #        messages=[
    #            {"role": "system", "content": "You are a helpful assistant."},
    #            {"role": "user", "content": f"{prompt}"},
    #            {"role": "assistant", "content":f"{msg_0}"},
    #            {"role": "user", "content": "Thank you! We want to be sure about that evaluation, so please double-check the previous evaluation. If necessary, revise the score accordingly. Please use the same formatting as before."}],
    #        temperature=temperature)

    #msg_1=response_1["choices"][0]["message"]["content"]

    #score_0=extract_score_and_print(msg_0,0)
    #score_1=extract_score_and_print(msg_1,1)
    #return score_1

    score_0 = extract_score_and_print(msg_0, 0)

    return score_0


def ask_user(img, prompt):

    img = Image.fromarray(img)
    window = tk.Tk()

    tk_img = ImageTk.PhotoImage(img)

    score = [-1]

    def on_radio_button_click(value):
        score[0] = value
        window.destroy()
        return score

    upper_frame = tk.Frame(window)
    upper_frame.pack()
    text_label = tk.Label(upper_frame, text=prompt)
    text_label.pack()

    lower_frame = tk.Frame(window)
    lower_frame.pack()

    img_label = tk.Label(lower_frame, image=tk_img)
    img_label.pack()

    score_var = tk.IntVar()
    score_var.set(
        -1
    )  #to avoid a button being preselected when the window is first displayed
    for ii in range(11):  # 1 to 10
        radio_button = tk.Radiobutton(
            lower_frame,
            text=str(round(ii / 10, ndigits=1)),
            variable=score_var,
            value=ii,
            command=lambda val=round(ii / 10, ndigits=1
                                     ): on_radio_button_click(val))
        radio_button.pack(side=tk.LEFT)

    # Run the tkinter event loop
    window.mainloop()

    return score[0]


def visualize_dicts(*args):

    keys = [x.keys() for x in args if len(x.keys()) > 2][0]

    for dd in args:

        if len(dd.keys()) == 2:
            continue

        assert dd.keys() == keys

        coef = 1.0 if not dd["flipped_axis"] else -1.0
        zz = [
            coef * x for x in dd.values()
            if not isinstance(x, str) and not isinstance(x, bool)
        ]
        #plt.barh( #barh has a weird disappearing value bug
        plt.bar(range(len(dd.values()) - 2), zz, label=dd["type"], alpha=0.5)

    #plt.xticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1],
    #           [1.0, 0.8, 0.6, 0.4, 0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.legend(fontsize=14)
    plt.show()
    #pdb.set_trace()


def joint_distrib(scores_a, scores_b, num_scores=11):
    """
    assumes discrete scores in np.linspace(0,1,num_scores)

    scores_a, scores_b => vertical, horizontal
    """

    n = len(scores_a)
    assert n == len(scores_b)

    dist = np.zeros([num_scores, num_scores])
    for x, y in zip(scores_a, scores_b):

        dist[int(10 * x), int(10 * y)] += 1 / n

    scores = np.linspace(0, 1, num_scores).round(3).tolist()
    return dist, scores


def conditional(joint_in, dim, scores):
    """
    joint_in (np.array) discrete 2d distrib 
    dim   (int)      variable to condition on
    scores (list)    values of scores, so that joint[i,j] corresponds to the joint probability of scores[i] and scores[j]
    """

    joint_c = joint_in.copy()
    if dim == 0:
        p_y = joint_c.sum(1).reshape(-1, 1)  #p(y)
        joint_c /= p_y  #p(x,y)/p(y)
        distribs = {s: joint_c[int(10 * s), :].tolist()
                    for s in scores}  #per value
    elif dim == 1:
        p_x = joint_c.sum(0).reshape(1, -1)
        joint_c /= p_x
        distribs = {s: joint_c[:, int(10 * s)].tolist() for s in scores}

    return joint_c, distribs


def E_and_V(distrib_1d, score_values):

    NN = len(distrib_1d)
    expectation = functools.reduce(
        lambda acc, x: acc + score_values[x[0]] * x[1], enumerate(distrib_1d),
        0
    )  #this is actually the same as np.array(distrib_1d)*np.array(score_values)
    #print(expectation)
    variance = functools.reduce(
        lambda acc, x: acc + score_values[x[0]] * x[1],
        enumerate([(i - expectation)**2 for i in range(NN)]), 0)

    return expectation, variance


def sample_discrete(distrib, num_samples):
    """
    given a discrete distribution as a 1d array, returns n_samples indexes sampled according to their weights in distrib
    """

    N = len(distrib)

    cumulative = np.array([np.sum(distrib[:i + 1]) for i in range(N)])

    rr = np.random.rand(num_samples)
    samples = []
    for r_s in rr:

        samples.append((cumulative > r_s).argmax())

    return np.array(samples)


if __name__ == "__main__":

    _parser = argparse.ArgumentParser(
        description='evaluate trajectories generated in toy env')

    ### evaluation arguments
    _parser.add_argument(
        '--human_eval',
        action='store_true',
        help=
        'prompt user to evaluate how the trajectory pngs and descriptions match'
    )

    _parser.add_argument(
        '--llm_eval',
        action='store_true',
        help=
        'prompt an llm to evaluate how well the trajectory description and annotations match'
    )

    _parser.add_argument(
        '--dict_stats',
        type=str,
        default="",
        help=
        'displays stats about the dictionary. For stats about how two or more dicts correlate, see --visualize_dicts'
    )

    _parser.add_argument(
        '--input_path',
        type=str,
        help=
        'path to a directory containing pairs of files fig_<int>.png, pair_<int>.png',
        default="")

    _parser.add_argument(
        '--index_range',
        type=int,
        nargs=2,
        default=(),
        help=
        'Loading the input directory will result in N pairs p_0, ..., p_{N-1} after sorting. The arguments provided to index_range are treated as an interval [a,b) that specifies the p_a, ..., p_b pairs to be processed. Note that those are indexes into the sorted array of pairs. '
    )

    ### visualization and misc arguments
    _parser.add_argument(
        '--visualize_dicts',
        nargs="*",
        type=str,
        default=(),
        help=
        'visualizes all given dictionaries. Dictionaries of the same type (i.e. human or llm) will be merged in a single one which avergaes the scores for each datum. Note that this option assumes that at least one dictionary is supplied for each type (human, llm).'
    )

    _parser.add_argument(
        '--merge_non_overlapping_dicts',
        nargs="*",
        type=str,
        default=(),
        help=
        'dicts that are expected to have no overlap in their keys will be merded'
    )

    ### logging arguments
    _parser.add_argument('--output_dir', type=str, default="/tmp/eval_dir")

    _parser.add_argument(
        '--bd_dir',
        type=str,
        default="",
        help=
        "only taken into account when --dict_stats is given, will also retrieve the corresponding bd distance of the trajectories based on the keys of the dict"
    )

    _args = _parser.parse_args()

    if _args.merge_non_overlapping_dicts:

        dict_lst = []
        d_type = None
        for dd_fn in _args.merge_non_overlapping_dicts:

            with open(dd_fn, "r") as fl:
                dict_lst.append(json.load(fl))
                if d_type is None:
                    d_type = dict_lst[-1]["type"]
                else:
                    assert dict_lst[-1][
                        "type"] == d_type, "dictionaries must have the same type (human or llm)"

        final_dict = {
            k: max(0, min(v, 1.0))
            for dd in dict_lst for k, v in dd.items()
            if not (isinstance(v, str) or isinstance(v, bool))
        }  #we also clamp values in [-1,1] as in very rate situations, the LLM produces larger scores
        final_dict["type"] = d_type
        final_dict["flipped_axis"] = dict_lst[-1]["flipped_axis"]

        final_dict_path = f"{_args.output_dir}/final_dict_{d_type}"
        with open(final_dict_path, "w") as fl:
            json.dump(final_dict, fl)
        print(f"dictionary was saved to {final_dict_path}")

    elif _args.visualize_dicts:

        dicts_human = []
        dicts_llm = []

        for dic_path in _args.visualize_dicts:
            with open(dic_path, "r") as fl:
                dic_cur = json.load(fl)
                if dic_cur["type"] == "llm":
                    dicts_llm.append(dic_cur)
                else:
                    dicts_human.append(dic_cur)

        def dict_list_to_merged_dict(dicts_lst):
            dicts_merged = {
                k: []
                for k, v in dicts_lst[0].items() if not isinstance(v, str)
            }
            for kk in dicts_merged.keys():

                for dd in dicts_lst:
                    dicts_merged[kk].append(dd[kk])

            dicts_merged = {k: np.mean(v) for k, v in dicts_merged.items()}
            std = np.std(list(dicts_merged.values()))
            mean = np.mean(list(dicts_merged.values()))
            dicts_merged = {
                k: np.round(np.mean(v), decimals=1)
                for k, v in dicts_merged.items()
            }
            dicts_merged["type"] = dicts_lst[0]["type"]
            dicts_merged["flipped_axis"] = dicts_lst[0]["flipped_axis"]

            return dicts_merged, std, mean

        dicts_human_merged, std_human, mean_human = dict_list_to_merged_dict(
            dicts_human)
        dicts_llm_merged, std_llm, mean_llm = dict_list_to_merged_dict(
            dicts_llm)

        visualize_dicts(dicts_human_merged, dicts_llm_merged)

        assert dicts_llm_merged.keys() == dicts_human_merged.keys()
        mse = [(x - y)**2 for x, y in zip([
            u for u in dicts_human_merged.values()
            if not (isinstance(u, str) or isinstance(u, bool))
        ], [
            u for u in dicts_llm_merged.values()
            if not (isinstance(u, str) or isinstance(u, bool))
        ])]

        mae = [
            np.abs(x - y) for x, y in zip([
                u for u in dicts_human_merged.values()
                if not (isinstance(u, str) or isinstance(u, bool))
            ], [
                u for u in dicts_llm_merged.values()
                if not (isinstance(u, str) or isinstance(u, bool))
            ])
        ]

        correlation = np.mean([
            ((x - mean_human) * (y - mean_llm)) / (std_human * std_llm)
            for x, y in zip([
                u for u in dicts_human_merged.values()
                if not (isinstance(u, str) or isinstance(u, bool))
            ], [
                u for u in dicts_llm_merged.values()
                if not (isinstance(u, str) or isinstance(u, bool))
            ])
        ])

        plt.hist(mse, bins=8, density=True)
        plt.xlabel("Score MSE", fontsize=_label_fontsize)
        plt.xticks(fontsize=_tick_fontsize)
        plt.yticks(fontsize=_tick_fontsize)
        plt.ylabel("Frequency", fontsize=_label_fontsize)
        plt.xlim([0, 1])
        plt.grid("on")
        plt.tight_layout()
        plt.show()
        plt.hist(mae, bins=8, density=True)
        plt.xlabel("Score MAE", fontsize=_label_fontsize)
        plt.ylabel("Frequency", fontsize=_label_fontsize)
        plt.xlim([0, 1])
        plt.xticks(fontsize=_tick_fontsize)
        plt.yticks(fontsize=_tick_fontsize)
        plt.grid("on")
        plt.tight_layout()
        plt.show()

        scores_human = [
            u for u in dicts_human_merged.values()
            if not (isinstance(u, str) or isinstance(u, bool))
        ]
        scores_llm = [
            u for u in dicts_llm_merged.values()
            if not (isinstance(u, str) or isinstance(u, bool))
        ]

        mm_set = set(zip(scores_human, scores_llm))

        tau, p_value = kendalltau(scores_human, scores_llm)

        spearman_corr, _ = spearmanr(scores_human, scores_llm)

        plt.plot(scores_human, scores_llm, "bo", alpha=0.2)
        #markersize=10)
        stats_str = f"Pearson=={correlation}, Kendall's Tau=={tau}, P-value=={p_value}, Spearman=={spearman_corr},mse=={np.mean(mse)}, mae=={np.mean(mae)}"
        print(stats_str)
        plt.title(stats_str)
        #plt.axis("equal")
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plt.show()
        plt.close()

        correlations = [correlation, tau, spearman_corr]
        correlations = [(x + 1.0) / 2 for x in correlations]
        names = ['Kendall\'s $\\tau$', 'Pearson\'s $\\rho$', 'Spearman\'s $r$']

        plt.bar(names, correlations)
        plt.ylabel('Correlation Coefficient \n (normalized)',
                   fontsize=_label_fontsize)
        plt.xlabel('Correlation type', fontsize=_label_fontsize)
        plt.ylim(0, 1)
        plt.grid(True)
        plt.xticks(fontsize=_tick_fontsize - 4)
        plt.yticks(fontsize=_tick_fontsize)
        plt.tight_layout()

        plt.show()

        j_d, scores_lst = joint_distrib(scores_llm, scores_human)
        P_h_knowing_l_2d, P_h_knowing_l_per_value = conditional(
            j_d, dim=0, scores=scores_lst)

        plt.imshow(P_h_knowing_l_2d)
        plt.show()

        conditional_Es = []

        for scr, distrib in P_h_knowing_l_per_value.items():

            assert np.abs(np.sum(distrib) - 1.0) < 1e-8
            #pdb.set_trace()
            conditional_E, conditional_V = E_and_V(distrib, scores_lst)

            conditional_Es.append(conditional_E)

            if 0:
                data = sample_discrete(distrib, num_samples=1000) / 10
                kde = KDEUnivariate(data)
                kde.fit()  # Fit the kernel density estimate

                # generate smooth curve points
                x_grid = np.linspace(0, 1, 1000)
                pdf_est = kde.evaluate(x_grid)

                plt.plot(x_grid, pdf_est, label='KDE', linewidth=5)
                plt.hist(data, density=True, alpha=0.5, label='Histogram')
                plt.legend()
                plt.show()

            plt.bar(range(len(distrib)), distrib, color="orange")
            plt.title(
                f"P(H|L={round(scr,2)}), E[H|L]={round(conditional_E,2)}, var={round(conditional_V,2)}"
            )
            plt.xticks(range(len(distrib)), scores_lst)
            plt.show()

        plt.bar(range(11), conditional_Es, color="red", alpha=0.5)
        plt.xticks(range(len(distrib)), scores_lst, fontsize=_tick_fontsize)
        plt.yticks(fontsize=_tick_fontsize)
        plt.ylabel("$\mathbb{E} [S_{human}|S_{LLM}]$",
                   fontsize=_label_fontsize)
        plt.xlabel("$S_{LLM}$", fontsize=_label_fontsize)
        plt.grid("on")
        plt.tight_layout()
        plt.show()

    elif _args.dict_stats:

        with open(_args.dict_stats, "r") as fl:
            _dd = json.load(fl)

        _zz = [
            x for x in _dd.values()
            if not isinstance(x, str) and not isinstance(x, bool)
        ]
        _zz_mean = np.mean(_zz)
        _zz_median = np.median(_zz)
        _zz_std = np.std(_zz)

        if _args.bd_dir:

            _bd_dists = []
            _dist_stds_over_trials = []
            _dist_means_over_trials = []
            for kk, vv in _dd.items():
                if isinstance(vv, str) or isinstance(vv, bool):
                    continue

                with open(_args.bd_dir + f"/bds_{kk}.json", "r") as fl:
                    cur_dd = json.load(fl)
                    dist_cur = cur_dd["dist"] / np.sqrt(2 * (200**2))
                    _bd_dists.append(dist_cur)

                    _dist_means_over_trials.append(
                        cur_dd["dist_mean_over_trials"])
                    _dist_stds_over_trials.append(
                        cur_dd["dist_std_over_trials"])

            mean_bd_dist = np.mean(_bd_dists)
            std_bd_dist = np.std(_bd_dists)
            mean_llm_score = np.mean(_zz)
            std_llm_score = np.std(_zz)
            correlation_bd_and_score = np.mean([
                ((x - mean_bd_dist) *
                 (y - mean_llm_score)) / (std_bd_dist * std_llm_score)
                for x, y in zip(_bd_dists, _zz)
            ])
            print("correlation==", correlation_bd_and_score)

            kendal_tau, pval = kendalltau(_zz, _bd_dists)
            print("kendal_tau==", kendal_tau, "  pval==", pval)

            plt.hist(_bd_dists, density=True, bins=14, range=[0, 1])
            ylim_dist = 8
            plt.vlines(np.median(_bd_dists),
                       0,
                       ylim_dist,
                       color="red",
                       linewidth=3,
                       label="median error")
            plt.ylim(0, ylim_dist)
            plt.grid("on")
            plt.legend(fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.xlabel("error as a percentage of the maze's diagonal",
                       fontsize=18)
            plt.ylabel("normalized frequency", fontsize=18)
            plt.tight_layout()
            plt.show()

            plt.plot(_zz, [-x for x in _bd_dists], "b*")
            plt.show()

        print("mean, median, std==", _zz_mean, _zz_median, _zz_std)
        plt.hist(_zz, bins=11, density=True)
        ylim = 2.5
        plt.ylim(0, ylim)
        plt.vlines(np.median(_zz),
                   0,
                   ylim,
                   color="red",
                   linewidth=3,
                   label="median score")
        plt.grid("on")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel("scores given by the LLM", fontsize=18)
        plt.ylabel("normalized frequency", fontsize=18)
        plt.tight_layout()
        plt.legend(fontsize=14)
        plt.show()

    else:

        if not _args.input_path:
            raise Exception(
                "Unless you're using --visualize_dicts, --dict_stats or --merge_non_overlapping_dicts, you need to specify --input_path and --index_range"
            )

        fns_imgs = [
            _args.input_path + "/" + x for x in os.listdir(_args.input_path)
            if "fig_" in x
        ]
        fns_dics = [
            _args.input_path + "/" + x for x in os.listdir(_args.input_path)
            if "pair_" in x
        ]

        fns_imgs = sorted(fns_imgs, key=lambda x: get_file_index(x))
        fns_dics = sorted(fns_dics, key=lambda x: get_file_index(x))

        data = list(zip(fns_imgs, fns_dics))

        if not _args.index_range:
            raise Exception("No range was provided, see --index_range")

        data = data[_args.index_range[0]:_args.index_range[1]]

        llm_dict = {
            "type": "llm",
            "flipped_axis": True
        }  #flipped_axis is just for display
        human_dict = {"type": "human", "flipped_axis": False}

        for ii in range(len(data)):

            file_index = get_file_index(data[ii][0])
            print(
                colored(f"evaluating {ii}, which has file index {file_index}",
                        "magenta",
                        attrs=["bold"]))
            assert get_file_index(data[ii][0]) == get_file_index(data[ii][1])

            with open(data[ii][1], "r") as fl:
                cur_dict = json.load(fl)

            if _args.human_eval:

                im = cv2.merge(cv2.split(cv2.imread(data[ii][0]))[::-1])
                human_score = ask_user(
                    im, "Please select a matching score below.")
                human_dict[file_index] = human_score

            if _args.llm_eval:

                llm_scores_lst = []
                for _ in range(1):
                    while True:
                        try:
                            llm_scores_lst.append(eval_with_llm(cur_dict))
                            break
                        except Exception as e:
                            print(
                                f"An error occurred (probably formatting error on the LLM's side): {e}. Retrying with index..."
                            )

                llm_dict[file_index] = np.mean(llm_scores_lst)

        with open(
                f"{_args.output_dir}/human_scores_{_args.index_range[0]}_{_args.index_range[1]}.json",
                "w") as fl:
            json.dump(human_dict, fl)
        with open(
                f"{_args.output_dir}/llm_scores_{_args.index_range[0]}_{_args.index_range[1]}.json",
                "w") as fl:
            json.dump(llm_dict, fl)

        visualize_dicts(human_dict, llm_dict)

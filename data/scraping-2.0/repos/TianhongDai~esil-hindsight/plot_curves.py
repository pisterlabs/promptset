import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib

"""
example code for generate the curves
"""

sns.set(rc={"figure.figsize": (8, 7)})
sns.set_context(rc={"lines.linewidth": 2})

"""
same smooth function from openai baselines to make sure consistent

https://github.com/openai/baselines/blob/master/baselines/her/experiment/plot.py

"""
font_size = 25

def smooth_reward_curve(x, y):
    halfwidth = int(np.ceil(len(x) / 60))  # Halfwidth of our smoothing convolution
    k = halfwidth
    xsmoo = x
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='same') / np.convolve(np.ones_like(y), np.ones(2 * k + 1),
        mode='same')
    return xsmoo, ysmoo

def pad(xs, value=np.nan):
    maxlen = np.max([len(x) for x in xs])
    padded_xs = []
    for x in xs:
        if x.shape[0] >= maxlen:
            padded_xs.append(x)
        padding = np.ones((maxlen - x.shape[0],) + x.shape[1:]) * value
        x_padded = np.concatenate([x, padding], axis=0)
        assert x_padded.shape[1:] == x.shape[1:]
        assert x_padded.shape[0] == maxlen
        padded_xs.append(x_padded)
    return np.array(padded_xs)

# extract info from hsl logs
def extract_info_esil(path):    
    file_content = open(path)
    reward_temp, episode_temp = [], []
    # start to process now...
    while True:
        content = file_content.readline()
        if len(content) == 0:
            break
        if content.find('Update:') != -1:
            split_sentence = content.split()
            index = split_sentence.index('Update:')
            get_str = split_sentence[index + 1]
            episode_temp.append(int(get_str))
            # collect the reward information...
            index = split_sentence.index('Success:')
            get_str = split_sentence[index + 1]
            reward_temp.append(float(get_str[:-1]))
    episode = np.array(episode_temp) + 1
    reward = np.array(reward_temp)
    return episode, reward

# process data
def process_data(logs_path, seeds_number, data_len):
    episode_total = []
    reward_total = []
    for idx in range(seeds_number):
        # extract information
        ep, rewards = extract_info_esil('{}/seed_{}.log'.format(logs_path, idx+1))
        # smooth the curve
        ep, rewards = smooth_reward_curve(ep[:data_len], rewards[:data_len])
        # store the data into...
        episode_total.append(ep)
        reward_total.append(rewards)
    episode_total = np.array(episode_total)
    reward_total = np.array(reward_total)
    reward_median = np.median(reward_total, axis=0)
    return episode_total[0], reward_median, reward_total

def plot_results(task_name, title, seeds_number=5, data_len=1000):
    esil_logs_path = 'example_logs/esil-logs/' + task_name
    ep, reward_median_esil, reward_total_esil = process_data(esil_logs_path, seeds_number, data_len)
    # after load data
    plt.figure()
    _, ax = plt.subplots(1, 1)
    # plot the hsl
    plt.plot(ep, reward_median_esil, label='PPO + ESIL (Ours)')
    plt.fill_between(ep, np.nanpercentile(reward_total_esil, 25, axis=0), np.nanpercentile(reward_total_esil, 75, axis=0), alpha=0.25)
    # some format
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.xaxis.get_offset_text().set_fontsize(font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xlim([0, 1000])
    plt.ylim([0, 1.05])
    plt.xlabel('Epochs', fontsize=font_size)
    plt.ylabel('Success Rate', fontsize=font_size)
    plt.title(title, fontsize=font_size)
    plt.tight_layout()
    plt.legend(loc='center right', prop={'size': font_size})
    plt.savefig('{}_baseline.pdf'.format(task_name))
    
if __name__ == '__main__':
    plot_results(task_name='push', title='FetchPush-v1')
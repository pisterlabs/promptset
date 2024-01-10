import numpy as np
import seaborn as sns


def plot_qmat(q_mat):
    """ plot individual q_mat for each pattern """
    sns.heatmap(q_mat, cmap="YlGnBu")
    sns.plt.title("Q Matrix Heatmap", fontweight='bold')
    sns.plt.ylabel("States")
    sns.plt.xlabel("Action (Next states)")
    sns.plt.show()

def running_average(x, window_size, mode='valid'):
    """ Adopted from openAI gym """
    return np.convolve(x, np.ones(window_size) / window_size, mode=mode)


def display_policy(icounter, q_mat):
    """ Display final policy for self visiting allowed experiments"""
    policy = 'policy_' + str(icounter) + '.txt'
    state_size = 20
    for i in range(state_size):
        tmp = np.argmax(q_mat[i, :])
        log = "echo " + "state: " + str(i) + " next state " + str(tmp) + " >> " + policy
        os.system(log)

def plot_average_td_curve_scaffolding(td_avg_mag, win_size, fname):
    """ Plot reward curve for all patters in full iteration for scafolding scenerio """

    nof_runs = 2
    run_paths =['Nao-Pepper without trust and the theory of mind', 'Nao-Pepper with trust and the theory of mind']
    for i in range(nof_runs):
        td_error = running_average(td_avg_mag[i], win_size, mode='valid')
        sns.plt.plot(td_error, linewidth=3, label=run_paths[i])
        sns.plt.ylabel("Average Temporal Difference Error", fontweight='bold')
        sns.plt.xlabel("Iterations (window size = 150)", fontweight='bold')
        sns.plt.hold(True)

    sns.plt.xlim(0, len(td_error))
    sns.plt.ylim(0.0025, 0.06)
    sns.plt.legend(loc='upper right', shadow=True)
    sns.plt.savefig(fname)
    sns.plt.show()


def plot_avg_reward_curve_scaffolding(rew_mat):
    """ Plot reward curve for all patters in full iteration for scafolding scenerio """
    nof_runs = 2
    run_paths = ['Nao-Pepper without trust and the theory of mind', 'Nao-Pepper with trust and the theory of mind']
    for i in range(nof_runs):
        sns.plt.plot(rew_mat[i], linewidth=3, label=run_paths[i])
        sns.plt.ylabel("Average cumulative reward", fontweight='bold')
        sns.plt.xlabel("Iteration steps", fontweight='bold')
        sns.plt.hold(True)
        sns.plt.legend(loc='upper left', shadow=True)

        sns.plt.savefig('reward_plots_nao/average_reward.png')
        sns.plt.xlim(0, 300)
    sns.plt.show()

def plot_avg_reward_curve(rew_mat):
    """ Plot reward curve for all patters in full iteration """

    nof_runs = 3
    run_paths = ['Random instructor', 'Less reliable instructor', 'Reliable instructor']
    for i in range(nof_runs):
        sns.plt.plot(rew_mat[i], linewidth=3, label=run_paths[i])
        sns.plt.ylabel("Average cumulative reward", fontweight='bold')
        sns.plt.xlabel("Iteration steps", fontweight='bold')
        sns.plt.hold(True)
        sns.plt.legend(loc='upper left', shadow=True)

        sns.plt.savefig('reward_plots/average_reward.png')
        sns.plt.xlim(0, 500)
    sns.plt.show()


def plot_reward_curve(rew_mat, fig_title):
    """ Plot reward curve for all patters in full iteration """

    nof_runs = 5
    for i in range(nof_runs):
        sns.plt.plot(rew_mat[i], label="Run: "+str(i+1))

        #sns.plt.xlim(0, len(reward_all_patterns))
    avg_rew_curve = rew_mat.sum(axis=0, dtype=np.float32) / nof_runs

    sns.plt.plot(avg_rew_curve, linewidth=4,  label="Avg run")
    sns.plt.ylabel("Cumulative reward", fontweight='bold')
    sns.plt.xlabel("Iteration steps", fontweight='bold')
    sns.plt.hold(True)
    sns.plt.legend(loc='upper left', shadow=True)
    #sns.plt.title(fig_title, fontweight='bold')
    sns.plt.xlim(0, 500)
    sns.plt.savefig('reward_plots/'+fig_title+'.png')
    sns.plt.show()


def plot_energy_state(state_energy, fig_title, datatype):
    """ Plot the energy consumption for a given pattern """
    
    axis = sns.plt.gca()
    sns.heatmap(state_energy,  annot=True, fmt=datatype, cbar=False, cmap="YlGnBu", linewidths=1)
    sns.plt.title(fig_title, fontweight='bold')
    axis.axes.get_xaxis().set_visible(False)
    axis.axes.get_yaxis().set_visible(False)
    sns.plt.show()



def display_policy(icounter, energy_over_visits, run_paths, run_id, q_mat):
    """ Display final policy with correct/wrong actions. """

    policy = 'policy_logs_correct_steps/policy_' + str(icounter) +'_'+str(run_paths)+'_'+str(run_id)+'.txt'
    state_size = 20
    nof_correct = 0
    nof_wrong = 0

    for i in range(state_size):
        tmp = np.argmax(q_mat[i, :])
        h2l_check =  energy_over_visits[i] >= energy_over_visits[tmp]
        if h2l_check:
            nof_correct += 1
        else:
            nof_wrong += 1
        log = "echo " + "state: " + str(i) + " next state " + str(tmp) + " correct action HtoL: " + str(h2l_check)  + " >> " + policy
        os.system(log)
    flog =  "echo " + "num of correct actions: " + str(nof_correct)+ " num of wrong actions: " + str(nof_wrong)+ " | tee -a " + policy
    os.system(flog)


def plot_average_td_curve(td_avg_mag, win_size, fname):
    """ Plot average td error for all patters """

    nof_runs = 3
    run_paths = ['Random instructor', 'Less reliable instructor', 'Reliable instructor']
    for i in range(nof_runs):
        td_error = running_average(td_avg_mag[i], win_size, mode='valid')
        sns.plt.plot(td_error, linewidth=3, label=run_paths[i])
        sns.plt.ylabel("Temporal Difference Error", fontweight='bold')
        sns.plt.xlabel("Iterations (window size = 200)", fontweight='bold')
        sns.plt.hold(True)

    sns.plt.xlim(0, len(td_error))
    sns.plt.ylim(0, 0.180)
    sns.plt.legend(loc='upper right', shadow=True)
    sns.plt.savefig(fname)
    sns.plt.show()


def plot_td_curve(td_mat, win_size, fname):
    """ Plot td error for all patters in full iteration """

    nof_runs = 5
    for i in range(nof_runs):
        td_val = running_average(td_mat[i], win_size, mode='valid')
        sns.plt.plot(td_val, label="Run: "+str(i+1))

    avg_td_curve = td_mat.sum(axis=0, dtype=np.float32) / nof_runs
    r_avg = running_average(avg_td_curve, win_size, mode='valid')
    sns.plt.plot(r_avg, linewidth=4, label="Avg run")

    sns.plt.ylabel("TD Error", fontweight='bold')
    sns.plt.xlabel("Iteration steps", fontweight='bold')
    sns.plt.xlim(0, len(td_val))
    sns.plt.legend(loc='upper right', shadow=True)
    sns.plt.savefig(fname)
    sns.plt.show()

from util import set_seed, plt, plot
from hiive.mdptoolbox.example import openai
from solvers import q as q_solver, policy as policy_solver, value as value_solver, frozen_lake_map


def run(name, solver):
    P, R = openai("FrozenLake-v9", desc=frozen_lake_map)
    stats, i, p, t, v_mean, v_max, e = solver(P, R)
    return stats


def policy():
    return run("Frozen Lake Policy", policy_solver)


def value():
    return run("Frozen Lake Value", value_solver)


def q():
    q_x, _, q_mean_r, q_max_r, q_e, q_t = run("Frozen Lake Q", q_solver)
    plt(q_x, q_e, label="Q Error")
    plot(
        'Frozen Lake Q Solutions per Episode',
        xlabel="Episode",
        ylabel="Error",
    )
    plt(q_x, q_t, label="Q Time")
    plot(
        'Frozen Lake Q Times per Episode',
        xlabel="Episode",
        ylabel="Time (s)",
    )
    plt(q_x, q_mean_r, label="Q Mean Reward")
    plt(q_x, q_max_r, label="Q Max Reward")
    plot(
        'Frozen Lake Q Values per Episode',
        xlabel="Episode",
        ylabel="Rewards",
    )


def runner():
    policy_x, _, policy_mean_r, policy_max_r, policy_e, policy_t = policy()
    value_x, _, value_mean_r, value_max_r, value_e, value_t = value()

    plt(policy_x, policy_e, label="Policy Error")
    plt(value_x, value_e, label="Value Error")
    plot(
        'Frozen Lake Value & Policy Solutions per Episode',
        xlabel="Episode",
        ylabel="Error",
    )

    plt(policy_x, policy_t, label="Policy Time")
    plt(value_x, value_t, label="Value Time")
    plot(
        'Frozen Lake Value & Policy Times per Episode',
        xlabel="Episode",
        ylabel="Time (s)",
    )

    plt(policy_x, policy_mean_r, label="Policy Mean Reward")
    plt(value_x, value_mean_r, label="Value Mean Reward")
    plt(policy_x, policy_max_r, label="Policy Max Reward")
    plt(value_x, value_max_r, label="Value Max Reward")
    plot(
        'Frozen Lake Value & Policy Values per Episode',
        xlabel="Episode",
        ylabel="Rewards",
    )


def main():
    runner()
    q()


if __name__ == "__main__":
    set_seed()
    main()

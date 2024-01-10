from model import DeepQNet, DQNTrainer
from openai_gym import RafRpg
import torch
import time
import random

if __name__ == "__main__":
    # starting with 1 ends with 5
    agent = 2
    map_number = 5
    epochs = 100
    turn_rate = 10
    input_size = 3
    batch_size = 1
    epsilon = 0.4
    epsilon_decay = 0.8

    cnt = 0

    game = RafRpg(input_size, map_number, agent)
    input = game.tactics.agent_two_input(game.tactics.current_position, game.tactics.current_map)
    model = DeepQNet(len(input), 5)
    trainer = DQNTrainer(model, lr=0.001, gamma=0.2)
    should_print = False
    model_over_epochs = []
    file_path = "logs_step2.txt"
    file_path2 = "logs_loss2.txt"

    # remove content from logs_step2.txt
    with open(file_path, "w") as f:
        f.write("")
    # remove content from logs_loss2.txt
    with open(file_path2, "w") as f:
        f.write("")

    for i in range(epochs):
        game.reset(map_number)
        start_time = time.time()
        print('\n')
        old_inputs = []
        actions = []
        rewards = []
        new_inputs = []
        dones = []

        while not game.tactics.over:

            old_input = game.tactics.agent_two_input(game.tactics.current_position, game.tactics.current_map)
            
            if random.random() < epsilon:
                print("Random action!")
                action_idx = random.randint(0, 4)
            else:
                action = trainer.model(torch.tensor(old_input, dtype=torch.float).unsqueeze(0))
                action_idx = torch.argmax(action).item()
            action = game.tactics.convert_idx_to_action(action_idx)
            map, reward, done, _ = game.step(action)

            if should_print:
                print(f"Epoch: {i}")
                print(f"Reward: {reward}")

            new_input = game.tactics.agent_two_input(game.tactics.current_position, game.tactics.current_map)
            
            old_inputs.append([old_input])
            actions.append(action)
            rewards.append(reward)
            new_inputs.append([new_input])
            dones.append(done)

            if len(old_inputs) == batch_size:
                trainer.train_step(old_inputs, actions, rewards, new_inputs, dones)
                old_inputs = []
                actions = []
                rewards = []
                new_inputs = []
                dones = []





################### kraj epohe ################
        epsilon *= epsilon_decay

        if len(trainer.cum_loss) != 0:
            avg_loss = sum(trainer.cum_loss)/len(trainer.cum_loss)
            # save loss in logs_loss2.txt
            with open(file_path2, "a") as f:
                f.write(f"{avg_loss}, {i}\n")
        trainer.cum_loss = []

        end_time = time.time()
        metric = game.tactics.eval()
        model_over_epochs.append(metric)
        # save model in logs_step2.txt
        with open(file_path, "a") as f:
            f.write(f"{metric}, {i}\n")
        print(f"\nEpoch {i} finished in {end_time - start_time} seconds")
        print(f"Epoch Metric: {metric}\n")

        # change map number
        cnt +=1
        if cnt == turn_rate:
            cnt = 0
            map_number += 1
            if map_number > 5:
                map_number = 1



# done training
    overall_metric = sum(model_over_epochs)/len(model_over_epochs)
    print(f"Overall metric: {overall_metric}")
    trainer.model.save(file_name="rl2_model.pth")

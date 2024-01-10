from model import DeepQNet, DQNTrainer
from openai_gym import RafRpg
import torch
import time
import random


# promeniti input (koje informacije se prosledjuju mrezi) - x, y udaljenost samo
# promeniti reward? -> promeniti fiksnu nagradu za bandita
# promeniti model -> uprostili ga
# uvesti metriku za merenje modela
# raditi u batchevima

#########################################

# promeniti nagradu za seljaka i novac uvezati u jednu nagradu

# reward = reward - dist(merchant) - (2/55)*inventory_value (bude oko 50)

# reward = reward/inventory_value

# kada je inventar manji od ~50, treba da nam se isplati da skupkljamo stvari
# kada je inventar veci od ~50, treba da nam se isplati da idemo kod trgovca

#########################################
# hiperparametri: lr, gamma, broj epoha, broj koraka po epohi


#########################################
#########################################

# ideja: 2 mreze, jedna za nagrade, jedna da ode do trgovca
# I mreza: cilj je da se sto brze stigne do 60 zlata
# kraj je kada stigne do 60 zlata ili istekne vreme
# input: x-y do seljaka, x-y do bandita, x-y do neotrivenog polja, x-y do nevalidnog polja, number of moves
# 



if __name__ == "__main__":
    # starting with 1 ends with 5
    agent = 1
    map_number = 1
    epochs = 300
    turn_rate = 10
    input_size = 3
    batch_size = 1
    epsilon = 0.3
    epsilon_decay = 0.95

    cnt = 0

    game = RafRpg(input_size, map_number, agent)
    input = game.tactics.agent_one_input(game.tactics.current_position, game.tactics.current_map)
    model = DeepQNet(len(input), 5)
    trainer = DQNTrainer(model, lr=0.001, gamma=0.2)
    should_print = False
    model_over_epochs = []
    file_path = "logs_step.txt"
    file_path2 = "logs_loss.txt"

    # remove content from logs_step.txt
    with open(file_path, "w") as f:
        f.write("")
    # remove content from logs_loss.txt
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
            # if game.tactics.current_moves % 10 == 0:
            #     print("#################")
            #     print(game.tactics.current_map)
            #     print("#################")
            #     should_print = True
            # else:
            #     should_print = False

            old_input = game.tactics.agent_one_input(game.tactics.current_position, game.tactics.current_map)
            
            if random.random() < epsilon:
                print("Random action!")
                action_idx = random.randint(0, 4)
            else:
                action = trainer.model(torch.tensor(old_input, dtype=torch.float).unsqueeze(0))
                action_idx = torch.argmax(action).item()
            action = game.tactics.convert_idx_to_action(action_idx)
            map, reward, done, _ = game.step(action)

            # NOTE: reward is decreased for each move
            # reward -= (0.2*game.tactics.current_moves)
            if should_print:
                print(f"Epoch: {i}")
                print(f"Reward: {reward}")
            new_input = game.tactics.agent_one_input(game.tactics.current_position, game.tactics.current_map)
            # trainer.train_step(old_input, action, reward, new_input, done)
            
            
            old_inputs.append([old_input])
            actions.append(action)
            rewards.append(reward)
            new_inputs.append([new_input])
            dones.append(done)

            if len(old_inputs) == batch_size:
                # print("\nModel training\n")
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
            # save loss in logs_loss.txt
            with open(file_path2, "a") as f:
                f.write(f"{avg_loss}, {i}\n")
        trainer.cum_loss = []

        end_time = time.time()
        metric = game.tactics.eval()
        model_over_epochs.append(metric)
        # save model in logs.txt
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




    overall_metric = sum(model_over_epochs)/len(model_over_epochs)
    print(f"Overall metric: {overall_metric}")
    trainer.model.save(file_name="rl1_model.pth")
    # 92
    # 70
    # 33 (big network, (192, 128), big lr (0.01))

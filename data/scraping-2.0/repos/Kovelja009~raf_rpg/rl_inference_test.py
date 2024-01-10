from model import DeepQNet
from openai_gym import RafRpg
import torch


if __name__ == "__main__":
    agent = 3
    map_number = 1
    input_size = 3
    batch_size = 1

    game = RafRpg(input_size, map_number, agent)
    input = game.return_nn_input(game.tactics.current_position, game.tactics.current_map)

    # load torch model
    model = DeepQNet(len(input), 5)
    if agent == 1:
        model.load_state_dict(torch.load('./models/rl1_model.pth'))
    if agent == 2 or agent == 3:
        model.load_state_dict(torch.load('./models/rl2_model.pth'))
    model.eval()

    while not game.tactics.over:

            input = game.return_nn_input(game.tactics.current_position, game.tactics.current_map)
            
            action = model(torch.tensor(input, dtype=torch.float).unsqueeze(0))
            action_idx = torch.argmax(action).item()
            action = game.tactics.convert_idx_to_action(action_idx)
            map, reward, done, _ = game.step(action)

    print(f"Game over! Gold collected: {game.tactics.current_gold}")            

import torch
from game.wrapped_flappy_bird import GameState
from train import NeuralNetwork, select_action


def test():
    score = 0
    episode_num = 0
    max_score = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load('pretrained_model/best_model.pt', weights_only=False)

    while score < 100:
        env = GameState()
        coords, _, _ = env.frame_step([1, 0])
        state = coords
        while True:
            action_idx = select_action(state, 0.01, model, device)
            action_arr = [0,0]
            action_arr[action_idx] = 1

            next_coords, reward, done = env.frame_step(action_arr)

            state = next_coords
            if done:
                score = done-1
                episode_num += 1
                max_score = max(score, max_score)
                print("Episonde", episode_num, "Score", score, "Max", max_score)
                break


if __name__ == '__main__':
     test()

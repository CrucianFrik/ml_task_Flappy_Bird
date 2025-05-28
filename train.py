import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import wandb
from config import config
from game.wrapped_flappy_bird_edited import GameState


class NeuralNetwork(nn.Module):
    def __init__(self, n_states, n_actions):
        super(NeuralNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return np.array(states), actions, rewards, np.array(next_states), dones

    def __len__(self):
        return len(self.buffer)


def select_action(state, epsilon, model, device):
    if random.random() < epsilon:
        return random.randrange(config['N_ACTIONS'])
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        q = model(state)
        return int(q.argmax(1)[0])


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run = wandb.init(
            project="FlappyBird",
            config=config,
    )

    policy_net = NeuralNetwork(config['N_STATES'], config['N_ACTIONS']).to(device)
    target_net = NeuralNetwork(config['N_STATES'], config['N_ACTIONS']).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=config['LR'])
    memory = ReplayBuffer(config['MEM_SIZE'])

    epsilon = config['EPSILON_START']
    total_steps = 0
    flag = True

    for episode in range(10000):
        env = GameState(flag)
        coords, _, _ = env.frame_step([1, 0])
        state = coords
        score = 0

        while True:
            action_idx = select_action(state, epsilon, policy_net, device)
            action = [0, 1] if action_idx else [1, 0]
            action[action_idx] = 1

            next_coords, reward, done = env.frame_step(action)
            next_state = next_coords
            memory.push(state, action_idx, reward, next_state, done)

            state = next_state
            score += reward
            total_steps += 1

            if len(memory) > config['BATCH_SIZE']:
                batch = memory.sample(config['BATCH_SIZE'])
                states, actions, rewards, next_states, dones = batch
                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

                q_values = policy_net(states).gather(1, actions)
                next_q_values = target_net(next_states).max(1, keepdim=True)[0]
                expected_q = rewards + config['GAMMA'] * next_q_values * (1 - dones)

                loss = nn.MSELoss()(q_values, expected_q.detach())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Обновление target network
            if total_steps % config['TARGET_UPDATE'] == 0:
                target_net.load_state_dict(policy_net.state_dict())
        
            if done:
                if done > 100:
                    flag = False
                    torch.save(policy_net, 'model.pt')
                print(f"Episode: {episode}, Score: {score}, Epsilon: {epsilon:.3f}, Score: {done-1}")
                run.log({'reward': score, 'score': done-1}, step=episode+1)
                break

        if (episode+1) % 100 == 0:
            torch.save(policy_net, f'pretrained_model/model_{episode}.pt')

        # Decay epsilon
        if epsilon > config['EPSILON_MIN']:
            epsilon *= config['EPSILON_DECAY']


if __name__ == '__main__':
    train()

import gymnasium as gym

import ale_py 
import numpy as np
import random
import torch
from collections import deque
import cv2

import torch.nn as nn
import torch.optim as optim

# Hyperparameters
ENV_NAME = "ALE/SpaceInvaders-v5"
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 32
MEM_SIZE = 100_000
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 1_000_000
TARGET_UPDATE = 10_000
STACK_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocess frame
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    return frame / 255.0

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return (
            torch.tensor(state, dtype=torch.float32, device=DEVICE),
            torch.tensor(action, dtype=torch.int64, device=DEVICE),
            torch.tensor(reward, dtype=torch.float32, device=DEVICE),
            torch.tensor(next_state, dtype=torch.float32, device=DEVICE),
            torch.tensor(done, dtype=torch.float32, device=DEVICE),
        )

    def __len__(self):
        return len(self.buffer)

# DQN Network
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        return self.net(x)

# Epsilon decay
def get_epsilon(step):
    return max(EPS_END, EPS_START - (EPS_START - EPS_END) * step / EPS_DECAY)

# Stack frames
def stack_frames(stacked_frames, state, is_new_episode):
    frame = preprocess_frame(state)
    if is_new_episode:
        stacked_frames = deque([np.zeros((84, 84), dtype=np.float32) for _ in range(STACK_SIZE)], maxlen=STACK_SIZE)
        for _ in range(STACK_SIZE):
            stacked_frames.append(frame)
    else:
        stacked_frames.append(frame)
    return np.stack(stacked_frames, axis=0), stacked_frames

# Main training loop
def train():
    env = gym.make(ENV_NAME, render_mode=None)
    n_actions = env.action_space.n
    policy_net = DQN((STACK_SIZE, 84, 84), n_actions).to(DEVICE)
    target_net = DQN((STACK_SIZE, 84, 84), n_actions).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEM_SIZE)
    stacked_frames = deque([np.zeros((84, 84), dtype=np.float32) for _ in range(STACK_SIZE)], maxlen=STACK_SIZE)
    total_steps = 0
    episode_rewards = []

    for episode in range(1, 1001):
        state, _ = env.reset()
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        episode_reward = 0
        done = False

        while not done:
            epsilon = get_epsilon(total_steps)
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(np.array([state]), dtype=torch.float32, device=DEVICE)
                    q_values = policy_net(state_tensor)
                    action = int(torch.argmax(q_values).item())

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state_proc, stacked_frames = stack_frames(stacked_frames, next_state, False)
            memory.push(state, action, reward, next_state_proc, done)
            state = next_state_proc
            episode_reward += reward
            total_steps += 1

            # Training step
            if len(memory) > BATCH_SIZE:
                states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q_values = target_net(next_states).max(1)[0]
                    expected_q = rewards + GAMMA * next_q_values * (1 - dones)
                loss = nn.MSELoss()(q_values, expected_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update target network
            if total_steps % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

        episode_rewards.append(episode_reward)
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.3f}")

    env.close()

if __name__ == "__main__":
    train()
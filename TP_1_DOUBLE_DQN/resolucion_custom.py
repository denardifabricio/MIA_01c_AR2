# train_dqn.py

import ale_py  # importante para asegurar que ALE esté registrado
import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Configuración general
env = gym.make("ALE/Boxing-v5", obs_type="ram")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Observar forma de entrada y salida
n_observations = np.prod(env.observation_space.shape)  
n_actions = env.action_space.n

# Crear red neuronal Q
def create_q_network():
    return nn.Sequential(
        nn.Linear(n_observations, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, n_actions)
    ).to(device)

# Hiperparámetros
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.1
lr = 1e-4
batch_size = 64
max_episodes = 1000
max_steps = 5000
memory_size = 100_000
target_update_freq = 10

# Inicialización
policy_net = create_q_network()
target_net = create_q_network()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=lr)
memory = deque(maxlen=memory_size)

# Función para seleccionar acción
def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = policy_net(state_tensor)
        return torch.argmax(q_values).item()

# Función para entrenar una minibatch
def train_step():
    if len(memory) < batch_size:
        return

    minibatch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*minibatch)

    states = torch.tensor(states, dtype=torch.float32, device=device)
    actions = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
    next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

    q_values = policy_net(states).gather(1, actions)
    with torch.no_grad():
        max_next_q = target_net(next_states).max(1, keepdim=True)[0]
        target_q = rewards + (1 - dones) * gamma * max_next_q

    loss = nn.functional.mse_loss(q_values, target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Entrenamiento
for episode in range(max_episodes):
    state, _ = env.reset()
    state = state.flatten()
    total_reward = 0

    for step in range(max_steps):
        action = select_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = next_state.flatten()

        memory.append((state, action, reward, next_state, terminated or truncated))
        state = next_state
        total_reward += reward

        train_step()

        if terminated or truncated:
            break

    # Actualizar red objetivo
    if episode % target_update_freq == 0:
        target_net.load_state_dict(policy_net.state_dict())

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    print(f"Ep {episode}, Total reward: {total_reward}, Epsilon: {epsilon:.3f}")

# Guardar modelo
torch.save(policy_net.state_dict(), "dqn_boxing.pt")
env.close()

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import random
import numpy as np
import collections 
import matplotlib.pyplot as plt 

import os

# --- Hiperparámetros ---
max_episodes = 5000
max_steps_per_episode = 200
learning_rate = 0.001
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay_episodes = 3000
buffer_size = 100_000
batch_size = 64
target_update_freq = 1000
print_every = 100
smoothing_window = 100

# --- Entorno ---
env = gym.make('MountainCar-v0')
n_observations = env.observation_space.shape[0]
n_actions = env.action_space.n

print(f"Observaciones: {n_observations}, Acciones: {n_actions}")

# --- Red Neuronal (Q-Network) ---
def create_q_network():
    return nn.Sequential(
        nn.Linear(n_observations, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, n_actions))

q_network = create_q_network()
target_network = create_q_network()
target_network.load_state_dict(q_network.state_dict())
target_network.eval()

optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()


# --- Replay Buffer ---
Transition = collections.namedtuple('Transition',
                                    ('state', 'action', 'reward', 'next_state', 'done'))
replay_buffer = collections.deque(maxlen=buffer_size)


# --- Función para actualizar la Red Objetivo ---
def update_target_network():
    target_network.load_state_dict(q_network.state_dict())


# --- Entrenamiento ---
global_step = 0
epsilon = epsilon_start
episode_rewards_history = []


optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
print("--- Iniciando Entrenamiento (Double DQN) ---")
for episode in range(max_episodes):
    obs, info = env.reset()
    state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    episode_reward = 0
    episode_steps = 0

    for step in range(max_steps_per_episode):
        global_step += 1
        episode_steps += 1

        if random.random() < epsilon:
            action = torch.tensor([[env.action_space.sample()]], dtype=torch.long)
        else:
            with torch.no_grad():
                q_values = q_network(state)
                action = q_values.max(1)[1].view(1, 1)

        next_obs, reward, terminated, truncated, info = env.step(action.item())

        done = terminated or truncated
        episode_reward += reward

        reward_tensor = torch.tensor([reward], dtype=torch.float32)
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
        done_tensor = torch.tensor([done], dtype=torch.float32)

        replay_buffer.append(Transition(state, action, reward_tensor, next_state, done_tensor))

        state = next_state

        if len(replay_buffer) >= batch_size:
            transitions = random.sample(replay_buffer, batch_size)
            batch = Transition(*zip(*transitions))

            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            state_action_values = q_network(state_batch).gather(1, action_batch).squeeze(1)

            # --- INICIO DE LA MODIFICACIÓN DOUBLE DQN ---
            next_state_values = torch.zeros(batch_size)
            with torch.no_grad():
                # 1. Seleccionar la MEJOR acción para s' usando la red ONLINE
                online_next_q_values = q_network(non_final_next_states)
                online_best_next_actions = online_next_q_values.max(1)[1].unsqueeze(1) # Indices (argmax)

                # 2. Evaluar el valor Q de ESA acción seleccionada usando la red TARGET
                target_next_q_values = target_network(non_final_next_states)
                # Usamos gather para obtener Q_target(s', argmax_a' Q_online(s', a'))
                selected_target_next_q_values = target_next_q_values.gather(1, online_best_next_actions).squeeze(1)

                # Asignar los valores calculados con DDQN a las entradas correspondientes
                next_state_values[non_final_mask] = selected_target_next_q_values
            # --- FIN DE LA MODIFICACIÓN DOUBLE DQN ---

            expected_state_action_values = reward_batch + (gamma * next_state_values)

            loss = loss_fn(state_action_values, expected_state_action_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if global_step % target_update_freq == 0:
            update_target_network()

        if done:
            break

    episode_rewards_history.append(episode_reward)
    epsilon = max(epsilon_end, epsilon_start - (episode / epsilon_decay_episodes) * (epsilon_start - epsilon_end))

    if (episode + 1) % print_every == 0:
        avg_reward = np.mean(episode_rewards_history[-print_every:])
        print(f'Episodio: {episode + 1}/{max_episodes}, Pasos: {episode_steps}, Recompensa Promedio ({print_every} ep): {avg_reward:.2f}, Epsilon: {epsilon:.3f}')


print("--- Entrenamiento Finalizado ---")

base_path = os.path.dirname(os.path.abspath(__file__))
result_path = os.path.join(base_path, "result")

print(f"Directorio base: {base_path}")


# Guardar la Q-Network entrenada
model_path = os.path.join(result_path,"q_network_mountaincar.pth")

# Guardar el modelo completo (estructura + pesos)
torch.save(q_network, model_path)

print(f"Q-Network guardada en: {model_path}")

# --- Graficar Curva de Convergencia ---
print("\n--- Generando Gráfico de Convergencia ---")
plt.figure(figsize=(12, 6))
plt.plot(range(1, max_episodes + 1), episode_rewards_history, label='Recompensa por Episodio', alpha=0.4)
if len(episode_rewards_history) >= smoothing_window:
    rewards_smoothed = np.convolve(episode_rewards_history, np.ones(smoothing_window)/smoothing_window, mode='valid')
    plt.plot(range(smoothing_window, max_episodes + 1), rewards_smoothed, label=f'Media Móvil ({smoothing_window} episodios)', color='red', linewidth=2)
plt.xlabel("Episodio")
plt.ylabel("Recompensa Total")
plt.title("Convergencia de Recompensa Double DQN en MountainCar-V0 Base") # <-- Título actualizado
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(result_path, "convergencia_double_dqn.png"))
plt.show()


# Guardar historial de recompensas para comparación futura
np.save(os.path.join(result_path, "episode_rewards_history.npy"), np.array(episode_rewards_history))
import numpy as np
import gymnasium as gym
import os
import imageio
import matplotlib.pyplot as plt



base_path = os.path.dirname(os.path.abspath(__file__))
print(f"Directorio base: {base_path}")
result_dirpath = os.path.join(base_path, "result")


env = gym.make("MountainCar-v0")



# Discretización del espacio de observación
n_bins = (18, 14)
obs_space_low = env.observation_space.low
obs_space_high = env.observation_space.high
obs_bin_width = (obs_space_high - obs_space_low) / n_bins

def discretize(obs):
    return tuple(((obs - obs_space_low) / obs_bin_width).astype(int).clip(0, np.array(n_bins)-1))

# Hiperparámetros
alpha = 0.1      # tasa de aprendizaje
gamma = 0.99     # factor de descuento
epsilon = 0.2   # exploración inicial
epsilon_min = 0.01
epsilon_decay = 0.995
n_episodes = 1250
max_steps = 1000

# Inicializar Q-table
q_table = np.zeros(n_bins + (env.action_space.n,))

# Lista para almacenar las recompensas totales por episodio
episode_rewards_history = []

# Entrenamiento
for episode in range(n_episodes):
    obs, _ = env.reset()
    state = discretize(obs)
    done = False
    total_reward = 0  # Inicializar recompensa total para el episodio
    for step in range(max_steps):
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_state = discretize(next_obs)
        done = terminated or truncated

        # Actualización Q-learning
        best_next_action = np.argmax(q_table[next_state])
        td_target = reward + gamma * q_table[next_state][best_next_action]
        q_table[state][action] += alpha * (td_target - q_table[state][action])

        state = next_state
        total_reward += reward  # Acumular recompensa
        if done:
            break
    episode_rewards_history.append(total_reward)  # Guardar recompensa total del episodio
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    if (episode+1) % 1000 == 0:
        print(f"Episode {episode+1}/{n_episodes}")

# Guardar la Q-table
np.save(os.path.join(base_path, "q_table.npy"), q_table)

# --- Graficar Curva de Convergencia ---
print("\n--- Generando Gráfico de Convergencia ---")
plt.figure(figsize=(12, 6))
plt.plot(range(1, n_episodes + 1), episode_rewards_history, label='Recompensa por Episodio', alpha=0.4)

# Suavizar la curva con una media móvil
smoothing_window = 100
if len(episode_rewards_history) >= smoothing_window:
    rewards_smoothed = np.convolve(episode_rewards_history, np.ones(smoothing_window)/smoothing_window, mode='valid')
    plt.plot(range(smoothing_window, n_episodes + 1), rewards_smoothed, label=f'Media Móvil ({smoothing_window} episodios)', color='red', linewidth=2)

plt.xlabel("Episodio")
plt.ylabel("Recompensa Total")
plt.title("Convergencia de Recompensa Q-Learning en MountainCar-V0")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(result_dirpath, "convergencia_qlearning.png"))
plt.show()

# Guardar historial de recompensas para comparación futura
np.save(os.path.join(result_dirpath, "episode_rewards_history.npy"), np.array(episode_rewards_history))
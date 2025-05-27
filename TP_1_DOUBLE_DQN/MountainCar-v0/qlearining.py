import numpy as np
import gymnasium as gym
import os
import imageio
import matplotlib.pyplot as plt


env = gym.make("MountainCar-v0")

base_path = os.path.dirname(os.path.abspath(__file__))
print(f"Directorio base: {base_path}")


save_step_dirpath = os.path.join(base_path, "steps", "qlearning")


def generar_video(im_folder, output_path, fps=1):
    images = []
    filenames = sorted([f for f in os.listdir(im_folder) if f.endswith(".png")])
    for filename in filenames:
        img_path = os.path.join(im_folder, filename)
        images.append(imageio.imread(img_path))
    imageio.mimsave(output_path, images, fps=fps)



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
epsilon = 1.0    # exploración inicial
epsilon_min = 0.01
epsilon_decay = 0.995
n_episodes = 10000
max_steps = 200

# Inicializar Q-table
q_table = np.zeros(n_bins + (env.action_space.n,))

# Entrenamiento
for episode in range(n_episodes):
    obs, _ = env.reset()
    state = discretize(obs)
    done = False
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
        if done:
            break
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    if (episode+1) % 1000 == 0:
        print(f"Episode {episode+1}/{n_episodes}")

# Testeo
env = gym.make("MountainCar-v0", render_mode="rgb_array")
n_test_episodes = 1

episode_steps=0
step_number = 0

for episode in range(n_test_episodes):
    obs, _ = env.reset()
    state = discretize(obs)
    done = False
    total_reward = 0
    for step in range(max_steps):
        action = np.argmax(q_table[state])
        obs, reward, terminated, truncated, _ = env.step(action)
        state = discretize(obs)
        total_reward += reward
        done = terminated or truncated
        if done:
            break

        episode_steps += 1
        img = env.render()
        plt.imsave(os.path.join(save_step_dirpath, f"step_{step_number:04d}.png"), img)

        step_number += 1

    print(f"Test Episode {episode+1}: Total Reward = {total_reward}")

    video_path = os.path.join(
            save_step_dirpath, "simulacion_qlearning.gif"
        )
    
    
    generar_video(save_step_dirpath, video_path, fps=5)

env.close()
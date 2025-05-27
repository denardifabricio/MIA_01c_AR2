import numpy as np
import gymnasium as gym
import os
import imageio
import matplotlib.pyplot as plt


def generar_video(im_folder, output_path, fps=1):
    images = []
    filenames = sorted([f for f in os.listdir(im_folder) if f.endswith(".png")])
    for filename in filenames:
        img_path = os.path.join(im_folder, filename)
        images.append(imageio.imread(img_path))
    imageio.mimsave(output_path, images, fps=fps)


# Testeo
base_path = os.path.dirname(os.path.abspath(__file__))
print(f"Directorio base: {base_path}")

result_dir = os.path.join(base_path, "result")

q_table = np.load(os.path.join(result_dir, "q_table.npy"))


save_step_dirpath = os.path.join(result_dir, "steps")


# Borrar imágenes previas en la carpeta de pasos
if os.path.exists(save_step_dirpath):
    for filename in os.listdir(save_step_dirpath):
        file_path = os.path.join(save_step_dirpath, filename)
        if os.path.isfile(file_path) and filename.endswith(".png"):
            os.remove(file_path)

env = gym.make("MountainCar-v0", render_mode="rgb_array")
n_test_episodes = 1

episode_steps=0
step_number = 0
max_steps = 200

# Discretización del espacio de observación
n_bins = (18, 14)
obs_space_low = env.observation_space.low
obs_space_high = env.observation_space.high
obs_bin_width = (obs_space_high - obs_space_low) / n_bins

def discretize(obs):
    return tuple(((obs - obs_space_low) / obs_bin_width).astype(int).clip(0, np.array(n_bins)-1))

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
            result_dir, "simulacion_qlearning.gif"
        )
    
    
    generar_video(save_step_dirpath, video_path, fps=10)

env.close()
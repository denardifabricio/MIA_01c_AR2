import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import random
import numpy as np
import collections # Para el Replay Buffer (deque)
import matplotlib.pyplot as plt # Para graficar
import os

import imageio

def generar_video(im_folder, output_path, fps=1):
    images = []
    filenames = sorted([f for f in os.listdir(im_folder) if f.endswith(".png")])
    for filename in filenames:
        img_path = os.path.join(im_folder, filename)
        images.append(imageio.imread(img_path))
    imageio.mimsave(output_path, images, fps=fps)



base_path = os.path.dirname(os.path.abspath(__file__))
result_path = os.path.join(base_path, "result")
save_step_dirpath = os.path.join(result_path, "steps")

print(f"Directorio base: {base_path}")

model_path = os.path.join(result_path,"q_network_mountaincar.pth")
q_network = torch.load(model_path)


q_network.eval()


# Borrar imágenes previas en la carpeta de pasos
if os.path.exists(save_step_dirpath):
    for filename in os.listdir(save_step_dirpath):
        file_path = os.path.join(save_step_dirpath, filename)
        if os.path.isfile(file_path) and filename.endswith(".png"):
            os.remove(file_path)


print("\n--- prueba con la política aprendida (Evaluación) ---")

eval_env = gym.make('MountainCar-v0', render_mode="rgb_array")
total_eval_reward = 0
num_eval_episodes = 1

max_steps_per_episode = 500


step_number = 0

for i in range(num_eval_episodes):
    obs, info = eval_env.reset()
    state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    done = False
    episode_eval_reward = 0
    episode_steps = 0
    while not done and episode_steps < max_steps_per_episode:
        with torch.no_grad():
            q_values = q_network(state)
            action = q_values.max(1)[1].view(1, 1)

        obs, reward, terminated, truncated, info = eval_env.step(action.item())
        done = terminated or truncated
        episode_eval_reward += reward
        episode_steps += 1

        
        img = eval_env.render()
        plt.imsave(os.path.join(save_step_dirpath, f"step_{step_number:04d}.png"), img)

        if not done:
            state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

       
        print(f"Step {step_number} - Reward: {reward}")
        step_number += 1

        video_path = os.path.join(
            result_path, "simulacion.gif"
        )
        generar_video(save_step_dirpath, video_path, fps=10)

    print(f'Evaluación {i+1}, Recompensa: {episode_eval_reward}, Pasos: {episode_steps}')
    total_eval_reward += episode_eval_reward

print(f'\nRecompensa Promedio en Evaluación ({num_eval_episodes} episodios): {total_eval_reward / num_eval_episodes:.2f}')
eval_env.close()

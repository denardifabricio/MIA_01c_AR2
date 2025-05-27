
import os
import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder
import imageio
import matplotlib.pyplot as plt
print("Stable Baselines3 importado correctamente.")

base_path = os.path.dirname(os.path.abspath(__file__))
print(f"Directorio base: {base_path}")

result_path = os.path.join(base_path, "result")

save_step_dirpath = os.path.join(result_path, "steps")


env_id =  'MountainCar-v0'
video_folder = os.path.join(result_path)
video_length = 1000
log_dir = os.path.join(result_path, "log")
os.makedirs(log_dir, exist_ok=True)
os.makedirs(video_folder, exist_ok=True)

train_env = make_vec_env(env_id, n_envs=4)

model = PPO(
    policy='MlpPolicy',
    env=train_env,
    learning_rate=0.0005,
    n_steps=6000,
    batch_size=128,
    n_epochs=100,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
    seed=42,
)




print(f"Iniciando entrenamiento con PPO en {env_id}...")
model.learn(total_timesteps=500_000, progress_bar=True)
print("Entrenamiento completado.")


model.save(result_path)



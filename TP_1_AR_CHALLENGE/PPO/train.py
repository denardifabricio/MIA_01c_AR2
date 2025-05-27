import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt
print("Stable Baselines3 importado correctamente.")

base_path = os.path.dirname(os.path.abspath(__file__))
print(f"Directorio base: {base_path}")

result_path = os.path.join(base_path, "result")

save_step_dirpath = os.path.join(result_path, "steps")

env_id = 'MountainCar-v0'
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
    n_steps=12000,
    batch_size=128,
    n_epochs=200,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
    seed=42,
)

# --- Recolección de recompensas por episodio ---
episode_rewards_history = []
callback_rewards = []

def reward_callback(locals_, globals_):
    global callback_rewards
    callback_rewards.append(locals_['infos'][0]['episode']['r']) if 'episode' in locals_['infos'][0] else None
    return True

print(f"Iniciando entrenamiento con PPO en {env_id}...")
model.learn(total_timesteps=750_000, progress_bar=True, callback=reward_callback)
print("Entrenamiento completado.")

model.save(result_path)

# --- Graficar Curva de Convergencia ---
print("\n--- Generando Gráfico de Convergencia ---")
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(callback_rewards) + 1), callback_rewards, label='Recompensa por episodio', alpha=0.4)
smoothing_window = 50
if len(callback_rewards) >= smoothing_window:
    rewards_smoothed = np.convolve(callback_rewards, np.ones(smoothing_window)/smoothing_window, mode='valid')
    plt.plot(range(smoothing_window, len(callback_rewards) + 1), rewards_smoothed, label=f'Media Móvil ({smoothing_window} episodios)', color='red', linewidth=2)
plt.xlabel("Episodio")
plt.ylabel("Recompensa Total")
plt.title("Convergencia de Recompensa PPO en MountainCar-V0") 
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(result_path, "convergencia.png"))
plt.show()

# Guardar historial de recompensas para comparación futura
np.save(os.path.join(result_path, "episode_rewards_history.npy"), np.array(callback_rewards))
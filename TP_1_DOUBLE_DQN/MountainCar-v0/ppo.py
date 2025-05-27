
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


save_step_dirpath = os.path.join(base_path, "steps", "PPO")


env_id =  'MountainCar-v0'
video_folder = os.path.join(base_path,"step","PPO",'logs/videos/')
video_length = 1000
log_dir = os.path.join(base_path, "/tmp/gym/")
os.makedirs(log_dir, exist_ok=True)
os.makedirs(video_folder, exist_ok=True)

train_env = make_vec_env(env_id, n_envs=4)
record_env_raw = gymnasium.make(env_id, render_mode='rgb_array')

model = PPO(
    policy='MlpPolicy',
    env=train_env,
    learning_rate=0.0005,         # Lower learning rate for more stable updates
    n_steps=6000,               # Default value, smaller batch for more frequent updates
    batch_size=128,              # Smaller batch size for more frequent updates
    n_epochs=100,                # Default value, less overfitting per update
    gamma=0.99,
    gae_lambda=0.95,            # Default value, more bias but less variance
    clip_range=0.2,             # Default value, more conservative updates
    ent_coef=0.0,               # No entropy bonus, as exploration is less useful in sparse-reward
    vf_coef=0.5,                # Default value
    max_grad_norm=0.5,          # Default value
    verbose=1,
    seed=42,
)

def generar_video(im_folder, output_path, fps=1):
    images = []
    filenames = sorted([f for f in os.listdir(im_folder) if f.endswith(".png")])
    for filename in filenames:
        img_path = os.path.join(im_folder, filename)
        images.append(imageio.imread(img_path))
    imageio.mimsave(output_path, images, fps=fps)



print(f"Iniciando entrenamiento con PPO en {env_id}...")
model.learn(total_timesteps=500_000)
print("Entrenamiento completado.")

print(f"Grabando video de {env_id}...")
record_env = VecVideoRecorder(make_vec_env(lambda: record_env_raw, n_envs=1), video_folder,
                           record_video_trigger=lambda x: x == 0, video_length=video_length,
                         name_prefix=f"ppo-{env_id}")

step_number = 0
obs = record_env.reset()
for _ in range(video_length + 1):
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = record_env.step(action)
    if dones[0]:
        break

    img = record_env.render()
    plt.imsave(os.path.join(save_step_dirpath, f"step_{step_number:04d}.png"), img)
    step_number += 1

video_path = os.path.join(
    save_step_dirpath, "simulacion.gif"
)
generar_video(save_step_dirpath, video_path, fps=5)

record_env.close()
print(f"Video guardado en la carpeta: {video_folder}")
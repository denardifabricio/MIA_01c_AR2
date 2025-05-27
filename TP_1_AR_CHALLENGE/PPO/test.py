
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


# Borrar imágenes previas en la carpeta de pasos
if os.path.exists(save_step_dirpath):
    for filename in os.listdir(save_step_dirpath):
        file_path = os.path.join(save_step_dirpath, filename)
        if os.path.isfile(file_path) and filename.endswith(".png"):
            os.remove(file_path)

env_id =  'MountainCar-v0'
video_folder = os.path.join(result_path)
video_length = 1000
log_dir = os.path.join(result_path, "log")
os.makedirs(log_dir, exist_ok=True)
os.makedirs(video_folder, exist_ok=True)



def generar_video(im_folder, output_path, fps=1):
    images = []
    filenames = sorted([f for f in os.listdir(im_folder) if f.endswith(".png")])
    for filename in filenames:
        img_path = os.path.join(im_folder, filename)
        images.append(imageio.imread(img_path))
    imageio.mimsave(output_path, images, fps=fps)



train_env = make_vec_env(env_id, n_envs=4)

model = PPO.load(os.path.join(base_path, "result.zip"), env=train_env)





record_env_raw = gymnasium.make(env_id, render_mode='rgb_array')


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
    result_path, "simulacion.gif"
)
generar_video(save_step_dirpath, video_path, fps=15)

record_env.close()
print(f"Video guardado en la carpeta: {video_folder}")
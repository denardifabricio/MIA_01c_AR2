import os
import numpy as np
import matplotlib.pyplot as plt

base_path = os.path.dirname(os.path.abspath(__file__))


dql_base_result_path = os.path.join(base_path, "DQL_Base", "result","episode_rewards_history.npy")
dql_edited_result_path = os.path.join(base_path, "DQL_Edited", "result","episode_rewards_history.npy")
ql_result_path = os.path.join(base_path, "QLearning", "result","episode_rewards_history.npy")
ppo_result_path = os.path.join(base_path, "PPO", "result","episode_rewards_history.npy")

dql_base_rewards = np.load(dql_base_result_path)
dql_edited_rewards = np.load(dql_edited_result_path)
ql_rewards = np.load(ql_result_path)
ppo_rewards = np.load(ppo_result_path)

print("\n--- Generando Gráfico de Convergencia ---")
plt.figure(figsize=(12, 6))

smoothing_window = 20

rewards_dict = {
    "DQL Base": dql_base_rewards,
    "DQL Editado": dql_edited_rewards,
    "Q-Learning": ql_rewards,
    "PPO": ppo_rewards
}

colors = {
    "DQL Base": "blue",
    "DQL Editado": "green",
    "Q-Learning": "orange",
    "PPO": "purple"
}

for label, rewards in rewards_dict.items():
    max_episodes = len(rewards)
    if max_episodes >= smoothing_window:
        rewards_smoothed = np.convolve(rewards, np.ones(smoothing_window)/smoothing_window, mode='valid')
        plt.plot(
            range(smoothing_window, max_episodes + 1),
            rewards_smoothed,
            label=f"{label} (Media Móvil {smoothing_window})",
            linewidth=2,
            color=colors[label]
        )

plt.xlabel("Episodio")
plt.ylabel("Recompensa Total")
plt.title("Convergencia de Recompensa - Comparación de Algoritmos")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(base_path, "convergencia_combinada.png"))
plt.show()
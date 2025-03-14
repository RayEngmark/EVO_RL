import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from training.agent import Agent
from main.run_evo_rl import SimpleTrackManiaEnv  # Sikrer samme miljø

# Finn siste session automatisk
models_dir = "models"
session_folders = sorted(
    [d for d in os.listdir(models_dir) if d.startswith("session_")],
    key=lambda x: int(x.replace("session_", "")),
    reverse=True
)

if not session_folders:
    raise FileNotFoundError("🚨 Ingen treningssesjoner funnet i models/")

latest_session = session_folders[0]
model_path = os.path.join(models_dir, latest_session, "latest.pth")
best_model_path = os.path.join(models_dir, latest_session, "best.pth")

# Hvis best.pth finnes, bruk den
if os.path.exists(best_model_path):
    model_path = best_model_path

if not os.path.exists(model_path):
    raise FileNotFoundError(f"🚨 Modellfilen mangler: {model_path}")

# Last inn modellen
print(f"🔄 Laster inn modellen fra: {model_path}")
state_dim = 3
action_dim = 2
agent = Agent(state_dim, action_dim)
agent.model.load_state_dict(torch.load(model_path))
agent.model.eval()
agent.epsilon = 0.0  # Ingen utforskning under testing

# Sett opp testmiljøet
env = SimpleTrackManiaEnv()
initial_episodes = 100
max_episodes = 1000
min_std_threshold = 2.0  # Hvis standardavviket er lavere enn dette, stopp testen
reward_history = []

def evaluate_agent():
    """Adaptive evaluering med dynamisk episodedybde."""
    total_rewards = []
    for episode in range(max_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.select_action(state)  # ✅ Bruker samme metode som i trening
            next_state, reward, done = env.step(action)

            # Bruk samme reward-funksjon som i trening
            reward = env.reward_function(state, action, next_state)
            
            episode_reward += reward
            state = next_state

        total_rewards.append(episode_reward)

        # Juster antall episoder dynamisk
        if episode >= initial_episodes:
            reward_std = np.std(total_rewards[-initial_episodes:])
            if reward_std < min_std_threshold:
                print(f"✅ Stopping early at {episode} episodes (low variance: {reward_std:.2f})")
                break

    return total_rewards

# Kjør evaluering
test_rewards = evaluate_agent()

# Plot resultater
plt.figure(figsize=(10, 5))
plt.plot(test_rewards, label="Episode reward")
plt.axhline(y=np.mean(test_rewards), color='r', linestyle='--', label="Average reward")
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.legend()
plt.title("Agent Performance Over Test Episodes")
plt.show()

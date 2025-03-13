import numpy as np
import torch
import matplotlib.pyplot as plt
from training.agent import Agent
from memory.replay_buffer import ReplayBuffer
from main.run_evo_rl import SimpleTrackManiaEnv

# Last inn den trente modellen
state_dim = 3  # Samme som under trening
action_dim = 2  # Samme som under trening
agent = Agent(state_dim, action_dim)
agent.model.load_state_dict(torch.load("model.pth"))
agent.model.eval()
agent.epsilon = 1.0  # Reset epsilon manually after loading model


# Sett opp testmiljøet
env = SimpleTrackManiaEnv()
test_episodes = 100
reward_history = []

def evaluate_agent():
    """Kjører agenten på testepisoder og samler resultater."""
    total_rewards = []
    for episode in range(test_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = torch.argmax(agent.model(state_tensor)).item()
            next_state, reward, done = env.step(action)
            episode_reward += reward
            state = next_state
        total_rewards.append(episode_reward)
    return total_rewards

# Kjør evaluering
test_rewards = evaluate_agent()

# Plot resultatene
plt.figure(figsize=(10, 5))
plt.plot(test_rewards, label="Episode reward")
plt.axhline(y=np.mean(test_rewards), color='r', linestyle='--', label="Average reward")
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.legend()
plt.title("Agent Performance Over Test Episodes")
plt.show()

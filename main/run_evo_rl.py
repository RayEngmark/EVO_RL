from training.agent import Agent
from memory.replay_buffer import PrioritizedReplayBuffer
import numpy as np
import torch
import os
import json
import time
import matplotlib.pyplot as plt

# Opprett `models/` hvis den ikke finnes
models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(models_dir, exist_ok=True)

class SimpleTrackManiaEnv:
    """Forbedret simulering av TrackMania-miljøet."""
    
    def __init__(self):
        self.max_speed = 10.0
        self.track_length = 100.0
        self.reset()

    def reset(self):
        self.position = 0.0
        self.speed = 0.0
        self.done = False
        return np.array([self.position, self.speed, 0.0])

    def step(self, action):
        if self.done:
            return self.state, 0, self.done  

        if action == 1:
            self.speed = min(self.speed + 1.0, self.max_speed)  
        else:
            self.speed = max(self.speed - 1.0, 0.0)  

        self.position += self.speed  
        reward = self.reward_function(self.state, action, np.array([self.position, self.speed, 0.0]))

        if self.position >= self.track_length:
            self.done = True
            reward += 10  
        elif self.speed == 0:
            self.done = True
            reward = -5  

        self.state = np.array([self.position, self.speed, 0.0])
        return self.state, reward, self.done

    def reward_function(self, state, action, next_state):
        pos, speed, _ = next_state
        reward = (pos - state[0]) * 0.8  
        reward += (speed - state[1]) * 0.1  
        reward -= abs(speed - 5) * 0.05  

        if next_state[1] == 0:
            reward -= 5  
        if next_state[0] % 10 == 0:
            reward += 1  

        return reward

def train_evo_rl():
    session_id = max([int(f.replace("session_", "")) for f in os.listdir(models_dir) if f.startswith("session_")], default=0) + 1
    session_path = os.path.join(models_dir, f"session_{session_id}")
    os.makedirs(session_path, exist_ok=True)

    env = SimpleTrackManiaEnv()
    agent = Agent(state_dim=3, action_dim=2, epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.995)
    replay_buffer = PrioritizedReplayBuffer(capacity=10000, alpha=0.6)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.model.to(device)

    num_episodes = 1000
    min_std_threshold = 2.0
    rewards_log = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            reward = env.reward_function(state, action, next_state)
            replay_buffer.push(state, action, reward, next_state, done)

            if replay_buffer.size() > 128:
                states, actions, rewards, next_states, dones, indices, weights = replay_buffer.sample(128)
                agent.train(replay_buffer, states, actions, rewards, next_states, dones, indices, weights, 128)

            state = next_state
            total_reward += reward
            if done:
                break  

        rewards_log.append(total_reward)

        if episode >= 100:
            reward_std = np.std(rewards_log[-100:])
            if reward_std < min_std_threshold:
                print(f"✅ Stopping early at {episode} episodes (low variance: {reward_std:.2f})")
                break

    torch.save(agent.model.state_dict(), os.path.join(session_path, "latest.pth"))

if __name__ == "__main__":
    train_evo_rl()

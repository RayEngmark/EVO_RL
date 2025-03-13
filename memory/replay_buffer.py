import numpy as np
import collections
import random
import numpy as np
import torch.nn as nn

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # ✅ Beholder init

    def push(self, state, action, reward, next_state, done):  # ✅ Endret add() til push()
        self.buffer.append((state, action, reward, next_state, done))
        print(f"[DEBUG] Added to buffer: State: {state}, Action: {action}, Reward: {reward}, Done: {done}")

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def size(self):
        return len(self.buffer)

# === Simple Neural Network for Agent ===
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# === Agent with Basic DQN Logic ===
class Agent:
    def __init__(self, state_dim, action_dim, lr=0.001):
        self.model = NeuralNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
    
    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, 1)  # Example for discrete actions (Left/Right)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            return torch.argmax(self.model(state_tensor)).item()
    
    def train(self, replay_buffer, batch_size, gamma=0.99):
        if replay_buffer.size() < batch_size:
            return
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        states_tensor = torch.tensor(states, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32)
        dones_tensor = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        
        current_q_values = self.model(states_tensor).gather(1, actions_tensor)
        next_q_values = self.model(next_states_tensor).max(1, keepdim=True)[0]
        target_q_values = rewards_tensor + gamma * next_q_values * (1 - dones_tensor)
        
        loss = self.criterion(current_q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# === Simplified Environment Placeholder ===
class SimpleTrackManiaEnv:
    def __init__(self):
        self.state = np.array([0.0, 0.0, 0.0])
    
    def reset(self):
        self.state = np.array([0.0, 0.0, 0.0])
        return self.state
    
    def step(self, action):
        self.state += np.random.randn(3) * 0.1  # Simulated random state changes
        reward = 1.0 if action == 1 else -1.0  # Simple example reward
        done = np.random.rand() < 0.1  # Random episode termination
        return self.state, reward, done

# === Training Loop ===
def train_evo_rl():
    env = SimpleTrackManiaEnv()
    agent = Agent(state_dim=3, action_dim=2)
    replay_buffer = ReplayBuffer(capacity=10000)
    
    num_episodes = 1000
    batch_size = 32
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)
            agent.train(replay_buffer, batch_size)
            state = next_state
        
        print(f"Episode {episode} completed.")

if __name__ == "__main__":
    train_evo_rl()

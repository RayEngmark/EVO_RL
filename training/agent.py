import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os  # 🔹 Viktig! Trengs for filoperasjoner
from network.model import NeuralNetwork

class Agent:
    def __init__(self, state_dim, action_dim, lr=0.001, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.997):
        self.model = NeuralNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        # Epsilon-parametere
        self.epsilon = epsilon_start
        self.epsilon_min = 0.10  # Øker minimumsverdien for å sikre utforskning
        self.epsilon_decay = 0.999  # Gjør nedgangen enda tregere
        print(f"[DEBUG] Initial epsilon: {self.epsilon:.4f}, Min: {self.epsilon_min:.4f}, Decay: {self.epsilon_decay:.4f}")

        self.load_latest_model()  # 🔹 Flyttet modellinnlasting til egen funksjon

    def get_latest_model(self):
        """Henter siste tilgjengelige modell for automatisk opplasting."""
        if not os.path.exists("models"):  # 🔹 Unngå krasj hvis 'models/' ikke finnes
            return None

        existing_sessions = sorted(
            [int(folder.replace("session_", "")) for folder in os.listdir("models") if folder.startswith("session_")], 
            reverse=True
        )

        for session in existing_sessions:
            model_path = f"models/session_{session}/latest.pth"
            if os.path.exists(model_path):
                print(f"🔄 Loading model from {model_path}")
                return model_path
        
        return None  # Ingen tidligere modell funnet

    def load_latest_model(self):
        """Laster inn den nyeste modellen hvis en finnes."""
        latest_model = self.get_latest_model()
        if latest_model:
            self.model.load_state_dict(torch.load(latest_model))
            print("✅ Model loaded successfully!")
        else:
            print("🆕 No previous model found. Training from scratch.")

    def select_action(self, state):
        """Epsilon-greedy action selection with dynamic decay."""
        if random.random() < self.epsilon:
            action = random.randint(0, 1)  # Utforskning
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = torch.argmax(self.model(state_tensor)).item()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        print(f"[DEBUG] Epsilon: {self.epsilon:.4f}")

        return action

    def train(self, replay_buffer, batch_size, gamma=0.99):
        """Trener agenten basert på erfaringer fra replay-bufferet."""
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

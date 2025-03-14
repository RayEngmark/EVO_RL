import torch
import torch.nn as nn
import torch.optim as optim
import time
import random
import numpy as np
import os
import torch.nn.functional as F
from network.model import NeuralNetwork

class Agent:
    def __init__(self, state_dim, action_dim, lr=0.001, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.997):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NeuralNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # Epsilon-parametere
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        print(f"[DEBUG] Initial epsilon: {self.epsilon:.4f}, Min: {self.epsilon_min:.4f}, Decay: {self.epsilon_decay:.4f}")

        self.load_latest_model()  # 🔹 Flyttet modellinnlasting til egen funksjon

    def get_latest_model(self):
        """Henter siste tilgjengelige modell for automatisk opplasting."""
        if not os.path.exists("models"):
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
            self.model.load_state_dict(torch.load(latest_model, map_location=self.device))  # ✅ Laster rett til GPU/CPU
            print("✅ Model loaded successfully!")
        else:
            print("🆕 No previous model found. Training from scratch.")

    def select_action(self, state):
        """Forbedret handlingvalg med entropy-styrt utforskning."""
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)  # ✅ Riktig enhet

        with torch.no_grad():
            q_values = self.model(state_tensor)

        # Utforskning vs utnyttelse med entropy-vekting
        if random.random() < self.epsilon:
            action = random.randint(0, q_values.shape[1] - 1)  # Utforskning
        else:
            probs = F.softmax(q_values / (self.epsilon + 1e-5), dim=1)
            action = torch.multinomial(probs, num_samples=1).item()  

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return action

    def train(self, replay_buffer, states, actions, rewards, next_states, dones, indices, weights, batch_size, gamma=0.99):
        """Trener agenten basert på samples fra Prioritized Experience Replay."""
        
        # ✅ Alle tensorer sendes til GPU/CPU basert på tilgjengelig enhet
        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Beregn Q-verdier
        current_q_values = self.model(states_tensor).gather(1, actions_tensor)
        next_q_values = self.model(next_states_tensor).max(1, keepdim=True)[0]
        target_q_values = rewards_tensor + gamma * next_q_values * (1 - dones_tensor)

        # 🚨 Sjekk for ekstremt høye Q-verdier
        if torch.max(current_q_values).item() > 1e6:
            print(f"🚨 Advarsel: Q-verdi eksploderer! {torch.max(current_q_values).item()}")

        # Beregn TD-error og oppdater Prioritized Replay Buffer
        td_errors = (current_q_values - target_q_values.detach()).squeeze().abs().detach().cpu().numpy()
        replay_buffer.update_priorities(indices, td_errors)

        # Beregn loss med vekting fra PER
        loss = (weights_tensor * (current_q_values - target_q_values.detach())**2).mean()
        
        # 🔥 Start tidtaking for backpropagation
        start_time = time.time()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ⏳ Print treningsbatchens tid
        print(f"⏳ Batch train time: {time.time() - start_time:.4f}s")
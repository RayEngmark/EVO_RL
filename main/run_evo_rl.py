from training.agent import Agent
from memory.replay_buffer import PrioritizedReplayBuffer  # Bruk PER
import numpy as np
import torch
import os
import json
import csv

# Opprett `models/` hvis den ikke finnes
models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(models_dir, exist_ok=True)

def get_next_session():
    """Finn neste ledige session-nummer basert på eksisterende mapper."""
    existing_sessions = [int(folder.replace("session_", "")) for folder in os.listdir(models_dir) if folder.startswith("session_")]
    return max(existing_sessions, default=0) + 1


class SimpleTrackManiaEnv:
    """Forbedret simulering av TrackMania-miljøet med logging."""
    
    def __init__(self):
        self.max_speed = 10.0
        self.track_length = 100.0
        self.reset()

    def reset(self):
        """Resett miljøet til startverdier."""
        self.position = 0.0
        self.speed = 0.0
        self.done = False
        self.state = np.array([self.position, self.speed, 0.0])  # Posisjon, fart, dummy-verdi
        
        return self.state

    def step(self, action):
        """Simulerer AI-ens bevegelse basert på handlingen."""
        if self.done:
            print("[WARNING] step() called after episode completed.")
            return self.state, 0, self.done  

        # Handlinger: 0 = Bremse, 1 = Akselerere
        if action == 1:
            self.speed = min(self.speed + 1.0, self.max_speed)  
        else:
            self.speed = max(self.speed - 1.0, 0.0)  

        self.position += self.speed  
        reward = (self.speed / self.max_speed) + (self.position / self.track_length)

        # Sluttbetingelser
        if self.position >= self.track_length:
            self.done = True
            reward += 10  # Bonus for å nå slutten
            print(f"[INFO] AI completed track with total reward: {reward:.2f}")

        elif self.speed == 0:
            self.done = True
            reward = -5  # Straff for å stoppe helt
            print(f"[INFO] AI came to a complete stop (straff)")

        # Debugging-logg
        #print(f"[STEP] pos: {self.position:.1f}, speed: {self.speed:.1f}, reward: {reward:.2f}, done: {self.done}")

        self.state = np.array([self.position, self.speed, 0.0])
        return self.state, reward, self.done

    def reward_function(self, state, action, next_state):
        """Ekstern belønningsfunksjon med bedre balanse mellom fart og progresjon."""
        pos, speed, _ = next_state

        reward = (pos - state[0]) * 0.8  # Økt vekt på progresjon
        reward += (speed - state[1]) * 0.1  # Mindre vekt på akselerasjon
        reward -= abs(speed - 5) * 0.05  # Straff for ikke å holde jevn fart

        if next_state[1] == 0:  
            reward -= 5  # Straff for full stopp
        if next_state[0] % 10 == 0:  
            reward += 1  # Bonus for sjekkpunkter

        return reward


def get_latest_session():
    """Hent siste session for å fortsette treningen."""
    existing_sessions = sorted(
        [int(folder.replace("session_", "")) for folder in os.listdir(models_dir) if folder.startswith("session_")],
        reverse=True
    )
    return f"session_{existing_sessions[0]}" if existing_sessions else None

latest_session = get_latest_session()

if latest_session is None:
    print("🆕 No previous sessions found, starting fresh.")
    session_path = os.path.join(models_dir, "session_1")
else:
    session_path = os.path.join(models_dir, latest_session)
    model_load_path = os.path.join(session_path, "latest.pth")
    metadata_path = os.path.join(session_path, "metadata.json")

    if os.path.exists(model_load_path):
        print(f"🔄 Loading model from {model_load_path}")
        agent = Agent(state_dim=3, action_dim=2, epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.995)
        agent.model.load_state_dict(torch.load(model_load_path))

        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                agent.epsilon = metadata.get("epsilon", agent.epsilon)

        print("✅ Model and metadata loaded!")
    else:
        print("⚠️ No previous model found, starting fresh.")


def train_evo_rl():
    """Trener AI-agenten og lagrer modellen med session-struktur."""
    session_id = get_next_session()
    session_path = os.path.join(models_dir, f"session_{session_id}")
    os.makedirs(session_path, exist_ok=True)
    print(f"🚀 Starting training in session {session_id}")

    # Opprett miljø og agent
    env = SimpleTrackManiaEnv()
    agent = Agent(state_dim=3, action_dim=2, epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.995)
    replay_buffer = PrioritizedReplayBuffer(capacity=10000, alpha=0.6)

    # 🚀 Bruk GPU hvis tilgjengelig
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.model.to(device)

    num_episodes = 1000
    max_timesteps = 200
    batch_size = 128  # Økt stabilitet og raskere konvergens

    # Opprett CSV-loggfil
    log_file = os.path.join(session_path, "training_log.csv")
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Total Reward", "Epsilon"])  # CSV-header

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        for t in range(max_timesteps):
            # 🚀 Sørg for at state er på riktig device
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

            # Velg handling
            action = agent.select_action(state)
            next_state, _, done = env.step(action)

            # Bruk belønningsfunksjonen
            reward = env.reward_function(state, action, next_state)

            # Legg til i replay buffer
            replay_buffer.push(state, action, reward, next_state, done)

            # Tren agenten hvis buffer er stor nok
            if replay_buffer.size() > batch_size:
                states, actions, rewards, next_states, dones, indices, weights = replay_buffer.sample(batch_size)
                agent.train(replay_buffer, states, actions, rewards, next_states, dones, indices, weights, batch_size)

            state = next_state
            total_reward += reward

            if done:  # 🚨 Episoden er ferdig, lagre data og gå til neste
                break  

        # 📊 Skriv episodens resultat til CSV etter hver episode
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode, total_reward, agent.epsilon])

        print(f"📊 Training log updated: {log_file}")

        # 🚀 Sjekk om dette er den beste modellen
        best_model_path = os.path.join(session_path, "best.pth")
        metadata_path = os.path.join(session_path, "metadata.json")

        if not os.path.exists(metadata_path):
            best_reward = -float("inf")  # Ingen tidligere reward
        else:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            best_reward = metadata.get("best_reward", -float("inf"))

        if total_reward > best_reward:
            print(f"🏆 New best model found! Reward: {total_reward:.2f}")
            torch.save(agent.model.state_dict(), best_model_path)
            best_reward = total_reward

        # Oppdater metadata
        metadata = {"epsilon": agent.epsilon, "best_reward": best_reward}
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

    # === Lagre modellen etter trening ===
    model_save_path = os.path.join(session_path, "latest.pth")
    metadata = {"epsilon": agent.epsilon}
    
    with open(os.path.join(session_path, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    torch.save(agent.model.state_dict(), model_save_path)
    print(f"✅ Model saved to {model_save_path}")


def evaluate_agent():
    """Evaluerer den trente agenten uten tilfeldig utforsking (epsilon = 0)."""
    session_id = get_next_session() - 1  # Bruk siste session
    session_path = os.path.join(models_dir, f"session_{session_id}")
    best_model_path = os.path.join(session_path, "best.pth")

    if not os.path.exists(best_model_path):
        print("❌ Ingen lagret modell for evaluering.")
        return

    print(f"🔍 Evaluating model from {best_model_path}...")
    env = SimpleTrackManiaEnv()
    agent = Agent(state_dim=3, action_dim=2, epsilon_start=0.0, epsilon_min=0.0, epsilon_decay=1.0)
    agent.model.load_state_dict(torch.load(best_model_path))
    
    total_reward = 0
    state = env.reset()
    done = False

    while not done:
        action = agent.select_action(state)
        state, reward, done = env.step(action)
        total_reward += reward

    print(f"🏁 Evaluation completed. Total Reward: {total_reward:.2f}")



if __name__ == "__main__":
    train_evo_rl()
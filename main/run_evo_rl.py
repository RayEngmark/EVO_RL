from training.agent import Agent
from memory.replay_buffer import ReplayBuffer
import numpy as np
import torch

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
        
        print("[RESET] AI is starting over.")
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
        reward = self.speed / self.max_speed  

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
        print(f"[STEP] pos: {self.position:.1f}, speed: {self.speed:.1f}, reward: {reward:.2f}, done: {self.done}")

        self.state = np.array([self.position, self.speed, 0.0])
        return self.state, reward, self.done


# === Training Loop ===
def train_evo_rl():
    env = SimpleTrackManiaEnv()
    agent = Agent(state_dim=3, action_dim=2, epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.995)

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

    # === Lagre modellen etter trening ===
    print("🚀 model saved to model.pth...")
    torch.save(agent.model.state_dict(), "model.pth")
    print("✅ Model saved!")

if __name__ == "__main__":
    train_evo_rl()

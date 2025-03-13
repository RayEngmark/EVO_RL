import numpy as np
import random

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        """Prioritert replay buffer med kapasitet og alpha for vekting."""
        self.capacity = capacity
        self.alpha = alpha  # Hvordan vektene påvirker sannsynlighet
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """Legger til en erfaring i bufferet med høyeste prioritet."""
        max_priority = self.priorities.max() if len(self.buffer) > 0 else 1.0
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """Henter en batch basert på prioritert sampling."""
        if len(self.buffer) == 0:
            return [], [], [], [], [], [], []

        priorities = self.priorities[: len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        states, actions, rewards, next_states, dones = zip(*samples)

        # Importance-Sampling vekter for å korrigere for bias
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones), indices, np.array(weights, dtype=np.float32))

    def update_priorities(self, indices, td_errors, epsilon=1e-5):
        """Oppdaterer prioriteter basert på TD-error."""
        self.priorities[indices] = np.abs(td_errors) + epsilon

    def size(self):
        return len(self.buffer)

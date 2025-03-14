import numpy as np
import random

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = np.empty((capacity, 5), dtype=object)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.current_size = 0  # Bruk en eksplisitt variabel for størrelse

    def push(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.current_size > 0 else 1.0
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.current_size = min(self.current_size + 1, self.capacity)  # Oppdater størrelse

    def sample(self, batch_size, beta=0.4):
        if self.current_size == 0:
            return [], [], [], [], [], [], []

        priorities = self.priorities[:self.current_size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(self.current_size, batch_size, p=probabilities)
        samples = self.buffer[indices]

        states, actions, rewards, next_states, dones = zip(*samples)

        weights = (self.current_size * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.int64),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.bool_),
                indices,
                np.array(weights, dtype=np.float32))

    def update_priorities(self, indices, td_errors, epsilon=1e-5):
        self.priorities[indices] = np.abs(td_errors) + epsilon

    def get_size(self):  # Endret fra size() til get_size()
        return self.current_size

    def __len__(self):
        return self.size

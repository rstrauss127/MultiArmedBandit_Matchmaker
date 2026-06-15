import numpy as np
from scipy.stats import beta

class ThompsonSamplingBandit:
    def __init__(self, num_queues: int):
        """
        Initializes the Multi-Armed Bandit engine with uniform priors.
        alphas = 1, betas = 1 represents a flat probability (total uncertainty).
        """
        self.num_queues = num_queues
        self.alphas = np.ones(num_queues)  # Success counters + 1
        self.betas = np.ones(num_queues)   # Failure counters + 1

    def select_queue(self) -> int:
        """
        Samples from the Beta distribution of each queue and 
        returns the index of the queue with the highest value.
        """
        # Draw a random sample from the Beta distribution for each arm
        sampled_probabilities = np.random.beta(self.alphas, self.betas)
        
        # Select the queue (arm) that yielded the maximum sampled probability
        selected_queue = int(np.argmax(sampled_probabilities))
        return selected_queue

    def update_feedback(self, queue_index: int, player_retained: bool):
        """
        Updates the statistical distribution of the selected queue 
        based on the observed player outcome.
        """
        if player_retained:
            # Player stayed online -> Increment alpha (Success)
            self.alphas[queue_index] += 1
        else:
            # Player quit early -> Increment beta (Failure)
            self.betas[queue_index] += 1

    def get_expected_probabilities(self) -> np.ndarray:
        """
        Returns the current expected conversion rate for each queue.
        The mean of a Beta distribution is alpha / (alpha + beta).
        """
        return self.alphas / (self.alphas + self.betas)
import numpy as np
from scipy.stats import beta

"""
How the Math Works
        The Beta distribution is defined by two parameters, $\alpha$ (alphas) and $\beta$ (betas):
            $\alpha$: Representing cumulative "successes" (plus a prior of 1).
            $\beta$: Representing cumulative "failures" (plus a prior of 1).
        Every time the engine needs to route a player, it draws a random sample from the Beta distribution of each queue. 
        Whichever queue pulls the highest random sample wins the player. 
        Over time, winning queues shift their distributions toward higher probabilities, 
        naturally earning more traffic, while losing queues get narrowed down.
"""
class ThompsonSamplingBandit:
    def __init__(self, num_queues: int):
        self.num_queues = num_queues
        self.alphas = np.ones(num_queues)  # Success counters + 1
        self.betas = np.ones(num_queues)   # Failure counters + 1

    def select_queue(self) -> int:
        """
        Samples from the Beta distribution of each queue using NumPy.
        """
        # CORRECTED: Cleanly using NumPy's vectorized beta random sampler
        sampled_probabilities = np.random.beta(self.alphas, self.betas)
        
        # Select the queue that yielded the highest sampled probability
        return int(np.argmax(sampled_probabilities))

    def update_feedback(self, queue_index: int, player_retained: bool):
        if player_retained:
            self.alphas[queue_index] += 1
        else:
            self.betas[queue_index] += 1
            
    def get_expected_probabilities(self) -> np.ndarray:
        """
        Returns the current expected conversion rate for each queue.
        The mean of a Beta distribution is alpha / (alpha + beta).
        """
        return self.alphas / (self.alphas + self.betas)
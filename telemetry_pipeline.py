"""
This module acts as an infrastructure layer. 
It simulates a stream of incoming players, routes them using the bandit engine, 
determines if they stayed or quit based on a hidden "true" retention rate, and logs everything.
"""
import logging
import numpy as np
from typing import Dict
from bandit_engine import ThompsonSamplingBandit

# Set up clean structured logging to mimic a real production stream
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MatchmakingTelemetryPipeline:
    def __init__(self, bandit_engine: ThompsonSamplingBandit):
        self.bandit = bandit_engine
        
        # Hidden "True" ground-truth retention rates for our simulation.
        # Queue 0: Connection-priority (Fast but poor matches -> 45% retention)
        # Queue 1: Strict Skill-based (Good matches but long wait times -> 55% retention)
        # Queue 2: Engagement-Optimized (The true winner -> 70% retention)
        self.TRUE_RETENTION_RATES = [0.45, 0.55, 0.70]

    def simulate_player_connection(self) -> Dict:
        """
        Simulates an incoming player request, asks the bandit for a queue route,
        evaluates the outcome, and passes feedback back to the engine.
        """
        # 1. Select the optimal queue route using the bandit engine
        assigned_queue = self.bandit.select_queue()
        
        # 2. Simulate the player outcome based on the hidden true probability
        # np.random.random() picks a float between 0.0 and 1.0
        true_probability = self.TRUE_RETENTION_RATES[assigned_queue]
        player_retained = bool(np.random.random() < true_probability)
        
        # 3. Update the bandit engine instantly (the streaming feedback loop)
        self.bandit.update_feedback(assigned_queue, player_retained)
        
        # 4. Construct telemetry log payload
        payload = {
            "assigned_queue": assigned_queue,
            "player_retained": player_retained
        }
        
        return payload

    def run_streaming_pipeline(self, total_sessions: int):
        """
        Runs the simulation loop across a huge volume of streaming data blocks.
        """
        logger.info(f"Starting telemetry pipeline for {total_sessions} gameplay sessions...")
        
        for session_id in range(total_sessions):
            payload = self.simulate_player_connection()
            
            # Log milestones to show the platform learning in real-time
            if (session_id + 1) % (total_sessions // 5) == 0:
                current_estimates = self.bandit.get_expected_probabilities()
                logger.info(
                    f"Session {session_id + 1}/{total_sessions} completed. "
                    f"Engine Convergence Estimates -> Q0: {current_estimates[0]:.2%}, "
                    f"Q1: {current_estimates[1]:.2%}, Q2: {current_estimates[2]:.2%}"
                )
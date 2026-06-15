"""
This file runs a side-by-side simulation comparing the Multi-Armed Bandit (MAB) approach against a Traditional A/B Test (Static 33% split across all paths).
We will evaluate by calculating Cumulative Regret. 
"Regret" is defined as the difference between the maximum possible retention you could have achieved (if you knew Queue 2 was the best from day one) and what your routing engine actually achieved.
"""
import numpy as np
import matplotlib.pyplot as plt
from bandit_engine import ThompsonSamplingBandit
from telemetry_pipeline import MatchmakingTelemetryPipeline

def run_evaluation(total_sessions: int = 10000):
    # Hidden true retention rates from our telemetry system
    true_rates = [0.45, 0.55, 0.70]
    optimal_rate = max(true_rates)
    
    # --- 1. SIMULATE TRADITIONAL A/B TEST (Static Control Group) ---
    ab_retained_count = 0
    ab_regret_over_time = []
    cumulative_ab_regret = 0.0
    
    for _ in range(total_sessions):
        # A/B testing splits traffic equally across the 3 queues
        assigned_queue = np.random.choice([0, 1, 2])
        actual_outcome = np.random.random() < true_rates[assigned_queue]
        
        if actual_outcome:
            ab_retained_count += 1
            
        # Calculate regret: optimal possibility minus what this queue offered
        regret = optimal_rate - true_rates[assigned_queue]
        cumulative_ab_regret += regret
        ab_regret_over_time.append(cumulative_ab_regret)

    # --- 2. SIMULATE MULTI-ARMED BANDIT (Adaptive Variant) ---
    bandit_engine = ThompsonSamplingBandit(num_queues=3)
    pipeline = MatchmakingTelemetryPipeline(bandit_engine)
    
    bandit_retained_count = 0
    bandit_regret_over_time = []
    cumulative_bandit_regret = 0.0
    
    for _ in range(total_sessions):
        # Bandit pipeline dynamically chooses the queue based on probability distributions
        payload = pipeline.simulate_player_connection()
        assigned_queue = payload["assigned_queue"]
        actual_outcome = payload["player_retained"]
        
        if actual_outcome:
            bandit_retained_count += 1
            
        # Calculate regret
        regret = optimal_rate - true_rates[assigned_queue]
        cumulative_bandit_regret += regret
        bandit_regret_over_time.append(cumulative_bandit_regret)

    # --- 3. PRINT ANALYTICAL OUTCOMES ---
    print("\n=== EXPERIMENTATION PLATFORM METRICS ===")
    print(f"Traditional A/B Test Total Retention: {ab_retained_count} sessions")
    print(f"Multi-Armed Bandit Total Retention:    {bandit_retained_count} sessions")
    print(f"Additional Retained Sessions via MAB:   {bandit_retained_count - ab_retained_count}")
    print("========================================\n")
    #print(f"bandit_regret_over_time: {bandit_regret_over_time}")
    print(f"cumulative_bandit_regret: {cumulative_bandit_regret}")
    # --- 4. PLOT CUMULATIVE REGRET (SCIENTIFIC PROOF) ---
    plt.figure(figsize=(10, 6))
    plt.plot(ab_regret_over_time, label="Traditional Fixed A/B Test (Linear Regret)", color='red', linestyle='--')
    plt.plot(bandit_regret_over_time, label="Thompson Sampling Bandit (Sublinear Regret)", color='green', linewidth=2)
    plt.title("Platform Evaluation: Cumulative Regret Minimization")
    plt.xlabel("Gameplay Sessions Processed")
    plt.ylabel("Cumulative Regret (Lost Player Retention)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_evaluation(total_sessions=15000)
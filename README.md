# Dynamic Matchmaking Optimization Engine

A compact Python project demonstrating how a real-time matchmaking system can use a Multi-Armed Bandit (MAB) strategy to optimize player retention. This repository contrasts a classical fixed A/B testing approach against an adaptive Thompson Sampling bandit, showing how the bandit can learn the best matchmaking queue and reduce lost retention over time.

## Project Structure

- `bandit_engine.py`
  - Implements `ThompsonSamplingBandit`.
  - Tracks success/failure counts for each queue using Beta priors.
  - Selects a queue by sampling from each queue's Beta distribution.
  - Updates the engine from feedback after each simulation.

- `telemetry_pipeline.py`
  - Implements `MatchmakingTelemetryPipeline`.
  - Simulates player arrivals, routes them through the bandit engine, and evaluates retention.
  - Logs periodic convergence estimates for the engine.

- `eval.py`
  - Runs side-by-side simulations comparing a static A/B test to the adaptive MAB solution.
  - Calculates total retained sessions and cumulative regret.
  - Plots regret over time for both strategies.

## Key Concepts

- **Multi-Armed Bandit (MAB)**
  - An online decision-making algorithm that balances exploration and exploitation.
  - Uses observed outcomes to adaptively favor better-performing options.

- **Thompson Sampling**
  - A Bayesian algorithm for MAB problems.
  - Models each queue's retention rate with a Beta distribution.
  - Samples from each distribution and selects the queue with the highest sampled value.

- **Regret**
  - The difference between the reward of always choosing the best queue and the reward achieved by the selection strategy.
  - The project demonstrates that the bandit strategy accumulates less regret than a static A/B test.

## Installation

1. Create a Python environment (recommended):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install numpy matplotlib
   ```

## Usage

Run the evaluation script to compare the two approaches and display the cumulative regret plot:

```bash
python eval.py
```

## Example Output

- Total retention count for the static A/B test.
- Total retention count for the Thompson Sampling bandit.
- Additional retained sessions due to adaptive routing.
- Cumulative regret values for both strategies.
- Visualization of regret over time.

### Actual Example output
- Traditional A/B Test Total Retention: 8477 sessions
- Multi-Armed Bandit Total Retention:    10451 sessions
- Additional Retained Sessions via MAB:   1974

- Cumulative A/B Test Regret: 1986.20
- Cumulative Bandit Regret: 36.89

![Results Graph](<Cumulative Regret Graph.png>)

### Analysis
1. Traditional A/B Testing exhibits a perpetual linear upward trajectory due to continuous traffic misallocation.

2. The Thompson Sampling Framework displays an initial exploratory curve before successfully identifying the optimal topology and asymptotically flattening out, preserving business critical KPIs.

## Why This Matters

This project illustrates the benefit of moving from rigid experimentation to an adaptive routing engine for matchmaking. In live systems, reducing regret means fewer players are sent to suboptimal queues, improving overall retention and user experience.

## Notes

- The hidden true retention rates are defined in `telemetry_pipeline.py`.
- The bandit starts with equal prior belief for all queues.
- The simulation uses synthetic traffic and retention outcomes for evaluation.


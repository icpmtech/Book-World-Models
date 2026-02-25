# Chapter 4

## The V-M-C Architecture

World Models are structured around three core components — Vision (V), Memory (M), and Controller (C) — forming a closed-loop system that observes, simulates, and acts.

### Vision Model (V)

Compresses high-dimensional input into a compact latent representation `z_t`.

In finance, inputs include:

- Price and volume time series across assets
- Macro indicators (inflation, rates, GDP forecasts)
- Options implied volatility surfaces
- Credit spreads and fixed income data
- Earnings and fundamental data
- Sentiment signals

The encoder (typically a Variational Autoencoder) produces a stochastic latent vector that captures regime, risk appetite, rate environment, and volatility state.

### Memory Model (M)

Learns temporal dynamics — how the latent state evolves over time:

    z_t → z_{t+1}

Implemented as a Recurrent State Space Model (RSSM), it maintains:

- A **deterministic path** capturing structural dynamics (rate cycles, earnings momentum)
- A **stochastic path** capturing inherent uncertainty (shocks, regime shifts)

This captures:

- Market cycles and momentum
- Shock propagation through interlinked variables
- Regime transitions (bull/bear, high/low volatility)
- Cross-asset correlation dynamics

The Memory Model can be unrolled to generate a full distribution of future paths — not a single forecast, but calibrated uncertainty across many scenarios.

### Controller (C)

A reinforcement-learning policy trained inside the Memory Model's simulation.

It selects portfolio actions that maximize risk-adjusted reward across the simulated distribution of future states:

    π(z_t) → action a_t

In markets, actions include:

- Adjust equity/bond allocation
- Rotate sector weights
- Add or reduce hedge positions
- Extend or shorten portfolio duration
- Manage cash buffer

The Controller is trained via model-based RL — generating millions of simulated trajectories from the Memory Model without requiring additional real market data.

### How They Work Together

1. **Vision** encodes current market data into latent state `z_t`
2. **Memory** simulates k steps forward, producing a probability distribution over future states
3. **Controller** selects the action that maximizes expected risk-adjusted reward across those simulated paths
4. The action is executed; new observations arrive; the cycle repeats

This closed loop continuously updates the world model's view of reality and adapts portfolio strategy accordingly — without retraining the system.

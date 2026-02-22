# Chapter 2

## What Is a World Model?

The modern formulation of World Models was introduced in 2018 by David Ha and Jürgen Schmidhuber.

A world model is an internal simulation engine.

Instead of predicting the next word, it predicts:

The next state of the environment.

Formally:

    State(t) → State(t+1)

A state can represent:

- Physical position of objects
- Economic conditions
- Market liquidity
- Investor sentiment
- Volatility regimes

In robotics, companies like NVIDIA use world models to simulate thousands of scenarios before deploying a robot in the real world.

In finance, this approach can be transformative.

---

## The V-M-C Architecture

Most World Models are organized around three core components: **Vision (V)**, **Memory (M)**, and **Controller (C)**. Together, they form a closed-loop system that observes, simulates, and acts.

### Vision Model (V) — The Encoder

The Vision component compresses high-dimensional, noisy input data into a compact **latent representation** `z_t`. In finance, this input might include:

- Price and volume time series across hundreds of assets
- Macro indicators (inflation, rate expectations, GDP forecasts)
- Options market implied volatility surfaces
- Credit spreads and fixed income term structure
- Earnings and fundamental data
- Sentiment signals from news and social data

The output is a dense latent vector `z_t ∈ ℝⁿ` that efficiently encodes the current state of the financial world.

### Memory Model (M) — The Dynamics Engine

The Memory component is the heart of the World Model. It learns how latent states evolve over time:

    z_t → z_{t+1}

This component must capture:

- **Temporal dependencies** — how today's regime shapes tomorrow's volatility
- **Momentum effects** — trend persistence and mean reversion patterns
- **Shock propagation** — how a credit event cascades through the financial system
- **Regime transitions** — the shift from expansion to contraction

Modern implementations use **Recurrent State Space Models (RSSM)**, **Mamba** (selective state space), or **transformer-based** memory modules.

### Controller (C) — The Decision Maker

The Controller is the policy component. Given the current latent state `z_t` and a simulated distribution of future states, it selects the optimal portfolio action:

    π(z_t) → action a_t

Actions may include:

- Adjusting equity allocation (reduce beta)
- Rotating sector weights (from growth to defensives)
- Increasing hedge positions (buying puts or VIX calls)
- Adjusting duration in fixed income
- Setting position-level stop-loss thresholds

The Controller is typically trained via **reinforcement learning** to maximize a risk-adjusted reward.

---

## Future Architectures for Financial World Models

Research in world model architectures is advancing rapidly. The next generation of financial world models will draw on several emerging paradigms:

### Transformer-Based World Models

Architectures like **GAIA** and **UniSim** extend world modeling to handle very long context windows. In finance, this allows the model to:

- Attend to multi-year historical state sequences
- Capture long-horizon macro cycles spanning decades
- Integrate global macro context alongside asset-specific dynamics

### Hierarchical World Models

Markets operate simultaneously at multiple time scales — intraday, daily, weekly, monthly, macro cycles. Hierarchical World Models maintain **separate latent representations at each temporal scale**, enabling reasoning from high-frequency mean reversion to multi-year macro trends.

### Diffusion-Based World Models

Applied to finance, **diffusion-based World Models** generate richer, more calibrated distributions over future paths — better capturing the fat tails and asymmetric return distributions characteristic of financial markets.

### Multi-Agent World Models

**Multi-agent World Models** represent multiple market participants simultaneously, enabling:

- Game-theoretic reasoning about counterparty behavior
- Modeling of crowding effects
- Simulation of market impact under large order flow
- Reflexivity modeling

---

## The Formal State-Space Framework

A Financial World Model can be described formally as a **partially observable Markov decision process (POMDP)**:

    ( S, A, T, R, Ω, O )

Where:
- `S` — the (unobserved) true financial state space
- `A` — the action space (portfolio allocations)
- `T(s' | s, a)` — the transition model (learned dynamics)
- `R(s, a)` — the reward function (risk-adjusted return)
- `Ω` — the observation space (market data)
- `O(o | s)` — the observation model

The World Model learns `T` and `O` simultaneously, enabling it to infer the current state from noisy observations, simulate forward trajectories, and optimize the policy.

---

## World Models vs. Traditional Quantitative Finance

| Approach | Dynamics | Uncertainty | Causal | Planning |
|---|---|---|---|---|
| Mean-Variance Optimization | Static | Partial | No | No |
| Black-Scholes | Partial | Closed-form | No | No |
| Hidden Markov Models | Partial | Regime probs | No | No |
| Monte Carlo Simulation | Parametric | Full distribution | No | No |
| RL Agent (model-free) | None | Implicit | No | Yes |
| **World Model (model-based RL)** | **Learned** | **Full distribution** | **Yes** | **Yes** |

The World Model is the only approach that simultaneously models learned dynamics, full uncertainty, causal structure, and active planning.

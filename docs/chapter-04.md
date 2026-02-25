---
id: chapter-04
title: The V-M-C Architecture
sidebar_label: "Chapter 4 — The V-M-C Architecture"
sidebar_position: 5
---

# Chapter 4

## The V-M-C Architecture

The previous chapter established why World Models differ fundamentally from language models: they simulate future **states**, not future **words**. But how is such a simulation system actually built?

The answer lies in the **V-M-C architecture** — a three-component design that separates the problem of understanding the world (Vision), predicting its dynamics (Memory), and acting optimally within it (Controller).

This architecture, originally introduced in the context of reinforcement learning environments by Ha and Schmidhuber (2018), maps directly onto the structure of financial markets.

![World Model Architecture and Future Extensions](/img/world-model-architecture.svg)

---

## The Three Components at a Glance

| Component | Role | Financial Analogy |
|---|---|---|
| **Vision (V)** | Compresses raw inputs into a compact latent state | The analyst who reads all data and forms a market view |
| **Memory (M)** | Models how the latent state evolves over time | The risk model that forecasts dynamics forward |
| **Controller (C)** | Chooses actions using the simulated future | The portfolio manager who allocates based on the simulation |

Each component is differentiable, allowing the entire system to be trained end-to-end. They operate in sequence at each time step, forming a closed-loop system.

---

## Vision Model (V) — The Encoder

The Vision Model is responsible for transforming high-dimensional, heterogeneous financial data into a compact **latent representation** `z_t`.

### The Compression Problem

Raw financial data is high-dimensional, noisy, and multi-modal:

- Thousands of asset price series with complex cross-correlations
- Macro indicators measured at different frequencies (daily, monthly, quarterly)
- Options surfaces: a 2D grid of implied volatilities across strikes and maturities
- Earnings reports, credit ratings, and fundamental ratios
- Sentiment signals derived from news flow and social media

Feeding all of this directly into a dynamics model would be computationally intractable and statistically fragile. The Vision Model solves this by learning the most informative lower-dimensional encoding of the current financial environment.

### Architecture Options

The Vision Model is typically implemented as a **Variational Autoencoder (VAE)** or a **transformer-based encoder**:

```
Input data x_t  →  Encoder E(x_t)  →  Latent z_t ~ N(μ_t, σ_t²)
                                           ↓
                                     Decoder D(z_t) → Reconstructed x̂_t
```

The VAE objective trains the encoder to produce a **stochastic latent space** — a probability distribution over encodings rather than a single point. This is critical for uncertainty propagation downstream.

For financial data, the encoder typically uses:

- **Temporal convolutional networks** for time-series price data
- **Cross-asset attention** for modeling inter-asset correlations
- **Separate embedding heads** for each data modality, merged before compression

### What the Latent Space Encodes

The latent vector `z_t ∈ ℝⁿ` (typically 32–256 dimensions) implicitly encodes:

| Latent Dimension (Conceptual) | Financial Meaning |
|---|---|
| Growth/recession axis | Macro regime — expansion vs contraction |
| Risk-on / risk-off axis | Investor sentiment and risk appetite |
| Rate regime axis | Rate environment — easing vs tightening |
| Volatility regime axis | Low-vol vs high-vol, calm vs stress |
| Sector rotation axis | Cyclical vs defensive positioning |
| Liquidity axis | Credit availability and market depth |

The model does not assign these labels explicitly — they emerge from learning the structure of financial dynamics. Techniques such as **principal component analysis** of the trained latent space often reveal that the first few latent dimensions align naturally with known macro factors.

### Example: Encoding a Market State

Given observations at a specific date:

```
Inflation:       7.2%
Fed Funds Rate:  2.0%
VIX:             24.3
Credit Spread:   180 bps
GDP Growth:      2.1%
10Y Yield:       3.1%
Equity PE:       19.4x
USD Index:       102.5
...
```

The Vision Model compresses these (and many more inputs) into a latent vector:

```
z_t = [–1.2, 0.8, 2.1, –0.4, 1.3, 0.1, ...]
         ↑      ↑     ↑      ↑
       growth  risk  rate  liquidity
       (neg)   on   tension  (ok)
```

This 32-dimensional encoding captures the essential financial state in a form the Memory Model can process efficiently.

---

## Memory Model (M) — The Dynamics Engine

The Memory Model is the heart of the World Model. It learns the **transition function** — how the latent state evolves from one time step to the next:

```
z_t, a_t  →  M  →  z_{t+1}
```

Where `a_t` is an optional action (e.g., a policy decision or exogenous shock being tested).

### The Recurrent State Space Model (RSSM)

The dominant architecture for the Memory Model in world modeling is the **Recurrent State Space Model (RSSM)**, which maintains two complementary representations:

![RSSM Architecture](/img/rssm-architecture.svg)

1. **Deterministic state `h_t`** — a hidden state propagated by a recurrent network (GRU or LSTM), capturing the deterministic history of dynamics
2. **Stochastic state `s_t`** — a latent variable sampled from a learned distribution, capturing inherent unpredictability

```
h_t = f(h_{t-1}, s_{t-1}, a_{t-1})          [deterministic path]
s_t ~ P(s_t | h_t)                            [stochastic path — prior]
s_t ~ Q(s_t | h_t, z_t)                       [posterior — given observation]
```

The separation into deterministic and stochastic paths is crucial for financial modeling:

- The **deterministic path** captures structural dynamics: rate cycles, earnings momentum, macro trends
- The **stochastic path** captures inherent randomness: idiosyncratic shocks, sudden regime shifts, black swan events

### What the Memory Model Learns

Through training on historical financial data, the Memory Model learns the full complexity of market dynamics:

#### Temporal Dependencies

```
VIX today → VIX tomorrow:  mean-reverting with persistence parameter θ ≈ 0.85
Rate change today → credit spreads next month:  positive correlation, 3–6 week lag
Earnings surprise → price response:  largest in first 5 days, fades over 30 days
```

#### Momentum and Mean Reversion

The model learns which variables exhibit momentum (trend persistence) vs. mean reversion, and at which time horizons — without being told explicitly.

#### Regime Transitions

Perhaps most importantly, the Memory Model learns the **transition probabilities between market regimes**:

![Market Regime Cycle](/img/regime-cycle.svg)

```
From Bull Market:
  → Bull continues:   P = 0.78 (monthly)
  → Correction:       P = 0.16
  → Bear Market:      P = 0.06

From Bear Market:
  → Recovery begins:  P = 0.22 (monthly)
  → Bear continues:   P = 0.68
  → Severe crisis:    P = 0.10
```

These probabilities are not hard-coded — they are learned from data and adapt as market structure evolves.

#### Shock Propagation

The Memory Model captures how shocks propagate through the financial system over time:

```
t=0:  Oil shock +30% (exogenous input)
t=1:  Inflation expectations +0.8%, energy equities +12%
t=2:  Core CPI revision upward; real yields reprice
t=3:  Rate hike probability increases; growth stocks fall
t=4:  USD strengthens; EM outflows begin
t=5:  Credit conditions tighten; corporate capex plans delayed
...
```

![Causal Chain Diagram](/img/causal-chain.svg)

### Simulation: Unrolling the Memory Model

Because the Memory Model is differentiable and autoregressive, it can be **unrolled in time** to generate multi-step simulations:

```
z_t → z_{t+1} → z_{t+2} → ... → z_{t+k}
```

At each step, the stochastic component introduces calibrated uncertainty. The result is a **fan of simulated trajectories** — a probability distribution over future paths:

![Simulation Fan Chart](/img/simulation-fan-chart.svg)

This simulation capability — not prediction of a single path, but generation of a full distribution of possible futures — is what makes the Memory Model so powerful for financial risk management.

---

## Controller (C) — The Decision Maker

The Controller is the policy component of the V-M-C architecture. It takes the current latent state `z_t` and outputs an action `a_t` that maximizes the risk-adjusted expected reward over the simulated distribution of future states.

```
π(z_t) → a_t
```

### The Financial Action Space

In portfolio management, the action space includes:

| Action Category | Examples |
|---|---|
| **Allocation** | Increase equity from 60% to 70%; reduce duration |
| **Sector rotation** | Rotate from growth to defensives; overweight energy |
| **Risk overlay** | Buy put protection; add VIX calls; reduce gross exposure |
| **Factor tilts** | Increase value tilt; reduce momentum exposure |
| **Asset class shifts** | Move from IG credit to Treasuries; increase gold allocation |
| **Cash management** | Increase cash buffer from 5% to 15% |

The action `a_t` is a vector specifying the magnitude of each change — a continuous, multi-dimensional action space that reflects the full complexity of portfolio management.

### Training the Controller: Reinforcement Learning

The Controller is trained via **reinforcement learning (RL)** — but critically, it learns *inside the Memory Model's simulation*, not from direct market interaction.

This is the key advantage of model-based RL over model-free RL:

```
Model-free RL:  requires millions of real environment steps
Model-based RL: generates millions of simulated steps from the Memory Model
```

The training loop is:

1. **Encode:** Vision Model produces latent state `z_t` from market observations
2. **Imagine:** Memory Model simulates `k` steps forward from `z_t` under proposed actions
3. **Evaluate:** Compute reward across simulated trajectories (Sharpe, CVaR, drawdown)
4. **Update:** Backpropagate through the simulation to improve the Controller policy

The reward function is designed to capture real portfolio management objectives:

```
R(z_t, a_t) = λ₁ · Sharpe(path) 
             – λ₂ · Max_Drawdown(path) 
             – λ₃ · Turnover_Cost(a_t)
             – λ₄ · CVaR₀.₀₅(path)
```

The hyperparameters `λ₁ ... λ₄` allow risk preferences to be tuned without retraining the entire model.

### Controller Optimization Algorithms

Two families of algorithms are commonly used:

| Algorithm | Description | Best For |
|---|---|---|
| **CEM (Cross-Entropy Method)** | Sample-based planning: generate many action sequences, keep the best, refine | Low-dimensional, fast planning at inference time |
| **PPO (Proximal Policy Optimization)** | Gradient-based policy learning with clipped updates | High-dimensional, continuous action spaces |
| **SAC (Soft Actor-Critic)** | Entropy-regularized RL for maximum exploration | Noisy, non-stationary environments like finance |
| **MPPI (Model Predictive Path Integral)** | Physics-inspired trajectory sampling | Constrained portfolio problems (weight limits, factor bounds) |

For financial applications, **SAC** and **MPPI** are typically preferred due to the non-stationary nature of markets and the presence of portfolio constraints.

### Decision Loop

At inference time, the Controller operates in a continuous loop:

![Investment Decision Loop](/img/investment-decision-loop.svg)

```
1. Observe: gather latest market data
2. Encode:  z_t = V(x_t)
3. Plan:    simulate k paths forward under candidate actions
4. Select:  a_t = argmax E[R(z_t, ...z_{t+k})]
5. Act:     execute portfolio rebalance
6. Update:  incorporate new observations into model
```

This loop runs continuously — the system is always planning ahead, always updating its simulation of the world.

---

## How V, M, and C Work Together

The power of the V-M-C architecture lies in the **tight integration** of all three components. They share a common latent space and are trained jointly, creating a coherent internal model of financial dynamics.

### A Worked Example: Navigating a Rate Hike Cycle

Consider the following scenario as it unfolds in real time:

**Month 0 — Initial State**

```
Vision Model input:
  Inflation: 4.1%, Rate: 0.25%, VIX: 18, Credit spreads: 95bps
  → Latent z_0: [growth=+1.2, risk_on=+0.9, rate_tension=-0.1, liquidity=+0.8]

Memory Model forecast (6-month simulation):
  P(rate hikes ≥ 3) = 67%
  P(equity correction > 10%) = 34%
  P(recession within 18 months) = 12%

Controller action:
  Reduce equity allocation 5% → Increase TIPS 3%, increase cash 2%
  Add tail-risk hedges (1-month puts at 95% strike)
```

**Month 3 — Mid-Cycle**

```
Vision Model input:
  Inflation: 7.2%, Rate: 1.50%, VIX: 28, Credit spreads: 145bps
  → Latent z_3: [growth=-0.3, risk_on=-0.2, rate_tension=+1.8, liquidity=+0.3]

Memory Model forecast (6-month simulation):
  P(rate hikes ≥ 3 more) = 89%
  P(equity correction > 20%) = 51%
  P(recession within 12 months) = 29%

Controller action:
  Reduce equity allocation further 10% → Rotate to defensives
  Extend put protection, add VIX calls
  Increase gold to 8% (inflation + uncertainty hedge)
```

**Month 6 — Peak Tightening**

```
Vision Model input:
  Inflation: 8.9%, Rate: 3.0%, VIX: 33, Credit spreads: 210bps
  → Latent z_6: [growth=-1.1, risk_on=-1.3, rate_tension=+2.4, liquidity=-0.6]

Memory Model forecast (6-month simulation):
  P(inflation declining) = 72%
  P(recession within 12 months) = 44%
  P(Fed pivot within 9 months) = 61%

Controller action:
  Begin extending duration (add Treasuries in anticipation of rate cuts)
  Maintain equity underweight but reduce hedging costs
  Position for recovery: add cyclicals on weakness
```

This example illustrates how the three components work in concert — the Vision Model continuously re-encodes reality, the Memory Model updates its simulation of future paths, and the Controller adapts its actions as new information arrives.

---

## Training the V-M-C Architecture

Training the full V-M-C system involves three interleaved objectives:

### 1. Representation Learning (Vision Model)

The Vision Model is trained to minimize reconstruction loss — the ability to decode accurate market observations from the latent representation:

```
L_V = E[||x_t – D(z_t)||²] + β · KL(Q(z_t|x_t) || P(z_t))
```

The KL divergence term regularizes the latent space, ensuring that it forms a smooth, continuous distribution that the Memory Model can interpolate across.

### 2. World Model Training (Memory Model)

The Memory Model is trained to predict future latent states accurately:

```
L_M = E[||z_{t+1} – M(z_t, a_t)||²] + KL(posterior || prior)
```

In addition, the Memory Model is trained with **observation prediction loss** — predicting actual market observables from the latent trajectory — ensuring that the simulation remains grounded in financial reality.

### 3. Policy Optimization (Controller)

The Controller is trained via RL within the Memory Model's simulation:

```
L_C = –E_simulation[Σ γᵏ R(z_{t+k}, a_{t+k})]
```

Where `γ` is a discount factor weighting near-term vs. long-term rewards.

### Training Data Requirements

| Component | Primary Data | Training Horizon |
|---|---|---|
| Vision Model | All available asset prices, macro data, alternatives | As far back as available (30+ years) |
| Memory Model | Cross-asset daily data; regime-labeled episodes | 20+ years including multiple regimes |
| Controller | Simulated trajectories generated by Memory Model | Millions of simulated steps |

A key advantage of the V-M-C approach is that once the Vision and Memory Models are trained, the Controller can be trained and retrained rapidly using simulated data — without requiring additional real market exposure.

---

## Performance Characteristics in Finance

The V-M-C architecture demonstrates several properties that make it particularly well-suited to financial markets:

### Regime Adaptation

Because the Memory Model maintains a stochastic latent state, it can **infer regime shifts in real time** as new observations arrive. The latent state updates to reflect the new regime, and the Controller's policy adapts accordingly — without retraining.

### Calibrated Uncertainty

The stochastic components of both the Vision and Memory Models ensure that uncertainty estimates grow appropriately as simulation horizon extends:

```
1-month forecast uncertainty:   ±5.2% (equity return)
3-month forecast uncertainty:   ±9.8%
6-month forecast uncertainty:   ±14.3%
12-month forecast uncertainty:  ±21.7%
```

This calibration is critical: a model that is overconfident at long horizons will produce dangerous portfolio recommendations.

### Data Efficiency

Unlike model-free RL methods that require direct market interaction, the V-M-C system can train the Controller on simulated experience. This dramatically reduces the amount of real historical data needed and allows the model to learn from scenarios that have not yet occurred in history.

### Robustness to Distribution Shift

Markets change over time — correlations shift, new instruments emerge, regulatory regimes change. The V-M-C architecture handles this through:

- **Online updating** of the Vision and Memory Models as new data arrives
- **Latent space regularization** that prevents the model from memorizing specific historical episodes
- **Uncertainty calibration** that widens forecast distributions when market structure has changed

---

## Key Concepts Introduced in This Chapter

- **Vision Model (V):** a variational encoder that compresses high-dimensional financial data into a compact latent state `z_t`
- **Latent space:** a low-dimensional representation that captures the essential structure of the financial environment
- **Memory Model (M):** a recurrent state space model that learns the causal dynamics of market evolution
- **RSSM:** the Recurrent State Space Model — separating deterministic history from stochastic uncertainty
- **Controller (C):** a reinforcement-learning policy that selects optimal portfolio actions using simulated future states
- **Model-based RL:** training the policy inside the World Model's simulation rather than through direct market interaction
- **Closed-loop system:** V, M, and C operating in continuous integration — encoding, simulating, and acting in sequence

---

## Chapter Summary

The V-M-C architecture provides a principled, modular framework for financial world modeling:

- The **Vision Model** solves the compression problem: converting the overwhelming complexity of financial data into a tractable latent representation
- The **Memory Model** solves the dynamics problem: learning how that representation evolves causally, capturing cycles, shocks, momentum, and regime transitions
- The **Controller** solves the decision problem: using the simulated distribution of futures to select risk-optimized portfolio actions

Together, these three components form a system that does not merely describe financial markets — it **simulates them, reasons over them, and acts within them**.

The next chapter examines how this architecture is applied in practice — building a complete financial world model, defining the state space, and training the system on real market data.


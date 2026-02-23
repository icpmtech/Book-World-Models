---
id: chapter-08
title: Portfolio Simulation Engines
sidebar_label: "Chapter 8 — Portfolio Simulation Engines"
sidebar_position: 9
---

# Chapter 8

## Portfolio Simulation Engines

Traditional backtesting is linear and historical.

World simulation is **probabilistic and forward-looking**.

Instead of replaying the past, the model generates:

> 10,000 possible futures.

For each future, it measures:

- Drawdown
- Recovery time
- Dividend sustainability
- Bankruptcy probability
- Volatility clustering

The portfolio is evaluated not by past performance but by **future resilience**.

---

## The Limits of Traditional Backtesting

Backtesting is the standard method for evaluating investment strategies: apply a set of trading rules to historical data and measure the result. It is easy to understand, easy to communicate, and widely accepted.

But it carries a fundamental flaw: **the past is not the future**.

### Why Backtesting Fails

1. **Single-path problem:** Backtesting returns one number — the performance of the strategy on one historical sequence of events. The strategy is judged by what *did* happen, not by the distribution of what *could* happen.

2. **Survivorship bias:** Historical data over-represents assets, markets, and regimes that survived. Strategies that look robust in backtests often fail when markets encounter regimes not represented in the historical sample.

3. **Overfitting:** With enough parameters, any model can be fitted to past data. The resulting strategy may have zero out-of-sample validity.

4. **Regime blindness:** A strategy may perform brilliantly during a decade of expansion and catastrophically during contraction — but if the backtest covers only one regime, the risk is invisible.

5. **No tail distribution:** A backtest produces a point estimate of performance (e.g., Sharpe ratio = 1.2). It does not produce a probability distribution over future outcomes or a quantification of tail risk.

| Property | Backtesting | World Model Simulation |
|---|---|---|
| Paths | One (historical) | 10,000 (generated) |
| Future coverage | Past regimes only | Full regime distribution |
| Uncertainty | None | Full distribution |
| Tail risk | Invisible | Explicitly measured |
| Overfitting risk | High | Low (generative) |
| Actionability | Sharpe ratio | Resilience score |

---

## The Portfolio Simulation Engine

A World Model simulation engine evaluates a portfolio not by replaying history but by generating a **probability distribution over future portfolio trajectories**.

![Portfolio Simulation Engine Architecture](/img/portfolio-simulation-engine.svg)

### Architecture Overview

The simulation pipeline operates in six stages:

1. **Encode the current state:** The World Model encoder compresses current portfolio holdings, market prices, yield curves, volatility surfaces, and macro indicators into a latent state `z_t`.

2. **Infer the current regime:** Using the latent state, the model estimates the probability distribution over market regimes (Expansion, Overheating, Contraction, Recovery).

3. **Simulate 10,000 futures:** Starting from `z_t`, the model rolls the latent dynamics forward, sampling 10,000 independent trajectories. Each trajectory is a plausible future evolution of the financial system, conditioned on the inferred current regime.

4. **Evaluate resilience metrics:** For each of the 10,000 trajectories, five resilience metrics are computed (see below).

5. **Aggregate the distribution:** The five metric distributions are combined into a composite resilience score, representing the probability that the portfolio survives and compounds capital across the full distribution of futures.

6. **Optimize:** The portfolio weights are adjusted to maximize resilience-weighted expected return, explicitly trading off return against tail risk.

### The Simulation Loop

Formally, the simulation engine executes the following loop for each of the `N = 10,000` paths:

```
For i = 1 to N:
  z_0 = Encoder(portfolio_state, market_data)
  For t = 1 to T:
    z_t = Dynamics(z_{t-1}, ε_t)    // ε_t ~ learned noise distribution
    o_t = Decoder(z_t)               // market observables at step t
    r_t = Portfolio(o_t, weights)    // portfolio return at step t
  Metrics[i] = Evaluate(r_1, ..., r_T)
```

Where `T` is the simulation horizon (typically 12–36 months) and `ε_t` is sampled from the learned noise distribution, which includes fat tails and regime-specific volatility clusters.

---

## Generating 10,000 Possible Futures

The core of the simulation engine is the generation of 10,000 plausible forward trajectories, each representing a coherent, internally consistent future for the market and the portfolio.

![10,000 Simulated Portfolio Futures — Fan Chart](/img/simulation-fan-chart.svg)

### What the Fan Chart Shows

The fan chart above illustrates the output of a single simulation run:

- The **solid dark line** shows the historical portfolio value — one observed path.
- The **shaded bands** show the distribution of 10,000 simulated future paths diverging from "Now."
- The **25th–75th percentile band** (darker blue) contains the central half of outcomes.
- The **5th–95th percentile band** (lighter blue) contains 90% of outcomes.
- The **dashed line** shows the median simulated path.

The widening of the bands over time reflects **growing uncertainty** as the simulation horizon extends. This uncertainty is not an artifact — it is a mathematically correct representation of the limits of predictability in complex financial systems.

### How the Paths Are Generated

Each simulated path is generated by:

1. **Starting from the current latent state** `z_0 = Encoder(market_data_now)`.

2. **Sampling regime transitions:** At each step, the model samples a regime transition probability matrix conditioned on the current latent state. High-entropy latent states produce more varied regime transitions.

3. **Sampling return innovations:** Within each regime, returns are sampled from a learned distribution that captures:
   - **Volatility clustering** (GARCH-like persistence)
   - **Fat tails** (Student-t or mixture distributions)
   - **Cross-asset correlation** (full covariance structure, regime-conditioned)

4. **Propagating portfolio value:** The sampled returns are applied to portfolio weights, updating portfolio value at each time step.

5. **Repeating N=10,000 times:** Each repetition uses independently sampled noise, producing a distinct trajectory.

The result is not a parametric model with assumed normal distributions — it is a data-driven, regime-aware, fat-tailed distribution of outcomes learned directly from historical financial dynamics.

---

## The Five Resilience Metrics

Traditional portfolio evaluation uses a single score — the Sharpe ratio — which is a ratio of expected return to standard deviation. This captures one dimension of performance.

A simulation engine evaluates portfolios on **five resilience dimensions**, each measuring a different aspect of how the portfolio behaves across the full distribution of futures.

![Portfolio Resilience Metrics](/img/portfolio-resilience-metrics.svg)

### 1. Maximum Drawdown

**Definition:** The worst peak-to-trough decline in portfolio value observed across all simulated paths.

Rather than a single historical drawdown figure, the simulation produces a **distribution of drawdowns**:

- The **median drawdown** across 10,000 paths represents the expected worst-case decline.
- The **95th percentile drawdown** represents the tail scenario — the outcome in 5% of simulated futures.

```
Drawdown(i) = max_{t1 < t2} [V(t1) - V(t2)] / V(t1)
P(Drawdown > 20%) = #{i : Drawdown(i) > 20%} / N
```

A World-Model-optimized portfolio minimizes not just the expected drawdown but the entire right tail of the drawdown distribution.

### 2. Recovery Time

**Definition:** The number of months required for the portfolio to recover from its maximum drawdown back to its previous peak.

Recovery time is critical for investors with **liability-driven constraints** — pension funds, endowments, and income-generating portfolios that cannot tolerate prolonged capital impairment.

The simulation produces the full distribution of recovery times:

- **Short recovery time** signals a portfolio that is resilient to temporary dislocations.
- **Long recovery time** signals structural exposure to persistent adverse regimes.

```
RecoveryTime(i) = min{t > t_trough : V(t) >= V(t_peak)}
```

A portfolio with a short median recovery time but a long 95th-percentile recovery time may have hidden tail exposure that backtesting would not reveal.

### 3. Dividend Sustainability

**Definition:** The probability that dividends are maintained or grown across all simulated futures.

For income-focused portfolios — dividend equity, REITs, infrastructure, fixed income — dividend sustainability is a primary risk dimension that standard Sharpe-based optimization completely ignores.

The simulation evaluates dividend sustainability by:

1. For each simulated path, computing the **payout ratio** and **free cash flow coverage** of dividend-paying holdings at each time step.
2. Flagging paths where dividends are cut, suspended, or eliminated due to earnings compression or cash flow stress.
3. Computing `P(dividend sustained) = #{i : dividend maintained through horizon} / N`.

A high dividend sustainability score signals a portfolio whose income stream is robust to adverse economic scenarios, not just to historical averages.

### 4. Bankruptcy Probability

**Definition:** The probability that any holding in the portfolio reaches insolvency within the simulation horizon.

Bankruptcy risk is typically invisible in backtests because companies that went bankrupt often disappear from the historical data entirely (survivorship bias). The simulation makes this risk explicit:

1. For each simulated path, the model tracks the **financial health indicators** of each holding — interest coverage ratios, debt-to-equity, cash burn rates.
2. When these indicators cross insolvency thresholds, the holding is marked as defaulted.
3. The portfolio impact is computed: loss of principal, dividend elimination, potential contagion effects.

```
P(default in portfolio) = 1 - ∏ P(holding_k solvent)   over all holdings k
```

> **Note:** This formula assumes independent default probabilities across holdings. In practice, defaults are correlated — particularly during systemic crises when multiple holdings face stress simultaneously. The simulation engine captures this correlation by conditioning each holding's insolvency threshold on the shared latent state `z_t`, which encodes the regime-level stress experienced across the entire portfolio.

This metric is particularly important for credit-heavy portfolios, high-yield bond allocations, and equity positions in cyclically leveraged sectors.

### 5. Volatility Clustering

**Definition:** The fraction of simulated futures in which the portfolio enters a persistent high-volatility regime, and the expected duration of that regime.

Volatility is not constant — it **clusters**. A single large shock tends to be followed by more large shocks. This phenomenon, well-documented empirically and captured by GARCH models, is critical for option pricing, margin requirements, and investor psychology.

The simulation engine explicitly models volatility clustering through its learned dynamics. For each path, it measures:

- The fraction of time steps spent in high-volatility regimes
- The autocorrelation of simulated volatility (persistence coefficient)
- The expected duration of a high-volatility episode once entered

Portfolios with lower volatility clustering exposure are less likely to trigger stop-losses, margin calls, or forced liquidations during stress events.

---

## Simulation-Driven Portfolio Construction

The five metrics above are combined into a **resilience-weighted optimization objective**:

```
Maximize: E[R_portfolio] - λ₁·P(DD > threshold)
                         - λ₂·E[RecoveryTime]
                         - λ₃·P(dividend cut)
                         - λ₄·P(default)
                         - λ₅·VolClusterExposure
```

Where `λ₁...λ₅` are investor-specific risk penalty weights reflecting the portfolio's mandate, liability structure, and risk tolerance.

This formulation differs fundamentally from mean-variance optimization:

- **Mean-variance** optimizes expected return for a given variance — a single number.
- **Resilience optimization** optimizes expected return for a given *profile* of tail risks across five dimensions.

The resulting portfolio is not the one that would have performed best historically — it is the one that is most likely to survive and compound capital across the widest distribution of possible futures.

### Portfolio Construction Example

A simulation-based optimization might produce the following allocation shifts relative to a standard 60/40 portfolio:

| Asset Class | 60/40 Baseline | Resilience-Optimized | Rationale |
|---|---|---|---|
| Global Equities | 50% | 42% | Reduce drawdown tail risk |
| Government Bonds | 30% | 28% | Maintain duration hedge |
| Dividend Equities | 10% | 18% | Improve dividend sustainability |
| Inflation-Linked Bonds | 5% | 7% | Reduce purchasing-power risk |
| Gold / Alternatives | 5% | 5% | Maintain crisis hedge |

The shift from 50% to 42% global equities and the increase in dividend equities is driven not by a view on expected returns but by the **resilience profile** of each asset class across the 10,000 simulated futures.

---

## Formal Framework

The portfolio simulation engine can be written formally as:

```
State:        z_t = Encoder(o_t, h_{t-1})
Dynamics:     z_{t+1} ~ p_θ(z_{t+1} | z_t, r_t)
Observation:  o_{t+1} = Decoder(z_{t+1})
Return:       r_t = Portfolio(o_t, w)

Simulation:   {τ_i}_{i=1}^{N} ~ p_θ(τ | z_0)    // N trajectories
Metrics:      M_k = f_k({τ_i})                    // per metric k
Optimize:     w* = argmax_w Σ_k α_k · M_k(w)
```

Where:
- `z_t` is the latent state at time `t`
- `p_θ` is the learned transition distribution
- `τ_i` is the i-th simulated trajectory
- `M_k` is the k-th resilience metric function
- `α_k` are metric weights reflecting the investor mandate
- `w*` is the resilience-optimized portfolio weight vector

---

## Chapter Summary

- Traditional backtesting evaluates portfolios on **one historical path** — making it blind to regimes not in the training sample, tail risks, and the full distribution of future outcomes.
- A **Portfolio Simulation Engine** uses a World Model to generate 10,000 probabilistic future trajectories, each representing a coherent, regime-aware evolution of the financial system.
- The **fan chart** visualization shows how the distribution of outcomes widens over time — correctly capturing the limits of predictability.
- Portfolios are evaluated on **five resilience metrics**: Maximum Drawdown, Recovery Time, Dividend Sustainability, Bankruptcy Probability, and Volatility Clustering Exposure.
- **Resilience optimization** replaces Sharpe-ratio maximization with a multi-dimensional objective that minimizes tail risk across the full distribution of simulated futures.
- The result is a portfolio optimized not for past performance but for **future resilience** — the ability to survive adverse scenarios while compounding capital in favorable ones.

The next chapter explores scenario generation and counterfactual reasoning — how the simulation engine can be used to answer "what if?" questions and stress-test portfolios against specific hypothetical shocks.

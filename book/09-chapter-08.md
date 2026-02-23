# Chapter 8

## Portfolio Simulation Engines

Traditional backtesting is linear and historical.

World simulation is probabilistic and forward-looking.

Instead of replaying the past, the model generates:

10,000 possible futures.

For each future, it measures:

- Drawdown
- Recovery time
- Dividend sustainability
- Bankruptcy probability
- Volatility clustering

The portfolio is evaluated not by past performance but by future resilience.

---

## The Limits of Traditional Backtesting

Backtesting is the standard method for evaluating investment strategies: apply a set of trading rules to historical data and measure the result. It is easy to understand, easy to communicate, and widely accepted.

But it carries a fundamental flaw: the past is not the future.

### Why Backtesting Fails

1. **Single-path problem:** Backtesting returns one number — the performance of the strategy on one historical sequence of events. The strategy is judged by what *did* happen, not by the distribution of what *could* happen.

2. **Survivorship bias:** Historical data over-represents assets, markets, and regimes that survived. Strategies that look robust in backtests often fail when markets encounter regimes not represented in the historical sample.

3. **Overfitting:** With enough parameters, any model can be fitted to past data. The resulting strategy may have zero out-of-sample validity.

4. **Regime blindness:** A strategy may perform brilliantly during a decade of expansion and catastrophically during contraction — but if the backtest covers only one regime, the risk is invisible.

5. **No tail distribution:** A backtest produces a point estimate of performance. It does not produce a probability distribution over future outcomes or a quantification of tail risk.

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

A World Model simulation engine evaluates a portfolio not by replaying history but by generating a probability distribution over future portfolio trajectories.

### Architecture Overview

The simulation pipeline operates in six stages:

1. **Encode the current state:** The World Model encoder compresses current portfolio holdings, market prices, yield curves, volatility surfaces, and macro indicators into a latent state `z_t`.

2. **Infer the current regime:** Using the latent state, the model estimates the probability distribution over market regimes (Expansion, Overheating, Contraction, Recovery).

3. **Simulate 10,000 futures:** Starting from `z_t`, the model rolls the latent dynamics forward, sampling 10,000 independent trajectories.

4. **Evaluate resilience metrics:** For each of the 10,000 trajectories, five resilience metrics are computed.

5. **Aggregate the distribution:** The five metric distributions are combined into a composite resilience score.

6. **Optimize:** The portfolio weights are adjusted to maximize resilience-weighted expected return.

### The Simulation Loop

```
For i = 1 to N:
  z_0 = Encoder(portfolio_state, market_data)
  For t = 1 to T:
    z_t = Dynamics(z_{t-1}, ε_t)
    o_t = Decoder(z_t)
    r_t = Portfolio(o_t, weights)
  Metrics[i] = Evaluate(r_1, ..., r_T)
```

---

## The Five Resilience Metrics

### 1. Maximum Drawdown

The worst peak-to-trough decline in portfolio value observed across all simulated paths — expressed as a full probability distribution, not a single historical figure.

```
P(Drawdown > 20%) = #{i : Drawdown(i) > 20%} / N
```

### 2. Recovery Time

The number of months required for the portfolio to recover from its maximum drawdown. Critical for liability-driven investors such as pension funds and endowments.

```
RecoveryTime(i) = min{t > t_trough : V(t) >= V(t_peak)}
```

### 3. Dividend Sustainability

The probability that dividends are maintained or grown across all simulated futures. Evaluated by tracking payout ratios and free cash flow coverage under each simulated economic scenario.

### 4. Bankruptcy Probability

The probability that any holding in the portfolio reaches insolvency within the simulation horizon. Explicitly captures survivorship bias that backtesting conceals.

```
P(default in portfolio) = 1 - ∏ P(holding_k solvent)
```

### 5. Volatility Clustering

The fraction of simulated futures in which the portfolio enters a persistent high-volatility regime. Captures the risk of stop-loss triggers, margin calls, and forced liquidations during stress events.

---

## Simulation-Driven Portfolio Construction

The five metrics are combined into a resilience-weighted optimization objective:

```
Maximize: E[R_portfolio] - λ₁·P(DD > threshold)
                         - λ₂·E[RecoveryTime]
                         - λ₃·P(dividend cut)
                         - λ₄·P(default)
                         - λ₅·VolClusterExposure
```

Where `λ₁...λ₅` are investor-specific risk penalty weights reflecting the portfolio's mandate, liability structure, and risk tolerance.

### Portfolio Construction Example

| Asset Class | 60/40 Baseline | Resilience-Optimized | Rationale |
|---|---|---|---|
| Global Equities | 50% | 42% | Reduce drawdown tail risk |
| Government Bonds | 30% | 28% | Maintain duration hedge |
| Dividend Equities | 10% | 18% | Improve dividend sustainability |
| Inflation-Linked Bonds | 5% | 7% | Reduce purchasing-power risk |
| Gold / Alternatives | 5% | 5% | Maintain crisis hedge |

---

## Formal Framework

```
State:        z_t = Encoder(o_t, h_{t-1})
Dynamics:     z_{t+1} ~ p_θ(z_{t+1} | z_t, r_t)
Observation:  o_{t+1} = Decoder(z_{t+1})
Return:       r_t = Portfolio(o_t, w)

Simulation:   {τ_i}_{i=1}^{N} ~ p_θ(τ | z_0)
Metrics:      M_k = f_k({τ_i})
Optimize:     w* = argmax_w Σ_k α_k · M_k(w)
```

---

## Chapter Summary

- Traditional backtesting evaluates portfolios on one historical path — making it blind to regimes not in the training sample, tail risks, and the full distribution of future outcomes.
- A Portfolio Simulation Engine uses a World Model to generate 10,000 probabilistic future trajectories, each representing a coherent, regime-aware evolution of the financial system.
- Portfolios are evaluated on five resilience metrics: Maximum Drawdown, Recovery Time, Dividend Sustainability, Bankruptcy Probability, and Volatility Clustering Exposure.
- Resilience optimization replaces Sharpe-ratio maximization with a multi-dimensional objective that minimizes tail risk across the full distribution of simulated futures.
- The result is a portfolio optimized not for past performance but for future resilience — the ability to survive adverse scenarios while compounding capital in favorable ones.

The next chapter explores scenario generation and counterfactual reasoning — how the simulation engine can be used to answer "what if?" questions and stress-test portfolios against specific hypothetical shocks.

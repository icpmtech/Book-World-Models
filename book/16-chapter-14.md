# Chapter 14

## Price Prediction World Model: Forecasting Asset Prices with World Model Concepts

The previous chapters established how World Models simulate financial environments and support portfolio decisions. This chapter applies those foundations to one of the most fundamental challenges in quantitative finance: **predicting future asset prices**.

Rather than treating price prediction as a regression problem, a World Model reframes it as a **latent-state inference and rollout problem** — generating a full probability distribution over future price trajectories.

---

## Why Traditional Price Prediction Falls Short

Conventional approaches to price prediction map inputs directly to output prices without modelling the underlying generative process:

- **No uncertainty quantification** — the model returns a point estimate, not a distribution
- **No causal structure** — the model learns correlations, not the mechanisms driving prices
- **Regime blindness** — a model trained in a bull market will fail in a bear market
- **No compositionality** — changes in macro conditions cannot be injected into the model

World Models address all four limitations through explicit state representation, generative simulation, and causal dynamics modelling.

---

## The World Model Framework for Price Prediction

A Price Prediction World Model (PPWM) implements the standard Vision–Memory–Controller architecture adapted for financial time series:

    o_t  →  [Encoder]  →  z_t  →  [Dynamics]  →  z_{t+1}, ..., z_{t+H}
                                                        ↓
                                                  [Decoder]
                                                        ↓
                                        p̂_{t+1}, ..., p̂_{t+H}  (price distributions)

### Vision: Encoding Market Observations

The encoder maps market observations into a compact latent state that captures hidden structure — current regime, implied volatility, and cross-asset correlation — none of which are directly observable.

### Memory: Simulating Price Dynamics

The dynamics model propagates the latent state forward, generating thousands of possible future trajectories. Each trajectory samples from the stochastic transition distribution:

    z_{t+1} ~ p_θ(z_{t+1} | z_t)

### Controller: Decoding to Prices

The decoder maps latent trajectories to return distributions `N(μ, σ²)`, not point predictions. Cumulative returns over decoded paths yield a full distribution of future price paths.

---

## Building the Price Distribution

A single run of the PPWM generates a distributional forecast:

| Horizon | Median Price | 80% Interval      | 95% Interval      | P(Price > 10% up) |
|---------|--------------|-------------------|-------------------|-------------------|
| 5 days  | $153.20      | $148.10–$158.40   | $143.60–$163.90   | 18%               |
| 10 days | $155.80      | $146.20–$165.90   | $138.50–$174.10   | 28%               |
| 21 days | $159.40      | $143.70–$175.80   | $132.40–$190.20   | 34%               |
| 63 days | $166.10      | $141.20–$193.40   | $124.80–$214.60   | 41%               |

The widening intervals reflect genuine uncertainty growth over the horizon — not an assumption of constant volatility.

---

## Regime-Conditioned Price Forecasting

Price forecasts are conditioned on the inferred market regime, producing fundamentally different dynamics across regimes:

| Regime | Trend Persistence | Volatility | Mean-Reversion Speed | Tail Shape |
|---|---|---|---|---|
| **Bull / Expansion** | High (momentum) | Low | Slow | Light left tail |
| **Overheating** | Moderate | Rising | Moderate | Fat right + left tails |
| **Bear / Contraction** | Negative (momentum) | Very high | Moderate | Very fat left tail |
| **Recovery** | Low → building | Declining | Fast | Symmetric but fat |

The PPWM encodes regime as part of the latent state, so the dynamics model applies regime-appropriate price dynamics automatically.

---

## Uncertainty Quantification

The PPWM explicitly separates two types of uncertainty:

| Uncertainty Type | Source | Managed By |
|---|---|---|
| **Aleatoric** | Fundamental price randomness | Stochastic dynamics model |
| **Epistemic (model)** | Limited training data | Monte Carlo Dropout / ensemble |
| **Epistemic (regime)** | Ambiguous current regime | Entropy of regime probabilities |
| **Epistemic (structural)** | Unseen regime | Out-of-distribution detection on `z_t` |

This decomposition allows portfolio managers to distinguish between uncertainty that cannot be reduced (aleatoric) and uncertainty that may decrease with more data or model refinement (epistemic).

---

## Training the Price Prediction World Model

The PPWM is trained with a multi-component loss:

    L_total = L_reconstruction + β · L_KL + L_temporal + L_tail

- **L_reconstruction** — log-likelihood of observed prices under the decoded distribution
- **L_KL** — KL divergence regularising the latent space
- **L_temporal** — consistency of latent dynamics over time
- **L_tail** — calibration penalty for under-estimating extreme events

---

## Evaluating Price Prediction Quality

| Metric | Definition | Interpretation |
|---|---|---|
| **CRPS** | Continuous Ranked Probability Score | Overall distribution quality |
| **Coverage** | Fraction of realised prices within N% interval | Calibration check |
| **Sharpness** | Width of 80% prediction interval | Narrower is better if coverage holds |
| **Pinball loss** | Quantile-specific accuracy | Tail and body accuracy |
| **Directional accuracy** | P(correct sign of price move) | Trend prediction quality |

---

## Real-World Applications

### Intraday Price Path Prediction

For high-frequency trading, the PPWM generates short-horizon price distributions conditioned on order book state and micro-regime signals. Explicit uncertainty output prevents overconfident position sizing during regime ambiguity.

### Earnings-Day Price Prediction

Around earnings announcements, the PPWM generates a bimodal price distribution reflecting the possibility of a positive or negative surprise — weighted by the model's inferred probability of each outcome.

### Options Pricing Support

The PPWM's simulated price paths provide a model-implied return distribution that can replace the Black-Scholes log-normal assumption, yielding option prices consistent with the empirically fat-tailed return distribution.

### Cross-Asset Price Prediction

The shared latent state captures cross-asset dependencies automatically — when equity volatility rises in the latent state, the model automatically increases uncertainty in bond and commodity forecasts.

---

## Limitations

1. **Non-stationarity** — dynamics change over time; continuous retraining is required.
2. **Latent identifiability** — multiple latent configurations may explain the same observations; regularisation is essential.
3. **Computational cost** — generating thousands of paths at inference time requires GPU resources and approximate inference techniques.
4. **No fundamental value anchor** — the PPWM models price dynamics, not intrinsic value; bubbles and crises may generate misleading distributions.
5. **Reflexivity** — if model predictions are widely acted upon, the dynamics the model learned may change.

---

## Looking Ahead

Price prediction is not the end goal — it is an intermediate representation on the path to **decision-making under uncertainty**. The PPWM's distributional output feeds directly into portfolio optimisers, risk engines, and execution systems.

The complete chain — from raw observations through latent state inference, probabilistic price simulation, and resilience-weighted portfolio construction — is the architecture of next-generation financial intelligence.

    "A point forecast is a claim about the future.
     A distributional forecast is an honest account of what we know and do not know."

The World Model's contribution to price prediction is not greater accuracy in a narrow sense — it is **calibrated honesty** about the full range of what may come.

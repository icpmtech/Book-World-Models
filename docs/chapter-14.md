---
id: chapter-14
title: Price Prediction World Model — Forecasting Asset Prices with World Model Concepts
sidebar_label: "Chapter 14 — Price Prediction World Model"
sidebar_position: 15
---

# Chapter 14

## Price Prediction World Model: Forecasting Asset Prices with World Model Concepts

The previous chapters established how World Models simulate financial environments and support portfolio decisions. This chapter applies those foundations to one of the most fundamental challenges in quantitative finance: **predicting future asset prices**.

Rather than treating price prediction as a regression problem, a World Model reframes it as a **latent-state inference and rollout problem** — generating a full probability distribution over future price trajectories.

---

## Why Traditional Price Prediction Falls Short

Conventional approaches to price prediction share a critical weakness: they map inputs directly to output prices without modelling the underlying generative process.

### The Supervised-Regression Trap

A standard machine learning approach trains a model:

```
f(x_t) → p_{t+1}
```

Where `x_t` is a feature vector (price history, volume, indicators) and `p_{t+1}` is the next price. This setup has fundamental limitations:

- **No uncertainty quantification** — the model returns a point estimate, not a distribution
- **No causal structure** — the model learns correlations, not the mechanisms driving prices
- **Regime blindness** — a model trained in a bull market will fail in a bear market
- **No compositionality** — changes in macro conditions cannot be injected into the model

World Models address all four limitations through explicit state representation, generative simulation, and causal dynamics modelling.

---

## The World Model Framework for Price Prediction

A Price Prediction World Model (PPWM) implements the standard Vision–Memory–Controller architecture adapted for financial time series:

```
o_t  →  [Vision/Encoder]  →  z_t  →  [Memory/Dynamics]  →  z_{t+1}, ..., z_{t+H}
                                                               ↓
                                                       [Decoder]
                                                               ↓
                                               p̂_{t+1}, ..., p̂_{t+H}  (price distributions)
```

### Vision: Encoding Market Observations

The encoder maps a high-dimensional market observation `o_t` into a compact latent state `z_t`:

```python
class MarketEncoder(nn.Module):
    """
    Encodes multi-asset market observations into a latent state vector.
    Inputs:  o_t — price returns, volume, macro signals, sentiment
    Output:  z_t — latent state (mean + log-variance for uncertainty)
    """
    def forward(self, o_t: Tensor) -> tuple[Tensor, Tensor]:
        h = self.feature_extractor(o_t)   # shared representation
        z_mean    = self.mean_head(h)
        z_log_var = self.log_var_head(h)
        return z_mean, z_log_var           # parameterise N(z_mean, exp(z_log_var))
```

The encoder does not simply compress data — it **infers hidden state**: the current market regime, implied volatility level, and cross-asset correlation structure, none of which are directly observable.

### Memory: Simulating Price Dynamics

The dynamics model propagates the latent state forward in time:

```python
class FinancialDynamicsModel(nn.Module):
    """
    Recurrent State Space Model for financial dynamics.
    Implements:  z_{t+1} ~ p_θ(z_{t+1} | z_t, a_t)
    where a_t represents optional action / portfolio intervention.
    """
    def step(self, z_t: Tensor, a_t: Tensor | None = None) -> tuple[Tensor, Tensor]:
        deterministic_h = self.gru(z_t, self.h_prev)
        prior_mean, prior_log_var = self.prior_head(deterministic_h)
        if a_t is not None:
            prior_mean = prior_mean + self.action_modulation(a_t)
        return prior_mean, prior_log_var

    def rollout(self, z_0: Tensor, horizon: int, n_samples: int = 1000) -> Tensor:
        """Roll out n_samples trajectories over the given horizon."""
        trajectories = []
        z = z_0.unsqueeze(0).expand(n_samples, -1)
        for _ in range(horizon):
            mean, log_var = self.step(z)
            z = mean + torch.randn_like(mean) * torch.exp(0.5 * log_var)
            trajectories.append(z)
        return torch.stack(trajectories, dim=1)   # (n_samples, horizon, latent_dim)
```

### Controller: Decoding Latent States to Prices

The decoder maps latent trajectories back to observable price distributions:

```python
class PriceDecoder(nn.Module):
    """
    Maps latent state z_t to a price return distribution N(μ, σ²).
    Output is a distribution over returns, not a point prediction.
    """
    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        return_mean = self.mean_head(z)
        return_std  = F.softplus(self.std_head(z))   # ensure positivity
        return return_mean, return_std
```

---

## Building the Price Distribution

The PPWM does not predict "the price." It generates a **distribution over future prices**, capturing both expected direction and uncertainty.

### From Latent Rollouts to Price Paths

Given `N` sampled latent trajectories, the decoder produces `N` return paths. Cumulative returns yield price paths:

```python
def generate_price_paths(
    world_model,
    current_state: Tensor,
    current_price: float,
    horizon: int = 60,       # trading days
    n_paths: int = 5000,
) -> dict:
    """
    Generate a distribution of future price paths using the World Model.
    Returns percentile bands and summary statistics.
    """
    # 1. Roll out latent trajectories
    latent_paths = world_model.dynamics.rollout(current_state, horizon, n_paths)

    # 2. Decode to return distributions
    return_means, return_stds = world_model.decoder(latent_paths)

    # 3. Sample returns from decoded distributions
    returns = return_means + torch.randn_like(return_means) * return_stds

    # 4. Convert to price paths via cumulative product
    price_relatives = 1.0 + returns
    price_paths = current_price * torch.cumprod(price_relatives, dim=1)

    return {
        'p05': price_paths.quantile(0.05, dim=0),
        'p25': price_paths.quantile(0.25, dim=0),
        'p50': price_paths.quantile(0.50, dim=0),   # median forecast
        'p75': price_paths.quantile(0.75, dim=0),
        'p95': price_paths.quantile(0.95, dim=0),
        'mean': price_paths.mean(dim=0),
        'std':  price_paths.std(dim=0),
        'all_paths': price_paths,
    }
```

### Example Price Distribution Output

A single run of the PPWM on a given date might produce:

```
Horizon  |  Median Price  |  80% Interval     |  95% Interval     |  P(Price > 10% up)
---------|----------------|-------------------|-------------------|-----------------------
5 days   |  $153.20       |  $148.10–$158.40  |  $143.60–$163.90  |  18%
10 days  |  $155.80       |  $146.20–$165.90  |  $138.50–$174.10  |  28%
21 days  |  $159.40       |  $143.70–$175.80  |  $132.40–$190.20  |  34%
63 days  |  $166.10       |  $141.20–$193.40  |  $124.80–$214.60  |  41%
```

The widening intervals reflect **genuine uncertainty growth** over the horizon — not an assumption of constant volatility.

---

## Regime-Conditioned Price Forecasting

A key advantage of the PPWM is that price forecasts are **conditioned on the inferred market regime**, not on a single unconditional model.

### Regime-Specific Price Dynamics

Different market regimes produce qualitatively different price dynamics:

| Regime | Trend Persistence | Volatility | Mean-Reversion Speed | Tail Shape |
|---|---|---|---|---|
| **Bull / Expansion** | High (momentum) | Low | Slow | Light left tail |
| **Overheating** | Moderate | Rising | Moderate | Fat right + left tails |
| **Bear / Contraction** | Negative (momentum) | Very high | Moderate | Very fat left tail |
| **Recovery** | Low → building | Declining | Fast | Symmetric but fat |

The PPWM encodes regime as part of the latent state `z_t`, so the dynamics model automatically applies regime-appropriate price dynamics without any manual switching.

### Forecasting Across Regimes

When the model infers ambiguous regime probabilities, it generates a **mixture of regime-conditional forecasts**:

```python
def regime_conditional_forecast(world_model, z_t, price, horizon=21):
    """
    Decompose the price forecast by regime contribution.
    """
    regime_probs = world_model.regime_classifier(z_t)
    forecasts = {}

    for regime in ['expansion', 'overheating', 'contraction', 'recovery']:
        # Generate forecast conditioned on this regime being certain
        z_conditioned = world_model.condition_on_regime(z_t, regime)
        paths = generate_price_paths(world_model, z_conditioned, price, horizon)
        forecasts[regime] = {
            'weight':   regime_probs[regime].item(),
            'median':   paths['p50'][-1].item(),
            'p05':      paths['p05'][-1].item(),
            'p95':      paths['p95'][-1].item(),
        }

    # Mixture forecast (weighted by regime probability)
    weighted_median = sum(
        v['weight'] * v['median'] for v in forecasts.values()
    )
    return {'regime_forecasts': forecasts, 'mixture_median': weighted_median}
```

---

## Multi-Scale Price Prediction

Price dynamics operate at multiple time scales simultaneously. The PPWM uses a **hierarchical latent structure** to represent these scales:

### Scale Decomposition

```
High-frequency layer  (minutes–hours):  microstructure, order flow imbalance
Daily layer           (days):            momentum, mean-reversion, earnings catalysts
Macro layer           (weeks–months):   rate cycles, sector rotation, macro surprises
Structural layer      (months–years):   valuation mean-reversion, secular trends
```

Each layer maintains its own latent state, and the prediction for any horizon aggregates contributions from all relevant scales:

```python
class HierarchicalPPWM(nn.Module):
    """
    Multi-scale World Model for price prediction.
    Each layer operates at a different temporal resolution.
    """
    def predict(self, observations: dict[str, Tensor], horizon_days: int) -> Tensor:
        # Encode at each scale
        z_micro  = self.micro_encoder(observations['intraday'])
        z_daily  = self.daily_encoder(observations['daily'])
        z_macro  = self.macro_encoder(observations['macro'])
        z_struct = self.structural_encoder(observations['quarterly'])

        # Fuse across scales (gating by relevance to horizon)
        gate = self.horizon_gate(torch.tensor([horizon_days], dtype=torch.float32))
        z_fused = gate[0]*z_micro + gate[1]*z_daily + gate[2]*z_macro + gate[3]*z_struct

        # Roll out and decode
        paths = self.dynamics.rollout(z_fused, horizon=horizon_days)
        return self.decoder(paths)
```

---

## Uncertainty Quantification in Price Prediction

A PPWM explicitly separates **aleatoric uncertainty** (irreducible randomness in markets) from **epistemic uncertainty** (model uncertainty due to limited data):

### Sources of Uncertainty

| Uncertainty Type | Source | Managed By |
|---|---|---|
| **Aleatoric** | Fundamental price randomness | Stochastic dynamics model output |
| **Epistemic (model)** | Limited training data, model mis-specification | Monte Carlo Dropout or ensemble |
| **Epistemic (regime)** | Ambiguous current regime | Entropy of regime probability distribution |
| **Epistemic (structural)** | Regime never seen in training | Out-of-distribution detection on `z_t` |

### Uncertainty-Aware Prediction Output

```python
def predict_with_uncertainty(
    world_model,
    obs: Tensor,
    horizon: int,
    n_mc_dropout: int = 50,
) -> dict:
    """
    Generate price forecast with full uncertainty decomposition.
    """
    world_model.train()   # enable dropout for epistemic uncertainty

    mc_medians = []
    for _ in range(n_mc_dropout):
        z_mean, z_log_var = world_model.encoder(obs)
        z = z_mean + torch.randn_like(z_mean) * torch.exp(0.5 * z_log_var)
        paths = generate_price_paths(world_model, z, obs['price'], horizon)
        mc_medians.append(paths['p50'])

    world_model.eval()

    mc_medians = torch.stack(mc_medians)
    aleatoric  = paths['std'].mean()                          # within-sample spread
    epistemic  = mc_medians.std(dim=0).mean()                 # across-MC-sample spread

    return {
        'forecast_median':     mc_medians.mean(dim=0),
        'aleatoric_std':       aleatoric,
        'epistemic_std':       epistemic,
        'total_std':           (aleatoric**2 + epistemic**2).sqrt(),
        'prediction_interval': (
            mc_medians.mean(dim=0) - 1.96 * (aleatoric**2 + epistemic**2).sqrt(),
            mc_medians.mean(dim=0) + 1.96 * (aleatoric**2 + epistemic**2).sqrt(),
        ),
    }
```

---

## Training the Price Prediction World Model

The PPWM is trained end-to-end using a multi-component loss function that balances reconstruction accuracy, latent regularisation, and temporal consistency:

### Loss Function

```
L_total = L_reconstruction + β · L_KL + L_temporal + L_tail
```

Where:

- **L_reconstruction** — log-likelihood of observed prices under the decoded distribution
- **L_KL** — KL divergence between posterior and prior latent distributions (β-VAE style)
- **L_temporal** — consistency of latent dynamics across consecutive time steps
- **L_tail** — explicit penalty for under-estimating tail probabilities (calibration)

```python
def ppwm_loss(
    model,
    obs_sequence: Tensor,
    beta_kl: float = 1.0,
    gamma_tail: float = 0.5,
) -> Tensor:
    """
    Full PPWM training loss over an observation sequence.
    """
    posterior_mean, posterior_log_var = model.encoder(obs_sequence)
    prior_mean, prior_log_var         = model.dynamics(posterior_mean)

    # Reconstruction loss (NLL under decoded distribution)
    pred_mean, pred_std = model.decoder(posterior_mean)
    L_recon = -Normal(pred_mean, pred_std).log_prob(obs_sequence['returns']).mean()

    # KL regularisation
    L_kl = kl_divergence(
        Normal(posterior_mean, torch.exp(0.5 * posterior_log_var)),
        Normal(prior_mean,     torch.exp(0.5 * prior_log_var)),
    ).mean()

    # Temporal consistency: latent states should evolve smoothly
    L_temporal = (posterior_mean[:, 1:] - prior_mean[:, :-1]).pow(2).mean()

    # Tail calibration: penalise under-coverage at 5th / 95th percentile
    realized = obs_sequence['returns']
    L_tail = (
        F.relu(realized.quantile(0.05) - pred_mean - 1.645 * pred_std) +
        F.relu(pred_mean + 1.645 * pred_std - realized.quantile(0.95))
    ).mean()

    return L_recon + beta_kl * L_kl + L_temporal + gamma_tail * L_tail
```

---

## Evaluating Price Prediction Quality

Because the PPWM produces distributions, evaluation requires **probabilistic scoring metrics** beyond standard RMSE:

### Key Evaluation Metrics

| Metric | Definition | Interpretation |
|---|---|---|
| **CRPS** | Continuous Ranked Probability Score | Overall distribution quality; lower is better |
| **Coverage** | Fraction of realised prices within N% interval | Calibration: should equal N% |
| **Sharpness** | Width of 80% prediction interval | Narrower is better if coverage is maintained |
| **Pinball loss** | Quantile-specific accuracy at τ = `{0.05, 0.25, 0.75, 0.95}` | Tail and body accuracy |
| **Directional accuracy** | P(correct sign of price move at horizon H) | Trend prediction quality |
| **Regime-conditional CRPS** | CRPS stratified by inferred regime | Detects regime-specific weaknesses |

### Calibration Plot

A well-calibrated PPWM satisfies:

```
For all α ∈ [0, 1]:
  P(p_{t+H} ≤ F^{-1}(α)) ≈ α
```

Where `F^{-1}(α)` is the α-quantile of the predicted distribution. Deviation from this line indicates systematic over- or under-confidence.

---

## Real-World Price Prediction Applications

### 1. Intraday Price Path Prediction

For high-frequency trading, the PPWM generates short-horizon price distributions (minutes to hours) conditioned on order book state, recent trade flow, and micro-regime signals.

**Key advantage over LSTM/Transformer baselines:** explicit uncertainty output prevents overconfident position sizing during regime ambiguity.

### 2. Earnings-Day Price Prediction

Around earnings announcements, the PPWM generates a **bimodal price distribution** reflecting the possibility of a positive or negative surprise:

```
P(price jump | earnings) = w_+ · N(μ_+, σ_+) + w_- · N(μ_-, σ_-)
```

Where `w_+`, `w_-` are the model's inferred probabilities of a positive and negative surprise, learned from historical earnings dynamics encoded in the latent state.

### 3. Options Pricing Support

The PPWM's simulated price paths provide a model-implied return distribution that can substitute for the Black-Scholes assumption of log-normal returns:

```python
def world_model_option_price(
    world_model,
    current_state: Tensor,
    spot: float,
    strike: float,
    expiry_days: int,
    option_type: str = 'call',
) -> float:
    """
    Price a European option using World Model simulated paths.
    No distributional assumptions required.
    """
    paths = generate_price_paths(world_model, current_state, spot, expiry_days)
    terminal_prices = paths['all_paths'][:, -1]

    if option_type == 'call':
        payoffs = F.relu(terminal_prices - strike)
    else:
        payoffs = F.relu(strike - terminal_prices)

    # Discount at risk-free rate
    discount = torch.exp(torch.tensor(-world_model.risk_free_rate * expiry_days / 252))
    return (payoffs.mean() * discount).item()
```

### 4. Cross-Asset Price Prediction

The PPWM's shared latent state captures **cross-asset dependencies** automatically. When equity volatility rises in the latent state, the model automatically increases uncertainty in bond and commodity forecasts — reflecting the empirical correlation structure learned during training.

---

## Integrating Price Prediction with Portfolio Management

The PPWM's price distributions feed directly into the portfolio simulation engine described in Chapter 8:

```python
def build_portfolio_from_ppwm(
    world_model,
    universe: list[str],
    current_states: dict[str, Tensor],
    current_prices: dict[str, float],
    horizon_days: int = 21,
    risk_target: float = 0.10,   # annualised volatility target
) -> dict[str, float]:
    """
    Construct a portfolio allocation from PPWM price forecasts.
    """
    # 1. Generate price distribution for each asset
    forecasts = {
        ticker: generate_price_paths(
            world_model, current_states[ticker], current_prices[ticker], horizon_days
        )
        for ticker in universe
    }

    # 2. Compute expected returns and covariance from simulated paths
    final_returns = torch.stack([
        forecasts[t]['all_paths'][:, -1] / current_prices[t] - 1
        for t in universe
    ], dim=1)   # (n_paths, n_assets)

    mu     = final_returns.mean(dim=0)
    Sigma  = torch.cov(final_returns.T)

    # 3. Solve risk-targeted mean-variance problem
    weights = risk_targeted_optimizer(mu, Sigma, risk_target)
    return dict(zip(universe, weights.tolist()))
```

---

## Limitations of Price Prediction World Models

Despite their advantages over point-estimate approaches, PPWMs carry important limitations:

1. **Non-stationarity.** Financial price dynamics change over time. A PPWM trained on historical data may not capture novel regimes (e.g., zero-interest-rate policy, pandemic dislocations). Continuous retraining and out-of-distribution monitoring are essential.

2. **Latent state identifiability.** Multiple latent configurations may explain the same observations equally well. Regularisation and careful architecture design are needed to ensure that the latent state captures meaningful structure.

3. **Computationally intensive.** Generating thousands of sample paths at inference time requires GPU resources. For real-time applications, techniques such as distillation, path caching, and approximate inference are necessary.

4. **No fundamental value anchor.** The PPWM models price dynamics, not intrinsic value. In extreme cases (bubbles, crises), price dynamics can deviate sharply from fundamental value for extended periods — a regime where a dynamics-only model may generate misleading distributions.

5. **Adversarial market participants.** If the model's predictions become widely known and acted upon, the price dynamics it was trained on may change — the reflexivity problem discussed in Chapter 10.

---

## Chapter Summary

- The Price Prediction World Model (PPWM) reframes price forecasting as a **latent-state inference and generative rollout** problem, replacing point estimates with full probability distributions over future price trajectories
- The **Vision–Memory–Controller** architecture encodes market observations into latent state, propagates the state forward with learned dynamics, and decodes it into return distributions
- **Regime conditioning** allows the model to apply fundamentally different price dynamics depending on the inferred market regime — capturing the non-stationary nature of financial returns
- **Multi-scale hierarchical architecture** decomposes price dynamics across microstructure, daily, macro, and structural time scales, generating forecasts appropriate for each horizon
- **Uncertainty quantification** explicitly separates aleatoric (irreducible) from epistemic (model) uncertainty — enabling uncertainty-aware position sizing and risk management
- **Probabilistic evaluation metrics** (CRPS, coverage, pinball loss) replace RMSE as the appropriate benchmark for distributional forecasts
- Real-world applications include intraday path prediction, earnings-day bimodal forecasting, options pricing without distributional assumptions, and cross-asset portfolio construction
- Limitations include non-stationarity, computational cost, latent identifiability, and the reflexivity problem — all requiring active monitoring and mitigation in production systems

---

## Looking Ahead

Price prediction is not the end goal — it is an intermediate representation on the path to **decision-making under uncertainty**. The PPWM's output (a distribution over future prices) feeds directly into portfolio optimisers, risk engines, and execution systems.

The complete chain — from raw market observations through latent state inference, probabilistic price simulation, and resilience-weighted portfolio construction — is the architecture of **next-generation financial intelligence**.

> *"A point forecast is a claim about the future. A distributional forecast is an honest account of what we know and do not know."*
>
> The World Model's contribution to price prediction is not greater accuracy in a narrow sense — it is **calibrated honesty** about the full range of what may come.

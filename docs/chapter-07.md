---
id: chapter-07
title: Regime Shifts and Hidden States
sidebar_label: "Chapter 7 — Regime Shifts and Hidden States"
sidebar_position: 8
---

# Chapter 7

## Regime Shifts and Hidden States

Financial markets do not move randomly through a single statistical environment. They cycle through distinct **regimes** — structural states in which the statistical properties of returns, volatility, correlations, and macro dynamics are fundamentally different from one another.

A World Model can infer these hidden regime variables as **latent state variables**, enabling regime-aware simulation, early transition detection, and regime-conditioned portfolio optimization.

:::tip Interactive Simulator
Explore how macro conditions drive regime probabilities and early warning signals in the **[Regime Shift Simulator →](/regime-simulator)** companion tool for this chapter.
:::

---

## The Four Market Regimes

Markets cycle through four primary regimes that correspond to phases of the business and credit cycle:

![Market Regime Cycle](/img/regime-cycle.svg)

### 1. Expansion

The economy grows steadily. Corporate earnings rise. Credit conditions are accommodative. Equity markets trend upward with low volatility.

**Typical characteristics:**
- GDP growth above trend
- Moderate inflation (1.5%–3%)
- Accommodative or neutral monetary policy
- Low credit spreads (IG spreads below 100bps)
- Low implied volatility (VIX below 20)
- Broad equity market leadership across sectors

**Historical example — 2010–2019 US expansion:** Following the Global Financial Crisis, the US economy entered the longest expansion in recorded history (128 months). The S&P 500 returned over 400% with annualised volatility below 15%. VIX averaged 15.4; credit spreads remained compressed; the Fed Funds rate rose only gradually from 0% to 2.5%.

### 2. Overheating

Growth peaks and imbalances build. Inflation rises above target. Central banks tighten policy aggressively to cool demand. Risk assets become vulnerable.

**Typical characteristics:**
- GDP at or above potential output
- Inflation elevated and rising (above 4–5%)
- Central banks raising rates aggressively
- Yield curve flattening or inverting
- Elevated commodity prices
- Mixed equity performance — cyclicals struggle, defensives outperform

**Historical example — 2021–2022 post-COVID overheating:** Unprecedented fiscal stimulus, supply-chain disruption, and energy shocks drove US CPI to 9.1% by June 2022. The Federal Reserve responded with the fastest rate-hiking cycle since the 1980s — raising rates by 525bps over 16 months. The S&P 500 fell 25% peak-to-trough and the NASDAQ declined over 35%.

### 3. Contraction

The economy slows sharply or enters recession. Credit conditions tighten. Earnings disappoint. Risk aversion increases significantly.

**Typical characteristics:**
- GDP falling below trend or negative
- Unemployment rising sharply
- Credit spreads widening sharply (HY above 700bps)
- High and rising volatility (VIX above 30)
- Broad equity bear market (drawdowns typically 30–55%)
- Flight to quality (government bonds, gold, USD)

**Historical example — 2008–2009 Global Financial Crisis:** The collapse of structured credit markets triggered a systemic liquidity crisis. US GDP contracted 4.3% peak-to-trough. The S&P 500 fell 57% from October 2007 to March 2009. VIX hit an intraday high of 89.5. HY credit spreads exceeded 2,000bps. LIBOR-OIS spread — a key liquidity stress indicator — reached 364bps.

### 4. Recovery

Growth returns from a trough. Policy remains supportive. Risk appetite recovers. Asset prices begin to re-price higher.

**Typical characteristics:**
- GDP stabilizing and beginning to recover
- Inflation low or falling (often below target)
- Monetary policy still accommodative (rates low, QE active)
- Improving credit conditions
- Early equity bull market — cyclicals, financials and small caps lead
- Declining volatility from stressed levels

**Historical example — 2009 recovery:** The Fed's first round of QE (QE1, launched March 2009) combined with TARP stabilisation marked the turning point. The S&P 500 rallied 68% in the 12 months following the March 2009 trough. High-yield spreads compressed from 2,000bps to below 600bps within 18 months.

---

## Regime Characteristics at a Glance

| Indicator | Expansion | Overheating | Contraction | Recovery |
|---|---|---|---|---|
| GDP Growth | ↑ Rising | High / Peaking | ↓ Falling | Recovering |
| Inflation | Moderate (1–3%) | ↑ Elevated (&gt;4%) | ↓ Falling | Low (&lt;2%) |
| Policy Rates | Low / Rising | ↑ High | ↓ Falling | Low |
| VIX | Low (&lt; 20) | Moderate (20–28) | High (&gt; 30) | Declining |
| Equity Trend | ↑ Bullish | Mixed / Topping | ↓ Bearish | ↑ Recovering |
| Credit Spreads | Tight (&lt;100bps IG) | Widening | ↑ Wide (&gt;700bps HY) | ↓ Narrowing |
| Yield Curve | Normal / Steep | Flattening | Flat / Inverted | Steepening |
| Typical Duration | 3–8 years | 1–2 years | 1–3 years | 1–3 years |

---

## Hidden States and Latent Variables

The four regimes described above are **not directly observable**. No single indicator unambiguously signals which regime is active. Regimes must be *inferred* from a combination of noisy, delayed, and sometimes contradictory signals.

This is the classical **hidden state problem**. The true regime is a latent variable — hidden from direct observation but detectable through its influence on observable data.

![Hidden State Inference in a Financial World Model](/img/hidden-state-inference.svg)

### Why Hidden States Explain More Variance

A World Model that encodes the current regime as a latent variable `z_t` can explain variance that observable indicators cannot:

- **Cross-asset correlation structure** changes dramatically across regimes. In Expansion, equity–bond correlations are typically negative (bonds hedge equities). In Crisis, correlations converge toward 1 — diversification fails precisely when it is most needed.
- **Volatility clustering** is regime-dependent. A single high-volatility observation is far more informative when the model knows it is in a Contraction regime than during an Expansion where the same spike may quickly revert.
- **Factor exposures** shift — value stocks outperform in Recovery versus Contraction, even when their fundamentals are identical. The *same* security has different risk and return characteristics in different regimes.
- **Tail risk** is asymmetric. In Expansion, the distribution of returns is approximately normal. In Contraction, it exhibits heavy left-tail skew with fat tails — standard Gaussian risk models dramatically underestimate loss probability.

Latent regime variables capture this structural shift in a way that observable indicators alone cannot.

---

## Probabilistic Regime Modeling

### Hidden Markov Models (HMMs)

The classical statistical approach to regime detection uses **Hidden Markov Models**. In a HMM:

- The hidden state `s_t ∈ {Expansion, Overheating, Contraction, Recovery}` evolves according to a **transition matrix** `A` where `A[i][j] = P(s_t = j | s_{t-1} = i)`
- Observable data `o_t` (e.g., equity returns, yield curve slope, credit spreads) is generated from an emission distribution conditioned on `s_t`
- The **Viterbi algorithm** infers the most probable hidden state sequence given all observations
- The **forward-backward algorithm** computes the posterior probability of each regime at each time step

```
P(s_t | o_1, ..., o_t) ∝ P(o_t | s_t) · Σ P(s_t | s_{t-1}) · P(s_{t-1} | o_1, ..., o_{t-1})
```

The transition matrix encodes the expected persistence of each regime and the likely transition paths:

![HMM Regime Transition Matrix](/img/hmm-transition-matrix.svg)

**Reading the matrix:** Each row shows the quarterly probability of remaining in or transitioning from a given regime. For example, a portfolio currently in the Expansion regime has a 72% chance of remaining there next quarter, a 20% chance of moving to Overheating, and only a 5% chance of jumping directly to Contraction.

**Limitations of HMMs:** They assume linear Gaussian emission distributions, fixed time-invariant transition probabilities, and strict Markovian dynamics — approximations that may not hold in real markets where regime dynamics are non-stationary and feedback-driven.

#### Python Example: Fitting a Two-Regime HMM

```python
import numpy as np
from hmmlearn import hmm

# Monthly observations: [equity_return, vix_change, credit_spread_change]
# Shape: (n_months, n_features)
obs = np.column_stack([equity_returns, vix_changes, spread_changes])

# Fit a 4-state Gaussian HMM
model = hmm.GaussianHMM(
    n_components=4,
    covariance_type="full",
    n_iter=200,
    random_state=42
)
model.fit(obs)

# Decode the most probable hidden state sequence
hidden_states = model.predict(obs)

# Compute posterior regime probabilities
regime_probs = model.predict_proba(obs)  # shape: (n_months, 4)

# Print transition matrix
print("Transition Matrix A:")
print(np.round(model.transmat_, 3))
```

### Recurrent State Space Models (RSSM)

A World Model replaces the classical HMM with a deep **Recurrent State Space Model**. The RSSM maintains both a *deterministic recurrent state* `h_t` and a *stochastic latent state* `z_t`:

```
Encoder:    z_t ~ q_φ(z_t | o_t, h_{t-1})      # posterior inference
Prior:      z_t ~ p_θ(z_t | h_{t-1})            # prior prediction
Dynamics:   h_t  = f_dyn(z_t, h_{t-1})          # GRU update
Decoder:    ô_t ~ p_θ(o_t | z_t, h_t)           # reconstruction
```

![RSSM Architecture](/img/rssm-architecture.svg)

The training objective is a variational lower bound (ELBO) that balances reconstruction accuracy with KL divergence between posterior and prior:

```
L(θ, φ) = Σ_t [ E_q[log p_θ(o_t | z_t, h_t)] - β · KL(q_φ(z_t | o_t, h_{t-1}) || p_θ(z_t | h_{t-1})) ]
```

This approach:
- Learns **non-linear** emission and transition dynamics from data — no Gaussian assumption required
- Maintains a continuous, high-dimensional latent state `z_t ∈ ℝ^d` that can encode richer structure than four discrete regimes
- Supports **uncertainty quantification** — the model outputs a *distribution* over latent states `p(z_t)`, not a point estimate
- Enables **simulation**: given the current latent state `z_t`, the RSSM can roll forward to generate thousands of plausible future regime paths
- Can be queried for **counterfactuals**: "What is the distribution of market outcomes if the Fed raises rates by 200bps?"

#### Python Example: RSSM Rollout for Regime Simulation

```python
import torch

def simulate_regime_paths(rssm_model, z_current, h_current, n_steps=12, n_paths=1000):
    """
    Roll out n_paths simulated futures from the current latent state.
    Returns: (n_paths, n_steps, latent_dim) trajectory tensor
    """
    paths = []
    for _ in range(n_paths):
        z, h = z_current.clone(), h_current.clone()
        path = []
        for _ in range(n_steps):
            # Sample next latent state from prior
            z_prior_dist = rssm_model.prior(h)
            z = z_prior_dist.rsample()
            # Update recurrent state
            h = rssm_model.dynamics(z, h)
            path.append(z.detach())
        paths.append(torch.stack(path))
    return torch.stack(paths)  # (n_paths, n_steps, latent_dim)

# Classify each simulated path into regimes
regime_logits = regime_classifier(simulated_paths)      # (n_paths, n_steps, 4)
regime_probs  = torch.softmax(regime_logits, dim=-1)

# Probability of entering Contraction within 6 months
p_contraction = (regime_probs[:, :6, 2].max(dim=1).values > 0.5).float().mean()
print(f"P(Contraction within 6 months) = {p_contraction:.1%}")
```

---

## Detecting Regime Transitions

Early detection of regime transitions is one of the most commercially valuable applications of financial World Models.

### Why Transitions Are Difficult

- **Lagging indicators** (e.g., unemployment, GDP) confirm regime changes only after significant delay — often 2–3 quarters after the fact
- **Reflexivity** — market participants' attempts to front-run transitions can obscure or even accelerate them
- **Gradual versus abrupt transitions** — some transitions (Expansion → Overheating) occur slowly over many months; others (Overheating → Contraction) can be abrupt, triggered by a single shock event

### World Model Early Warning Signals

A trained World Model provides three complementary early-warning mechanisms:

![Early Warning Signals — Regime Transition Detection](/img/early-warning-signals.svg)

1. **Latent state drift:** Continuous monitoring of `z_t` shows gradual drift toward contraction-like latent regions before a macro turn is confirmed by any observable indicator. This is the earliest signal — detectable 4–8 weeks before traditional models.

2. **Regime probability shifts:** The posterior distribution `P(regime | z_t)` can shift from "80% Expansion" to "55% Expansion, 40% Overheating" weeks before any observable indicator confirms the change. The *rate of change* in regime probabilities is itself a signal.

3. **Predictive entropy:** As the model approaches a transition, its uncertainty about the next state increases — a rising predictive entropy `H(z_{t+1} | z_t)` is itself an early warning signal, even before any single regime becomes dominant.

4. **Counterfactual simulation:** The model can be queried: "If rates rise by 100bps in the next two quarters, what is the probability of entering Contraction within 12 months?" This produces an actionable probability — not a qualitative description.

### Case Study: Detecting the 2022 Regime Shift

The 2021–2022 transition from Expansion to Overheating to incipient Contraction provides a clear illustration. A World Model trained on 1990–2021 data would have generated the following signal sequence:

| Date | Observable Signal | World Model Signal |
|---|---|---|
| Nov 2021 | CPI rising, Fed "transitory" language | Latent drift toward Overheating; P(Overheat) rises from 12% → 31% |
| Jan 2022 | Fed pivots, signals March hike | P(Overheat) reaches 58%; entropy spike |
| Mar 2022 | First hike (+25bps) | Regime probability: E:28%, O:52%, C:18% |
| Jun 2022 | CPI peaks at 9.1% | P(Contraction) crosses 35%; early warning triggered |
| Oct 2022 | S&P 500 trough | Traditional models confirm bear market retrospectively |

The World Model's early warning window was approximately **8–12 weeks ahead** of the point at which traditional rule-based models would confirm the regime shift.

---

## Regime-Conditioned Portfolio Strategy

A World Model that infers the current regime enables **regime-conditioned portfolio optimization**:

| Regime | Equity Weight | Fixed Income | Commodities | Alternatives | Active Hedges |
|---|---|---|---|---|---|
| Expansion | ↑ High (60–70%) | Neutral / Short dur. | Moderate (5–10%) | Low vol alts | Minimal |
| Overheating | ↓ Reduce (35–50%) | ↓ Short duration | ↑ Commodities (15–20%) | Real assets | Vol exposure (VIX calls) |
| Contraction | ↓ Low (15–30%) | ↑ Long government bonds | Gold (10–15%) | Defensive equity | Active (puts, VIX longs) |
| Recovery | ↑ Building (45–60%) | Neutral | Moderate | ↑ Cyclicals, EM | Minimal |

Rather than applying fixed allocation rules, a World Model computes the **probability-weighted expected return** of each allocation across all possible regimes:

```
E[R_portfolio] = Σ_r P(regime = r | z_t) · E[R_portfolio | regime = r]
```

This produces a **distribution-aware** portfolio allocation rather than a point-estimate allocation, with explicit quantification of regime uncertainty in the optimization objective.

#### Example: Probability-Weighted Regime Allocation

```python
import numpy as np

# Current regime posterior probabilities from World Model
regime_probs = np.array([0.28, 0.52, 0.18, 0.02])  # E, O, C, R

# Expected equity returns by regime (annualised %)
equity_returns_by_regime = np.array([12.0, 2.0, -18.0, 15.0])

# Expected bond returns by regime (annualised %)
bond_returns_by_regime = np.array([2.0, -4.0, 8.0, 3.0])

# Probability-weighted expected returns
equity_expected = np.dot(regime_probs, equity_returns_by_regime)
bond_expected   = np.dot(regime_probs, bond_returns_by_regime)

# Regime-conditioned equity target weight (example rule)
# Reduce equity weight proportionally to P(Overheating) + P(Contraction)
bear_prob = regime_probs[1] + regime_probs[2]  # 0.70
equity_weight = max(0.20, 0.65 * (1 - bear_prob))

print(f"Bear regime probability:        {bear_prob:.0%}")
print(f"Probability-weighted eq. return: {equity_expected:.1f}%")
print(f"Probability-weighted bd. return: {bond_expected:.1f}%")
print(f"Regime-conditioned equity weight: {equity_weight:.0%}")
```

---

## Formal Framework

The regime-detection component of a financial World Model can be written formally as:

```
Encoder:     z_t ~ q_φ(z_t | o_{1:t}, h_{t-1})     # posterior inference
Dynamics:    h_t  = f_dyn(z_t, h_{t-1})              # recurrent update
Prior:       z_t ~ p_θ(z_t | h_{t-1})                # prior transition
Regime:      r_t  = argmax_r P(r | z_t; ψ)           # discrete regime label
Warning:     H_t  = H(z_{t+1} | z_t, h_t)           # predictive entropy
```

Where:
- `z_t ∈ ℝ^d` is the continuous stochastic latent state at time `t`
- `h_t ∈ ℝ^m` is the deterministic recurrent state (GRU hidden state)
- `r_t ∈ {Expansion, Overheating, Contraction, Recovery}` is the inferred discrete regime label
- `o_{1:t}` is the full history of multi-asset observations up to `t`
- `H(·)` is the Shannon entropy of the predictive next-state distribution

The RSSM can generate a **joint distribution** over future regime paths and asset returns simultaneously — enabling portfolio optimisation under full uncertainty rather than a single-scenario assumption.

---

## Chapter Summary

- Markets cycle through four structural regimes: **Expansion, Overheating, Contraction, and Recovery**, each with distinct statistical properties and historical examples from real market cycles
- Regimes are **hidden states** — they must be inferred from noisy, delayed observable signals through probabilistic inference
- Latent regime variables capture cross-asset correlation shifts, volatility clustering, tail-risk asymmetry, and factor rotation that observable indicators alone cannot explain
- Classical **Hidden Markov Models** provide a baseline approach with interpretable transition matrices; World Model **Recurrent State Space Models** offer non-linear, uncertainty-aware, simulatable regime inference
- World Models provide **three complementary early warning signals**: latent state drift, regime probability shifts, and predictive entropy — with a typical 4–12 week lead over traditional macro indicators
- The 2022 rate-hiking cycle illustrates how a World Model could have identified the Expansion-to-Overheating transition weeks before traditional indicators confirmed it
- Regime-conditioned portfolio optimization produces probability-weighted allocation strategies that adapt dynamically to the inferred market environment

The next chapter explores how portfolio simulation engines use these regime-aware world models to stress-test and optimize portfolios across thousands of simulated futures.

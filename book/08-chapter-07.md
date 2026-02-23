# Chapter 7

## Regime Shifts and Hidden States

Financial markets do not move randomly through a single statistical environment. They cycle through distinct **regimes** — structural states in which the statistical properties of returns, volatility, correlations, and macro dynamics are fundamentally different from one another.

A World Model can infer these hidden regime variables as **latent state variables**, enabling regime-aware simulation, early transition detection, and regime-conditioned portfolio optimization.

---

## The Four Market Regimes

Markets cycle through four primary regimes that correspond to phases of the business and credit cycle:

### 1. Expansion

The economy grows steadily. Corporate earnings rise. Credit conditions are accommodative. Equity markets trend upward with low volatility.

**Typical characteristics:**
- GDP growth above trend
- Moderate inflation
- Accommodative or neutral monetary policy
- Low credit spreads
- Low implied volatility (VIX below 20)
- Broad equity market leadership

### 2. Overheating

Growth peaks and imbalances build. Inflation rises above target. Central banks tighten policy aggressively to cool demand. Risk assets become vulnerable.

**Typical characteristics:**
- GDP at or above potential
- Inflation elevated and rising
- Central banks raising rates
- Yield curve flattening or inverting
- Elevated commodity prices
- Mixed equity performance — cyclicals struggle, defensives outperform

### 3. Contraction

The economy slows sharply or enters recession. Credit conditions tighten. Earnings disappoint. Risk aversion increases significantly.

**Typical characteristics:**
- GDP falling below trend or negative
- Unemployment rising
- Credit spreads widening sharply
- High and rising volatility (VIX above 30)
- Broad equity bear market
- Flight to quality (government bonds, gold)

### 4. Recovery

Growth returns from a trough. Policy remains supportive. Risk appetite recovers. Asset prices begin to re-price higher.

**Typical characteristics:**
- GDP stabilizing and beginning to recover
- Inflation low or falling
- Monetary policy still accommodative
- Improving credit conditions
- Early equity bull market — cyclicals and small caps lead
- Declining volatility

---

## Regime Characteristics at a Glance

| Indicator | Expansion | Overheating | Contraction | Recovery |
|---|---|---|---|---|
| GDP Growth | ↑ Rising | High / Peaking | ↓ Falling | Recovering |
| Inflation | Moderate | ↑ Elevated | ↓ Falling | Low |
| Policy Rates | Low / Rising | ↑ High | ↓ Falling | Low |
| VIX | Low (< 20) | Moderate | High (> 30) | Declining |
| Equity Trend | ↑ Bullish | Mixed | ↓ Bearish | ↑ Recovering |
| Credit Spreads | Tight | Widening | ↑ Wide | ↓ Narrowing |
| Yield Curve | Normal / Steep | Flattening | Flat / Inverted | Steepening |

---

## Hidden States and Latent Variables

The four regimes described above are **not directly observable**. No single indicator unambiguously signals which regime is active. Regimes must be *inferred* from a combination of noisy, delayed, and sometimes contradictory signals.

This is the classical **hidden state problem**. The true regime is a latent variable — hidden from direct observation but detectable through its influence on observable data.

### Why Hidden States Explain More Variance

A World Model that encodes the current regime as a latent variable `z_t` can explain variance that observable indicators cannot:

- **Cross-asset correlation structure** changes dramatically across regimes. In Expansion, equity–bond correlations are typically negative. In Crisis, correlations converge toward 1.
- **Volatility clustering** is regime-dependent. A single high-volatility observation is far more informative when the model knows it is in a Contraction regime.
- **Factor exposures** shift — value stocks behave differently in Recovery versus Contraction, even when their fundamentals are identical.

Latent regime variables capture this structural shift in a way that observable indicators alone cannot.

---

## Probabilistic Regime Modeling

### Hidden Markov Models (HMMs)

The classical statistical approach to regime detection uses **Hidden Markov Models**. In a HMM:

- The hidden state `s_t ∈ {Expansion, Overheating, Contraction, Recovery}` evolves according to a transition matrix `A`
- Observable data `o_t` is generated from an emission distribution conditioned on `s_t`
- The Viterbi algorithm infers the most probable hidden state sequence
- The forward-backward algorithm computes the posterior probability of each regime at each time step

```
P(s_t | o_1, ..., o_t) ∝ P(o_t | s_t) · Σ P(s_t | s_{t-1}) · P(s_{t-1} | o_1, ..., o_{t-1})
```

**Limitations of HMMs:** They assume linear emission distributions, fixed transition probabilities, and Markovian dynamics — approximations that may not hold in real markets.

### Recurrent State Space Models (RSSM)

A World Model replaces the classical HMM with a deep **Recurrent State Space Model**:

```
Encoder:    z_t = f_enc(o_t, h_{t-1})
Dynamics:   h_t = f_dyn(z_t, h_{t-1})
Decoder:    ô_t = f_dec(z_t, h_t)
```

This approach:
- Learns non-linear emission and transition dynamics from data
- Maintains a continuous, high-dimensional latent state that can encode more than four discrete regimes
- Supports uncertainty quantification — the model outputs a *distribution* over latent states, not a point estimate
- Enables simulation: given the current latent state, the model can roll forward to generate plausible future regime paths

---

## Detecting Regime Transitions

Early detection of regime transitions is one of the most commercially valuable applications of financial World Models.

### Why Transitions Are Difficult

- **Lagging indicators** (e.g., unemployment, GDP) confirm regime changes only after significant delay
- **Reflexivity** — market participants' attempts to front-run transitions can obscure them
- **Gradual versus abrupt transitions** — some transitions (Expansion → Overheating) occur slowly; others (Overheating → Contraction) can be abrupt

### World Model Early Warning Signals

A trained World Model provides several early-warning mechanisms:

1. **Latent state drift:** Continuous monitoring of `z_t` shows gradual drift toward contraction-like latent regions before a macro turn is confirmed
2. **Regime probability shifts:** The posterior distribution `P(regime | z_t)` can shift from "80% Expansion" to "55% Expansion, 40% Overheating" weeks before any observable indicator confirms the change
3. **Predictive entropy:** As the model approaches a transition, its uncertainty about the next state increases — a rising predictive entropy is itself an early warning signal
4. **Counterfactual simulation:** The model can be queried: "If rates rise by 100bps in the next two quarters, what is the probability of entering Contraction within 12 months?"

---

## Regime-Conditioned Portfolio Strategy

A World Model that infers the current regime enables **regime-conditioned portfolio optimization**:

| Regime | Equity Weight | Duration | Alternatives | Hedges |
|---|---|---|---|---|
| Expansion | ↑ High | Neutral | Moderate | Minimal |
| Overheating | ↓ Reduce | ↓ Short | Commodities | Vol exposure |
| Contraction | ↓ Low | ↑ Long | Gold, Defensives | Active (puts, VIX) |
| Recovery | ↑ Building | Neutral | Cyclicals | Minimal |

Rather than applying fixed allocation rules, a World Model computes the **probability-weighted expected return** of each allocation across all possible regimes, producing a distribution-aware portfolio rather than a point-estimate allocation.

---

## Formal Framework

The regime-detection component of a financial World Model can be written formally as:

```
Encoder:     z_t = q_φ(z_t | o_{1:t})
Regime:      r_t = argmax P(r | z_t)
Transition:  P(z_{t+1} | z_t, a_t) — learned dynamics
Warning:     H(z_{t+1} | z_t) — predictive entropy as early warning
```

Where:
- `z_t` is the continuous latent state at time `t`
- `r_t` is the inferred discrete regime label
- `o_{1:t}` is the full history of observations up to `t`
- `H(·)` is the predictive entropy of the next-state distribution

---

## Chapter Summary

- Markets cycle through four structural regimes: **Expansion, Overheating, Contraction, and Recovery**, each with distinct statistical properties
- Regimes are **hidden states** — they must be inferred from noisy, delayed observable signals
- Latent regime variables capture cross-asset correlation shifts, volatility clustering, and factor rotation that observable indicators alone cannot explain
- Classical Hidden Markov Models provide a baseline; World Model **Recurrent State Space Models** offer non-linear, uncertainty-aware, simulatable regime inference
- World Models provide early warning signals through **latent state drift**, **regime probability shifts**, and **predictive entropy**
- Regime-conditioned portfolio optimization produces allocation strategies that adapt dynamically to the inferred market environment

The next chapter explores how portfolio simulation engines use these regime-aware world models to stress-test and optimize portfolios across thousands of simulated futures.

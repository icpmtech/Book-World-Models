# Chapter 11

## Toward Financial AGI

Artificial General Intelligence requires:

- Causal reasoning
- Temporal continuity
- Environmental modeling

Financial markets are one of the most complex dynamic systems in existence.

Building accurate financial world models is not just an investment breakthrough.

It is **a step toward general intelligence**.

---

## What Is Artificial General Intelligence?

Artificial General Intelligence (AGI) refers to a system capable of performing **any intellectual task** that a human can perform — not by brute-force memorisation of patterns, but by building an internal model of the world and reasoning about it.

The defining capabilities of AGI are:

| Capability | Description | Current AI | World Models |
|---|---|---|---|
| **Causal reasoning** | Understand *why* events happen, not just correlate them | Absent in LLMs | Explicit in causal graphs |
| **Temporal continuity** | Maintain and update a consistent world state over time | No persistent state | Latent state `z_t` |
| **Environmental modeling** | Simulate hypothetical futures and counterfactuals | Narrative only | Generative simulation |
| **Transfer learning** | Apply knowledge from one domain to another | Narrow | Partial (regime transfer) |
| **Uncertainty quantification** | Know what you don't know | Not quantified | Full posterior |

Financial world models implement the first three capabilities — and partially address the fourth and fifth.

---

## Why Financial Markets Are an AGI Test Environment

### The Complexity Argument

Financial markets are among the most complex adaptive systems ever created by humans:

- **Millions of interacting agents**, each with heterogeneous beliefs, incentives, and information sets
- **Non-stationary dynamics**: the statistical properties of returns change over time as regimes shift
- **Reflexivity**: participant behavior changes the system they are trying to predict
- **Sparse rewards**: the feedback signal (profit or loss) arrives with substantial delay and noise
- **Partial observability**: the true state of the market is never directly observable

A system that can build an accurate, updatable, probabilistic model of financial markets would need to solve all of these challenges. This is precisely why financial markets represent a **demanding benchmark** for progress toward AGI.

### The Three AGI Pillars in Financial Context

#### 1. Causal Reasoning

A financial world model encodes an explicit **causal structure** linking macro variables, sector dynamics, and asset prices.

```
Rate Rise → Bond Yield ↑ → Equity Discount Rate ↑ → Growth Stock Price ↓
         → Bank Net Interest Margin ↑ → Bank Equity Price ↑
         → Mortgage Rate ↑ → Housing Activity ↓ → Construction Sector ↓
```

This causal graph allows the model to distinguish correlation from causation and simulate realistic propagation paths through the financial system.

#### 2. Temporal Continuity

A world model maintains a **persistent latent state** `z_t` that is continuously updated as new observations arrive:

```
z_t = Encoder(o_t, z_{t-1})
```

This temporal continuity enables tracking slow-moving regime variables (credit cycles unfold over years) and detecting early warning signals before they appear in headline data.

#### 3. Environmental Modeling

The world model simulates the future evolution of the financial environment as a **full probability distribution** over possible futures:

```
p_θ(o_{t+1}, ..., o_{t+H} | z_t)
```

This generative capacity is what distinguishes a world model from a discriminative forecasting model.

---

## The RSSM as an AGI Building Block

The Recurrent State Space Model (RSSM) implements the core loop required for environmental modeling:

```
Prior:      p_θ(z_t | z_{t-1}, a_{t-1})           # predict next state
Posterior:  q_φ(z_t | z_{t-1}, a_{t-1}, o_t)      # update from observation
Decoder:    p_θ(o_t | z_t)                         # reconstruct observation
Reward:     p_θ(r_t | z_t)                         # predict outcome
```

This four-component loop — **predict, update, decode, reward** — is a minimal implementation of the cognition cycle attributed to intelligent agents in cognitive science.

---

## Regime Intelligence: The Hidden State

A key capability of a financial world model is its ability to infer **hidden regimes** — the underlying state of the economy not directly observable but determining the statistical behavior of all assets.

### The Four-Regime Model

| Regime | Macro Indicators | Asset Behavior | Latent State Signature |
|---|---|---|---|
| **Expansion** 🟢 | GDP rising, unemployment falling | Equities outperform, credit spreads tight | High `z_growth`, low `z_stress` |
| **Overheating** 🟠 | Inflation rising, labor tight | Commodities outperform, rates rise | High `z_inflation`, moderate `z_stress` |
| **Contraction** 🔴 | GDP falling, earnings compressing | Bonds outperform, equity drawdowns | High `z_stress`, low `z_growth` |
| **Recovery** 🟡 | Stimulus active, re-leveraging | Cyclicals outperform, credit recovers | Falling `z_stress`, rising `z_growth` |

The world model maintains a **continuous posterior** over regime probabilities:

```
P(regime_t = Contraction | o_1, ..., o_t) = 0.73
P(regime_t = Recovery    | o_1, ..., o_t) = 0.21
P(regime_t = Expansion   | o_1, ..., o_t) = 0.06
```

---

## Testing a Financial World Model for AGI Properties

### Test 1: Causal Intervention Test

**Purpose:** Verify that the model correctly propagates a causal shock through the learned graph.

```python
def test_causal_rate_shock(model, z_baseline):
    """
    Test that a rate shock propagates correctly through the world model.
    Expected: bonds down, growth stocks down, banks up, USD up.
    """
    delta_rate = torch.zeros_like(z_baseline)
    delta_rate[model.rate_dim] = +2.0  # 200bp rate shock

    z_shocked = z_baseline + delta_rate
    obs_baseline = model.decode(z_baseline)
    obs_shocked  = model.decode(z_shocked)

    assert obs_shocked['bond_return']   < obs_baseline['bond_return'],   "Bonds should fall on rate rise"
    assert obs_shocked['growth_return'] < obs_baseline['growth_return'], "Growth stocks should fall on rate rise"
    assert obs_shocked['bank_return']   > obs_baseline['bank_return'],   "Banks should benefit from rate rise"
    assert obs_shocked['usd_index']     > obs_baseline['usd_index'],     "USD should strengthen on rate rise"
```

### Test 2: Temporal Consistency Test

**Purpose:** Verify that the latent state evolves consistently over time.

```python
def test_temporal_consistency(model, expansion_obs, contraction_obs):
    """
    Test that the model correctly tracks a regime transition over time.
    """
    z = model.initial_state()
    expansion_probs  = []
    contraction_probs = []

    for obs in expansion_obs:
        z = model.update(z, obs)
        expansion_probs.append(model.regime_probabilities(z)['expansion'])

    for obs in contraction_obs:
        z = model.update(z, obs)
        contraction_probs.append(model.regime_probabilities(z)['contraction'])

    assert sum(expansion_probs[-5:]) / 5 > 0.6, "Should infer expansion during expansion observations"
    assert contraction_probs[-1] > contraction_probs[0], "Contraction probability should increase over time"
```

### Test 3: Counterfactual Consistency Test

**Purpose:** Verify that the model produces internally consistent counterfactual scenarios.

```python
def test_counterfactual_consistency(model, z_0, n_paths=1000):
    """
    Test that counterfactual shocks produce coherent distributional shifts.
    """
    baseline_returns = model.simulate(z_0, n_paths=n_paths, horizon=12)
    z_stressed       = model.inject_credit_stress(z_0, severity=3.0)
    stressed_returns = model.simulate(z_stressed, n_paths=n_paths, horizon=12)

    assert stressed_returns.mean() < baseline_returns.mean(), "Stressed scenario must underperform baseline"
    assert stressed_returns.std()  > baseline_returns.std(),  "Stress increases uncertainty"
    assert stressed_returns.quantile(0.05) < baseline_returns.quantile(0.05), "Tail risk must be worse under stress"
```

### Test 4: Hidden State Inference Test

**Purpose:** Verify that the hidden state captures latent market information not visible in headline prices.

```python
def test_hidden_state_inference(model, pre_crisis_obs):
    """
    Test that hidden state warns before observable price moves occur.
    """
    z = model.initial_state()
    stress_signals = []
    price_signals  = []

    for obs in pre_crisis_obs:
        z = model.update(z, obs)
        stress_signals.append(model.stress_indicator(z))
        price_signals.append(obs['price_momentum'])

    stress_warning_t = next((t for t, s in enumerate(stress_signals) if s > 0.5), None)
    price_warning_t  = next((t for t, p in enumerate(price_signals)  if p < -0.1), None)

    if stress_warning_t is not None and price_warning_t is not None:
        assert stress_warning_t <= price_warning_t, \
            f"Hidden state warned at t={stress_warning_t}, but prices only at t={price_warning_t}"
```

---

## From Financial Intelligence to General Intelligence

### What Financial AGI Achieves

A financial world model that passes the four tests above has demonstrated:

1. **Causal understanding** of how economic variables propagate through a complex system
2. **Temporal memory** maintained over months or years of evolving market conditions
3. **Generative simulation** of coherent, probabilistic futures
4. **Hidden state inference** — detecting latent structure not visible in surface observations

These are domain instances of **general cognitive capabilities**.

### The Transfer Question

The RSSM architecture, regime inference engine, and causal graph are all **domain-agnostic** components. The financial domain provides a richly observable, densely sampled environment in which to develop and validate these components — components that transfer directly to robotics, climate modeling, and epidemiology.

### The Measurement Standard

```
Financial AGI Benchmark:
Score = α₁ · CausalAccuracy
      + α₂ · TemporalConsistency
      + α₃ · CounterfactualCoherence
      + α₄ · EarlyWarningLead
      + α₅ · RegimeInferenceAccuracy
```

Where each component is measured on held-out data across at least two complete economic cycles.

---

## Chapter Summary

- **AGI requires** causal reasoning, temporal continuity, and environmental modeling — precisely the three capabilities that financial world models implement.
- Financial markets are an **ideal AGI test environment** due to their complexity, reflexivity, partial observability, and the richness of their observable signals.
- The **RSSM architecture** implements a minimal cognition loop — predict, update, decode, reward — that is domain-agnostic and transferable beyond finance.
- **Regime inference** is a concrete implementation of hidden state reasoning: inferring unobservable causes from observable effects.
- Four **AGI validation tests** — causal intervention, temporal consistency, counterfactual coherence, and hidden state inference — provide a concrete benchmark for measuring progress toward financial AGI.
- Financial world models are not merely investment tools: they are **prototypes for general-purpose environmental reasoning**, demonstrating that causal, temporal, and generative intelligence can be built from real-world data.

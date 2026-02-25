---
id: chapter-05
title: Building a Financial World Model
sidebar_label: "Chapter 5 — Building a Financial World Model"
sidebar_position: 6
---

# Chapter 5

## Building a Financial World Model

A financial world model must define its state space carefully.

Example state vector:

- Interest rates
- Inflation
- Oil price
- Volatility index
- Market liquidity
- Earnings growth
- Portfolio allocation

The model learns:

```
State(t) → Distribution of State(t+1)
```

This is not a price predictor.

It is a **market simulator**.

---

## State Transitions

At each time step, the world model takes the current state and produces a **probability distribution** over future states — not a single forecast, but a full distribution.

![State Transition Diagram](/img/state-transition.svg)

The transition function encodes:

- Causal relationships between variables
- Lag structures (how long effects take to propagate)
- Non-linearities (e.g. regime-dependent correlations)

---

## Simulated Futures

Because the model is probabilistic, it can generate thousands of possible future paths from any current state. This fan of outcomes captures the full uncertainty of the forecast.

![Simulation Fan Chart](/img/simulation-fan-chart.svg)

The width of the fan at any horizon reflects the **model's calibrated uncertainty** — wider bands at longer horizons, tighter bands where the current regime constrains outcomes.

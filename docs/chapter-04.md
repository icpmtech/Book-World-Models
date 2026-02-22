---
id: chapter-04
title: The V-M-C Architecture
sidebar_label: "Chapter 4 — The V-M-C Architecture"
sidebar_position: 5
---

# Chapter 4

## The V-M-C Architecture

Most World Models are structured around three core components:

### Vision Model (V)

Compresses high-dimensional input into a latent representation.

In finance, this could encode:

- Price history
- Macro indicators
- Volatility structure
- Earnings data

### Memory Model (M)

Learns temporal dynamics.

It models how the latent state evolves over time:

```
Latent(t) → Latent(t+1)
```

This captures:

- Market cycles
- Momentum decay
- Shock propagation
- Regime transitions

### Controller (C)

Uses the simulated future to choose optimal actions.

In markets, this means:

- Adjust allocation
- Hedge exposure
- Increase dividend weighting
- Rotate sectors

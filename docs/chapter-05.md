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

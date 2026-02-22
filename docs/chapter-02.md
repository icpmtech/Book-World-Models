---
id: chapter-02
title: What Is a World Model?
sidebar_label: "Chapter 2 — What Is a World Model?"
sidebar_position: 3
---

# Chapter 2

## What Is a World Model?

The modern formulation of World Models was introduced in 2018 by David Ha and Jürgen Schmidhuber.

A world model is an internal simulation engine.

Instead of predicting the next word, it predicts:

> The next state of the environment.

Formally:

```
State(t) → State(t+1)
```

A state can represent:

- Physical position of objects
- Economic conditions
- Market liquidity
- Investor sentiment
- Volatility regimes

In robotics, companies like NVIDIA use world models to simulate thousands of scenarios before deploying a robot in the real world.

In finance, this approach can be transformative.

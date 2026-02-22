# Financial World Models

## Simulating the Future of Capital Markets Beyond LLMs

Pedro Martins

---

## Copyright

© 2026 Pedro Martins
All rights reserved.

No part of this publication may be reproduced, distributed, or transmitted in any form or by any means without prior written permission of the author.

---

## Table of Contents

1. The Ceiling of Large Language Models
2. What Is a World Model?
3. From Words to Worlds
4. The V-M-C Architecture
5. Building a Financial World Model
6. Market Dynamics as Physics
7. Regime Shifts and Hidden States
8. Portfolio Simulation Engines
9. Scenario Generation and Counterfactual Futures
10. Risk, Ethics, and Market Reflexivity
11. Toward Financial AGI
12. The Future of Intelligent Capital Allocation
13. World Models in Finance: Improving Investment Returns and Decision Making in Stock Markets
14. References

---

## Chapter 1

### The Ceiling of Large Language Models

Large Language Models (LLMs) revolutionized artificial intelligence by mastering language prediction. Systems built on transformer architectures can summarize documents, answer complex questions, and even generate code.

However, they share a fundamental limitation:

They predict tokens — not reality.

An LLM can describe what might happen in a recession.
It cannot internally simulate the causal mechanics of one.

Financial markets are not text streams. They are dynamic systems shaped by feedback loops, delayed reactions, nonlinear transitions, and regime shifts.

To model markets properly, AI must move beyond sequence prediction toward state prediction.

This is where World Models enter the picture.

#### Why LLMs Fall Short in Stock Markets

LLMs face five structural limitations in financial markets:

1. **Token Prediction ≠ Price Dynamics** — LLMs learn linguistic correlations, not causal mechanics. A rate hike propagates simultaneously through bond yields, equity discount rates, currency markets, and credit spreads. An LLM describes this in prose; it cannot simulate it numerically.

2. **The Stale Knowledge Problem** — LLMs are trained on data up to a cutoff date and carry no live market state. They cannot respond dynamically to intraday events.

3. **No Uncertainty Quantification** — Portfolio managers need probability distributions and confidence intervals. LLMs produce qualitative narratives.

4. **No Regime Awareness** — Correlations that hold in bull markets break in crises. LLMs trained on mixed-regime data produce blended, averages-of-history responses.

5. **No Counterfactual Simulation** — LLMs can approximate answers linguistically but cannot simulate the causal mechanics of counterfactual interventions.

The tasks where LLMs fall short — forecasting price trajectories, simulating portfolio outcomes, quantifying tail risk, modeling regime transitions — are precisely the tasks that matter most for investment return and risk management.

---

## Chapter 2

### What Is a World Model?

The modern formulation of World Models was introduced in 2018 by David Ha and Jürgen Schmidhuber.

A world model is an internal simulation engine.

Instead of predicting the next word, it predicts:

The next state of the environment.

Formally:

    State(t) → State(t+1)

A state can represent:

- Physical position of objects
- Economic conditions
- Market liquidity
- Investor sentiment
- Volatility regimes

In robotics, companies like NVIDIA use world models to simulate thousands of scenarios before deploying a robot in the real world.

In finance, this approach can be transformative.

#### The V-M-C Architecture

Most World Models are organized around three core components:

**Vision Model (V) — The Encoder:** Compresses high-dimensional financial inputs (price series, macro indicators, options surfaces, earnings data, sentiment) into a compact latent vector `z_t ∈ ℝⁿ`.

**Memory Model (M) — The Dynamics Engine:** Learns how latent states evolve over time (`z_t → z_{t+1}`), capturing temporal dependencies, momentum, shock propagation, and regime transitions. Modern implementations use RSSM, Mamba, or transformer-based memory modules.

**Controller (C) — The Decision Maker:** Policy component trained via reinforcement learning that translates latent state and simulated futures into optimal portfolio actions (allocation adjustments, hedges, sector rotations, stop-loss levels).

#### Future Architectures for Financial World Models

Four emerging paradigms will define the next generation of financial world models:

- **Transformer-Based World Models** (GAIA, UniSim): Handle very long context windows for multi-year macro cycle modeling
- **Hierarchical World Models**: Maintain separate latent representations at intraday, daily, weekly, and macro time scales
- **Diffusion-Based World Models**: Generate richer, fat-tailed distributions over future paths
- **Multi-Agent World Models**: Represent multiple market participants simultaneously for game-theoretic reasoning and reflexivity modeling

---

## Chapter 3

### From Words to Worlds

LLMs are statistical mirrors of language.

World Models are generative simulators of dynamics.

The difference is subtle but profound:

**LLM:** "What usually happens after inflation rises?"

**World Model:** "If inflation rises 2%, rates increase 0.5%, and liquidity tightens, what does the market look like in 6 months?"

This shift moves AI from descriptive intelligence to anticipatory intelligence.

---

## Chapter 4

### The V-M-C Architecture

Most World Models are structured around three core components:

#### Vision Model (V)

Compresses high-dimensional input into a latent representation.

In finance, this could encode:

- Price history
- Macro indicators
- Volatility structure
- Earnings data

#### Memory Model (M)

Learns temporal dynamics.

It models how the latent state evolves over time:

    Latent(t) → Latent(t+1)

This captures:

- Market cycles
- Momentum decay
- Shock propagation
- Regime transitions

#### Controller (C)

Uses the simulated future to choose optimal actions.

In markets, this means:

- Adjust allocation
- Hedge exposure
- Increase dividend weighting
- Rotate sectors

---

## Chapter 5

### Building a Financial World Model

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

    State(t) → Distribution of State(t+1)

This is not a price predictor.

It is a market simulator.

---

## Chapter 6

### Market Dynamics as Physics

Markets behave like complex physical systems:

- Energy = Liquidity
- Friction = Transaction costs
- Momentum = Trend persistence
- Phase transitions = Regime shifts

Simulation allows the system to internally test:

- Crashes
- Recessions
- Geopolitical shocks
- Liquidity squeezes

Before capital is exposed.

---

## Chapter 7

### Regime Shifts and Hidden States

Markets operate in regimes:

- Expansion
- Overheating
- Contraction
- Recovery

A world model can infer hidden regime variables in latent space.

These latent variables often explain more variance than observable indicators.

Detecting regime transitions early is one of the most powerful applications of financial world modeling.

---

## Chapter 8

### Portfolio Simulation Engines

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

## Chapter 9

### Scenario Generation and Counterfactual Futures

World models allow counterfactual reasoning:

- What if oil drops 20%?
- What if rates spike unexpectedly?
- What if earnings compress simultaneously across sectors?

The system can:

- Inject shocks
- Simulate propagation
- Measure systemic impact

This transforms risk management into experimental science.

---

## Chapter 10

### Risk, Ethics, and Market Reflexivity

Simulating markets introduces reflexivity risk.

If enough agents use similar world models:

They may shape the very futures they predict.

This creates:

- Feedback loops
- Flash crashes
- AI-driven volatility clusters

Ethical deployment requires transparency and robustness testing.

---

## Chapter 11

### Toward Financial AGI

Artificial General Intelligence requires:

- Causal reasoning
- Temporal continuity
- Environmental modeling

Financial markets are one of the most complex dynamic systems in existence.

Building accurate financial world models is not just an investment breakthrough.

It is a step toward general intelligence.

---

## Chapter 12

### The Future of Intelligent Capital Allocation

The next generation of investment systems will:

- Simulate before allocating
- Stress-test before deploying
- Optimize across probabilistic futures

The question is no longer:

"What will the market do?"

The question becomes:

"Across thousands of possible worlds, where does capital survive and compound?"

That is the promise of Financial World Models.

---

## Chapter 13

### World Models in Finance: Improving Investment Returns and Decision Making in Stock Markets

The preceding chapters established the theoretical and architectural foundations of Financial World Models. This chapter focuses on the practical application: how World Models can concretely improve investment returns and support decision-making in stock markets.

Every investment decision is, at its core, a bet on future states. Traditional approaches — fundamental analysis, technical analysis, factor models, macroeconomic forecasting — generate point estimates or qualitative scenarios, not probability distributions over the full space of possible futures.

World Models change this. They transform the investment problem into a **simulation problem**.

#### The Decision Loop

The World Model decision loop operates in four stages:

1. **Observe** — Ingest price data, macro signals, earnings data, microstructure, sentiment, and alternative data
2. **Encode State** — Compress to latent vector capturing regime, risk state, and uncertainty
3. **Simulate** — Roll out thousands of forward trajectories with macro shocks and tail events
4. **Decide and Optimize** — Select the allocation that maximizes risk-adjusted return across paths

#### The Simulation Advantage

The simulation advantage manifests in four concrete improvements:

**Better Entry and Exit Timing:** Probability-weighted expected returns identify asymmetrically favorable entry points.

**Regime-Conditional Position Sizing:** Dynamic scaling of equity exposure with regime confidence enables proactive risk management rather than reactive response to realized losses.

**Tail Risk Mitigation:** Explicit representation of tail mass transforms drawdown risk from an afterthought into a first-class constraint.

**Counterfactual Stress Testing:** Interventional simulation enables structured testing of macro scenarios, regulatory stress tests, and client-facing scenario reports.

#### Decision Support

When a portfolio manager proposes a trade, the World Model generates a full quantitative decision brief: expected return distribution, VaR/CVaR, regime context, macro sensitivity, correlation impact, optimal sizing, and exit conditions.

#### Real-World Applications

- **Multi-asset tactical allocation** — dynamic allocation adapting to time-varying correlations across regimes
- **Earnings surprise prediction** — probability distributions over earnings outcomes and price reactions
- **IPO and event-driven investing** — latent-space inference from comparable company dynamics
- **Systemic risk monitoring** — early warning when latent state drifts toward crisis-associated regimes

#### The Path Forward

The firms that will lead the next decade of investment performance are those that move earliest and most deliberately from **descriptive AI** (LLMs telling stories about markets) to **anticipatory AI** (World Models simulating them).

"Across thousands of possible worlds, where does capital survive and compound?"

That is the question World Models are built to answer.

---

## References

### Foundational World Models

1. Ha, D., & Schmidhuber, J. (2018). **World Models**. *arXiv preprint arXiv:1803.10122*. https://arxiv.org/abs/1803.10122

2. LeCun, Y. (2022). **A Path Towards Autonomous Machine Intelligence**. *OpenReview*. https://openreview.net/forum?id=BZ5a1r-kVsf

3. Hafner, D., Lillicrap, T., Ba, J., & Norouzi, M. (2019). **Dream to Control: Learning Behaviors by Latent Imagination**. *arXiv preprint arXiv:1912.01603*. https://arxiv.org/abs/1912.01603

4. Hafner, D., Lillicrap, T., Norouzi, M., & Ba, J. (2020). **Mastering Atari with Discrete World Models (DreamerV2)**. *arXiv preprint arXiv:2010.02193*. https://arxiv.org/abs/2010.02193

5. Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2023). **Mastering Diverse Domains through World Models (DreamerV3)**. *arXiv preprint arXiv:2301.04104*. https://arxiv.org/abs/2301.04104

### Large Language Models and Their Limitations

6. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). **Attention Is All You Need**. *Advances in Neural Information Processing Systems, 30*. https://arxiv.org/abs/1706.03762

7. Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., ... & Amodei, D. (2020). **Language Models are Few-Shot Learners (GPT-3)**. *Advances in Neural Information Processing Systems, 33*. https://arxiv.org/abs/2005.14165

8. Bommasani, R., Hudson, D. A., Raghunathan, A., Altman, R., Arora, S., Koreeda, Y., ... & Liang, P. (2021). **On the Opportunities and Risks of Foundation Models**. *arXiv preprint arXiv:2108.07258*. https://arxiv.org/abs/2108.07258

### Reinforcement Learning and State Prediction

9. Sutton, R. S., & Barto, A. G. (2018). **Reinforcement Learning: An Introduction** (2nd ed.). *MIT Press*. http://incompleteideas.net/book/the-book-2nd.html

10. Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Hassabis, D. (2015). **Human-level control through deep reinforcement learning**. *Nature, 518*(7540), 529–533. https://doi.org/10.1038/nature14236

11. Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., Van Den Driessche, G., ... & Hassabis, D. (2016). **Mastering the game of Go with deep neural networks and tree search**. *Nature, 529*(7587), 484–489. https://doi.org/10.1038/nature16961

### Financial Modeling and AI

12. López de Prado, M. (2018). **Advances in Financial Machine Learning**. *Wiley*.

13. Ritter, G. (2017). **Machine Learning for Trading**. *arXiv preprint arXiv:1708.05765*. https://arxiv.org/abs/1708.05765

14. Dixon, M. F., Halperin, I., & Bilokon, P. (2020). **Machine Learning in Finance: From Theory to Practice**. *Springer*.

15. Cont, R. (2001). **Empirical properties of asset returns: stylized facts and statistical issues**. *Quantitative Finance, 1*(2), 223–236. https://doi.org/10.1080/713665670

### Regime Shifts and Hidden Markov Models

16. Hamilton, J. D. (1989). **A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle**. *Econometrica, 57*(2), 357–384. https://doi.org/10.2307/1912559

17. Ang, A., & Bekaert, G. (2002). **Regime Switches in Interest Rates**. *Journal of Business & Economic Statistics, 20*(2), 163–182. https://doi.org/10.1198/073500102317351949

### Portfolio Simulation and Risk Management

18. Markowitz, H. (1952). **Portfolio Selection**. *The Journal of Finance, 7*(1), 77–91. https://doi.org/10.2307/2975974

19. Glasserman, P. (2003). **Monte Carlo Methods in Financial Engineering**. *Springer*.

20. Black, F., & Scholes, M. (1973). **The Pricing of Options and Corporate Liabilities**. *Journal of Political Economy, 81*(3), 637–654. https://doi.org/10.1086/260062

### Market Reflexivity and Ethics

21. Soros, G. (1987). **The Alchemy of Finance**. *Wiley*.

22. Taleb, N. N. (2007). **The Black Swan: The Impact of the Highly Improbable**. *Random House*.

23. Bookstaber, R. (2017). **The End of Theory: Financial Crises, the Failure of Economics, and the Sweep of Human Interaction**. *Princeton University Press*.

### Artificial General Intelligence

24. Goertzel, B., & Pennachin, C. (Eds.). (2007). **Artificial General Intelligence**. *Springer*.

25. Russell, S. (2019). **Human Compatible: Artificial Intelligence and the Problem of Control**. *Viking*.

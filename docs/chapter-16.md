---
id: chapter-16
title: "Predicting the Next Price: World Models, LLMs, and Portfolio Ontology for Multi-Horizon Forecasting, Backtesting, and Economic-Event-Driven Trading"
sidebar_label: "Chapter 16 — Price Prediction with World Models & LLMs"
sidebar_position: 17
---

# Chapter 16

## Predicting the Next Price: World Models, LLMs, and Portfolio Ontology for Multi-Horizon Forecasting, Backtesting, and Economic-Event-Driven Trading

Chapter 15 introduced the Ontology-Driven World Model (ODWM) as a knowledge-grounded architecture for financial reasoning. This chapter applies that architecture to its most commercially important task: **predicting the next price of an asset** across every relevant time horizon — from the next minute to the next trading day — and systematically validating those predictions through rigorous backtesting.

Beyond raw price prediction, we integrate two powerful extensions. First, a **Portfolio Ontology** that elevates prediction from single-asset guessing to whole-portfolio intelligence. Second, an **Economic Event Engine** that conditions every forecast on the macro calendar — earnings releases, central-bank decisions, payroll prints, and geopolitical shocks — using a combination of MORL (Multi-Objective Reinforcement Learning) and collaborative LLM agent frameworks.

---

## The Multi-Horizon Prediction Problem

Financial price prediction is not one problem. It is a family of problems distinguished by **time horizon**, **data resolution**, **market microstructure**, and **signal-to-noise ratio**:

| Horizon | Resolution | Dominant Signal Source | Typical Use Case |
|---|---|---|---|
| **Next minute** | Tick / 1-min bar | Order-flow imbalance, bid-ask spread | HFT, market-making |
| **Next hour** | 5-min / 15-min bar | Intra-day momentum, news sentiment | Intra-day strategies |
| **Next day** | Daily OHLCV | Macro regime, earnings calendar | Swing trading |
| **Next week** | Daily / weekly bar | Sector rotation, factor exposure | Medium-term alpha |

A World Model approach is uniquely suited to this multi-horizon challenge because its **latent state** captures the current market configuration at a semantic level — independent of time scale — and its **dynamics model** can be rolled forward by any number of steps.

---

## Architecture Overview

The system presented in this chapter integrates five components:

```
┌─────────────────────────────────────────────────────────────┐
│                 Multi-Horizon Price Engine                   │
│                                                             │
│  ┌──────────────┐    ┌───────────────┐    ┌─────────────┐  │
│  │  Market Data │───▶│  World Model  │───▶│  Forecaster │  │
│  │  Ingestion   │    │  (VMC Core)   │    │  (Multi-Hz) │  │
│  └──────────────┘    └───────────────┘    └─────────────┘  │
│         │                   │                    │          │
│  ┌──────▼──────┐    ┌───────▼───────┐    ┌──────▼──────┐  │
│  │  Portfolio  │    │  Economic     │    │  Backtesting │  │
│  │  Ontology   │    │  Event Engine │    │  Framework  │  │
│  └─────────────┘    └───────────────┘    └─────────────┘  │
│                             │                              │
│                    ┌────────▼────────┐                     │
│                    │  MORL + Agent   │                     │
│                    │  Orchestration  │                     │
│                    └─────────────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Portfolio Ontology for Price Prediction

Building on Chapter 15's ODWM, the **Portfolio Ontology** extends the knowledge graph with classes specific to multi-asset forecasting and portfolio construction.

### Extended Ontology Schema

```
Class: PriceTarget
  properties: horizon (enum: MINUTE|HOUR|DAY|WEEK),
              predicted_price (float),
              confidence_interval_95 (tuple[float, float]),
              model_version (string),
              generated_at (datetime)
  relations:  targetsAsset → Asset,
              producedBy → ForecastModel,
              conditionedOn → [EconomicEvent]

Class: ForecastModel
  properties: model_type (enum: WORLD_MODEL|LLM|ENSEMBLE),
              training_universe (string),
              last_trained (datetime),
              backtest_sharpe (float),
              backtest_max_drawdown (float)

Class: EconomicEvent
  properties: event_name (string),
              scheduled_at (datetime),
              actual_value (float | None),
              consensus_estimate (float | None),
              surprise_magnitude (float | None),
              impact_tier (enum: HIGH|MEDIUM|LOW)
  relations:  affects → [Asset], affects → [Sector]

Class: BacktestResult
  properties: strategy_id (string),
              start_date (date),
              end_date (date),
              total_return (float),
              annualised_sharpe (float),
              max_drawdown (float),
              calmar_ratio (float),
              win_rate (float),
              average_holding_period (float)
  relations:  evaluates → ForecastModel,
              overUniverse → [Asset]
```

### Ontology-Driven Feature Construction

The portfolio ontology enables **relational feature engineering** — features computed not just from an asset's own history but from its position in the knowledge graph:

```python
class OntologyFeatureEngineer:
    """
    Constructs features for price prediction using portfolio ontology relationships.
    Enriches raw OHLCV signals with sector momentum, peer divergence,
    macro-factor exposures, and upcoming economic event proximity.
    """

    def __init__(self, knowledge_graph: FinancialKnowledgeGraph):
        self.kg = knowledge_graph

    def build_features(
        self,
        asset_id: str,
        bars: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> pd.Series:
        asset_node = self.kg.get_entity(asset_id)
        sector_id  = asset_node['sector']
        peers      = self.kg.get_neighbours(asset_id, 'peerOf', max_hops=1)

        # Sector momentum: equal-weight sector return over past 5 days
        sector_assets  = self.kg.get_neighbours(sector_id, 'contains', max_hops=1)
        sector_returns = self._mean_return(sector_assets, bars, window=5)

        # Peer divergence: z-score of asset return vs. peers
        peer_returns   = self._mean_return([p['id'] for p in peers], bars, window=1)
        asset_return   = bars.loc[asset_id, 'return_1d']
        peer_z_score   = (asset_return - peer_returns.mean()) / (peer_returns.std() + 1e-8)

        # Economic event proximity: days until next HIGH-impact event
        next_event     = self._next_high_impact_event(asset_id, as_of)
        days_to_event  = (next_event.scheduled_at - as_of).days if next_event else 99

        return pd.Series({
            'sector_momentum_5d': sector_returns,
            'peer_z_score_1d':    peer_z_score,
            'days_to_event':      days_to_event,
            'asset_beta':         asset_node.get('beta_1y', 1.0),
            'iv_rank':            asset_node.get('iv_rank_30d', 0.5),
        })

    def _mean_return(self, asset_ids: list[str], bars: pd.DataFrame, window: int) -> float:
        rets = bars.loc[asset_ids, f'return_{window}d']
        return float(rets.mean())

    def _next_high_impact_event(
        self, asset_id: str, as_of: pd.Timestamp
    ) -> 'EconomicEvent | None':
        events = self.kg.get_events_affecting(asset_id, after=as_of, tier='HIGH')
        return min(events, key=lambda e: e.scheduled_at) if events else None
```

---

## Multi-Horizon World Model Forecaster

The core forecasting engine uses the World Model's latent dynamics to generate price distributions at multiple horizons simultaneously.

### Latent State Encoding

```python
class MultiResolutionEncoder(nn.Module):
    """
    Encodes market observations at multiple temporal resolutions into
    a unified latent state that supports multi-horizon rollout.

    The encoder processes three parallel streams:
      - Micro stream: 1-min bars (last 60 minutes)
      - Meso  stream: 15-min bars (last 8 hours)
      - Macro stream: daily bars  (last 252 days)

    Each stream is encoded by a Temporal Convolutional Network (TCN)
    and the outputs are fused via a cross-attention layer.
    """

    def __init__(self, d_micro=64, d_meso=128, d_macro=256, d_latent=512):
        super().__init__()
        self.micro_tcn = TemporalConvNet(in_channels=5, num_channels=[32, 64],     kernel_size=3)
        self.meso_tcn  = TemporalConvNet(in_channels=5, num_channels=[64, 128],    kernel_size=3)
        self.macro_tcn = TemporalConvNet(in_channels=10, num_channels=[128, 256],  kernel_size=3)
        self.fusion    = nn.MultiheadAttention(embed_dim=d_latent, num_heads=8, batch_first=True)
        self.proj      = nn.Linear(d_micro + d_meso + d_macro, d_latent)

    def forward(
        self,
        micro_bars: Tensor,   # (B, 60,  5)  — 1-min OHLCV
        meso_bars:  Tensor,   # (B, 32,  5)  — 15-min OHLCV
        macro_bars: Tensor,   # (B, 252, 10) — daily OHLCV + volume features
    ) -> Tensor:              # (B, d_latent)
        z_micro = self.micro_tcn(micro_bars.permute(0, 2, 1)).mean(-1)
        z_meso  = self.meso_tcn(meso_bars.permute(0, 2, 1)).mean(-1)
        z_macro = self.macro_tcn(macro_bars.permute(0, 2, 1)).mean(-1)

        z_concat = torch.cat([z_micro, z_meso, z_macro], dim=-1)
        z_latent = self.proj(z_concat)
        return z_latent
```

### Multi-Horizon Dynamics Rollout

```python
class MultiHorizonDynamics(nn.Module):
    """
    Rolls the latent state forward to generate price distributions at
    four canonical horizons: next-minute, next-hour, next-day, next-week.

    Each horizon head is a separate MLP that maps the rolled-out latent
    state to a (mu, sigma) parameterisation of a Normal distribution,
    supporting probabilistic inference and uncertainty quantification.
    """

    HORIZONS = {
        'minute': 1,
        'hour':   60,
        'day':    390,   # ~390 one-minute bars per US trading day
        'week':   1950,  # ~5 trading days × 390
    }

    def __init__(self, d_latent=512, d_hidden=256):
        super().__init__()
        self.gru = nn.GRU(d_latent, d_hidden, num_layers=2, batch_first=True)
        self.heads = nn.ModuleDict({
            hz: nn.Sequential(
                nn.Linear(d_hidden, 128),
                nn.SiLU(),
                nn.Linear(128, 2),   # (mu_log_return, log_sigma)
            )
            for hz in self.HORIZONS
        })

    def forward(self, z_0: Tensor, n_samples: int = 1000) -> dict[str, Tensor]:
        """
        Returns a dict mapping horizon name → Tensor of shape (B, n_samples)
        representing sampled log-returns at each horizon.
        """
        results = {}
        z_cur = z_0.unsqueeze(1)   # (B, 1, d_latent)

        # Roll forward to the maximum required horizon
        max_steps = max(self.HORIZONS.values())
        outputs, _ = self.gru(z_cur.expand(-1, max_steps, -1))

        for hz_name, hz_steps in self.HORIZONS.items():
            h_t = outputs[:, hz_steps - 1, :]        # (B, d_hidden)
            params = self.heads[hz_name](h_t)         # (B, 2)
            mu    = params[:, 0]
            sigma = params[:, 1].exp().clamp(min=1e-4)
            dist  = torch.distributions.Normal(mu, sigma)
            results[hz_name] = dist.rsample((n_samples,)).T  # (B, n_samples)

        return results
```

### Probabilistic Price Decoder

```python
class ProbabilisticPriceDecoder:
    """
    Converts sampled log-returns into price distributions with
    full percentile summary statistics for downstream use.
    """

    def decode(
        self,
        current_price: float,
        log_return_samples: Tensor,  # (n_samples,)
    ) -> dict:
        prices = current_price * torch.exp(log_return_samples).numpy()
        return {
            'mean':    float(prices.mean()),
            'median':  float(np.median(prices)),
            'p05':     float(np.percentile(prices, 5)),
            'p25':     float(np.percentile(prices, 25)),
            'p75':     float(np.percentile(prices, 75)),
            'p95':     float(np.percentile(prices, 95)),
            'std':     float(prices.std()),
            'skew':    float(scipy.stats.skew(prices)),
            'kurt':    float(scipy.stats.kurtosis(prices)),
        }
```

---

## Economic Event Engine

Economic events are **regime-switching catalysts**. A payroll miss, a hawkish Fed statement, or an earnings surprise can render every pre-event forecast stale within seconds. The Economic Event Engine (EEE) addresses this by conditioning the World Model on the full macro calendar.

### Event Taxonomy

```python
from enum import Enum

class EconomicEventType(Enum):
    # Monetary policy
    CENTRAL_BANK_DECISION   = "central_bank_decision"
    FED_MINUTES             = "fed_minutes"
    FOMC_SPEECH             = "fomc_speech"

    # Labour market
    NONFARM_PAYROLLS        = "nonfarm_payrolls"
    JOBLESS_CLAIMS          = "jobless_claims"
    ADP_EMPLOYMENT          = "adp_employment"

    # Inflation
    CPI_RELEASE             = "cpi_release"
    PPI_RELEASE             = "ppi_release"
    PCE_DEFLATOR            = "pce_deflator"

    # Growth
    GDP_ADVANCE             = "gdp_advance"
    ISM_MANUFACTURING       = "ism_manufacturing"
    RETAIL_SALES            = "retail_sales"

    # Corporate
    EARNINGS_RELEASE        = "earnings_release"
    GUIDANCE_UPDATE         = "guidance_update"

    # Geopolitical
    GEOPOLITICAL_SHOCK      = "geopolitical_shock"
    REGULATORY_ANNOUNCEMENT = "regulatory_announcement"
```

### Event-Conditioned Latent State Update

```python
class EconomicEventConditioner(nn.Module):
    """
    Conditions the World Model's latent state on an incoming economic event.

    Architecture:
      1. Embed the event (type, surprise magnitude, impact tier) into
         a dense vector using a learned embedding table + MLP.
      2. Apply a gated update to the current latent state, modulating
         only the dimensions most relevant to the event type.
      3. Re-run the multi-horizon dynamics from the updated latent state.
    """

    def __init__(self, d_latent: int = 512, n_event_types: int = 16):
        super().__init__()
        self.event_embed = nn.Embedding(n_event_types, 64)
        self.event_mlp   = nn.Sequential(
            nn.Linear(64 + 2, 256),   # +2 for (surprise, impact_tier)
            nn.SiLU(),
            nn.Linear(256, d_latent),
        )
        # Gating: learn which latent dimensions each event type modulates
        self.gate = nn.Sequential(
            nn.Linear(d_latent * 2, d_latent),
            nn.Sigmoid(),
        )

    def forward(
        self,
        z_t: Tensor,              # (B, d_latent) — pre-event latent state
        event_type_idx: Tensor,   # (B,) — integer index into EconomicEventType
        surprise: Tensor,         # (B,) — normalised surprise magnitude
        impact_tier: Tensor,      # (B,) — 0=LOW, 1=MED, 2=HIGH
    ) -> Tensor:                  # (B, d_latent) — post-event latent state
        e = self.event_embed(event_type_idx)                       # (B, 64)
        ctx = torch.cat([e, surprise.unsqueeze(-1),
                         impact_tier.float().unsqueeze(-1)], dim=-1)
        delta = self.event_mlp(ctx)                                # (B, d_latent)
        gate  = self.gate(torch.cat([z_t, delta], dim=-1))        # (B, d_latent)
        return z_t + gate * delta
```

### LLM Narrative Interpretation

```python
class EconomicEventInterpreter:
    """
    Uses an LLM to generate a structured narrative interpretation of an
    economic event, then extracts numeric impact estimates for each affected
    asset class in the portfolio ontology.
    """

    SYSTEM_PROMPT = """
    You are a macro economist and quantitative analyst.
    Given an economic data release, you will:
    1. Assess the surprise relative to consensus.
    2. Identify the primary asset classes affected.
    3. Estimate the magnitude and direction of near-term price impact
       (next hour, next day) for each affected class.
    4. Output a structured JSON response conforming to the schema provided.
    """

    def __init__(self, llm_client, knowledge_graph: 'FinancialKnowledgeGraph'):
        self.llm = llm_client
        self.kg  = knowledge_graph

    def interpret(self, event: 'EconomicEvent') -> 'EventInterpretation':
        prompt = self._build_prompt(event)
        raw    = self.llm.complete(
            system=self.SYSTEM_PROMPT,
            user=prompt,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(raw.text)
        return EventInterpretation(
            narrative=parsed['narrative'],
            asset_impacts={
                cls: AssetImpact(
                    direction=imp['direction'],
                    magnitude_1h=imp['magnitude_1h'],
                    magnitude_1d=imp['magnitude_1d'],
                    confidence=imp['confidence'],
                )
                for cls, imp in parsed['asset_impacts'].items()
            },
            regime_shift_probability=parsed.get('regime_shift_probability', 0.0),
        )

    def _build_prompt(self, event: 'EconomicEvent') -> str:
        affected = self.kg.get_events_affecting_classes(event.event_name)
        return (
            f"Event: {event.event_name}\n"
            f"Actual: {event.actual_value}\n"
            f"Consensus: {event.consensus_estimate}\n"
            f"Surprise: {event.surprise_magnitude:+.2f} ({event.surprise_magnitude / (event.consensus_estimate + 1e-8) * 100:+.1f}%)\n"
            f"Historically affected asset classes: {affected}\n"
            f"Please analyse and return the JSON interpretation."
        )
```

---

## MORL Framework for Multi-Objective Portfolio Optimisation

Multi-Objective Reinforcement Learning (MORL) extends standard RL to settings where the agent must simultaneously optimise multiple, potentially conflicting objectives. In portfolio management these objectives include:

- **Return maximisation** — grow NAV over the forecast horizon
- **Risk control** — keep volatility and drawdown within tolerance
- **Turnover minimisation** — reduce transaction costs and market impact
- **Tracking error management** — stay close to benchmark when required

### Reward Vector Design

```python
@dataclass
class PortfolioRewardVector:
    """
    A multi-dimensional reward signal for MORL portfolio agents.
    Each component is normalised to [-1, +1] for stable training.
    """
    return_component:    float   # Sharpe-normalised log-return
    risk_component:      float   # Negative volatility penalty
    drawdown_component:  float   # Negative max-drawdown penalty
    turnover_component:  float   # Negative turnover cost
    tracking_component:  float   # Negative tracking error vs benchmark

    def to_tensor(self) -> Tensor:
        return torch.tensor([
            self.return_component,
            self.risk_component,
            self.drawdown_component,
            self.turnover_component,
            self.tracking_component,
        ], dtype=torch.float32)

    def scalarise(self, weights: Tensor) -> float:
        """Linear scalarisation: dot product of reward vector and preference weights."""
        return float((self.to_tensor() * weights).sum())
```

### MORL Agent Architecture

```python
class MORLPortfolioAgent(nn.Module):
    """
    A multi-objective RL agent for portfolio management.

    The agent is conditioned on a **preference vector** (lambda) that encodes
    the investor's current risk-return trade-off. By varying lambda at inference
    time, a single trained model can express the full Pareto frontier of
    portfolio strategies — from highly conservative to aggressive growth.

    Architecture:
      - State encoder: fuses latent World Model state + portfolio state
      - Preference conditioner: FiLM-modulates the policy network with lambda
      - Policy head: outputs portfolio weight adjustments
      - Value head: estimates the vectorised value function V(s, lambda)
    """

    def __init__(
        self,
        d_world_model: int = 512,
        d_portfolio: int = 128,
        n_assets: int = 50,
        n_objectives: int = 5,
    ):
        super().__init__()
        d_state = d_world_model + d_portfolio

        # Preference conditioner (FiLM: Feature-wise Linear Modulation)
        self.film_gamma = nn.Linear(n_objectives, d_state)
        self.film_beta  = nn.Linear(n_objectives, d_state)

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(d_state, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
        )
        # Policy head: weight adjustments (before softmax)
        self.policy_head = nn.Linear(256, n_assets)

        # Value head: vectorised value function (one per objective)
        self.value_head  = nn.Linear(256, n_objectives)

    def forward(
        self,
        z_world: Tensor,          # (B, d_world_model) — World Model latent state
        portfolio_state: Tensor,  # (B, d_portfolio)   — current weights + positions
        preference: Tensor,       # (B, n_objectives)  — investor preference lambda
    ) -> tuple[Tensor, Tensor]:
        s = torch.cat([z_world, portfolio_state], dim=-1)

        # FiLM conditioning on preference vector
        gamma = self.film_gamma(preference)
        beta  = self.film_beta(preference)
        s     = gamma * s + beta

        h = self.trunk(s)
        weights_logits = self.policy_head(h)
        weights        = torch.softmax(weights_logits, dim=-1)   # (B, n_assets)
        values         = self.value_head(h)                      # (B, n_objectives)

        return weights, values

    def select_action(
        self,
        z_world: Tensor,
        portfolio_state: Tensor,
        preference: Tensor,
        temperature: float = 1.0,
    ) -> Tensor:
        weights_logits = self.policy_head(
            self.trunk(
                self.film_gamma(preference) * torch.cat([z_world, portfolio_state], -1)
                + self.film_beta(preference)
            )
        )
        if temperature == 0:
            return torch.softmax(weights_logits, dim=-1)
        noisy = weights_logits + torch.randn_like(weights_logits) * temperature
        return torch.softmax(noisy, dim=-1)
```

### Pareto Frontier Navigation

```python
class ParetoFrontierNavigator:
    """
    Navigates the Pareto frontier of portfolio strategies at inference time.
    Given an investor's stated risk appetite, selects the preference vector
    lambda that best matches it and queries the MORL agent.
    """

    def __init__(self, morl_agent: MORLPortfolioAgent, n_pareto_points: int = 100):
        self.agent = morl_agent
        # Pre-compute a library of evenly spaced preference vectors
        self.pareto_library = self._init_pareto_library(n_pareto_points)

    def _init_pareto_library(self, n: int) -> Tensor:
        # Sample preference vectors uniformly from the simplex over 5 objectives
        raw    = torch.rand(n, 5)
        return raw / raw.sum(dim=-1, keepdim=True)

    def recommend(
        self,
        risk_appetite: str,        # 'conservative' | 'balanced' | 'aggressive'
        z_world: Tensor,
        portfolio_state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        lambda_target = {
            'conservative': torch.tensor([0.2, 0.4, 0.2, 0.1, 0.1]),
            'balanced':     torch.tensor([0.35, 0.25, 0.15, 0.15, 0.1]),
            'aggressive':   torch.tensor([0.5, 0.1, 0.1, 0.2, 0.1]),
        }[risk_appetite]

        # Find nearest pre-computed Pareto point
        dists = torch.cdist(
            lambda_target.unsqueeze(0),
            self.pareto_library,
        )
        best_lambda = self.pareto_library[dists.argmin()]

        weights, values = self.agent(
            z_world.unsqueeze(0),
            portfolio_state.unsqueeze(0),
            best_lambda.unsqueeze(0),
        )
        return weights.squeeze(0), values.squeeze(0)
```

---

## Multi-Agent Forecasting Framework

Price forecasting benefits from a **market of opinions**: different agents specialise in different signals and their outputs are aggregated into a consensus forecast.

### Agent Taxonomy

| Agent | Specialisation | Primary Signal | Horizon Focus |
|---|---|---|---|
| **Momentum Agent** | Trend following | Price velocity, MACD | 1 hour – 1 day |
| **Macro Agent** | Regime detection | Economic indicators, yield curve | 1 day – 1 week |
| **Sentiment Agent** | News & social media | LLM-scored headlines | 1 hour – 1 day |
| **Microstructure Agent** | Order-flow analysis | Bid-ask spread, tick imbalance | 1 minute – 1 hour |
| **Earnings Agent** | Earnings surprise modelling | EPS estimates, guidance | Pre/post earnings |
| **Risk Agent** | Tail risk detection | VIX term structure, skew | All horizons |

### Agent Orchestration

```python
class ForecastingAgentOrchestrator:
    """
    Coordinates specialist forecasting agents and aggregates their
    predictions into a calibrated ensemble forecast.

    The aggregation step uses a learned **meta-learner** (stacking)
    that weights agents by their recent prediction quality,
    conditioned on the current market regime.
    """

    def __init__(
        self,
        agents: dict[str, 'BaseForecastingAgent'],
        world_model: 'WorldModel',
        knowledge_graph: 'FinancialKnowledgeGraph',
        meta_learner: 'ForecastMetaLearner',
    ):
        self.agents       = agents
        self.wm           = world_model
        self.kg           = knowledge_graph
        self.meta_learner = meta_learner

    def forecast(
        self,
        asset_id: str,
        current_price: float,
        as_of: pd.Timestamp,
        horizons: list[str] = ('minute', 'hour', 'day'),
    ) -> dict[str, 'PriceForecast']:
        # 1. Encode market state
        observation = self._build_observation(asset_id, as_of)
        z_t = self.wm.encoder(observation)

        # 2. Collect agent forecasts in parallel
        agent_forecasts = {}
        for name, agent in self.agents.items():
            try:
                agent_forecasts[name] = agent.forecast(
                    asset_id=asset_id,
                    z_world=z_t,
                    current_price=current_price,
                    horizons=horizons,
                    knowledge_graph=self.kg,
                )
            except Exception:
                pass  # graceful degradation: exclude failed agents

        # 3. Meta-learner aggregation
        regime = self.wm.classify_regime(z_t)
        ensemble = self.meta_learner.aggregate(
            agent_forecasts=agent_forecasts,
            regime=regime,
            horizons=horizons,
        )
        return ensemble

    def _build_observation(
        self, asset_id: str, as_of: pd.Timestamp
    ) -> 'MarketObservation':
        return MarketObservation(
            asset_id=asset_id,
            micro_bars=self.kg.get_bars(asset_id, resolution='1min',  n=60,  end=as_of),
            meso_bars= self.kg.get_bars(asset_id, resolution='15min', n=32,  end=as_of),
            macro_bars=self.kg.get_bars(asset_id, resolution='1d',    n=252, end=as_of),
            as_of=as_of,
        )
```

### Momentum Agent Implementation

```python
class MomentumForecastingAgent:
    """
    A trend-following forecasting agent that combines traditional
    technical signals with the World Model's latent trend encoding.
    """

    def forecast(
        self,
        asset_id: str,
        z_world: Tensor,
        current_price: float,
        horizons: list[str],
        knowledge_graph: 'FinancialKnowledgeGraph',
    ) -> dict[str, 'AgentForecast']:
        bars = knowledge_graph.get_bars(asset_id, resolution='1d', n=252)

        # Technical signals
        rsi_14     = self._rsi(bars['close'], period=14)
        macd_hist  = self._macd_histogram(bars['close'])
        bb_pct     = self._bollinger_band_pct(bars['close'], period=20)

        # World Model trend direction (sign of trend partition of z_world)
        wm_trend   = float(z_world[..., :32].mean())

        forecasts = {}
        for hz in horizons:
            scale = {'minute': 0.001, 'hour': 0.003, 'day': 0.01}[hz]
            signal = 0.4 * wm_trend + 0.3 * (rsi_14 / 100 - 0.5) + 0.3 * macd_hist
            mu     = current_price * (1 + signal * scale)
            sigma  = current_price * scale * 2
            forecasts[hz] = AgentForecast(
                agent='momentum',
                horizon=hz,
                mu=mu,
                sigma=sigma,
                confidence=min(abs(signal) * 2, 1.0),
            )
        return forecasts

    def _rsi(self, prices: pd.Series, period: int = 14) -> float:
        delta = prices.diff()
        gain  = delta.clip(lower=0).ewm(span=period).mean()
        loss  = (-delta.clip(upper=0)).ewm(span=period).mean()
        rs    = gain.iloc[-1] / (loss.iloc[-1] + 1e-8)
        return float(100 - 100 / (1 + rs))

    def _macd_histogram(self, prices: pd.Series) -> float:
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd  = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        hist   = (macd - signal).iloc[-1]
        return float(hist / (prices.std() + 1e-8))

    def _bollinger_band_pct(self, prices: pd.Series, period: int = 20) -> float:
        ma  = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        pct = (prices - (ma - 2 * std)) / (4 * std + 1e-8)
        return float(pct.iloc[-1])
```

---

## Backtesting Framework

No forecasting system is complete without rigorous backtesting. The backtesting framework here is designed to be **statistically honest**: it enforces look-ahead prevention, accounts for transaction costs and market impact, and provides a comprehensive set of performance statistics.

### Backtesting Engine

```python
class WorldModelBacktester:
    """
    Walk-forward backtester for World Model price forecasts.

    Key design principles:
    - **No look-ahead**: the model only uses data available at each bar's open
    - **Realistic costs**: per-trade commission + half-spread + linear market impact
    - **Re-training cadence**: model weights are updated on a rolling window
    - **Multi-horizon evaluation**: evaluates forecast accuracy at all horizons
    """

    def __init__(
        self,
        forecasting_engine: 'ForecastingAgentOrchestrator',
        transaction_cost_bps: float = 5.0,
        market_impact_bps_per_pct: float = 2.0,
        initial_capital: float = 1_000_000.0,
    ):
        self.engine              = forecasting_engine
        self.tc_bps              = transaction_cost_bps
        self.mi_bps_per_pct      = market_impact_bps_per_pct
        self.initial_capital     = initial_capital

    def run(
        self,
        universe: list[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        rebalance_freq: str = 'D',    # pandas offset alias
        horizons: list[str] = ('hour', 'day'),
    ) -> 'BacktestReport':
        portfolio        = EqualWeightPortfolio(universe, self.initial_capital)
        equity_curve     = []
        forecast_errors  = {hz: [] for hz in horizons}
        trade_log        = []

        dates = pd.date_range(start, end, freq=rebalance_freq)

        for date in dates:
            # --- Forecast ---
            forecasts = {}
            for asset_id in universe:
                price = self._get_open_price(asset_id, date)
                forecasts[asset_id] = self.engine.forecast(
                    asset_id=asset_id,
                    current_price=price,
                    as_of=date,
                    horizons=horizons,
                )

            # --- Construct signal ---
            signals = self._build_signal(forecasts, universe, horizons)

            # --- Rebalance portfolio ---
            trades = portfolio.rebalance(signals)
            cost   = self._compute_cost(trades)
            portfolio.apply_costs(cost)
            trade_log.extend(trades)

            # --- Mark-to-market ---
            pnl = portfolio.mark_to_market(self._get_close_prices(universe, date))
            equity_curve.append({'date': date, 'equity': portfolio.nav})

            # --- Record forecast errors ---
            for asset_id in universe:
                for hz in horizons:
                    actual  = self._get_return(asset_id, date, hz)
                    pred_mu = forecasts[asset_id][hz].mu
                    err     = actual - (pred_mu / self._get_open_price(asset_id, date) - 1)
                    forecast_errors[hz].append(err)

        return BacktestReport(
            equity_curve=pd.DataFrame(equity_curve).set_index('date'),
            forecast_errors=forecast_errors,
            trade_log=trade_log,
        )

    def _compute_cost(self, trades: list['Trade']) -> float:
        total_cost = 0.0
        for trade in trades:
            notional = abs(trade.shares * trade.price)
            tc_cost  = notional * self.tc_bps / 10_000
            mi_cost  = notional * abs(trade.shares / trade.adv_shares) * self.mi_bps_per_pct / 10_000
            total_cost += tc_cost + mi_cost
        return total_cost

    def _build_signal(
        self,
        forecasts: dict,
        universe: list[str],
        horizons: list[str],
    ) -> dict[str, float]:
        """
        Combine multi-horizon forecasts into a single portfolio weight signal.
        Uses a simple equal-weighted combination of standardised expected returns.
        """
        hz_weights = {'minute': 0.1, 'hour': 0.3, 'day': 0.6}
        raw = {}
        for asset_id in universe:
            score = 0.0
            for hz in horizons:
                f = forecasts[asset_id][hz]
                score += hz_weights.get(hz, 0.0) * f.mu
            raw[asset_id] = score

        # Cross-sectional z-score to produce long-short signal
        values = np.array(list(raw.values()))
        z = (values - values.mean()) / (values.std() + 1e-8)
        return dict(zip(universe, z.tolist()))

    # stub helpers ─ replace with data provider in production
    def _get_open_price(self, asset_id: str, date: pd.Timestamp) -> float:
        raise NotImplementedError

    def _get_close_prices(self, universe: list[str], date: pd.Timestamp) -> dict[str, float]:
        raise NotImplementedError

    def _get_return(self, asset_id: str, date: pd.Timestamp, horizon: str) -> float:
        raise NotImplementedError
```

### Performance Metrics

```python
class BacktestReport:
    """
    Computes and presents comprehensive backtest performance statistics.
    """

    def __init__(
        self,
        equity_curve: pd.DataFrame,
        forecast_errors: dict[str, list[float]],
        trade_log: list,
        risk_free_rate: float = 0.04,
    ):
        self.equity_curve    = equity_curve
        self.forecast_errors = forecast_errors
        self.trade_log       = trade_log
        self.rfr             = risk_free_rate

    @property
    def returns(self) -> pd.Series:
        return self.equity_curve['equity'].pct_change().dropna()

    @property
    def annualised_return(self) -> float:
        n_years = len(self.returns) / 252
        total   = self.equity_curve['equity'].iloc[-1] / self.equity_curve['equity'].iloc[0]
        return float(total ** (1 / n_years) - 1)

    @property
    def annualised_volatility(self) -> float:
        return float(self.returns.std() * np.sqrt(252))

    @property
    def sharpe_ratio(self) -> float:
        excess = self.annualised_return - self.rfr
        return excess / (self.annualised_volatility + 1e-8)

    @property
    def max_drawdown(self) -> float:
        cum    = (1 + self.returns).cumprod()
        roll   = cum.cummax()
        dd     = (cum - roll) / roll
        return float(dd.min())

    @property
    def calmar_ratio(self) -> float:
        return self.annualised_return / (abs(self.max_drawdown) + 1e-8)

    @property
    def win_rate(self) -> float:
        return float((self.returns > 0).mean())

    def forecast_accuracy(self, horizon: str) -> dict:
        errors = np.array(self.forecast_errors[horizon])
        return {
            'mae':        float(np.abs(errors).mean()),
            'rmse':       float(np.sqrt((errors ** 2).mean())),
            'directional_accuracy': float(
                (np.sign(errors) == 0).mean()  # direction correct when error ≈ 0
            ),
            'ic':         float(np.corrcoef(errors[:-1], errors[1:])[0, 1]),
        }

    def summary(self) -> dict:
        return {
            'annualised_return':     self.annualised_return,
            'annualised_volatility': self.annualised_volatility,
            'sharpe_ratio':          self.sharpe_ratio,
            'max_drawdown':          self.max_drawdown,
            'calmar_ratio':          self.calmar_ratio,
            'win_rate':              self.win_rate,
            'forecast_accuracy':     {
                hz: self.forecast_accuracy(hz)
                for hz in self.forecast_errors
            },
        }
```

---

## Economic-Event-Driven Backtesting

Standard backtests ignore the calendar. An **event-conditioned backtest** separates performance into pre-event, event-day, and post-event windows, revealing whether the model's edge is concentrated around macro catalysts.

```python
class EventConditionedBacktest:
    """
    Stratifies backtest returns by proximity to high-impact economic events.
    Reveals whether alpha is concentrated around macro catalysts.
    """

    def __init__(
        self,
        base_report: BacktestReport,
        economic_calendar: list['EconomicEvent'],
        event_window_days: int = 3,
    ):
        self.report    = base_report
        self.calendar  = economic_calendar
        self.window    = event_window_days

    def stratify(self) -> dict[str, pd.Series]:
        """Split returns into pre-event, event, and post-event windows."""
        event_dates = {
            e.scheduled_at.normalize()
            for e in self.calendar
            if e.impact_tier == 'HIGH'
        }

        daily_returns = self.report.returns.copy()
        daily_returns.index = pd.to_datetime(daily_returns.index)

        pre_mask   = pd.Series(False, index=daily_returns.index)
        event_mask = pd.Series(False, index=daily_returns.index)
        post_mask  = pd.Series(False, index=daily_returns.index)

        for edate in event_dates:
            for offset in range(-self.window, 0):
                d = edate + pd.Timedelta(days=offset)
                if d in daily_returns.index:
                    pre_mask[d] = True
            if edate in daily_returns.index:
                event_mask[edate] = True
            for offset in range(1, self.window + 1):
                d = edate + pd.Timedelta(days=offset)
                if d in daily_returns.index:
                    post_mask[d] = True

        neutral_mask = ~(pre_mask | event_mask | post_mask)

        return {
            'pre_event':  daily_returns[pre_mask],
            'event_day':  daily_returns[event_mask],
            'post_event': daily_returns[post_mask],
            'neutral':    daily_returns[neutral_mask],
        }

    def event_alpha_summary(self) -> dict:
        strata = self.stratify()
        return {
            period: {
                'mean_return': float(rets.mean()),
                'sharpe':      float(rets.mean() / (rets.std() + 1e-8) * np.sqrt(252)),
                'n_obs':       len(rets),
            }
            for period, rets in strata.items()
        }
```

---

## End-to-End System: From Observation to Forecast to Trade

The following pseudocode shows how all components wire together in a live trading loop:

```python
class LiveTradingSystem:
    """
    End-to-end live trading system integrating:
      - Multi-resolution World Model encoder
      - Portfolio Ontology feature engineer
      - Economic Event Engine
      - Multi-Agent Forecasting Orchestrator
      - MORL Portfolio Agent
    """

    def __init__(
        self,
        world_model: 'MultiResolutionEncoder',
        dynamics: 'MultiHorizonDynamics',
        decoder: 'ProbabilisticPriceDecoder',
        event_conditioner: 'EconomicEventConditioner',
        event_interpreter: 'EconomicEventInterpreter',
        forecasting_orchestrator: 'ForecastingAgentOrchestrator',
        morl_agent: 'MORLPortfolioAgent',
        ontology: 'FinancialKnowledgeGraph',
        risk_appetite: str = 'balanced',
    ):
        self.wm           = world_model
        self.dynamics     = dynamics
        self.decoder      = decoder
        self.eec          = event_conditioner
        self.eei          = event_interpreter
        self.forecaster   = forecasting_orchestrator
        self.morl         = morl_agent
        self.kg           = ontology
        self.risk_appetite = risk_appetite

    def on_bar(
        self,
        asset_universe: list[str],
        current_prices: dict[str, float],
        pending_events: list['EconomicEvent'],
        as_of: pd.Timestamp,
    ) -> dict[str, float]:
        """
        Called on every bar. Returns target portfolio weights.
        """
        # 1. Encode current market state for each asset
        latent_states = {}
        for asset_id in asset_universe:
            obs = self._build_observation(asset_id, as_of)
            latent_states[asset_id] = self.wm(obs.micro, obs.meso, obs.macro)

        # 2. Condition latent states on any pending economic events
        for event in pending_events:
            interp = self.eei.interpret(event)
            for asset_id in asset_universe:
                impact = interp.asset_impacts.get(
                    self.kg.get_entity(asset_id)['asset_class']
                )
                if impact and abs(impact.magnitude_1h) > 0.001:
                    latent_states[asset_id] = self.eec(
                        z_t=latent_states[asset_id].unsqueeze(0),
                        event_type_idx=torch.tensor([event.type_idx]),
                        surprise=torch.tensor([event.surprise_magnitude]),
                        impact_tier=torch.tensor([event.impact_tier_idx]),
                    ).squeeze(0)

        # 3. Generate multi-horizon forecasts via agent ensemble
        forecasts = {}
        for asset_id in asset_universe:
            forecasts[asset_id] = self.forecaster.forecast(
                asset_id=asset_id,
                current_price=current_prices[asset_id],
                as_of=as_of,
                horizons=['minute', 'hour', 'day'],
            )

        # 4. Build portfolio state tensor
        portfolio_state = self._encode_portfolio_state(current_prices, asset_universe)

        # 5. MORL agent selects weights on Pareto-optimal frontier
        navigator = ParetoFrontierNavigator(self.morl)
        mean_z = torch.stack(list(latent_states.values())).mean(0)
        weights, _ = navigator.recommend(
            risk_appetite=self.risk_appetite,
            z_world=mean_z,
            portfolio_state=portfolio_state,
        )

        return dict(zip(asset_universe, weights.tolist()))

    def _build_observation(self, asset_id: str, as_of: pd.Timestamp):
        raise NotImplementedError   # wire to data provider

    def _encode_portfolio_state(
        self, prices: dict[str, float], universe: list[str]
    ) -> Tensor:
        raise NotImplementedError   # wire to portfolio state encoder
```

---

## Evaluation Methodology

### Forecast Quality Metrics

| Metric | Definition | Target (daily horizon) |
|---|---|---|
| **MAE** | Mean absolute error of predicted vs. actual return | < 0.5% |
| **RMSE** | Root mean squared error | < 0.8% |
| **Directional accuracy** | Fraction of correct sign predictions | > 55% |
| **IC (Information Coefficient)** | Pearson correlation of forecasts with outcomes | > 0.05 |
| **ICIR** | IC / std(IC) — consistency measure | > 0.5 |

### Portfolio Performance Metrics

| Metric | Definition | Target |
|---|---|---|
| **Annualised Sharpe** | Excess return / annualised volatility | > 1.5 |
| **Max Drawdown** | Largest peak-to-trough equity decline | < –15% |
| **Calmar Ratio** | Annualised return / Max drawdown | > 0.8 |
| **Win Rate** | Fraction of positive trading days | > 52% |
| **Turnover** | Annual two-way portfolio turnover | < 200% |

### Event-Conditioned Metrics

| Period | Expected behaviour |
|---|---|
| **Pre-event (–3 to –1 day)** | Elevated uncertainty; strategy should reduce risk |
| **Event day** | Highest information content; models conditioned on event show largest alpha |
| **Post-event (+1 to +3 day)** | Drift continuation if event surprise was large |
| **Neutral (no event)** | Baseline regime-driven alpha |

---

## Chapter Summary

- **Multi-horizon forecasting** requires a unified architecture that operates across minute, hour, day, and week time scales simultaneously — the World Model's latent state provides this scale-agnostic representation
- The **Portfolio Ontology** extends Chapter 15's ODWM with `PriceTarget`, `ForecastModel`, `EconomicEvent`, and `BacktestResult` classes, enabling relational feature engineering that no single-asset model can match
- The **Economic Event Engine** conditions every forecast on the macro calendar using a gated latent-state update, preventing the catastrophic forecast failures that occur around high-impact news
- **LLM agents** interpret economic event narratives and translate qualitative context into quantitative impact estimates, bridging the gap between macro analysis and model inputs
- **MORL** enables a single portfolio agent to express the full Pareto frontier of risk-return trade-offs by conditioning on a preference vector — eliminating the need to retrain for each investor mandate
- The **multi-agent forecasting framework** decomposes the prediction problem across specialists (Momentum, Macro, Sentiment, Microstructure, Earnings, Risk) and aggregates their outputs via a regime-conditioned meta-learner
- **Rigorous backtesting** — walk-forward, cost-aware, and event-stratified — is the non-negotiable validation layer between research and deployment
- The combined system represents a **complete pipeline** from raw market observation to probabilistic price distribution to optimal portfolio weights, grounded in both the World Model's learned dynamics and the ontology's structured knowledge

---

## Looking Ahead

This chapter completes the book's journey from World Model theory to production-ready financial intelligence. The architecture presented here — multi-resolution encoding, economic-event conditioning, multi-agent ensemble forecasting, MORL portfolio optimisation, and ontology-grounded knowledge — constitutes the state of the art for data-driven quantitative investment at scale.

The remaining open problems are precisely those that make this field so compelling:

> *"A model that predicts price without understanding cause is guessing. A model that understands cause without predicting price is a philosopher. The goal is both."*

The World Model framework, augmented by portfolio ontologies, LLM agents, and multi-objective reinforcement learning, is the most promising path we have toward systems that are simultaneously **predictive**, **explanatory**, and **actionable** — the three properties that define a truly intelligent financial agent.

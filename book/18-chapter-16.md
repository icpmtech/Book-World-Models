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

The system integrates five components: a Market Data Ingestion layer, a World Model (VMC Core), a Multi-Horizon Forecaster, a Portfolio Ontology, an Economic Event Engine, a Backtesting Framework, and a MORL + Agent Orchestration layer. All components share the Portfolio Ontology as a common knowledge layer.

---

## Portfolio Ontology for Price Prediction

The **Portfolio Ontology** extends the ODWM knowledge graph with classes specific to multi-asset forecasting and portfolio construction.

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

```python
class OntologyFeatureEngineer:
    def __init__(self, knowledge_graph):
        self.kg = knowledge_graph

    def build_features(self, asset_id, bars, as_of):
        asset_node = self.kg.get_entity(asset_id)
        sector_id  = asset_node['sector']
        peers      = self.kg.get_neighbours(asset_id, 'peerOf', max_hops=1)
        sector_assets  = self.kg.get_neighbours(sector_id, 'contains', max_hops=1)
        sector_returns = self._mean_return(sector_assets, bars, window=5)
        peer_returns   = self._mean_return([p['id'] for p in peers], bars, window=1)
        asset_return   = bars.loc[asset_id, 'return_1d']
        peer_z_score   = (asset_return - peer_returns.mean()) / (peer_returns.std() + 1e-8)
        next_event     = self._next_high_impact_event(asset_id, as_of)
        days_to_event  = (next_event.scheduled_at - as_of).days if next_event else 99
        return pd.Series({
            'sector_momentum_5d': sector_returns,
            'peer_z_score_1d':    peer_z_score,
            'days_to_event':      days_to_event,
            'asset_beta':         asset_node.get('beta_1y', 1.0),
            'iv_rank':            asset_node.get('iv_rank_30d', 0.5),
        })
```

---

## Multi-Horizon World Model Forecaster

### Latent State Encoding

```python
class MultiResolutionEncoder(nn.Module):
    """
    Encodes market observations at multiple temporal resolutions into
    a unified latent state supporting multi-horizon rollout.
    Three parallel streams: 1-min (micro), 15-min (meso), daily (macro).
    Fused via cross-attention.
    """
    def __init__(self, d_micro=64, d_meso=128, d_macro=256, d_latent=512):
        super().__init__()
        self.micro_tcn = TemporalConvNet(in_channels=5,  num_channels=[32, 64],    kernel_size=3)
        self.meso_tcn  = TemporalConvNet(in_channels=5,  num_channels=[64, 128],   kernel_size=3)
        self.macro_tcn = TemporalConvNet(in_channels=10, num_channels=[128, 256],  kernel_size=3)
        self.fusion    = nn.MultiheadAttention(embed_dim=d_latent, num_heads=8, batch_first=True)
        self.proj      = nn.Linear(d_micro + d_meso + d_macro, d_latent)

    def forward(self, micro_bars, meso_bars, macro_bars):
        z_micro = self.micro_tcn(micro_bars.permute(0, 2, 1)).mean(-1)
        z_meso  = self.meso_tcn(meso_bars.permute(0, 2, 1)).mean(-1)
        z_macro = self.macro_tcn(macro_bars.permute(0, 2, 1)).mean(-1)
        return self.proj(torch.cat([z_micro, z_meso, z_macro], dim=-1))
```

### Multi-Horizon Dynamics Rollout

```python
class MultiHorizonDynamics(nn.Module):
    HORIZONS = {'minute': 1, 'hour': 60, 'day': 390, 'week': 1950}

    def __init__(self, d_latent=512, d_hidden=256):
        super().__init__()
        self.gru   = nn.GRU(d_latent, d_hidden, num_layers=2, batch_first=True)
        self.heads = nn.ModuleDict({
            hz: nn.Sequential(nn.Linear(d_hidden, 128), nn.SiLU(), nn.Linear(128, 2))
            for hz in self.HORIZONS
        })

    def forward(self, z_0, n_samples=1000):
        max_steps = max(self.HORIZONS.values())
        outputs, _ = self.gru(z_0.unsqueeze(1).expand(-1, max_steps, -1))
        results = {}
        for hz_name, hz_steps in self.HORIZONS.items():
            h_t   = outputs[:, hz_steps - 1, :]
            params = self.heads[hz_name](h_t)
            mu, sigma = params[:, 0], params[:, 1].exp().clamp(min=1e-4)
            dist  = torch.distributions.Normal(mu, sigma)
            results[hz_name] = dist.rsample((n_samples,)).T
        return results
```

---

## Economic Event Engine

### Event Taxonomy

```python
class EconomicEventType(Enum):
    CENTRAL_BANK_DECISION   = "central_bank_decision"
    NONFARM_PAYROLLS        = "nonfarm_payrolls"
    CPI_RELEASE             = "cpi_release"
    GDP_ADVANCE             = "gdp_advance"
    EARNINGS_RELEASE        = "earnings_release"
    GEOPOLITICAL_SHOCK      = "geopolitical_shock"
    # … (full list in docs/chapter-16.md)
```

### Event-Conditioned Latent State Update

```python
class EconomicEventConditioner(nn.Module):
    def __init__(self, d_latent=512, n_event_types=16):
        super().__init__()
        self.event_embed = nn.Embedding(n_event_types, 64)
        self.event_mlp   = nn.Sequential(
            nn.Linear(64 + 2, 256), nn.SiLU(), nn.Linear(256, d_latent)
        )
        self.gate = nn.Sequential(nn.Linear(d_latent * 2, d_latent), nn.Sigmoid())

    def forward(self, z_t, event_type_idx, surprise, impact_tier):
        e     = self.event_embed(event_type_idx)
        ctx   = torch.cat([e, surprise.unsqueeze(-1), impact_tier.float().unsqueeze(-1)], dim=-1)
        delta = self.event_mlp(ctx)
        gate  = self.gate(torch.cat([z_t, delta], dim=-1))
        return z_t + gate * delta
```

---

## MORL Framework for Multi-Objective Portfolio Optimisation

MORL extends standard RL to settings where the agent must simultaneously optimise multiple, potentially conflicting objectives: return maximisation, risk control, turnover minimisation, and tracking error management.

### MORL Agent Architecture

```python
class MORLPortfolioAgent(nn.Module):
    """
    Preference-conditioned portfolio agent. A single trained model
    navigates the full Pareto frontier by varying the preference vector lambda.
    Uses FiLM (Feature-wise Linear Modulation) for preference conditioning.
    """
    def __init__(self, d_world_model=512, d_portfolio=128, n_assets=50, n_objectives=5):
        super().__init__()
        d_state = d_world_model + d_portfolio
        self.film_gamma  = nn.Linear(n_objectives, d_state)
        self.film_beta   = nn.Linear(n_objectives, d_state)
        self.trunk        = nn.Sequential(
            nn.Linear(d_state, 512), nn.LayerNorm(512), nn.SiLU(),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.SiLU(),
        )
        self.policy_head = nn.Linear(256, n_assets)
        self.value_head  = nn.Linear(256, n_objectives)

    def forward(self, z_world, portfolio_state, preference):
        s = torch.cat([z_world, portfolio_state], dim=-1)
        s = self.film_gamma(preference) * s + self.film_beta(preference)
        h = self.trunk(s)
        return torch.softmax(self.policy_head(h), dim=-1), self.value_head(h)
```

---

## Multi-Agent Forecasting Framework

| Agent | Specialisation | Horizon Focus |
|---|---|---|
| **Momentum Agent** | Trend following | 1 hour – 1 day |
| **Macro Agent** | Regime detection | 1 day – 1 week |
| **Sentiment Agent** | News & social media | 1 hour – 1 day |
| **Microstructure Agent** | Order-flow analysis | 1 minute – 1 hour |
| **Earnings Agent** | Earnings surprise | Pre/post earnings |
| **Risk Agent** | Tail risk detection | All horizons |

Agents are coordinated by a `ForecastingAgentOrchestrator` that collects their individual forecasts and aggregates them using a regime-conditioned meta-learner.

---

## Backtesting Framework

The backtesting engine enforces walk-forward look-ahead prevention, accounts for per-trade commissions and linear market impact, and produces a `BacktestReport` with the following statistics:

| Metric | Definition |
|---|---|
| **Annualised Sharpe** | Excess return / annualised volatility |
| **Max Drawdown** | Largest peak-to-trough equity decline |
| **Calmar Ratio** | Annualised return / Max drawdown |
| **Win Rate** | Fraction of positive trading days |
| **IC** | Pearson correlation of forecasts with outcomes |
| **ICIR** | IC / std(IC) — forecast consistency |

An **Event-Conditioned Backtest** further stratifies returns by proximity to high-impact economic events (pre-event, event-day, post-event, neutral), revealing whether the strategy's alpha is concentrated around macro catalysts.

---

## Chapter Summary

- **Multi-horizon forecasting** requires a unified latent-state representation that operates across minute, hour, day, and week time scales simultaneously
- The **Portfolio Ontology** adds `PriceTarget`, `ForecastModel`, `EconomicEvent`, and `BacktestResult` classes, enabling relational feature engineering and ontology-driven signal construction
- The **Economic Event Engine** conditions every forecast on the macro calendar using a gated latent-state update, preventing forecast failures around high-impact news
- **LLM agents** interpret economic event narratives and produce quantitative impact estimates, bridging macro analysis and model inputs
- **MORL** enables a single portfolio agent to navigate the full Pareto frontier of risk-return trade-offs by conditioning on a preference vector
- A **multi-agent forecasting framework** decomposes prediction across specialist agents aggregated by a regime-conditioned meta-learner
- **Rigorous event-conditioned backtesting** is the non-negotiable validation layer between research and deployment

---

## Looking Ahead

This chapter completes the book's journey from World Model theory to production-ready financial intelligence. The remaining open problems — causal discovery, non-stationarity adaptation, and safe RL under market impact — are precisely those that make this field so compelling.

> *"A model that predicts price without understanding cause is guessing. A model that understands cause without predicting price is a philosopher. The goal is both."*

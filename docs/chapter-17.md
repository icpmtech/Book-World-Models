---
id: chapter-17
title: Backtesting, Paper Trading, and Production Deployment of Trading World Models
sidebar_label: "Chapter 17 — Backtesting, Paper Trading & Deployment"
sidebar_position: 18
---

# Chapter 17

## Backtesting, Paper Trading, and Production Deployment of Trading World Models

Chapters 15 and 16 established the theoretical and architectural foundations for trading World Models — from ontology-driven systems to tick-level microstructure simulators for HFT and institutional execution desks. This chapter addresses the critical transition from *model* to *deployed strategy*: how do practitioners validate, stress-test, paper-trade, and finally run a trading World Model with real capital?

The gap between a model that performs well in simulation and one that survives contact with live markets is where most quantitative strategies fail. World Models introduce new validation challenges absent from conventional backtesting: the model is both a predictor *and* a simulator, so its errors can compound in ways that simple regression models cannot.

---

## The Validation Hierarchy

Before any trading World Model risks real capital, it must pass through a structured **validation hierarchy** — a sequence of increasingly realistic evaluation environments that surface different failure modes:

```
Level 0 — Unit tests on model components (encoder, dynamics, controller)
    ↓
Level 1 — Historical simulation on in-sample data (fit quality, latent diagnostics)
    ↓
Level 2 — Walk-forward backtest on held-out data (out-of-sample performance)
    ↓
Level 3 — Synthetic market simulation (stress scenarios, regime injection)
    ↓
Level 4 — Paper trading in live market feed (no capital, live microstructure)
    ↓
Level 5 — Shadow mode (model generates signals alongside live strategy, no execution)
    ↓
Level 6 — Limited live deployment (small position sizing, real P&L)
    ↓
Level 7 — Full production deployment
```

Each level is a filter. A model that fails at Level 2 is not promoted to Level 3. This prevents capital from reaching strategies that only appear to work due to in-sample overfitting.

---

## Level 1 — Historical Simulation and Latent Diagnostics

### Reconstruction Quality

The first validation asks: does the World Model accurately reconstruct the historical market states it was trained on?

```python
class WorldModelDiagnostics:
    """
    Diagnostic suite for evaluating World Model fit quality.
    Run after training before any out-of-sample evaluation.
    """
    def reconstruction_quality(
        self,
        model: WorldModel,
        data: MarketDataset,
    ) -> ReconstructionReport:
        z_encoded = model.encoder(data.observations)
        obs_hat   = model.decoder(z_encoded)

        return ReconstructionReport(
            mse          = F.mse_loss(obs_hat, data.observations).item(),
            r2_per_feat  = r2_score(data.observations, obs_hat, multioutput='raw_values'),
            latent_rank  = torch.linalg.matrix_rank(z_encoded.T @ z_encoded).item(),
            kl_divergence= self._latent_kl(z_encoded),
        )

    def dynamics_quality(
        self,
        model: WorldModel,
        data: MarketDataset,
        horizon: int = 21,
    ) -> DynamicsReport:
        """
        Evaluates how well the dynamics model predicts future latent states.
        Key metric: does prediction error grow sub-linearly with horizon?
        """
        errors = []
        for t in range(len(data) - horizon):
            z_t    = model.encoder(data.observations[t])
            z_true = model.encoder(data.observations[t + horizon])
            z_pred = model.dynamics.rollout(z_t, horizon=horizon)[-1]
            errors.append(F.mse_loss(z_pred, z_true).item())
        return DynamicsReport(
            mean_prediction_error   = np.mean(errors),
            horizon_error_profile   = self._horizon_error_curve(model, data, horizon),
            regime_conditional_rmse = self._regime_rmse(model, data),
        )

    def _latent_kl(self, z: Tensor) -> float:
        mu  = z.mean(0)
        std = z.std(0)
        return (0.5 * (mu**2 + std**2 - torch.log(std**2) - 1)).sum().item()
```

### Latent Space Diagnostics

```python
class LatentSpaceDiagnostics:
    """
    Analyses the structure of the learned latent space.
    A healthy latent space should be:
      - Smooth (nearby observations map to nearby latent vectors)
      - Disentangled (different factors correspond to different dimensions)
      - Interpretable (latent dimensions correlate with known market variables)
    """
    def regime_separation(
        self,
        z: Tensor,
        regime_labels: Tensor,
    ) -> float:
        """
        Measures how well regimes are separated in latent space.
        Uses silhouette score — higher is better (range: -1 to +1).
        """
        from sklearn.metrics import silhouette_score
        return silhouette_score(z.numpy(), regime_labels.numpy())

    def correlation_with_observables(
        self,
        z: Tensor,
        observables: dict[str, Tensor],
    ) -> dict[str, float]:
        """
        Computes Pearson correlation between each latent dimension
        and known market variables (e.g., VIX, yield spread, momentum).
        """
        result = {}
        for name, obs in observables.items():
            corrs = [
                float(torch.corrcoef(torch.stack([z[:, d], obs]))[0, 1])
                for d in range(z.shape[1])
            ]
            result[name] = max(corrs, key=abs)
        return result
```

---

## Level 2 — Walk-Forward Backtesting

Walk-forward backtesting is the gold standard for out-of-sample evaluation. It prevents the model from seeing future data during training at any step:

```python
class WalkForwardBacktester:
    """
    Implements a strict walk-forward backtest for trading World Models.

    Design principles:
      1. No lookahead bias — model only trained on data before each test window
      2. Realistic transaction costs — commission, bid-ask spread, market impact
      3. Retraining cadence — model updated at fixed intervals (e.g., monthly)
      4. Slippage simulation — execution price != mid-price
    """
    def __init__(
        self,
        train_window_days:  int = 252,
        test_window_days:   int = 21,
        retrain_every_days: int = 21,
        transaction_cost_bps: float = 5.0,
    ):
        self.train_window      = train_window_days
        self.test_window       = test_window_days
        self.retrain_every     = retrain_every_days
        self.transaction_cost  = transaction_cost_bps / 10_000

    def run(
        self,
        model_factory: Callable[[], WorldModel],
        data: MarketDataset,
        strategy: TradingStrategy,
    ) -> BacktestResult:
        results     = []
        train_end   = self.train_window

        while train_end + self.test_window <= len(data):
            # Strict temporal split
            train_data = data[:train_end]
            test_data  = data[train_end : train_end + self.test_window]

            # Train fresh model on available history
            model = model_factory()
            model.fit(train_data)

            # Evaluate on unseen test window
            pnl = self._simulate_test_window(model, strategy, test_data)
            results.append(pnl)

            train_end += self.retrain_every

        return BacktestResult(
            daily_returns         = np.concatenate(results),
            sharpe                = self._sharpe(np.concatenate(results)),
            max_drawdown          = self._max_drawdown(np.concatenate(results)),
            calmar                = self._calmar(np.concatenate(results)),
            hit_rate              = (np.concatenate(results) > 0).mean(),
        )

    def _simulate_test_window(
        self,
        model: WorldModel,
        strategy: TradingStrategy,
        data: MarketDataset,
    ) -> np.ndarray:
        daily_pnl = []
        for t in range(len(data)):
            obs    = data.observations[t]
            z_t    = model.encoder(obs)
            action = strategy.decide(z_t, model)

            # Simulate execution with realistic costs
            fill_price  = obs.mid_price * (1 + action.direction * self.transaction_cost)
            position_pnl = action.size * (data.next_mid_price(t) - fill_price)
            daily_pnl.append(position_pnl)
        return np.array(daily_pnl)
```

### Backtest Integrity Checks

Common pitfalls in backtesting World Models and how to detect them:

```python
class BacktestIntegrityChecker:
    """
    Runs a battery of integrity checks on backtest results to detect
    common sources of false alpha:
      - Lookahead bias
      - Survivorship bias
      - Data snooping (overfitting to the test set)
      - Excessive turnover (transaction costs underestimated)
      - Regime concentration (all alpha from one market regime)
    """
    def check_all(self, result: BacktestResult, data: MarketDataset) -> IntegrityReport:
        issues = []

        # 1. Turnover check: annualised turnover > 500% warrants scrutiny
        if result.annualised_turnover > 5.0:
            issues.append(IntegrityIssue(
                severity='warning',
                description=f'High turnover ({result.annualised_turnover:.0%} p.a.): '
                             'ensure transaction costs are realistic.',
            ))

        # 2. Regime concentration: > 80% of P&L from one regime
        regime_attribution = self._pnl_by_regime(result, data)
        dominant_share = max(regime_attribution.values()) / sum(regime_attribution.values())
        if dominant_share > 0.8:
            issues.append(IntegrityIssue(
                severity='warning',
                description=f'P&L concentrated in one regime ({dominant_share:.0%}): '
                             'strategy may not generalise across regimes.',
            ))

        # 3. Deflated Sharpe Ratio (Lopez de Prado, 2018)
        dsr = self._deflated_sharpe(result)
        if dsr < 0.95:
            issues.append(IntegrityIssue(
                severity='critical',
                description=f'Deflated Sharpe Ratio = {dsr:.2f}: '
                             'likely overfitted — insufficient out-of-sample evidence.',
            ))

        return IntegrityReport(issues=issues, passed=len(
            [i for i in issues if i.severity == 'critical']
        ) == 0)

    def _deflated_sharpe(self, result: BacktestResult) -> float:
        """
        Deflated Sharpe Ratio (DSR) adjusts for the number of trials tested,
        skewness, and kurtosis of returns.
        Reference: López de Prado & Bailey (2014).
        """
        sr       = result.sharpe
        n        = len(result.daily_returns)
        skew     = float(pd.Series(result.daily_returns).skew())
        kurt     = float(pd.Series(result.daily_returns).kurtosis())
        sr_star  = sr * np.sqrt(n) / np.sqrt(1 - skew * sr + ((kurt - 1) / 4) * sr**2)
        return float(norm.cdf(sr_star))
```

---

## Level 3 — Synthetic Market Simulation

Historical backtesting can only evaluate strategies on scenarios that actually occurred. A World Model's greatest strength is generating *novel* scenarios for stress testing:

```python
class SyntheticMarketSimulator:
    """
    Uses the trained World Model to generate synthetic market scenarios
    for stress testing and regime-specific evaluation.

    Key capability: inject scenarios that have never occurred historically
    but are plausible given the model's learned dynamics — e.g.,
    a 2008-style credit crisis overlaid on today's market structure.
    """
    def __init__(self, world_model: WorldModel):
        self.wm = world_model

    def stress_test(
        self,
        strategy: TradingStrategy,
        scenarios: list[StressScenario],
        n_paths_per_scenario: int = 1_000,
    ) -> StressTestReport:
        results = {}
        for scenario in scenarios:
            # Encode scenario starting conditions
            z_0 = self.wm.encoder(scenario.initial_market_state)

            # Inject scenario shock into latent space
            z_shocked = z_0 + scenario.latent_shock_vector

            # Simulate paths forward from shocked state
            paths = []
            for _ in range(n_paths_per_scenario):
                path = self.wm.dynamics.rollout(
                    z_0=z_shocked,
                    horizon=scenario.horizon_days,
                    stochastic=True,
                )
                price_path = self.wm.decoder(path)
                strategy_pnl = strategy.simulate(price_path)
                paths.append(strategy_pnl)

            arr = np.array(paths)
            results[scenario.name] = ScenarioResult(
                mean_return       = arr.mean(),
                var_99            = np.percentile(arr, 1),
                expected_shortfall= arr[arr < np.percentile(arr, 5)].mean(),
                max_drawdown      = self._max_dd_distribution(arr),
            )
        return StressTestReport(scenario_results=results)

    def generate_synthetic_history(
        self,
        n_days: int = 1_000,
        regime_sequence: list[str] | None = None,
    ) -> SyntheticDataset:
        """
        Generates a complete synthetic price history by sampling from the
        World Model's learned distribution. Useful for:
          - Augmenting limited historical data
          - Generating adversarial scenarios for model robustness testing
          - Creating synthetic data for training execution algorithms
        """
        z = self.wm.sample_initial_state(regime=regime_sequence[0] if regime_sequence else None)
        observations = []
        for day in range(n_days):
            obs = self.wm.decoder(z)
            observations.append(obs)
            regime = regime_sequence[day] if regime_sequence else None
            z      = self.wm.dynamics.step(z, regime_hint=regime)
        return SyntheticDataset(observations=observations)
```

### Standard Stress Scenarios

```python
STANDARD_STRESS_SCENARIOS = [
    StressScenario(
        name            = 'Flash Crash (2010-style)',
        latent_shock_vector = torch.tensor([-3.2, +1.8, +4.1, 0.0, -2.5]),
        horizon_days    = 5,
        description     = 'Sudden liquidity withdrawal and rapid mean-reversion',
    ),
    StressScenario(
        name            = 'Volatility Spike (VIX +20)',
        latent_shock_vector = torch.tensor([0.0, 0.0, +5.0, +3.0, 0.0]),
        horizon_days    = 21,
        description     = 'Volatility regime transition without directional price move',
    ),
    StressScenario(
        name            = 'Credit Crisis (2008-style)',
        latent_shock_vector = torch.tensor([-5.0, -3.5, +6.0, +4.0, -4.0]),
        horizon_days    = 63,
        description     = 'Sustained risk-off regime with cross-asset contagion',
    ),
    StressScenario(
        name            = 'Rate Shock (+100 bps overnight)',
        latent_shock_vector = torch.tensor([-1.5, +2.0, +2.5, -1.0, +3.0]),
        horizon_days    = 10,
        description     = 'Sudden tightening shock propagating through duration assets',
    ),
    StressScenario(
        name            = 'Liquidity Drought',
        latent_shock_vector = torch.tensor([0.0, -4.0, +3.0, +2.0, -3.5]),
        horizon_days    = 21,
        description     = 'Market-wide bid-ask spread widening and volume collapse',
    ),
]
```

---

## Level 4 — Paper Trading

Paper trading connects the World Model to a **live market data feed** without executing real orders. It is the first test of the model in a truly non-stationary environment:

```python
class PaperTradingEngine:
    """
    Paper trading environment for World Model strategies.

    Connects to live market data and simulates order execution
    without submitting real orders to the exchange.

    Captures failure modes invisible in backtesting:
      - Non-stationarity since training cutoff
      - Latency between signal and execution
      - Fill assumptions (backtests assume perfect fills)
      - Regime shifts post-training
    """
    def __init__(
        self,
        world_model: WorldModel,
        strategy: TradingStrategy,
        market_feed: MarketDataFeed,
        execution_simulator: ExecutionSimulator,
    ):
        self.wm        = world_model
        self.strategy  = strategy
        self.feed      = market_feed
        self.simulator = execution_simulator
        self.ledger    = PaperLedger()

    def run_session(self, session_date: date) -> SessionReport:
        """
        Runs a full trading session in paper mode.
        Logs all signals, simulated fills, and P&L in real time.
        """
        self.feed.subscribe(session_date)
        h_t = self.wm.dynamics.initial_hidden_state()

        for tick in self.feed:
            # Update World Model state with live tick
            z_t, h_t = self.wm.encode_tick(tick, h_t)

            # Generate signal
            action = self.strategy.decide(z_t, self.wm)

            if action.size != 0:
                # Simulate fill using current bid/ask
                fill = self.simulator.simulate_fill(action, tick.orderbook)
                self.ledger.record_fill(fill, tick.timestamp)

            # Log model state for diagnostics
            self.ledger.log_model_state(z_t, h_t, tick.timestamp)

        return self.ledger.generate_report(session_date)

    def detect_distribution_shift(
        self,
        paper_states: list[Tensor],
        train_states: list[Tensor],
    ) -> DistributionShiftReport:
        """
        Compares the distribution of latent states observed in paper trading
        against the training distribution.
        A large shift indicates the model is operating out-of-distribution.
        """
        mmd = self._maximum_mean_discrepancy(
            torch.stack(paper_states),
            torch.stack(train_states),
        )
        return DistributionShiftReport(
            mmd_statistic = mmd,
            shift_detected= mmd > 0.05,
            recommendation= (
                'Retrain model'      if mmd > 0.1  else
                'Monitor closely'    if mmd > 0.05 else
                'No action required'
            ),
        )

    def _maximum_mean_discrepancy(self, x: Tensor, y: Tensor) -> float:
        """
        Computes the Maximum Mean Discrepancy (MMD) between two sets of latent vectors.
        MMD > 0 indicates distributional shift between train and deployment environments.
        """
        def rbf_kernel(a: Tensor, b: Tensor, sigma: float = 1.0) -> Tensor:
            sq_dist = torch.cdist(a, b) ** 2
            return torch.exp(-sq_dist / (2 * sigma ** 2))

        k_xx = rbf_kernel(x, x).mean()
        k_yy = rbf_kernel(y, y).mean()
        k_xy = rbf_kernel(x, y).mean()
        return float(k_xx + k_yy - 2 * k_xy)
```

---

## Level 5 — Shadow Mode

Shadow mode runs the World Model strategy in parallel with the existing live strategy (or benchmark), without affecting execution:

```python
class ShadowModeRunner:
    """
    Runs the World Model strategy in shadow mode alongside a live production strategy.

    Shadow mode:
      - Generates all signals and would-be orders in real time
      - Does NOT submit orders to the exchange
      - Records the hypothetical P&L if orders had been executed
      - Enables direct comparison with the live strategy on identical market conditions
    """
    def __init__(
        self,
        shadow_strategy: TradingStrategy,
        live_strategy: TradingStrategy,
        market_feed: MarketDataFeed,
    ):
        self.shadow = shadow_strategy
        self.live   = live_strategy
        self.feed   = market_feed
        self.shadow_ledger = PaperLedger()
        self.live_ledger   = LiveLedger()

    def compare_strategies(
        self,
        start_date: date,
        end_date:   date,
    ) -> ShadowModeReport:
        for session_date in date_range(start_date, end_date):
            shadow_session = self.shadow.paper_session(session_date)
            live_session   = self.live.live_session(session_date)

            self.shadow_ledger.record(shadow_session)
            self.live_ledger.record(live_session)

        shadow_perf = self.shadow_ledger.performance_metrics()
        live_perf   = self.live_ledger.performance_metrics()

        return ShadowModeReport(
            shadow_sharpe          = shadow_perf.sharpe,
            live_sharpe            = live_perf.sharpe,
            alpha_vs_live          = shadow_perf.mean_daily_return - live_perf.mean_daily_return,
            promotion_recommended  = (
                shadow_perf.sharpe > live_perf.sharpe * 1.1
                and shadow_perf.max_drawdown < live_perf.max_drawdown * 0.9
            ),
        )
```

---

## Level 6 — Limited Live Deployment

Upon passing shadow mode, the World Model is promoted to limited live trading with strict guardrails:

```python
class LimitedLiveDeployment:
    """
    Manages the initial live deployment of a World Model trading strategy.

    Safety mechanisms:
      1. Position sizing: fraction of target live size (e.g., 10%)
      2. Daily drawdown limit: circuit breaker halts trading
      3. Latency monitor: alerts if inference time exceeds threshold
      4. Distribution shift monitor: pauses if model operates OOD
      5. Human override: risk manager can halt at any time
    """
    def __init__(
        self,
        model: WorldModel,
        strategy: TradingStrategy,
        execution_engine: ExecutionEngine,
        risk_manager: RiskManager,
        config: LiveConfig,
    ):
        self.model       = model
        self.strategy    = strategy
        self.execution   = execution_engine
        self.risk        = risk_manager
        self.config      = config
        self.circuit_breaker = CircuitBreaker(
            max_daily_drawdown_pct = config.max_daily_drawdown_pct,
        )

    def on_tick(self, tick: MarketTick) -> None:
        # 1. Circuit breaker check
        if self.circuit_breaker.is_tripped():
            return

        # 2. Latency budget check
        t0 = time.perf_counter_ns()

        # 3. Encode state
        z_t = self.model.encode_tick(tick)

        # 4. Distribution shift check
        if self.risk.is_ood(z_t):
            self.risk.alert('OOD state detected — pausing signal generation')
            return

        # 5. Generate signal
        action = self.strategy.decide(z_t, self.model)

        # 6. Apply position size fraction for limited deployment
        scaled_action = action.scale(self.config.live_size_fraction)

        # 7. Risk validation
        if not self.risk.approve(scaled_action):
            return

        # 8. Execute
        self.execution.submit(scaled_action)

        # 9. Log latency
        latency_ns = time.perf_counter_ns() - t0
        if latency_ns > self.config.max_latency_ns:
            self.risk.alert(f'Latency breach: {latency_ns / 1e6:.2f} ms')
```

---

## Continuous Learning and Model Maintenance

Markets change. A World Model trained on 2019–2022 data will degrade as market structure evolves. Continuous learning mechanisms keep the model current:

```python
class ContinuousLearningManager:
    """
    Manages the ongoing retraining and updating of a deployed World Model.

    Three update strategies:
      1. Periodic full retrain — monthly or quarterly on rolling window
      2. Online fine-tuning — incremental updates on recent data
      3. Regime-triggered retrain — automatic retrain when a new regime is detected
    """
    def __init__(
        self,
        model: WorldModel,
        data_store: MarketDataStore,
        training_config: TrainingConfig,
    ):
        self.model    = model
        self.store    = data_store
        self.config   = training_config
        self.version  = 1

    def periodic_retrain(self, reference_date: date) -> WorldModel:
        """
        Full retrain on a rolling window ending at reference_date.
        Runs offline; new model is A/B tested before promotion.
        """
        train_data = self.store.get_window(
            end_date    = reference_date,
            window_days = self.config.train_window_days,
        )
        new_model = self.model.__class__(self.config)
        new_model.fit(train_data)
        return new_model

    def online_fine_tune(
        self,
        new_observations: list[MarketObservation],
        learning_rate: float = 1e-5,
        max_steps: int = 100,
    ) -> None:
        """
        Fine-tunes the encoder and dynamics model on recent observations
        without full retraining. Uses a small learning rate to prevent
        catastrophic forgetting of learned dynamics.
        """
        optimizer = torch.optim.Adam(
            list(self.model.encoder.parameters())
            + list(self.model.dynamics.parameters()),
            lr=learning_rate,
        )
        loader = DataLoader(new_observations, batch_size=32, shuffle=True)
        for step, batch in enumerate(loader):
            if step >= max_steps:
                break
            z_t    = self.model.encoder(batch.obs_t)
            z_tp1  = self.model.encoder(batch.obs_tp1)
            z_pred = self.model.dynamics.step(z_t, batch.action)
            loss   = F.mse_loss(z_pred, z_tp1.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def should_retrain(
        self,
        recent_states: list[Tensor],
        train_states:  list[Tensor],
        mmd_threshold: float = 0.08,
    ) -> bool:
        """
        Returns True if the model is operating sufficiently out-of-distribution
        to warrant a retrain.
        """
        mmd = compute_mmd(torch.stack(recent_states), torch.stack(train_states))
        return mmd > mmd_threshold
```

---

## Monitoring and Observability in Production

A deployed trading World Model requires a comprehensive monitoring stack:

```python
class ProductionMonitor:
    """
    Real-time monitoring dashboard for a deployed trading World Model.

    Tracks four categories of signals:
      1. Model health — are predictions calibrated? Is the model in-distribution?
      2. Strategy health — are P&L metrics in line with backtest expectations?
      3. Execution health — are fills at expected prices with expected latency?
      4. Risk health — are positions and drawdowns within limits?
    """
    def compute_metrics(
        self,
        window: MonitoringWindow,
    ) -> ProductionMetrics:
        return ProductionMetrics(
            # Model health
            prediction_calibration = self._calibration_score(window),
            latent_mmd             = self._mmd_from_train(window.latent_states),
            regime_confidence      = window.regime_probs.max(dim=-1).values.mean().item(),
            inference_latency_p99_ms = np.percentile(window.latency_log_ms, 99),

            # Strategy health
            realised_sharpe_30d    = self._sharpe(window.daily_returns[-30:]),
            sharpe_vs_backtest     = (
                self._sharpe(window.daily_returns[-60:])
                / window.backtest_sharpe
            ),
            signal_autocorrelation = self._autocorr(window.signals),

            # Execution health
            avg_slippage_bps       = window.slippage_log.mean(),
            fill_rate              = window.fill_rate,
            dark_pool_alpha_bps    = window.dark_pool_pnl / window.dark_pool_volume,

            # Risk health
            current_drawdown_pct   = self._current_drawdown(window.cum_pnl),
            position_var_99        = self._position_var(window.positions),
            limit_utilisation      = window.current_exposure / window.exposure_limit,
        )
```

### Automated Alerting Rules

| Metric | Warning Threshold | Critical Threshold | Action |
|---|---|---|---|
| **Latent MMD** | > 0.05 | > 0.10 | Warning → retrain scheduled; Critical → pause |
| **Inference latency p99** | > 5 ms | > 20 ms | Warning → investigate; Critical → halt |
| **Daily drawdown** | > 1.5% | > 3.0% | Warning → reduce size; Critical → halt |
| **Sharpe vs. backtest** | < 0.7× | < 0.4× | Warning → review; Critical → halt |
| **Fill rate** | < 90% | < 70% | Warning → check routing; Critical → halt |
| **Prediction calibration** | ECE > 0.05 | ECE > 0.10 | Warning → fine-tune; Critical → retrain |

---

## Model Governance and Versioning

Trading World Models require rigorous governance — a clear chain of custody from training data through production decision:

```python
@dataclass
class ModelVersion:
    """
    Full provenance record for a deployed World Model version.
    Every production model must have a complete ModelVersion record.
    """
    version_id:          str         # e.g., 'wm-hft-v2.3.1'
    created_at:          datetime
    training_data_hash:  str         # SHA-256 of training dataset
    training_config:     dict        # full hyperparameter set
    validation_results:  dict        # backtest metrics, stress test results
    approval_sign_off:   list[str]   # names of approvers (quant + risk + compliance)
    deployed_at:         datetime | None
    retired_at:          datetime | None
    retirement_reason:   str | None

    def is_approved_for_production(self) -> bool:
        required_approvers = {'lead_quant', 'risk_manager', 'compliance_officer'}
        return required_approvers.issubset(set(self.approval_sign_off))


class ModelRegistry:
    """
    Central registry for all World Model versions.
    Provides immutable audit trail for regulatory purposes.
    """
    def register(self, model: WorldModel, version: ModelVersion) -> str:
        assert version.is_approved_for_production(), \
            'Model requires sign-off from lead_quant, risk_manager, and compliance_officer'
        self._store[version.version_id] = {
            'version': version,
            'weights': model.state_dict(),
        }
        return version.version_id

    def rollback(self, version_id: str) -> WorldModel:
        """
        Restores a previous model version in under 60 seconds.
        Critical capability: if a new model misbehaves in production,
        the previous version must be restorable immediately.
        """
        record = self._store[version_id]
        model  = WorldModel(record['version'].training_config)
        model.load_state_dict(record['weights'])
        return model
```

---

## Regulatory Considerations

Deployed trading World Models must comply with an evolving set of regulations:

### MiFID II / RTS 6 (Algorithmic Trading)

Under the EU's MiFID II framework (RTS 6), all algorithmic trading systems must:

- Maintain a documented description of the algorithm, including the decision logic
- Pass annual self-assessment testing across varied market conditions
- Maintain kill-switch capability to halt all trading immediately
- Store detailed order and execution logs for at least five years

A World Model's circuit breaker, model registry, and monitoring stack directly satisfy several of these requirements.

### SEC Market Access Rule (Rule 15c3-5)

US-listed algorithmic strategies must implement pre-trade risk controls:

- Maximum order size limits
- Maximum position limits
- Maximum notional exposure per strategy
- Credit exposure limits per counterparty

```python
class RegulatoryPreTradeChecks:
    """
    Pre-trade risk controls satisfying SEC Rule 15c3-5 and MiFID II RTS 6.
    All checks must pass before an order is submitted.
    """
    def validate(
        self,
        order: Order,
        portfolio_state: PortfolioState,
        limits: RegulatoryLimits,
    ) -> PreTradeValidationResult:
        violations = []

        if order.notional > limits.max_order_notional:
            violations.append(f'Order notional {order.notional:,.0f} '
                               f'exceeds limit {limits.max_order_notional:,.0f}')

        new_position = portfolio_state.position(order.symbol) + order.signed_quantity
        if abs(new_position) > limits.max_position_size:
            violations.append(f'Resulting position {new_position} '
                               f'exceeds limit ±{limits.max_position_size}')

        new_gross = portfolio_state.gross_exposure + order.notional
        if new_gross > limits.max_gross_exposure:
            violations.append(f'Gross exposure {new_gross:,.0f} '
                               f'exceeds limit {limits.max_gross_exposure:,.0f}')

        return PreTradeValidationResult(
            approved   = len(violations) == 0,
            violations = violations,
        )
```

---

## End-to-End Deployment Pipeline

The complete pipeline from trained model to production deployment:

```
┌─────────────────────────────────────────────────────────────┐
│                 Model Lifecycle Management                   │
├─────────────────────────────────────────────────────────────┤
│  Training                                                    │
│    ├── Data pipeline (clean, normalise, feature engineer)    │
│    ├── Walk-forward CV to tune hyperparameters               │
│    └── Final model trained on full history                   │
├─────────────────────────────────────────────────────────────┤
│  Validation Gate                                             │
│    ├── Reconstruction & dynamics quality checks              │
│    ├── Walk-forward backtest + integrity checks              │
│    ├── Stress testing (5 standard + custom scenarios)        │
│    └── Deflated Sharpe Ratio > 0.95                          │
├─────────────────────────────────────────────────────────────┤
│  Pre-Production                                              │
│    ├── Paper trading (≥ 21 trading days)                     │
│    ├── Shadow mode vs. live strategy (≥ 21 days)             │
│    ├── Distribution shift check (MMD < 0.05)                 │
│    └── Approval: quant + risk + compliance sign-off          │
├─────────────────────────────────────────────────────────────┤
│  Limited Live (10% of target size)                           │
│    ├── Latency profiling in production environment           │
│    ├── Fill quality assessment                               │
│    ├── Daily drawdown circuit breakers active                │
│    └── Promote if Sharpe > 0.8× backtest over 21 days        │
├─────────────────────────────────────────────────────────────┤
│  Full Production                                             │
│    ├── Continuous monitoring (model + strategy + execution)  │
│    ├── Automated alerting and circuit breakers               │
│    ├── Monthly performance review                            │
│    └── Regime-triggered retrain pipeline                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Chapter Summary

- The **validation hierarchy** — unit tests → historical simulation → walk-forward backtest → synthetic stress testing → paper trading → shadow mode → limited live → full production — is the structured filter that separates World Models that genuinely work from those that only appear to in hindsight

- **Walk-forward backtesting** enforces strict temporal separation at every step; models see only data from before the test window, and the **Deflated Sharpe Ratio** (López de Prado & Bailey, 2014) corrects for the number of trials tested to guard against overfitting

- **Synthetic market simulation** leverages the World Model's generative capability to construct stress scenarios that never occurred historically — flash crashes, volatility spikes, credit crises, rate shocks — and evaluates strategy resilience across all of them

- **Paper trading** is the first test of a World Model in a live, non-stationary environment; the **Maximum Mean Discrepancy** (MMD) statistic detects distributional shift between the training environment and the live market

- **Shadow mode** provides a direct controlled comparison between the World Model strategy and the existing live strategy under identical market conditions, generating the statistical evidence needed for promotion decisions

- **Continuous learning** — periodic retraining, online fine-tuning, and regime-triggered updates — keeps the deployed model current as market structure evolves; MMD monitoring triggers retraining before performance degrades visibly

- **Production monitoring** covers four health dimensions — model (calibration, MMD, latency), strategy (P&L, Sharpe), execution (slippage, fill rate), and risk (drawdown, position limits) — with automated alerting and circuit breakers for each

- **Model governance and versioning** through an immutable model registry ensures that every production decision is traceable to a specific set of training data, hyperparameters, and human approvals — satisfying the audit requirements of MiFID II and SEC Rule 15c3-5

- The end-to-end deployment pipeline treats a trading World Model not as a static artefact but as a **living system** requiring ongoing validation, monitoring, and adaptation — reflecting the non-stationarity of the financial markets it simulates

---

## Looking Ahead

This chapter closes the engineering loop of the book. The architectural foundations of Chapters 1–12 described *what* a financial World Model is; Chapters 13–16 showed *how* different trading participants apply World Models to their specific decision problems; this chapter described *how to deploy and maintain* those models responsibly.

The financial World Model is not a product to be shipped — it is a **living cognitive system** to be tended: validated before deployment, monitored in production, retrained as markets evolve, and governed with the rigour that real capital demands.

> *"A model that works in backtesting but fails in production is not a model — it is a hypothesis that happened to fit the past."*
>
> The validation hierarchy, continuous learning, and production monitoring frameworks described in this chapter are the scientific method applied to algorithmic trading — iterative, falsifiable, and continuously updated by new evidence.

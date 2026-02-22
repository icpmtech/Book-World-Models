import { useState, useMemo, useCallback } from 'react';
import type { ChangeEvent } from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

// ─── Types ───────────────────────────────────────────────────────────────────

interface SimParams {
  inflation: number;
  rateChange: number;
  liquidityIndex: number;
  timeHorizon: number;
}

interface SimResult {
  trajectory: number[];
  upperBand: number[];
  lowerBand: number[];
  metrics: {
    expectedReturn: number;
    annualizedVol: number;
    maxDrawdown: number;
    sharpeRatio: number;
    probabilityPositive: number;
  };
  insights: string[];
}

interface Scenario {
  label: string;
  params: SimParams;
  description: string;
}

// ─── Scenario Presets ────────────────────────────────────────────────────────

const SCENARIOS: Record<string, Scenario> = {
  rateHike: {
    label: '🏦 Aggressive Rate Hike',
    description: 'Central bank raises rates aggressively to combat high inflation — mirrors 2022 Fed cycle.',
    params: { inflation: 7.5, rateChange: 3.0, liquidityIndex: 30, timeHorizon: 18 },
  },
  recession: {
    label: '📉 2008-Style Credit Crisis',
    description: 'Severe liquidity contraction, credit freeze, and falling inflation signal systemic risk.',
    params: { inflation: -0.5, rateChange: -4.0, liquidityIndex: 8, timeHorizon: 24 },
  },
  recovery: {
    label: '🌱 Post-Crisis Recovery',
    description: 'Rates fall, liquidity improves, and inflation stabilizes — typical recovery dynamics.',
    params: { inflation: 2.5, rateChange: -1.5, liquidityIndex: 72, timeHorizon: 18 },
  },
  inflation: {
    label: '🔥 Inflation Surge',
    description: 'Supply-side shock drives inflation well above target while rates lag — 1970s analogue.',
    params: { inflation: 9.0, rateChange: 1.0, liquidityIndex: 42, timeHorizon: 12 },
  },
  goldilocks: {
    label: '✨ Goldilocks Environment',
    description: 'Inflation near target, stable rates, high liquidity — optimal conditions for risk assets.',
    params: { inflation: 2.2, rateChange: 0.0, liquidityIndex: 78, timeHorizon: 12 },
  },
  custom: {
    label: '⚙ Custom',
    description: 'Set your own parameters below.',
    params: { inflation: 3.5, rateChange: 0.5, liquidityIndex: 55, timeHorizon: 12 },
  },
};

// ─── Simulation Math ──────────────────────────────────────────────────────────

function simulateMarket(p: SimParams): SimResult {
  const { inflation, rateChange, liquidityIndex, timeHorizon } = p;

  // Annual expected return components
  const baseReturn = 7.0;
  const rateImpact = -rateChange * 3.2;
  const inflationImpact = -(Math.abs(inflation - 2.0)) * 0.9;
  const liquidityImpact = ((liquidityIndex - 50) / 50) * 5.0;
  const expectedAnnualReturn = baseReturn + rateImpact + inflationImpact + liquidityImpact;

  // Monthly volatility
  const baseVol = 2.2;
  const excessVol =
    Math.abs(rateChange) * 0.6 +
    Math.max(0, inflation - 4) * 0.35 +
    Math.max(0, 35 - liquidityIndex) * 0.06;
  const monthlyVol = baseVol + excessVol;

  const trajectory: number[] = [100];
  const upperBand: number[] = [100];
  const lowerBand: number[] = [100];

  for (let m = 1; m <= timeHorizon; m++) {
    const prev = trajectory[m - 1];

    // Initial shock (front-loaded rate effect, decays over ~4 months)
    const shockDecay = Math.exp(-m / 4);
    const initialShock = rateChange > 0 ? rateChange * -1.8 * shockDecay : rateChange * -1.2 * shockDecay;

    const monthlyReturn = expectedAnnualReturn / 12 + initialShock / timeHorizon;
    trajectory.push(prev * (1 + monthlyReturn / 100));

    // Uncertainty band widens with sqrt(t)
    const band = monthlyVol * Math.sqrt(m) * 0.55;
    upperBand.push(trajectory[m] * (1 + band / 100));
    lowerBand.push(trajectory[m] * (1 - band / 100));
  }

  // Metrics
  const finalLevel = trajectory[timeHorizon];
  const finalReturn = (finalLevel / 100 - 1) * 100;
  const annualizedReturn = (finalReturn / timeHorizon) * 12;
  const annualizedVol = monthlyVol * Math.sqrt(12);

  let peak = 100;
  let maxDrawdown = 0;
  for (const level of trajectory) {
    if (level > peak) peak = level;
    const dd = ((peak - level) / peak) * 100;
    if (dd > maxDrawdown) maxDrawdown = dd;
  }

  const riskFree = 4.0;
  const sharpeRatio = annualizedVol > 0 ? (annualizedReturn - riskFree) / annualizedVol : 0;

  const spread = upperBand[timeHorizon] - lowerBand[timeHorizon];
  const probabilityPositive =
    spread > 0
      ? Math.min(95, Math.max(5, 50 + (finalReturn / spread) * 50))
      : finalReturn > 0
        ? 75
        : 25;

  // Contextual insights
  const insights: string[] = [];
  if (rateChange > 2) {
    insights.push(`Aggressive rate hikes (${rateChange.toFixed(1)}%) create significant front-loaded equity pressure through P/E compression and rising discount rates.`);
  } else if (rateChange > 0) {
    insights.push(`Moderate rate increases (${rateChange.toFixed(1)}%) weigh on growth stocks but may signal economic confidence, supporting cyclical sectors.`);
  } else if (rateChange < -1) {
    insights.push(`Rate cuts (${rateChange.toFixed(1)}%) typically provide stimulus to equities, though at this depth they may signal serious economic concern.`);
  }
  if (inflation > 6) {
    insights.push(`Inflation at ${inflation.toFixed(1)}% significantly above target erodes real returns and compresses equity multiples, especially in duration-sensitive sectors.`);
  } else if (inflation < 0) {
    insights.push(`Deflationary conditions (${inflation.toFixed(1)}%) signal demand collapse and increase recession probability materially.`);
  } else if (Math.abs(inflation - 2.0) < 0.8) {
    insights.push(`Inflation near 2% target (${inflation.toFixed(1)}%) is a supportive backdrop for risk assets and central bank policy stability.`);
  }
  if (liquidityIndex < 25) {
    insights.push(`Severely stressed liquidity (index: ${liquidityIndex}) amplifies all risks and may trigger forced selling cascades.`);
  } else if (liquidityIndex > 65) {
    insights.push(`High liquidity (index: ${liquidityIndex}) reduces risk premium and supports asset prices across the spectrum.`);
  }
  if (maxDrawdown > 15) {
    insights.push(`Expect a peak drawdown of ~${maxDrawdown.toFixed(1)}% — sizing and hedging should reflect this potential loss.`);
  }
  if (insights.length === 0) {
    insights.push(`Balanced conditions produce a relatively moderate expected return with manageable downside risk.`);
  }

  return {
    trajectory,
    upperBand,
    lowerBand,
    metrics: {
      expectedReturn: Math.round(finalReturn * 10) / 10,
      annualizedVol: Math.round(annualizedVol * 10) / 10,
      maxDrawdown: Math.round(maxDrawdown * 10) / 10,
      sharpeRatio: Math.round(sharpeRatio * 100) / 100,
      probabilityPositive: Math.round(probabilityPositive),
    },
    insights,
  };
}

// ─── LLM Response Generator ───────────────────────────────────────────────────

function generateLLMResponse(p: SimParams): string[] {
  const { inflation, rateChange, liquidityIndex, timeHorizon } = p;
  const lines: string[] = [];

  if (rateChange > 2) {
    lines.push(`With aggressive rate increases of ${rateChange.toFixed(1)}%, we would expect significant pressure on equity valuations, particularly in rate-sensitive sectors such as real estate, utilities, and long-duration technology stocks.`);
    lines.push(`Historically, aggressive rate hiking cycles have been associated with elevated market volatility, potential earnings downgrades, and — in some cases — recession.`);
  } else if (rateChange > 0.25) {
    lines.push(`Moderate rate increases of ${rateChange.toFixed(1)}% may reflect central bank confidence in economic growth. While this creates some headwinds for equities — particularly growth-oriented names — value sectors and financials often benefit.`);
  } else if (rateChange < -1.5) {
    lines.push(`Significant rate cuts of ${Math.abs(rateChange).toFixed(1)}% are typically associated with economic stimulus, which can support equities over the medium term. However, such cuts usually indicate underlying economic weakness that may weigh on earnings in the near term.`);
  } else if (Math.abs(rateChange) <= 0.25) {
    lines.push(`A stable rate environment generally provides a neutral-to-positive backdrop for equities, allowing companies to plan investments with predictable financing costs.`);
  } else {
    lines.push(`A modest rate adjustment of ${rateChange.toFixed(1)}% is unlikely to materially alter the macro backdrop for markets.`);
  }

  if (inflation > 6) {
    lines.push(`Inflation at ${inflation.toFixed(1)}% represents a significant challenge for central banks and markets. Above-target inflation historically compresses equity multiples via higher discount rates, while eroding consumer purchasing power.`);
  } else if (inflation > 3.5) {
    lines.push(`Inflation running above ${inflation.toFixed(1)}% may prompt additional policy response and could keep real bond yields negative, complicating asset allocation.`);
  } else if (inflation < 0) {
    lines.push(`Deflationary conditions (${inflation.toFixed(1)}%) raise concerns about demand weakness. While superficially positive for consumers, deflation can trap economies in low-growth spirals, as Japan experienced in the 1990s.`);
  }

  if (liquidityIndex < 30) {
    lines.push(`Low market liquidity conditions increase the risk of sharp price dislocations. In thin markets, selling pressure tends to be amplified, and bid-ask spreads widen materially.`);
  } else if (liquidityIndex > 70) {
    lines.push(`Favorable liquidity conditions are generally supportive of asset prices, reducing risk premiums and facilitating efficient price discovery.`);
  }

  lines.push(`Over a ${timeHorizon}-month horizon, the interplay of these factors — monetary policy trajectory, inflation dynamics, and market liquidity — will likely be the key determinants of market direction. That said, markets may overshoot or undershoot fundamentals in the short term.`);
  lines.push(`*This analysis reflects historical patterns and general economic reasoning. It does not constitute investment advice, and actual market outcomes may differ significantly from historical analogues.*`);

  return lines;
}

// ─── SVG Chart ────────────────────────────────────────────────────────────────

interface ChartProps {
  trajectory: number[];
  upperBand: number[];
  lowerBand: number[];
  timeHorizon: number;
}

function SimChart({ trajectory, upperBand, lowerBand, timeHorizon }: ChartProps) {
  const W = 660;
  const H = 260;
  const PAD = { top: 20, right: 24, bottom: 40, left: 52 };
  const chartW = W - PAD.left - PAD.right;
  const chartH = H - PAD.top - PAD.bottom;

  const allValues = [...trajectory, ...upperBand, ...lowerBand];
  const yMin = Math.min(...allValues) * 0.985;
  const yMax = Math.max(...allValues) * 1.015;

  const xScale = (i: number) => PAD.left + (i / timeHorizon) * chartW;
  const yScale = (v: number) => PAD.top + chartH - ((v - yMin) / (yMax - yMin)) * chartH;

  const bandPath =
    upperBand
      .map((v, i) => `${i === 0 ? 'M' : 'L'}${xScale(i).toFixed(1)},${yScale(v).toFixed(1)}`)
      .join(' ') +
    ' ' +
    lowerBand
      .slice()
      .reverse()
      .map((v, i) => {
        const origIdx = timeHorizon - i;
        return `L${xScale(origIdx).toFixed(1)},${yScale(v).toFixed(1)}`;
      })
      .join(' ') +
    ' Z';

  const trajPath = trajectory
    .map((v, i) => `${i === 0 ? 'M' : 'L'}${xScale(i).toFixed(1)},${yScale(v).toFixed(1)}`)
    .join(' ');

  const baselineY = yScale(100);

  // Y-axis grid lines
  const yTicks = 5;
  const yStep = (yMax - yMin) / yTicks;
  const gridLines = Array.from({ length: yTicks + 1 }, (_, i) => yMin + i * yStep);

  // X-axis ticks
  const xTickInterval = timeHorizon <= 12 ? 2 : timeHorizon <= 18 ? 3 : 4;
  const xTicks = Array.from({ length: Math.floor(timeHorizon / xTickInterval) + 1 }, (_, i) =>
    Math.min(i * xTickInterval, timeHorizon),
  );

  const finalVal = trajectory[timeHorizon];
  const isPositive = finalVal >= 100;

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className={styles.chart}
      aria-label="Market trajectory simulation chart"
      role="img"
    >
      {/* Background */}
      <rect x="0" y="0" width={W} height={H} fill="#0a1628" rx="8" />
      <rect
        x={PAD.left}
        y={PAD.top}
        width={chartW}
        height={chartH}
        fill="#0d1f3c"
        rx="4"
      />

      {/* Grid lines */}
      {gridLines.map((val, i) => {
        const y = yScale(val);
        return (
          <g key={i}>
            <line
              x1={PAD.left}
              y1={y}
              x2={PAD.left + chartW}
              y2={y}
              stroke="#1e3a5f"
              strokeWidth="1"
            />
            <text
              x={PAD.left - 6}
              y={y + 4}
              textAnchor="end"
              fontSize="9"
              fill="#5b7fa6"
            >
              {val.toFixed(0)}
            </text>
          </g>
        );
      })}

      {/* Baseline at 100 */}
      <line
        x1={PAD.left}
        y1={baselineY}
        x2={PAD.left + chartW}
        y2={baselineY}
        stroke="#334d6e"
        strokeWidth="1.5"
        strokeDasharray="4 3"
      />
      <text x={PAD.left - 6} y={baselineY + 4} textAnchor="end" fontSize="9" fill="#4d90e8">
        100
      </text>

      {/* Confidence band */}
      <path d={bandPath} fill="rgba(77,144,232,0.15)" />

      {/* Band borders */}
      <path
        d={upperBand.map((v, i) => `${i === 0 ? 'M' : 'L'}${xScale(i).toFixed(1)},${yScale(v).toFixed(1)}`).join(' ')}
        fill="none"
        stroke="rgba(77,144,232,0.4)"
        strokeWidth="1"
        strokeDasharray="3 2"
      />
      <path
        d={lowerBand.map((v, i) => `${i === 0 ? 'M' : 'L'}${xScale(i).toFixed(1)},${yScale(v).toFixed(1)}`).join(' ')}
        fill="none"
        stroke="rgba(77,144,232,0.4)"
        strokeWidth="1"
        strokeDasharray="3 2"
      />

      {/* Main trajectory */}
      <path
        d={trajPath}
        fill="none"
        stroke={isPositive ? '#22c55e' : '#ef4444'}
        strokeWidth="2.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* Start dot */}
      <circle cx={xScale(0)} cy={yScale(100)} r="4" fill="#4d90e8" />

      {/* End dot */}
      <circle
        cx={xScale(timeHorizon)}
        cy={yScale(finalVal)}
        r="5"
        fill={isPositive ? '#22c55e' : '#ef4444'}
      />

      {/* End label */}
      <text
        x={xScale(timeHorizon) + 8}
        y={yScale(finalVal) + 4}
        fontSize="10"
        fill={isPositive ? '#22c55e' : '#ef4444'}
        fontWeight="bold"
      >
        {finalVal.toFixed(1)}
      </text>

      {/* X-axis ticks */}
      {xTicks.map((m) => (
        <g key={m}>
          <line
            x1={xScale(m)}
            y1={PAD.top + chartH}
            x2={xScale(m)}
            y2={PAD.top + chartH + 4}
            stroke="#5b7fa6"
            strokeWidth="1"
          />
          <text
            x={xScale(m)}
            y={PAD.top + chartH + 15}
            textAnchor="middle"
            fontSize="9"
            fill="#5b7fa6"
          >
            {m}m
          </text>
        </g>
      ))}

      {/* Axis labels */}
      <text
        x={PAD.left + chartW / 2}
        y={H - 4}
        textAnchor="middle"
        fontSize="10"
        fill="#5b7fa6"
      >
        Time Horizon (months)
      </text>
      <text
        x={12}
        y={PAD.top + chartH / 2}
        textAnchor="middle"
        fontSize="10"
        fill="#5b7fa6"
        transform={`rotate(-90, 12, ${PAD.top + chartH / 2})`}
      >
        Market Index
      </text>

      {/* Legend */}
      <g transform={`translate(${PAD.left + 8}, ${PAD.top + 6})`}>
        <line x1="0" y1="6" x2="18" y2="6" stroke={isPositive ? '#22c55e' : '#ef4444'} strokeWidth="2.5" />
        <text x="22" y="10" fontSize="9" fill="#93a3b8">
          Expected Path
        </text>
        <rect x="60" y="0" width="18" height="12" fill="rgba(77,144,232,0.2)" stroke="rgba(77,144,232,0.5)" strokeWidth="0.5" />
        <text x="82" y="10" fontSize="9" fill="#93a3b8">
          90% Confidence Interval
        </text>
      </g>
    </svg>
  );
}

// ─── Metric Card ──────────────────────────────────────────────────────────────

function MetricCard({
  label,
  value,
  unit,
  sentiment,
}: {
  label: string;
  value: string | number;
  unit?: string;
  sentiment: 'positive' | 'negative' | 'neutral';
}) {
  return (
    <div className={clsx(styles.metricCard, styles[`metric_${sentiment}`])}>
      <div className={styles.metricLabel}>{label}</div>
      <div className={styles.metricValue}>
        {value}
        {unit && <span className={styles.metricUnit}>{unit}</span>}
      </div>
    </div>
  );
}

// ─── Slider ───────────────────────────────────────────────────────────────────

function Slider({
  label,
  value,
  min,
  max,
  step,
  unit,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit: string;
  onChange: (v: number) => void;
}) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div className={styles.sliderRow}>
      <div className={styles.sliderLabel}>
        <span>{label}</span>
        <span className={styles.sliderValue}>
          {value > 0 && min < 0 ? '+' : ''}
          {value.toFixed(step < 1 ? 1 : 0)}
          {unit}
        </span>
      </div>
      <div className={styles.sliderTrack}>
        <div className={styles.sliderFill} style={{ width: `${pct}%` }} />
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e: ChangeEvent<HTMLInputElement>) => onChange(parseFloat(e.target.value))}
          className={styles.sliderInput}
          aria-label={label}
        />
      </div>
      <div className={styles.sliderMinMax}>
        <span>
          {min}
          {unit}
        </span>
        <span>
          {max}
          {unit}
        </span>
      </div>
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────

export default function WorldModelSimulator() {
  const [scenarioKey, setScenarioKey] = useState<string>('rateHike');
  const [params, setParams] = useState<SimParams>(SCENARIOS.rateHike.params);
  const [activeTab, setActiveTab] = useState<'llm' | 'worldmodel' | 'compare'>('compare');

  const setScenario = useCallback((key: string) => {
    setScenarioKey(key);
    if (key !== 'custom') {
      setParams(SCENARIOS[key].params);
    }
  }, []);

  const updateParam = useCallback(
    (field: keyof SimParams) => (val: number) => {
      setScenarioKey('custom');
      setParams((prev) => ({ ...prev, [field]: val }));
    },
    [],
  );

  const simulation = useMemo(() => simulateMarket(params), [params]);
  const llmResponse = useMemo(() => generateLLMResponse(params), [params]);

  const { metrics } = simulation;

  return (
    <div className={styles.simulator}>
      {/* Header */}
      <div className={styles.header}>
        <h2 className={styles.headerTitle}>🌍 World Model vs LLM Simulator</h2>
        <p className={styles.headerSubtitle}>
          Explore the difference between <strong>descriptive intelligence</strong> (LLM) and{' '}
          <strong>anticipatory intelligence</strong> (World Model). Set economic parameters and see
          how each approach responds.
        </p>
      </div>

      {/* Scenarios */}
      <div className={styles.scenarios}>
        <div className={styles.scenariosLabel}>Preset Scenarios:</div>
        <div className={styles.scenarioPills}>
          {Object.entries(SCENARIOS).map(([key, s]) => (
            <button
              key={key}
              className={clsx(styles.scenarioPill, scenarioKey === key && styles.scenarioPillActive)}
              onClick={() => setScenario(key)}
            >
              {s.label}
            </button>
          ))}
        </div>
        {SCENARIOS[scenarioKey] && (
          <p className={styles.scenarioDesc}>{SCENARIOS[scenarioKey].description}</p>
        )}
      </div>

      {/* Controls */}
      <div className={styles.controls}>
        <div className={styles.controlsTitle}>📊 Parameters</div>
        <Slider
          label="Inflation"
          value={params.inflation}
          min={-3}
          max={12}
          step={0.5}
          unit="%"
          onChange={updateParam('inflation')}
        />
        <Slider
          label="Rate Change (total)"
          value={params.rateChange}
          min={-5}
          max={5}
          step={0.25}
          unit="%"
          onChange={updateParam('rateChange')}
        />
        <Slider
          label="Liquidity Index"
          value={params.liquidityIndex}
          min={0}
          max={100}
          step={1}
          unit=""
          onChange={updateParam('liquidityIndex')}
        />
        <Slider
          label="Time Horizon"
          value={params.timeHorizon}
          min={3}
          max={36}
          step={1}
          unit=" mo"
          onChange={updateParam('timeHorizon')}
        />
      </div>

      {/* Tabs */}
      <div className={styles.tabs}>
        {(
          [
            { key: 'compare', label: '↔ Side-by-Side' },
            { key: 'worldmodel', label: '🌍 World Model' },
            { key: 'llm', label: '💬 LLM Response' },
          ] as { key: typeof activeTab; label: string }[]
        ).map(({ key, label }) => (
          <button
            key={key}
            className={clsx(styles.tab, activeTab === key && styles.tabActive)}
            onClick={() => setActiveTab(key)}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className={styles.content}>
        {/* World Model Panel */}
        {(activeTab === 'worldmodel' || activeTab === 'compare') && (
          <div className={clsx(styles.panel, styles.wmPanel)}>
            <div className={styles.panelHeader}>
              <span className={styles.panelBadgeWm}>World Model</span>
              <span className={styles.panelHeaderTitle}>Anticipatory Simulation</span>
            </div>

            {/* Metrics */}
            <div className={styles.metrics}>
              <MetricCard
                label="Expected Return"
                value={metrics.expectedReturn > 0 ? `+${metrics.expectedReturn}` : metrics.expectedReturn}
                unit="%"
                sentiment={metrics.expectedReturn > 2 ? 'positive' : metrics.expectedReturn < -5 ? 'negative' : 'neutral'}
              />
              <MetricCard
                label="Annualised Vol"
                value={metrics.annualizedVol}
                unit="%"
                sentiment={metrics.annualizedVol > 20 ? 'negative' : metrics.annualizedVol < 12 ? 'positive' : 'neutral'}
              />
              <MetricCard
                label="Max Drawdown"
                value={`−${metrics.maxDrawdown}`}
                unit="%"
                sentiment={metrics.maxDrawdown > 15 ? 'negative' : metrics.maxDrawdown < 6 ? 'positive' : 'neutral'}
              />
              <MetricCard
                label="Sharpe Ratio"
                value={metrics.sharpeRatio}
                sentiment={metrics.sharpeRatio > 0.5 ? 'positive' : metrics.sharpeRatio < 0 ? 'negative' : 'neutral'}
              />
              <MetricCard
                label="P(Positive)"
                value={metrics.probabilityPositive}
                unit="%"
                sentiment={metrics.probabilityPositive > 60 ? 'positive' : metrics.probabilityPositive < 40 ? 'negative' : 'neutral'}
              />
            </div>

            {/* Chart */}
            <SimChart
              trajectory={simulation.trajectory}
              upperBand={simulation.upperBand}
              lowerBand={simulation.lowerBand}
              timeHorizon={params.timeHorizon}
            />

            {/* Insights */}
            <div className={styles.insights}>
              <div className={styles.insightsTitle}>🔍 Causal Insights</div>
              {simulation.insights.map((insight, i) => (
                <p key={i} className={styles.insightItem}>
                  {insight}
                </p>
              ))}
            </div>

            <div className={styles.wmNote}>
              ✓ Quantified probability distribution · ✓ Causal state propagation · ✓ Actionable metrics
            </div>
          </div>
        )}

        {/* LLM Panel */}
        {(activeTab === 'llm' || activeTab === 'compare') && (
          <div className={clsx(styles.panel, styles.llmPanel)}>
            <div className={styles.panelHeader}>
              <span className={styles.panelBadgeLlm}>LLM</span>
              <span className={styles.panelHeaderTitle}>Descriptive Response</span>
            </div>

            <div className={styles.llmOutput}>
              <div className={styles.llmPrompt}>
                <strong>Prompt:</strong> Inflation is {params.inflation.toFixed(1)}%, rates change by{' '}
                {params.rateChange > 0 ? '+' : ''}
                {params.rateChange.toFixed(2)}%, liquidity index is {params.liquidityIndex}. What
                happens to the equity market over {params.timeHorizon} months?
              </div>
              <div className={styles.llmResponse}>
                {llmResponse.map((line, i) => (
                  <p key={i} className={styles.llmLine}>
                    {line}
                  </p>
                ))}
              </div>
            </div>

            <div className={styles.llmLimitations}>
              <div className={styles.limitationsTitle}>⚠ Limitations of this approach</div>
              <ul className={styles.limitationsList}>
                <li>No probability distributions — cannot quantify uncertainty</li>
                <li>No state simulation — reverts to historical averages</li>
                <li>Cannot model interventions or counterfactuals</li>
                <li>No causal chain — only statistical co-occurrence</li>
                <li>Does not adapt to novel initial conditions outside training distribution</li>
              </ul>
            </div>
          </div>
        )}
      </div>

      {/* Footer note */}
      <div className={styles.footerNote}>
        <strong>Educational simulator.</strong> The World Model simulation uses simplified causal
        dynamics for illustration. Real financial world models require large-scale training on
        historical market microstructure data.
      </div>
    </div>
  );
}

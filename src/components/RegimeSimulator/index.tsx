import { useState, useMemo, useCallback } from 'react';
import type { ChangeEvent } from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

// ─── Types ───────────────────────────────────────────────────────────────────

interface MacroParams {
  gdpGrowth: number;
  inflation: number;
  vix: number;
  creditSpread: number;
  yieldCurveSlope: number;
}

interface RegimeResult {
  probs: [number, number, number, number]; // E, O, C, R
  dominantRegime: number;
  entropy: number;
  earlyWarning: string | null;
  portfolioWeights: { equity: number; bonds: number; commodities: number; cash: number };
  insights: string[];
}

interface Preset {
  label: string;
  params: MacroParams;
  description: string;
}

// ─── Constants ────────────────────────────────────────────────────────────────

const REGIME_NAMES = ['Expansion', 'Overheating', 'Contraction', 'Recovery'] as const;
const REGIME_COLORS = ['#2ecc71', '#e67e22', '#e74c3c', '#3498db'] as const;
const REGIME_EMOJIS = ['📈', '🔥', '📉', '🌱'] as const;

const PRESETS: Record<string, Preset> = {
  expansion: {
    label: '📈 Expansion',
    description: 'Steady growth, moderate inflation, low volatility — typical mid-cycle conditions.',
    params: { gdpGrowth: 2.8, inflation: 2.1, vix: 14, creditSpread: 85, yieldCurveSlope: 120 },
  },
  overheating: {
    label: '🔥 Overheating',
    description: '2022-style: inflation surges, Fed hikes aggressively, yield curve flattens.',
    params: { gdpGrowth: 3.5, inflation: 7.5, vix: 24, creditSpread: 180, yieldCurveSlope: 10 },
  },
  contraction: {
    label: '📉 Contraction',
    description: '2008-style: GDP falling, VIX spiking, credit spreads blow out.',
    params: { gdpGrowth: -2.5, inflation: 1.0, vix: 45, creditSpread: 650, yieldCurveSlope: -20 },
  },
  recovery: {
    label: '🌱 Recovery',
    description: 'Post-crisis: growth returns, policy remains loose, risk appetite recovers.',
    params: { gdpGrowth: 1.8, inflation: 1.5, vix: 22, creditSpread: 220, yieldCurveSlope: 80 },
  },
  transition: {
    label: '⚠️ Transition Zone',
    description: 'Ambiguous signals — high predictive entropy, early warning triggered.',
    params: { gdpGrowth: 1.2, inflation: 4.2, vix: 26, creditSpread: 290, yieldCurveSlope: 25 },
  },
  custom: {
    label: '⚙ Custom',
    description: 'Set your own macro parameters below.',
    params: { gdpGrowth: 2.0, inflation: 2.5, vix: 18, creditSpread: 120, yieldCurveSlope: 90 },
  },
};

// ─── Regime Inference Engine ──────────────────────────────────────────────────

function inferRegime(p: MacroParams): RegimeResult {
  const { gdpGrowth, inflation, vix, creditSpread, yieldCurveSlope } = p;

  // Score for each regime [Expansion, Overheating, Contraction, Recovery]
  // Each score component is in [0, 1] — higher = more evidence for this regime

  // Expansion: good GDP, moderate inflation, low VIX, tight spreads, normal curve
  const expansionScore =
    sigmoid((gdpGrowth - 2.5) * 1.2) *
    sigmoid((3.5 - inflation) * 0.8) *
    sigmoid((22 - vix) * 0.15) *
    sigmoid((200 - creditSpread) * 0.008) *
    sigmoid((yieldCurveSlope - 60) * 0.02);

  // Overheating: high GDP, high inflation, moderate VIX, widening spreads, flat curve
  const overheatScore =
    sigmoid((gdpGrowth - 2.0) * 0.8) *
    sigmoid((inflation - 3.5) * 0.7) *
    sigmoid((30 - vix) * 0.10) *
    (1 - sigmoid((creditSpread - 400) * 0.008)) *
    sigmoid((50 - Math.abs(yieldCurveSlope - 20)) * 0.04);

  // Contraction: falling GDP, low inflation, high VIX, wide spreads, inverted/flat curve
  const contractionScore =
    sigmoid((-gdpGrowth + 0.5) * 1.5) *
    sigmoid((3.0 - inflation) * 0.5) *
    sigmoid((vix - 28) * 0.18) *
    sigmoid((creditSpread - 300) * 0.006) *
    sigmoid((60 - yieldCurveSlope) * 0.03);

  // Recovery: low-moderate GDP, below-target inflation, declining VIX, narrowing spreads
  const recoveryScore =
    sigmoid((gdpGrowth - 0.5) * 1.0) *
    sigmoid((3.0 - inflation) * 0.8) *
    sigmoid((28 - vix) * 0.12) *
    sigmoid((350 - creditSpread) * 0.006) *
    sigmoid((yieldCurveSlope - 30) * 0.025);

  const rawScores = [expansionScore, overheatScore, contractionScore, recoveryScore];
  const total = rawScores.reduce((a, b) => a + b, 0) || 1;
  const probs = rawScores.map((s) => s / total) as [number, number, number, number];

  const dominantRegime = probs.indexOf(Math.max(...probs));

  // Shannon entropy (normalised to [0,1] by dividing by log2(4) = 2)
  const entropy =
    probs.reduce((h, p) => (p > 0 ? h - p * Math.log2(p) : h), 0) / 2;

  // Early warning: high entropy OR rapid shift away from dominant
  let earlyWarning: string | null = null;
  const secondMax = [...probs].sort((a, b) => b - a)[1];
  if (entropy > 0.75) {
    earlyWarning = `⚠️ High predictive entropy (${(entropy * 100).toFixed(0)}%) — regime transition likely. Monitor latent state drift closely.`;
  } else if (probs[dominantRegime] < 0.55 && secondMax > 0.30) {
    earlyWarning = `⚠️ Dominant regime confidence is low (${(probs[dominantRegime] * 100).toFixed(0)}%) with significant probability mass on ${REGIME_NAMES[probs.indexOf(secondMax)]} (${(secondMax * 100).toFixed(0)}%).`;
  }

  // Regime-conditioned portfolio weights (probability-weighted)
  const regimeEquity  = [0.65, 0.40, 0.20, 0.52];
  const regimeBonds   = [0.20, 0.15, 0.45, 0.28];
  const regimeCommodity = [0.08, 0.25, 0.10, 0.08];
  const regimeCash    = [0.07, 0.20, 0.25, 0.12];

  const equity     = dot(probs, regimeEquity);
  const bonds      = dot(probs, regimeBonds);
  const commodities = dot(probs, regimeCommodity);
  const cash       = dot(probs, regimeCash);

  // Insights
  const insights: string[] = [];
  if (inflation > 5) {
    insights.push(`Inflation at ${inflation.toFixed(1)}% significantly above target — central bank reaction function likely to suppress equity multiples via P/E compression.`);
  } else if (inflation < 1.0 && gdpGrowth < 1.0) {
    insights.push(`Deflationary conditions with weak growth signal potential debt-deflation spiral. Historical analogues: Japan 1990s, Eurozone 2013.`);
  }
  if (vix > 35) {
    insights.push(`VIX at ${vix} indicates severe stress. Forced selling and margin calls can create non-linear price dislocations. Liquidity premium is elevated.`);
  }
  if (creditSpread > 500) {
    insights.push(`HY credit spreads above 500bps signal systemic stress. Historical median: 330bps. At these levels, market pricing implies ~15–20% default rate over 12 months.`);
  }
  if (yieldCurveSlope < 0) {
    insights.push(`Inverted yield curve (slope: ${yieldCurveSlope}bps) has preceded every US recession since 1955 — though with variable lead times of 6–24 months.`);
  }
  if (gdpGrowth > 3.5 && inflation > 4) {
    insights.push(`Simultaneous above-trend growth and elevated inflation creates a stagflationary pressure point — difficult for central banks to navigate without inducing recession.`);
  }
  if (insights.length === 0) {
    insights.push(`Current macro conditions are broadly consistent with a ${REGIME_NAMES[dominantRegime]} environment. No acute stress signals detected.`);
  }

  return {
    probs,
    dominantRegime,
    entropy,
    earlyWarning,
    portfolioWeights: { equity, bonds, commodities, cash },
    insights,
  };
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function dot(a: number[], b: number[]): number {
  return a.reduce((s, v, i) => s + v * b[i], 0);
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function RegimeProbBar({
  name,
  prob,
  color,
  emoji,
  isDominant,
}: {
  name: string;
  prob: number;
  color: string;
  emoji: string;
  isDominant: boolean;
}) {
  const pct = (prob * 100).toFixed(1);
  return (
    <div className={clsx(styles.regimeBar, isDominant && styles.regimeBarDominant)}>
      <div className={styles.regimeBarLabel}>
        <span className={styles.regimeEmoji}>{emoji}</span>
        <span className={styles.regimeName}>{name}</span>
        {isDominant && <span className={styles.dominantBadge}>dominant</span>}
      </div>
      <div className={styles.regimeTrack}>
        <div
          className={styles.regimeFill}
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
      <div className={styles.regimePct} style={{ color }}>
        {pct}%
      </div>
    </div>
  );
}

function EntropyGauge({ entropy }: { entropy: number }) {
  const pct = entropy * 100;
  const color = pct > 75 ? '#e74c3c' : pct > 50 ? '#e67e22' : '#2ecc71';
  const label = pct > 75 ? 'High — transition likely' : pct > 50 ? 'Moderate — monitor closely' : 'Low — regime stable';
  return (
    <div className={styles.entropyGauge}>
      <div className={styles.entropyLabel}>
        <span>Predictive Entropy</span>
        <span style={{ color, fontWeight: 700 }}>{pct.toFixed(0)}%</span>
      </div>
      <div className={styles.entropyTrack}>
        <div className={styles.entropyFill} style={{ width: `${pct}%`, background: color }} />
      </div>
      <div className={styles.entropyStatus} style={{ color }}>
        {label}
      </div>
    </div>
  );
}

function PortfolioAllocation({
  weights,
}: {
  weights: { equity: number; bonds: number; commodities: number; cash: number };
}) {
  const items = [
    { label: 'Equity', value: weights.equity, color: '#2ecc71' },
    { label: 'Bonds', value: weights.bonds, color: '#3498db' },
    { label: 'Commodities', value: weights.commodities, color: '#e67e22' },
    { label: 'Cash', value: weights.cash, color: '#95a5a6' },
  ];
  return (
    <div className={styles.allocation}>
      <div className={styles.allocationTitle}>📊 Regime-Conditioned Allocation</div>
      <div className={styles.allocationBar}>
        {items.map((item) => (
          <div
            key={item.label}
            className={styles.allocationSegment}
            style={{ width: `${(item.value * 100).toFixed(1)}%`, background: item.color }}
            title={`${item.label}: ${(item.value * 100).toFixed(0)}%`}
          />
        ))}
      </div>
      <div className={styles.allocationLegend}>
        {items.map((item) => (
          <div key={item.label} className={styles.allocationLegendItem}>
            <span className={styles.allocationDot} style={{ background: item.color }} />
            <span>{item.label}</span>
            <span className={styles.allocationPct}>{(item.value * 100).toFixed(0)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function MacroSlider({
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
        <span>{min}{unit}</span>
        <span>{max}{unit}</span>
      </div>
    </div>
  );
}

// ─── Transition Matrix Display ────────────────────────────────────────────────

const TRANSITION_MATRIX: number[][] = [
  [0.72, 0.20, 0.05, 0.03], // From Expansion
  [0.08, 0.55, 0.32, 0.05], // From Overheating
  [0.04, 0.06, 0.60, 0.30], // From Contraction
  [0.42, 0.07, 0.06, 0.45], // From Recovery
];

function TransitionMatrix({ dominantRegime }: { dominantRegime: number }) {
  return (
    <div className={styles.transitionMatrix}>
      <div className={styles.matrixTitle}>📐 Quarterly Transition Probabilities (from dominant regime)</div>
      <div className={styles.matrixGrid}>
        {REGIME_NAMES.map((name, j) => {
          const prob = TRANSITION_MATRIX[dominantRegime][j];
          const isHighest = TRANSITION_MATRIX[dominantRegime].indexOf(Math.max(...TRANSITION_MATRIX[dominantRegime])) === j;
          return (
            <div
              key={name}
              className={clsx(styles.matrixCell, isHighest && styles.matrixCellHighest)}
              style={{ borderColor: REGIME_COLORS[j] }}
            >
              <div className={styles.matrixCellEmoji}>{REGIME_EMOJIS[j]}</div>
              <div className={styles.matrixCellName}>{name}</div>
              <div className={styles.matrixCellProb} style={{ color: REGIME_COLORS[j] }}>
                {(prob * 100).toFixed(0)}%
              </div>
            </div>
          );
        })}
      </div>
      <p className={styles.matrixNote}>
        From <strong style={{ color: REGIME_COLORS[dominantRegime] }}>{REGIME_NAMES[dominantRegime]}</strong>: {(TRANSITION_MATRIX[dominantRegime][dominantRegime] * 100).toFixed(0)}% chance of staying this quarter.
      </p>
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────

export default function RegimeSimulator() {
  const [presetKey, setPresetKey] = useState<string>('expansion');
  const [params, setParams] = useState<MacroParams>(PRESETS.expansion.params);

  const setPreset = useCallback((key: string) => {
    setPresetKey(key);
    if (key !== 'custom') {
      setParams(PRESETS[key].params);
    }
  }, []);

  const updateParam = useCallback(
    (field: keyof MacroParams) => (val: number) => {
      setPresetKey('custom');
      setParams((prev) => ({ ...prev, [field]: val }));
    },
    [],
  );

  const result = useMemo(() => inferRegime(params), [params]);

  return (
    <div className={styles.simulator}>
      {/* Header */}
      <div className={styles.header}>
        <h2 className={styles.headerTitle}>🔍 Regime Shift Simulator</h2>
        <p className={styles.headerSubtitle}>
          Set macro conditions and observe how a World Model infers the{' '}
          <strong>hidden market regime</strong>, computes{' '}
          <strong>transition probabilities</strong>, detects{' '}
          <strong>early warning signals</strong>, and adjusts{' '}
          <strong>portfolio allocation</strong> accordingly.
        </p>
      </div>

      {/* Presets */}
      <div className={styles.scenarios}>
        <div className={styles.scenariosLabel}>Preset Scenarios:</div>
        <div className={styles.scenarioPills}>
          {Object.entries(PRESETS).map(([key, s]) => (
            <button
              key={key}
              className={clsx(styles.scenarioPill, presetKey === key && styles.scenarioPillActive)}
              onClick={() => setPreset(key)}
            >
              {s.label}
            </button>
          ))}
        </div>
        {PRESETS[presetKey] && (
          <p className={styles.scenarioDesc}>{PRESETS[presetKey].description}</p>
        )}
      </div>

      <div className={styles.layout}>
        {/* Left: Controls */}
        <div className={styles.leftPanel}>
          <div className={styles.controls}>
            <div className={styles.controlsTitle}>🌐 Macro Parameters</div>
            <MacroSlider
              label="GDP Growth (annual)"
              value={params.gdpGrowth}
              min={-5}
              max={6}
              step={0.1}
              unit="%"
              onChange={updateParam('gdpGrowth')}
            />
            <MacroSlider
              label="Inflation (CPI)"
              value={params.inflation}
              min={-2}
              max={12}
              step={0.1}
              unit="%"
              onChange={updateParam('inflation')}
            />
            <MacroSlider
              label="VIX (Implied Volatility)"
              value={params.vix}
              min={8}
              max={80}
              step={1}
              unit=""
              onChange={updateParam('vix')}
            />
            <MacroSlider
              label="HY Credit Spread"
              value={params.creditSpread}
              min={50}
              max={1200}
              step={10}
              unit="bps"
              onChange={updateParam('creditSpread')}
            />
            <MacroSlider
              label="Yield Curve Slope (10y−2y)"
              value={params.yieldCurveSlope}
              min={-100}
              max={250}
              step={5}
              unit="bps"
              onChange={updateParam('yieldCurveSlope')}
            />
          </div>

          <TransitionMatrix dominantRegime={result.dominantRegime} />
        </div>

        {/* Right: Results */}
        <div className={styles.rightPanel}>
          {/* Regime probabilities */}
          <div className={styles.resultCard}>
            <div className={styles.resultCardTitle}>🎯 Regime Posterior Probabilities</div>
            <div className={styles.regimeBars}>
              {REGIME_NAMES.map((name, i) => (
                <RegimeProbBar
                  key={name}
                  name={name}
                  prob={result.probs[i]}
                  color={REGIME_COLORS[i]}
                  emoji={REGIME_EMOJIS[i]}
                  isDominant={result.dominantRegime === i}
                />
              ))}
            </div>
          </div>

          {/* Entropy gauge */}
          <div className={styles.resultCard}>
            <EntropyGauge entropy={result.entropy} />
          </div>

          {/* Early warning */}
          {result.earlyWarning && (
            <div className={styles.warningCard}>
              {result.earlyWarning}
            </div>
          )}

          {/* Portfolio allocation */}
          <div className={styles.resultCard}>
            <PortfolioAllocation weights={result.portfolioWeights} />
          </div>

          {/* Insights */}
          <div className={styles.resultCard}>
            <div className={styles.insightsTitle}>🔍 Causal Insights</div>
            {result.insights.map((insight, i) => (
              <p key={i} className={styles.insightItem}>{insight}</p>
            ))}
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className={styles.footerNote}>
        <strong>Educational simulator.</strong> Regime probabilities are derived from a simplified
        causal scoring model for illustration. Real World Model regime inference uses deep Recurrent
        State Space Models trained on decades of multi-asset market data.
        See{' '}
        <a href="/docs/chapter-07" style={{ color: 'inherit', fontWeight: 600 }}>
          Chapter 7 — Regime Shifts and Hidden States
        </a>.
      </div>
    </div>
  );
}

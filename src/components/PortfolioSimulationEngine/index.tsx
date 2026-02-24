import { useState, useCallback, useRef } from 'react';
import type { ChangeEvent, FormEvent } from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

// ─── Types ────────────────────────────────────────────────────────────────────

interface AssetData {
  symbol: string;
  name: string;
  price: number;
  annualReturn: number; // percent
  annualVol: number;    // percent
  source: 'live' | 'demo';
}

interface PortfolioHolding {
  symbol: string;
  weight: number; // 0–100 percent
}

interface SimulationResult {
  medianPath: number[];
  p10Path: number[];
  p25Path: number[];
  p75Path: number[];
  p90Path: number[];
  metrics: {
    expectedReturn: number;
    annualizedVol: number;
    maxDrawdown: number;
    sharpeRatio: number;
    probPositive: number;
    recoveryMonths: number;
    resilienceScore: number;
  };
  regime: string;
  insights: string[];
}

// ─── Well-known stock characteristics (demo / fallback) ───────────────────────

const DEMO_ASSETS: Record<string, Omit<AssetData, 'source'>> = {
  AAPL:  { symbol: 'AAPL',  name: 'Apple Inc.',        price: 189.30, annualReturn: 14.2, annualVol: 22.8 },
  MSFT:  { symbol: 'MSFT',  name: 'Microsoft Corp.',   price: 415.20, annualReturn: 18.1, annualVol: 20.4 },
  GOOGL: { symbol: 'GOOGL', name: 'Alphabet Inc.',     price: 172.60, annualReturn: 16.3, annualVol: 24.1 },
  AMZN:  { symbol: 'AMZN',  name: 'Amazon.com Inc.',   price: 198.10, annualReturn: 20.5, annualVol: 27.3 },
  NVDA:  { symbol: 'NVDA',  name: 'NVIDIA Corp.',      price: 887.50, annualReturn: 48.3, annualVol: 52.6 },
  META:  { symbol: 'META',  name: 'Meta Platforms',    price: 502.40, annualReturn: 28.7, annualVol: 35.2 },
  TSLA:  { symbol: 'TSLA',  name: 'Tesla Inc.',        price: 175.80, annualReturn: 12.6, annualVol: 58.4 },
  JPM:   { symbol: 'JPM',   name: 'JPMorgan Chase',    price: 198.60, annualReturn: 15.8, annualVol: 19.2 },
  BRK_B: { symbol: 'BRK-B', name: 'Berkshire Hathaway',price: 366.40, annualReturn: 11.4, annualVol: 13.7 },
  SPY:   { symbol: 'SPY',   name: 'S&P 500 ETF',       price: 522.80, annualReturn: 10.8, annualVol: 14.9 },
  QQQ:   { symbol: 'QQQ',   name: 'Nasdaq 100 ETF',    price: 448.30, annualReturn: 15.2, annualVol: 19.3 },
  GLD:   { symbol: 'GLD',   name: 'Gold ETF',          price: 207.50, annualReturn:  6.4, annualVol: 12.8 },
};

// ─── Yahoo Finance fetch ──────────────────────────────────────────────────────

async function fetchYahooFinance(symbol: string): Promise<AssetData | null> {
  const proxies = [
    `https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?interval=1d&range=1y`,
    `https://corsproxy.io/?url=${encodeURIComponent(`https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?interval=1d&range=1y`)}`,
  ];

  for (const url of proxies) {
    try {
      const res = await fetch(url, { signal: AbortSignal.timeout(6000) });
      if (!res.ok) continue;
      const json = await res.json();
      const result = json?.chart?.result?.[0];
      if (!result) continue;

      const closes: number[] = result.indicators?.quote?.[0]?.close ?? [];
      const price: number = result.meta?.regularMarketPrice ?? closes[closes.length - 1] ?? 0;
      const longName: string = result.meta?.longName ?? result.meta?.shortName ?? symbol;

      // Compute daily returns
      const returns: number[] = [];
      for (let i = 1; i < closes.length; i++) {
        if (closes[i] && closes[i - 1]) {
          returns.push(closes[i] / closes[i - 1] - 1);
        }
      }
      if (returns.length < 30) continue;

      const mean = returns.reduce((s, r) => s + r, 0) / returns.length;
      const variance = returns.reduce((s, r) => s + (r - mean) ** 2, 0) / returns.length;
      const dailyVol = Math.sqrt(variance);

      const annualReturn = mean * 252 * 100;
      const annualVol = dailyVol * Math.sqrt(252) * 100;

      return { symbol, name: longName, price, annualReturn, annualVol, source: 'live' };
    } catch {
      // try next proxy
    }
  }
  return null;
}

// ─── World-Model Monte Carlo Simulation ──────────────────────────────────────

function runPortfolioSimulation(
  assets: AssetData[],
  holdings: PortfolioHolding[],
  timeHorizonMonths: number,
  nPaths = 5000,
): SimulationResult {
  // Build weighted portfolio drift and vol
  let portDrift = 0;
  let portVarMonthly = 0;

  for (const h of holdings) {
    const w = h.weight / 100;
    const asset = assets.find(a => a.symbol === h.symbol);
    if (!asset || w === 0) continue;
    const monthlyReturn = asset.annualReturn / 12 / 100;
    const monthlyVol = asset.annualVol / Math.sqrt(12) / 100;
    portDrift += w * monthlyReturn;
    portVarMonthly += w * w * monthlyVol * monthlyVol;
  }

  // Add cross-asset correlation benefit (diversification discount ~30%)
  portVarMonthly *= 0.7;
  const portMonthlyVol = Math.sqrt(portVarMonthly);

  // Detect regime from drift vs vol ratio (Sharpe-like)
  const annualDrift = portDrift * 12 * 100;
  const annualVol = portMonthlyVol * Math.sqrt(12) * 100;
  const sharpeApprox = annualVol > 0 ? (annualDrift - 4) / annualVol : 0;
  let regime: string;
  if (sharpeApprox > 0.8) regime = 'Expansion 🟢';
  else if (sharpeApprox > 0.3) regime = 'Recovery 🟡';
  else if (sharpeApprox < -0.3) regime = 'Contraction 🔴';
  else regime = 'Overheating 🟠';

  // Run Monte Carlo paths using GBM with regime-dependent vol multiplier
  const volMultiplier = regime.startsWith('Contraction') ? 1.4 : regime.startsWith('Overheating') ? 1.2 : 1.0;
  const adjVol = portMonthlyVol * volMultiplier;

  // Box-Muller normal random number generator
  function randn(): number {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  // Store final values and full paths for percentile calculation
  const allPaths: number[][] = [];
  const maxDrawdowns: number[] = [];
  const recoveryMonthsList: number[] = [];

  for (let i = 0; i < nPaths; i++) {
    const path: number[] = [100];
    let peak = 100;
    let maxDD = 0;
    let recovered = false;
    let recoveryMonth = timeHorizonMonths;

    for (let t = 1; t <= timeHorizonMonths; t++) {
      const shock = randn();
      // GBM: S(t+1) = S(t) * exp((μ - σ²/2)*dt + σ*√dt*Z)
      const monthlyLogReturn = (portDrift - 0.5 * adjVol * adjVol) + adjVol * shock;
      const newVal = path[t - 1] * Math.exp(monthlyLogReturn);
      path.push(newVal);

      if (newVal > peak) {
        peak = newVal;
        if (!recovered && t > 1 && path.some(v => v < path[0])) recovered = true;
      }
      const dd = ((peak - newVal) / peak) * 100;
      if (dd > maxDD) maxDD = dd;
    }

    // Recovery time: first time value returns to 100 after going below
    const troughIdx = path.indexOf(Math.min(...path));
    if (path[troughIdx] < 100) {
      for (let t = troughIdx + 1; t <= timeHorizonMonths; t++) {
        if (path[t] >= 100) { recoveryMonth = t; break; }
      }
    } else {
      recoveryMonth = 0;
    }

    allPaths.push(path);
    maxDrawdowns.push(maxDD);
    recoveryMonthsList.push(recoveryMonth);
  }

  // Compute percentile paths
  function percentilePath(p: number): number[] {
    return Array.from({ length: timeHorizonMonths + 1 }, (_, t) => {
      const vals = allPaths.map(path => path[t]).sort((a, b) => a - b);
      const idx = Math.floor((p / 100) * (vals.length - 1));
      return vals[idx];
    });
  }

  const medianPath = percentilePath(50);
  const p10Path = percentilePath(10);
  const p25Path = percentilePath(25);
  const p75Path = percentilePath(75);
  const p90Path = percentilePath(90);

  // Aggregate metrics
  const finalVals = allPaths.map(p => p[timeHorizonMonths]);
  finalVals.sort((a, b) => a - b);
  const medianFinal = finalVals[Math.floor(finalVals.length / 2)];
  const expectedReturn = (medianFinal / 100 - 1) * 100;
  const probPositive = (finalVals.filter(v => v > 100).length / nPaths) * 100;

  const sortedDD = [...maxDrawdowns].sort((a, b) => a - b);
  const p50DD = sortedDD[Math.floor(sortedDD.length / 2)];

  const sortedRecovery = [...recoveryMonthsList].sort((a, b) => a - b);
  const medianRecovery = sortedRecovery[Math.floor(sortedRecovery.length / 2)];

  const riskFreeRate = 4.0;
  const annualizedReturn = (expectedReturn / timeHorizonMonths) * 12;
  const sharpeRatio = annualVol > 0 ? (annualizedReturn - riskFreeRate) / annualVol : 0;

  // Resilience score (0–100): high return, low drawdown, high prob positive
  const resilienceScore = Math.round(
    Math.min(100, Math.max(0,
      probPositive * 0.4 +
      Math.max(0, 50 - p50DD) * 0.4 +
      Math.min(100, (sharpeRatio + 0.5) * 50) * 0.2
    ))
  );

  // Insights
  const insights: string[] = [];

  if (annualDrift > 12) {
    insights.push(`Strong expected portfolio drift of ${annualDrift.toFixed(1)}%/yr suggests high-growth allocation — verify concentration risk.`);
  } else if (annualDrift < 4) {
    insights.push(`Low expected drift (${annualDrift.toFixed(1)}%/yr) may lag inflation — consider increasing growth exposure.`);
  }
  if (annualVol > 25) {
    insights.push(`High annualised volatility (${annualVol.toFixed(1)}%) — the portfolio is sensitive to market regime shifts. Consider diversification.`);
  } else if (annualVol < 12) {
    insights.push(`Low volatility (${annualVol.toFixed(1)}%/yr) reflects conservative allocation. Risk-adjusted returns may be strong.`);
  }
  if (p50DD > 20) {
    insights.push(`Median simulated drawdown of ${p50DD.toFixed(1)}% suggests material downside exposure. Hedging or rebalancing may improve resilience.`);
  }
  if (probPositive > 75) {
    insights.push(`${probPositive.toFixed(0)}% of simulated paths end above break-even — the World Model forecasts a favourable probability distribution.`);
  } else if (probPositive < 50) {
    insights.push(`Only ${probPositive.toFixed(0)}% of simulated paths end positive — reconsider asset allocation or reduce time horizon.`);
  }
  if (insights.length === 0) {
    insights.push(`Balanced portfolio characteristics. The simulation shows moderate return potential with manageable downside risk.`);
  }

  return {
    medianPath, p10Path, p25Path, p75Path, p90Path,
    metrics: {
      expectedReturn: Math.round(expectedReturn * 10) / 10,
      annualizedVol: Math.round(annualVol * 10) / 10,
      maxDrawdown: Math.round(p50DD * 10) / 10,
      sharpeRatio: Math.round(sharpeRatio * 100) / 100,
      probPositive: Math.round(probPositive),
      recoveryMonths: medianRecovery,
      resilienceScore,
    },
    regime,
    insights,
  };
}

// ─── Fan Chart (SVG) ─────────────────────────────────────────────────────────

interface FanChartProps {
  result: SimulationResult;
  timeHorizon: number;
}

function FanChart({ result, timeHorizon }: FanChartProps) {
  const { medianPath, p10Path, p25Path, p75Path, p90Path } = result;
  const W = 680;
  const H = 260;
  const PAD = { top: 20, right: 28, bottom: 40, left: 52 };
  const chartW = W - PAD.left - PAD.right;
  const chartH = H - PAD.top - PAD.bottom;

  const allVals = [...p10Path, ...p90Path];
  const yMin = Math.min(...allVals) * 0.98;
  const yMax = Math.max(...allVals) * 1.02;

  const xs = (i: number) => PAD.left + (i / timeHorizon) * chartW;
  const ys = (v: number) => PAD.top + chartH - ((v - yMin) / (yMax - yMin)) * chartH;

  const makePath = (pts: number[]) =>
    pts.map((v, i) => `${i === 0 ? 'M' : 'L'}${xs(i).toFixed(1)},${ys(v).toFixed(1)}`).join(' ');

  const makeArea = (upper: number[], lower: number[]) =>
    makePath(upper) +
    ' ' +
    lower.slice().reverse().map((v, i) => `L${xs(timeHorizon - i).toFixed(1)},${ys(v).toFixed(1)}`).join(' ') +
    ' Z';

  const baselineY = ys(100);
  const yTicks = 5;
  const yStep = (yMax - yMin) / yTicks;
  const gridLines = Array.from({ length: yTicks + 1 }, (_, i) => yMin + i * yStep);
  const xTickInterval = timeHorizon <= 12 ? 2 : timeHorizon <= 18 ? 3 : 6;
  const xTicks = Array.from(
    { length: Math.floor(timeHorizon / xTickInterval) + 1 },
    (_, i) => Math.min(i * xTickInterval, timeHorizon),
  );

  const finalMedian = medianPath[timeHorizon];
  const isPos = finalMedian >= 100;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className={styles.chart} aria-label="Portfolio simulation fan chart" role="img">
      <rect x="0" y="0" width={W} height={H} fill="#0a1628" rx="8" />
      <rect x={PAD.left} y={PAD.top} width={chartW} height={chartH} fill="#0d1f3c" rx="4" />

      {/* Grid */}
      {gridLines.map((val, i) => (
        <g key={i}>
          <line x1={PAD.left} y1={ys(val)} x2={PAD.left + chartW} y2={ys(val)} stroke="#1e3a5f" strokeWidth="1" />
          <text x={PAD.left - 6} y={ys(val) + 4} textAnchor="end" fontSize="9" fill="#5b7fa6">{val.toFixed(0)}</text>
        </g>
      ))}

      {/* Baseline */}
      <line x1={PAD.left} y1={baselineY} x2={PAD.left + chartW} y2={baselineY} stroke="#334d6e" strokeWidth="1.5" strokeDasharray="4 3" />
      <text x={PAD.left - 6} y={baselineY + 4} textAnchor="end" fontSize="9" fill="#4d90e8">100</text>

      {/* P10–P90 band */}
      <path d={makeArea(p90Path, p10Path)} fill="rgba(77,144,232,0.10)" />
      {/* P25–P75 band */}
      <path d={makeArea(p75Path, p25Path)} fill="rgba(77,144,232,0.22)" />

      {/* Band borders */}
      {[p10Path, p90Path].map((band, bi) => (
        <path key={bi} d={makePath(band)} fill="none" stroke="rgba(77,144,232,0.35)" strokeWidth="1" strokeDasharray="3 2" />
      ))}
      {[p25Path, p75Path].map((band, bi) => (
        <path key={bi} d={makePath(band)} fill="none" stroke="rgba(77,144,232,0.55)" strokeWidth="1" />
      ))}

      {/* Median path */}
      <path d={makePath(medianPath)} fill="none" stroke={isPos ? '#22c55e' : '#ef4444'} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />

      {/* Dots */}
      <circle cx={xs(0)} cy={ys(100)} r="4" fill="#4d90e8" />
      <circle cx={xs(timeHorizon)} cy={ys(finalMedian)} r="5" fill={isPos ? '#22c55e' : '#ef4444'} />
      <text x={xs(timeHorizon) + 8} y={ys(finalMedian) + 4} fontSize="10" fill={isPos ? '#22c55e' : '#ef4444'} fontWeight="bold">
        {finalMedian.toFixed(1)}
      </text>

      {/* X-axis ticks */}
      {xTicks.map(m => (
        <g key={m}>
          <line x1={xs(m)} y1={PAD.top + chartH} x2={xs(m)} y2={PAD.top + chartH + 4} stroke="#5b7fa6" strokeWidth="1" />
          <text x={xs(m)} y={PAD.top + chartH + 15} textAnchor="middle" fontSize="9" fill="#5b7fa6">{m}m</text>
        </g>
      ))}

      {/* Axis labels */}
      <text x={PAD.left + chartW / 2} y={H - 4} textAnchor="middle" fontSize="10" fill="#5b7fa6">Time Horizon (months)</text>
      <text x={12} y={PAD.top + chartH / 2} textAnchor="middle" fontSize="10" fill="#5b7fa6" transform={`rotate(-90, 12, ${PAD.top + chartH / 2})`}>Portfolio Value</text>

      {/* Legend */}
      <g transform={`translate(${PAD.left + 8}, ${PAD.top + 6})`}>
        <line x1="0" y1="6" x2="18" y2="6" stroke={isPos ? '#22c55e' : '#ef4444'} strokeWidth="2.5" />
        <text x="22" y="10" fontSize="9" fill="#93a3b8">Median Path</text>
        <rect x="90" y="0" width="18" height="12" fill="rgba(77,144,232,0.22)" stroke="rgba(77,144,232,0.55)" strokeWidth="0.5" />
        <text x="112" y="10" fontSize="9" fill="#93a3b8">P25–P75</text>
        <rect x="165" y="0" width="18" height="12" fill="rgba(77,144,232,0.10)" stroke="rgba(77,144,232,0.35)" strokeWidth="0.5" />
        <text x="187" y="10" fontSize="9" fill="#93a3b8">P10–P90</text>
      </g>
    </svg>
  );
}

// ─── Metric Card ─────────────────────────────────────────────────────────────

function MetricCard({ label, value, unit, sentiment }: {
  label: string; value: string | number; unit?: string; sentiment: 'positive' | 'negative' | 'neutral';
}) {
  return (
    <div className={clsx(styles.metricCard, styles[`metric_${sentiment}`])}>
      <div className={styles.metricLabel}>{label}</div>
      <div className={styles.metricValue}>
        {value}{unit && <span className={styles.metricUnit}>{unit}</span>}
      </div>
    </div>
  );
}

// ─── Holding Row ──────────────────────────────────────────────────────────────

function HoldingRow({
  holding, asset, onWeightChange, onRemove,
}: {
  holding: PortfolioHolding;
  asset: AssetData | undefined;
  onWeightChange: (w: number) => void;
  onRemove: () => void;
}) {
  const pct = holding.weight;
  return (
    <div className={styles.holdingRow}>
      <div className={styles.holdingSymbol}>
        <span className={styles.holdingTicker}>{holding.symbol}</span>
        {asset && <span className={styles.holdingName}>{asset.name}</span>}
        {asset && (
          <span className={clsx(styles.holdingSource, asset.source === 'live' ? styles.sourceLive : styles.sourceDemo)}>
            {asset.source === 'live' ? '● Live' : '○ Demo'}
          </span>
        )}
      </div>
      <div className={styles.holdingControls}>
        <div className={styles.holdingSlider}>
          <div className={styles.sliderTrack}>
            <div className={styles.sliderFill} style={{ width: `${pct}%` }} />
            <input
              type="range" min={0} max={100} step={5} value={pct}
              onChange={(e: ChangeEvent<HTMLInputElement>) => onWeightChange(parseInt(e.target.value))}
              className={styles.sliderInput}
              aria-label={`Weight for ${holding.symbol}`}
            />
          </div>
        </div>
        <span className={styles.holdingWeight}>{pct}%</span>
        {asset && (
          <span className={styles.holdingPrice}>
            ${asset.price.toFixed(2)}
          </span>
        )}
        <button className={styles.removeBtn} onClick={onRemove} aria-label={`Remove ${holding.symbol}`}>×</button>
      </div>
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────

const DEFAULT_HOLDINGS: PortfolioHolding[] = [
  { symbol: 'AAPL', weight: 30 },
  { symbol: 'MSFT', weight: 25 },
  { symbol: 'SPY',  weight: 45 },
];

const QUICK_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'SPY', 'QQQ', 'GLD', 'JPM', 'TSLA', 'BRK-B'];

export default function PortfolioSimulationEngine() {
  const [holdings, setHoldings] = useState<PortfolioHolding[]>(DEFAULT_HOLDINGS);
  const [assets, setAssets] = useState<AssetData[]>(() =>
    DEFAULT_HOLDINGS.map(h => ({ ...DEMO_ASSETS[h.symbol] ?? DEMO_ASSETS.SPY, symbol: h.symbol, source: 'demo' as const }))
  );
  const [timeHorizon, setTimeHorizon] = useState(12);
  const [result, setResult] = useState<SimulationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [symbolInput, setSymbolInput] = useState('');
  const [fetchStatus, setFetchStatus] = useState<string>('');
  const inputRef = useRef<HTMLInputElement>(null);

  const totalWeight = holdings.reduce((s, h) => s + h.weight, 0);

  // Normalize weights so they sum to 100
  const normalizeWeights = useCallback((hs: PortfolioHolding[]): PortfolioHolding[] => {
    const total = hs.reduce((s, h) => s + h.weight, 0);
    if (total === 0) return hs.map(h => ({ ...h, weight: Math.round(100 / hs.length) }));
    return hs.map(h => ({ ...h, weight: Math.round((h.weight / total) * 100) }));
  }, []);

  const addSymbol = useCallback(async (raw: string) => {
    const symbol = raw.trim().toUpperCase().replace(/_/g, '-');
    if (!symbol || holdings.some(h => h.symbol === symbol)) return;

    setFetchStatus(`Fetching ${symbol}…`);

    // Try Yahoo Finance, fall back to demo
    let asset: AssetData;
    const live = await fetchYahooFinance(symbol);
    if (live) {
      asset = live;
      setFetchStatus(`✓ Live data loaded for ${symbol}`);
    } else {
      const demo = DEMO_ASSETS[symbol] ?? DEMO_ASSETS[symbol.replace('-', '_')];
      if (demo) {
        asset = { ...demo, source: 'demo' };
        setFetchStatus(`Using demo data for ${symbol} (Yahoo Finance unavailable)`);
      } else {
        // Unknown symbol: create synthetic asset with market-average characteristics
        asset = {
          symbol, name: symbol, price: 100,
          annualReturn: 10 + (Math.random() * 8 - 4),
          annualVol: 20 + (Math.random() * 10 - 5),
          source: 'demo',
        };
        setFetchStatus(`Using synthetic data for ${symbol}`);
      }
    }

    setAssets(prev => [...prev.filter(a => a.symbol !== symbol), asset]);
    setHoldings(prev => {
      const newHoldings = [...prev, { symbol, weight: 10 }];
      return normalizeWeights(newHoldings);
    });
    setSymbolInput('');
    setResult(null);
  }, [holdings, normalizeWeights]);

  const handleAddSubmit = useCallback((e: FormEvent) => {
    e.preventDefault();
    if (symbolInput) addSymbol(symbolInput);
  }, [symbolInput, addSymbol]);

  const removeHolding = useCallback((symbol: string) => {
    setHoldings(prev => {
      const next = prev.filter(h => h.symbol !== symbol);
      return next.length > 0 ? normalizeWeights(next) : next;
    });
    setResult(null);
  }, [normalizeWeights]);

  const updateWeight = useCallback((symbol: string, weight: number) => {
    setHoldings(prev => prev.map(h => h.symbol === symbol ? { ...h, weight } : h));
    setResult(null);
  }, []);

  const runSimulation = useCallback(async () => {
    if (holdings.length === 0) return;
    setLoading(true);
    setFetchStatus('Running World Model simulation…');

    // Refresh live data for current holdings
    const updatedAssets = [...assets];
    for (const h of holdings) {
      if (!updatedAssets.find(a => a.symbol === h.symbol)) {
        const live = await fetchYahooFinance(h.symbol);
        if (live) {
          updatedAssets.push(live);
        } else {
          const demo = DEMO_ASSETS[h.symbol];
          if (demo) updatedAssets.push({ ...demo, source: 'demo' });
        }
      }
    }
    setAssets(updatedAssets);

    // Normalise holdings for simulation
    const normHoldings = normalizeWeights(holdings.filter(h => h.weight > 0));

    // Small delay to allow UI to update
    await new Promise(r => setTimeout(r, 40));

    const sim = runPortfolioSimulation(updatedAssets, normHoldings, timeHorizon, 4000);
    setResult(sim);
    setFetchStatus('');
    setLoading(false);
  }, [holdings, assets, timeHorizon, normalizeWeights]);

  return (
    <div className={styles.engine}>
      {/* Header */}
      <div className={styles.header}>
        <h2 className={styles.headerTitle}>📊 Portfolio Simulation Engine</h2>
        <p className={styles.headerSubtitle}>
          A <strong>World Model simulation engine</strong> that evaluates a portfolio not by replaying
          history but by generating a <strong>probability distribution over future trajectories</strong>.
          Enter stock symbols, set weights, and run the Monte Carlo world-model forecast.
        </p>
      </div>

      <div className={styles.body}>
        {/* Left column: portfolio builder */}
        <div className={styles.leftCol}>
          {/* Symbol Search */}
          <div className={styles.section}>
            <div className={styles.sectionTitle}>🔍 Add Symbols</div>
            <form onSubmit={handleAddSubmit} className={styles.addForm}>
              <input
                ref={inputRef}
                className={styles.symbolInput}
                type="text"
                placeholder="e.g. AAPL, NVDA, BRK-B…"
                value={symbolInput}
                onChange={e => setSymbolInput(e.target.value.toUpperCase())}
                aria-label="Stock symbol input"
              />
              <button type="submit" className={styles.addBtn} disabled={!symbolInput}>Add</button>
            </form>
            <div className={styles.quickSymbols}>
              {QUICK_SYMBOLS.map(sym => (
                <button
                  key={sym}
                  className={clsx(styles.quickBtn, holdings.some(h => h.symbol === sym) && styles.quickBtnActive)}
                  onClick={() => addSymbol(sym)}
                  disabled={holdings.some(h => h.symbol === sym)}
                >
                  {sym}
                </button>
              ))}
            </div>
          </div>

          {/* Portfolio Holdings */}
          <div className={styles.section}>
            <div className={styles.sectionTitle}>
              📁 Portfolio Holdings
              <span className={clsx(styles.weightTotal, Math.abs(totalWeight - 100) > 5 ? styles.weightWarn : styles.weightOk)}>
                {totalWeight}% allocated
              </span>
            </div>
            {holdings.length === 0 && (
              <p className={styles.emptyMsg}>Add symbols above to build your portfolio.</p>
            )}
            {holdings.map(h => (
              <HoldingRow
                key={h.symbol}
                holding={h}
                asset={assets.find(a => a.symbol === h.symbol)}
                onWeightChange={w => updateWeight(h.symbol, w)}
                onRemove={() => removeHolding(h.symbol)}
              />
            ))}
          </div>

          {/* Time Horizon */}
          <div className={styles.section}>
            <div className={styles.sectionTitle}>⏱ Time Horizon</div>
            <div className={styles.sliderRow}>
              <div className={styles.sliderLabel}>
                <span>Forecast horizon</span>
                <span className={styles.sliderValue}>{timeHorizon} months</span>
              </div>
              <div className={styles.sliderTrack}>
                <div className={styles.sliderFill} style={{ width: `${((timeHorizon - 3) / 33) * 100}%` }} />
                <input
                  type="range" min={3} max={36} step={3} value={timeHorizon}
                  onChange={e => { setTimeHorizon(parseInt(e.target.value)); setResult(null); }}
                  className={styles.sliderInput}
                  aria-label="Time horizon"
                />
              </div>
              <div className={styles.sliderMinMax}><span>3m</span><span>36m</span></div>
            </div>
          </div>

          <button
            className={styles.runBtn}
            onClick={runSimulation}
            disabled={loading || holdings.length === 0}
          >
            {loading ? '⚙ Simulating…' : '▶ Run World Model Simulation'}
          </button>

          {fetchStatus && <p className={styles.fetchStatus}>{fetchStatus}</p>}
        </div>

        {/* Right column: results */}
        <div className={styles.rightCol}>
          {!result && !loading && (
            <div className={styles.placeholder}>
              <div className={styles.placeholderIcon}>🌍</div>
              <p className={styles.placeholderText}>
                Build your portfolio on the left, then click <strong>Run World Model Simulation</strong> to generate a probability
                distribution of future portfolio trajectories.
              </p>
              <p className={styles.placeholderSub}>
                The engine uses price data from Yahoo Finance and runs a regime-aware Monte Carlo
                simulation to produce a full fan chart of possible futures.
              </p>
            </div>
          )}

          {loading && (
            <div className={styles.placeholder}>
              <div className={styles.loadingSpinner} />
              <p className={styles.placeholderText}>Running 4,000 simulated futures…</p>
            </div>
          )}

          {result && !loading && (
            <>
              {/* Regime Badge */}
              <div className={styles.regimeBadge}>
                Detected Regime: <strong>{result.regime}</strong>
              </div>

              {/* Metrics */}
              <div className={styles.metrics}>
                <MetricCard
                  label="Expected Return"
                  value={result.metrics.expectedReturn > 0 ? `+${result.metrics.expectedReturn}` : result.metrics.expectedReturn}
                  unit="%"
                  sentiment={result.metrics.expectedReturn > 5 ? 'positive' : result.metrics.expectedReturn < 0 ? 'negative' : 'neutral'}
                />
                <MetricCard
                  label="Annualised Vol"
                  value={result.metrics.annualizedVol}
                  unit="%"
                  sentiment={result.metrics.annualizedVol > 25 ? 'negative' : result.metrics.annualizedVol < 14 ? 'positive' : 'neutral'}
                />
                <MetricCard
                  label="Median Drawdown"
                  value={`−${result.metrics.maxDrawdown}`}
                  unit="%"
                  sentiment={result.metrics.maxDrawdown > 20 ? 'negative' : result.metrics.maxDrawdown < 8 ? 'positive' : 'neutral'}
                />
                <MetricCard
                  label="Sharpe Ratio"
                  value={result.metrics.sharpeRatio}
                  sentiment={result.metrics.sharpeRatio > 0.6 ? 'positive' : result.metrics.sharpeRatio < 0 ? 'negative' : 'neutral'}
                />
                <MetricCard
                  label="P(Positive)"
                  value={result.metrics.probPositive}
                  unit="%"
                  sentiment={result.metrics.probPositive > 65 ? 'positive' : result.metrics.probPositive < 45 ? 'negative' : 'neutral'}
                />
                <MetricCard
                  label="Recovery (median)"
                  value={result.metrics.recoveryMonths === 0 ? 'N/A' : `${result.metrics.recoveryMonths}m`}
                  sentiment="neutral"
                />
                <MetricCard
                  label="Resilience Score"
                  value={result.metrics.resilienceScore}
                  unit="/100"
                  sentiment={result.metrics.resilienceScore > 65 ? 'positive' : result.metrics.resilienceScore < 40 ? 'negative' : 'neutral'}
                />
              </div>

              {/* Fan Chart */}
              <FanChart result={result} timeHorizon={timeHorizon} />

              {/* Insights */}
              <div className={styles.insights}>
                <div className={styles.insightsTitle}>🔍 World Model Insights</div>
                {result.insights.map((ins, i) => (
                  <p key={i} className={styles.insightItem}>{ins}</p>
                ))}
              </div>

              {/* Resilience bar */}
              <div className={styles.resilienceBar}>
                <div className={styles.resilienceLabel}>Portfolio Resilience Score</div>
                <div className={styles.resilienceTrack}>
                  <div
                    className={clsx(
                      styles.resilienceFill,
                      result.metrics.resilienceScore > 65 ? styles.resilienceGood :
                      result.metrics.resilienceScore < 40 ? styles.resiliencePoor :
                      styles.resilienceMed
                    )}
                    style={{ width: `${result.metrics.resilienceScore}%` }}
                  />
                </div>
                <div className={styles.resilienceValue}>{result.metrics.resilienceScore} / 100</div>
              </div>

              <div className={styles.simNote}>
                ✓ 4,000 Monte Carlo paths · ✓ Regime-aware GBM dynamics · ✓ Yahoo Finance price data
              </div>
            </>
          )}
        </div>
      </div>

      <div className={styles.footerNote}>
        <strong>Educational tool.</strong> Live prices sourced from Yahoo Finance where CORS permits;
        demo data used as fallback. Simulation uses simplified geometric Brownian motion with
        regime-dependent volatility scaling. Not investment advice.
      </div>
    </div>
  );
}

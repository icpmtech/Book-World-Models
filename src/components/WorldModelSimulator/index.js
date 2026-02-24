"use strict";
var __assign = (this && this.__assign) || function () {
    __assign = Object.assign || function(t) {
        for (var s, i = 1, n = arguments.length; i < n; i++) {
            s = arguments[i];
            for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p))
                t[p] = s[p];
        }
        return t;
    };
    return __assign.apply(this, arguments);
};
var __spreadArray = (this && this.__spreadArray) || function (to, from, pack) {
    if (pack || arguments.length === 2) for (var i = 0, l = from.length, ar; i < l; i++) {
        if (ar || !(i in from)) {
            if (!ar) ar = Array.prototype.slice.call(from, 0, i);
            ar[i] = from[i];
        }
    }
    return to.concat(ar || Array.prototype.slice.call(from));
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.default = WorldModelSimulator;
var react_1 = require("react");
var clsx_1 = require("clsx");
var styles_module_css_1 = require("./styles.module.css");
// ─── Scenario Presets ────────────────────────────────────────────────────────
var SCENARIOS = {
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
function simulateMarket(p) {
    var inflation = p.inflation, rateChange = p.rateChange, liquidityIndex = p.liquidityIndex, timeHorizon = p.timeHorizon;
    // Annual expected return components
    var baseReturn = 7.0;
    var rateImpact = -rateChange * 3.2;
    var inflationImpact = -(Math.abs(inflation - 2.0)) * 0.9;
    var liquidityImpact = ((liquidityIndex - 50) / 50) * 5.0;
    var expectedAnnualReturn = baseReturn + rateImpact + inflationImpact + liquidityImpact;
    // Monthly volatility
    var baseVol = 2.2;
    var excessVol = Math.abs(rateChange) * 0.6 +
        Math.max(0, inflation - 4) * 0.35 +
        Math.max(0, 35 - liquidityIndex) * 0.06;
    var monthlyVol = baseVol + excessVol;
    var trajectory = [100];
    var upperBand = [100];
    var lowerBand = [100];
    for (var m = 1; m <= timeHorizon; m++) {
        var prev = trajectory[m - 1];
        // Initial shock (front-loaded rate effect, decays over ~4 months)
        var shockDecay = Math.exp(-m / 4);
        var initialShock = rateChange > 0 ? rateChange * -1.8 * shockDecay : rateChange * -1.2 * shockDecay;
        var monthlyReturn = expectedAnnualReturn / 12 + initialShock / timeHorizon;
        trajectory.push(prev * (1 + monthlyReturn / 100));
        // Uncertainty band widens with sqrt(t)
        var band = monthlyVol * Math.sqrt(m) * 0.55;
        upperBand.push(trajectory[m] * (1 + band / 100));
        lowerBand.push(trajectory[m] * (1 - band / 100));
    }
    // Metrics
    var finalLevel = trajectory[timeHorizon];
    var finalReturn = (finalLevel / 100 - 1) * 100;
    var annualizedReturn = (finalReturn / timeHorizon) * 12;
    var annualizedVol = monthlyVol * Math.sqrt(12);
    var peak = 100;
    var maxDrawdown = 0;
    for (var _i = 0, trajectory_1 = trajectory; _i < trajectory_1.length; _i++) {
        var level = trajectory_1[_i];
        if (level > peak)
            peak = level;
        var dd = ((peak - level) / peak) * 100;
        if (dd > maxDrawdown)
            maxDrawdown = dd;
    }
    var riskFree = 4.0;
    var sharpeRatio = annualizedVol > 0 ? (annualizedReturn - riskFree) / annualizedVol : 0;
    var spread = upperBand[timeHorizon] - lowerBand[timeHorizon];
    var probabilityPositive = spread > 0
        ? Math.min(95, Math.max(5, 50 + (finalReturn / spread) * 50))
        : finalReturn > 0
            ? 75
            : 25;
    // Contextual insights
    var insights = [];
    if (rateChange > 2) {
        insights.push("Aggressive rate hikes (".concat(rateChange.toFixed(1), "%) create significant front-loaded equity pressure through P/E compression and rising discount rates."));
    }
    else if (rateChange > 0) {
        insights.push("Moderate rate increases (".concat(rateChange.toFixed(1), "%) weigh on growth stocks but may signal economic confidence, supporting cyclical sectors."));
    }
    else if (rateChange < -1) {
        insights.push("Rate cuts (".concat(rateChange.toFixed(1), "%) typically provide stimulus to equities, though at this depth they may signal serious economic concern."));
    }
    if (inflation > 6) {
        insights.push("Inflation at ".concat(inflation.toFixed(1), "% significantly above target erodes real returns and compresses equity multiples, especially in duration-sensitive sectors."));
    }
    else if (inflation < 0) {
        insights.push("Deflationary conditions (".concat(inflation.toFixed(1), "%) signal demand collapse and increase recession probability materially."));
    }
    else if (Math.abs(inflation - 2.0) < 0.8) {
        insights.push("Inflation near 2% target (".concat(inflation.toFixed(1), "%) is a supportive backdrop for risk assets and central bank policy stability."));
    }
    if (liquidityIndex < 25) {
        insights.push("Severely stressed liquidity (index: ".concat(liquidityIndex, ") amplifies all risks and may trigger forced selling cascades."));
    }
    else if (liquidityIndex > 65) {
        insights.push("High liquidity (index: ".concat(liquidityIndex, ") reduces risk premium and supports asset prices across the spectrum."));
    }
    if (maxDrawdown > 15) {
        insights.push("Expect a peak drawdown of ~".concat(maxDrawdown.toFixed(1), "% \u2014 sizing and hedging should reflect this potential loss."));
    }
    if (insights.length === 0) {
        insights.push("Balanced conditions produce a relatively moderate expected return with manageable downside risk.");
    }
    return {
        trajectory: trajectory,
        upperBand: upperBand,
        lowerBand: lowerBand,
        metrics: {
            expectedReturn: Math.round(finalReturn * 10) / 10,
            annualizedVol: Math.round(annualizedVol * 10) / 10,
            maxDrawdown: Math.round(maxDrawdown * 10) / 10,
            sharpeRatio: Math.round(sharpeRatio * 100) / 100,
            probabilityPositive: Math.round(probabilityPositive),
        },
        insights: insights,
    };
}
// ─── LLM Response Generator ───────────────────────────────────────────────────
function generateLLMResponse(p) {
    var inflation = p.inflation, rateChange = p.rateChange, liquidityIndex = p.liquidityIndex, timeHorizon = p.timeHorizon;
    var lines = [];
    if (rateChange > 2) {
        lines.push("With aggressive rate increases of ".concat(rateChange.toFixed(1), "%, we would expect significant pressure on equity valuations, particularly in rate-sensitive sectors such as real estate, utilities, and long-duration technology stocks."));
        lines.push("Historically, aggressive rate hiking cycles have been associated with elevated market volatility, potential earnings downgrades, and \u2014 in some cases \u2014 recession.");
    }
    else if (rateChange > 0.25) {
        lines.push("Moderate rate increases of ".concat(rateChange.toFixed(1), "% may reflect central bank confidence in economic growth. While this creates some headwinds for equities \u2014 particularly growth-oriented names \u2014 value sectors and financials often benefit."));
    }
    else if (rateChange < -1.5) {
        lines.push("Significant rate cuts of ".concat(Math.abs(rateChange).toFixed(1), "% are typically associated with economic stimulus, which can support equities over the medium term. However, such cuts usually indicate underlying economic weakness that may weigh on earnings in the near term."));
    }
    else if (Math.abs(rateChange) <= 0.25) {
        lines.push("A stable rate environment generally provides a neutral-to-positive backdrop for equities, allowing companies to plan investments with predictable financing costs.");
    }
    else {
        lines.push("A modest rate adjustment of ".concat(rateChange.toFixed(1), "% is unlikely to materially alter the macro backdrop for markets."));
    }
    if (inflation > 6) {
        lines.push("Inflation at ".concat(inflation.toFixed(1), "% represents a significant challenge for central banks and markets. Above-target inflation historically compresses equity multiples via higher discount rates, while eroding consumer purchasing power."));
    }
    else if (inflation > 3.5) {
        lines.push("Inflation running above ".concat(inflation.toFixed(1), "% may prompt additional policy response and could keep real bond yields negative, complicating asset allocation."));
    }
    else if (inflation < 0) {
        lines.push("Deflationary conditions (".concat(inflation.toFixed(1), "%) raise concerns about demand weakness. While superficially positive for consumers, deflation can trap economies in low-growth spirals, as Japan experienced in the 1990s."));
    }
    if (liquidityIndex < 30) {
        lines.push("Low market liquidity conditions increase the risk of sharp price dislocations. In thin markets, selling pressure tends to be amplified, and bid-ask spreads widen materially.");
    }
    else if (liquidityIndex > 70) {
        lines.push("Favorable liquidity conditions are generally supportive of asset prices, reducing risk premiums and facilitating efficient price discovery.");
    }
    lines.push("Over a ".concat(timeHorizon, "-month horizon, the interplay of these factors \u2014 monetary policy trajectory, inflation dynamics, and market liquidity \u2014 will likely be the key determinants of market direction. That said, markets may overshoot or undershoot fundamentals in the short term."));
    lines.push("*This analysis reflects historical patterns and general economic reasoning. It does not constitute investment advice, and actual market outcomes may differ significantly from historical analogues.*");
    return lines;
}
function SimChart(_a) {
    var trajectory = _a.trajectory, upperBand = _a.upperBand, lowerBand = _a.lowerBand, timeHorizon = _a.timeHorizon;
    var W = 660;
    var H = 260;
    var PAD = { top: 20, right: 24, bottom: 40, left: 52 };
    var chartW = W - PAD.left - PAD.right;
    var chartH = H - PAD.top - PAD.bottom;
    var allValues = __spreadArray(__spreadArray(__spreadArray([], trajectory, true), upperBand, true), lowerBand, true);
    var yMin = Math.min.apply(Math, allValues) * 0.985;
    var yMax = Math.max.apply(Math, allValues) * 1.015;
    var xScale = function (i) { return PAD.left + (i / timeHorizon) * chartW; };
    var yScale = function (v) { return PAD.top + chartH - ((v - yMin) / (yMax - yMin)) * chartH; };
    var bandPath = upperBand
        .map(function (v, i) { return "".concat(i === 0 ? 'M' : 'L').concat(xScale(i).toFixed(1), ",").concat(yScale(v).toFixed(1)); })
        .join(' ') +
        ' ' +
        lowerBand
            .slice()
            .reverse()
            .map(function (v, i) {
            var origIdx = timeHorizon - i;
            return "L".concat(xScale(origIdx).toFixed(1), ",").concat(yScale(v).toFixed(1));
        })
            .join(' ') +
        ' Z';
    var trajPath = trajectory
        .map(function (v, i) { return "".concat(i === 0 ? 'M' : 'L').concat(xScale(i).toFixed(1), ",").concat(yScale(v).toFixed(1)); })
        .join(' ');
    var baselineY = yScale(100);
    // Y-axis grid lines
    var yTicks = 5;
    var yStep = (yMax - yMin) / yTicks;
    var gridLines = Array.from({ length: yTicks + 1 }, function (_, i) { return yMin + i * yStep; });
    // X-axis ticks
    var xTickInterval = timeHorizon <= 12 ? 2 : timeHorizon <= 18 ? 3 : 4;
    var xTicks = Array.from({ length: Math.floor(timeHorizon / xTickInterval) + 1 }, function (_, i) {
        return Math.min(i * xTickInterval, timeHorizon);
    });
    var finalVal = trajectory[timeHorizon];
    var isPositive = finalVal >= 100;
    return (<svg viewBox={"0 0 ".concat(W, " ").concat(H)} className={styles_module_css_1.default.chart} aria-label="Market trajectory simulation chart" role="img">
      {/* Background */}
      <rect x="0" y="0" width={W} height={H} fill="#0a1628" rx="8"/>
      <rect x={PAD.left} y={PAD.top} width={chartW} height={chartH} fill="#0d1f3c" rx="4"/>

      {/* Grid lines */}
      {gridLines.map(function (val, i) {
            var y = yScale(val);
            return (<g key={i}>
            <line x1={PAD.left} y1={y} x2={PAD.left + chartW} y2={y} stroke="#1e3a5f" strokeWidth="1"/>
            <text x={PAD.left - 6} y={y + 4} textAnchor="end" fontSize="9" fill="#5b7fa6">
              {val.toFixed(0)}
            </text>
          </g>);
        })}

      {/* Baseline at 100 */}
      <line x1={PAD.left} y1={baselineY} x2={PAD.left + chartW} y2={baselineY} stroke="#334d6e" strokeWidth="1.5" strokeDasharray="4 3"/>
      <text x={PAD.left - 6} y={baselineY + 4} textAnchor="end" fontSize="9" fill="#4d90e8">
        100
      </text>

      {/* Confidence band */}
      <path d={bandPath} fill="rgba(77,144,232,0.15)"/>

      {/* Band borders */}
      <path d={upperBand.map(function (v, i) { return "".concat(i === 0 ? 'M' : 'L').concat(xScale(i).toFixed(1), ",").concat(yScale(v).toFixed(1)); }).join(' ')} fill="none" stroke="rgba(77,144,232,0.4)" strokeWidth="1" strokeDasharray="3 2"/>
      <path d={lowerBand.map(function (v, i) { return "".concat(i === 0 ? 'M' : 'L').concat(xScale(i).toFixed(1), ",").concat(yScale(v).toFixed(1)); }).join(' ')} fill="none" stroke="rgba(77,144,232,0.4)" strokeWidth="1" strokeDasharray="3 2"/>

      {/* Main trajectory */}
      <path d={trajPath} fill="none" stroke={isPositive ? '#22c55e' : '#ef4444'} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>

      {/* Start dot */}
      <circle cx={xScale(0)} cy={yScale(100)} r="4" fill="#4d90e8"/>

      {/* End dot */}
      <circle cx={xScale(timeHorizon)} cy={yScale(finalVal)} r="5" fill={isPositive ? '#22c55e' : '#ef4444'}/>

      {/* End label */}
      <text x={xScale(timeHorizon) + 8} y={yScale(finalVal) + 4} fontSize="10" fill={isPositive ? '#22c55e' : '#ef4444'} fontWeight="bold">
        {finalVal.toFixed(1)}
      </text>

      {/* X-axis ticks */}
      {xTicks.map(function (m) { return (<g key={m}>
          <line x1={xScale(m)} y1={PAD.top + chartH} x2={xScale(m)} y2={PAD.top + chartH + 4} stroke="#5b7fa6" strokeWidth="1"/>
          <text x={xScale(m)} y={PAD.top + chartH + 15} textAnchor="middle" fontSize="9" fill="#5b7fa6">
            {m}m
          </text>
        </g>); })}

      {/* Axis labels */}
      <text x={PAD.left + chartW / 2} y={H - 4} textAnchor="middle" fontSize="10" fill="#5b7fa6">
        Time Horizon (months)
      </text>
      <text x={12} y={PAD.top + chartH / 2} textAnchor="middle" fontSize="10" fill="#5b7fa6" transform={"rotate(-90, 12, ".concat(PAD.top + chartH / 2, ")")}>
        Market Index
      </text>

      {/* Legend */}
      <g transform={"translate(".concat(PAD.left + 8, ", ").concat(PAD.top + 6, ")")}>
        <line x1="0" y1="6" x2="18" y2="6" stroke={isPositive ? '#22c55e' : '#ef4444'} strokeWidth="2.5"/>
        <text x="22" y="10" fontSize="9" fill="#93a3b8">
          Expected Path
        </text>
        <rect x="60" y="0" width="18" height="12" fill="rgba(77,144,232,0.2)" stroke="rgba(77,144,232,0.5)" strokeWidth="0.5"/>
        <text x="82" y="10" fontSize="9" fill="#93a3b8">
          90% Confidence Interval
        </text>
      </g>
    </svg>);
}
// ─── Metric Card ──────────────────────────────────────────────────────────────
function MetricCard(_a) {
    var label = _a.label, value = _a.value, unit = _a.unit, sentiment = _a.sentiment;
    return (<div className={(0, clsx_1.default)(styles_module_css_1.default.metricCard, styles_module_css_1.default["metric_".concat(sentiment)])}>
      <div className={styles_module_css_1.default.metricLabel}>{label}</div>
      <div className={styles_module_css_1.default.metricValue}>
        {value}
        {unit && <span className={styles_module_css_1.default.metricUnit}>{unit}</span>}
      </div>
    </div>);
}
// ─── Slider ───────────────────────────────────────────────────────────────────
function Slider(_a) {
    var label = _a.label, value = _a.value, min = _a.min, max = _a.max, step = _a.step, unit = _a.unit, onChange = _a.onChange;
    var pct = ((value - min) / (max - min)) * 100;
    return (<div className={styles_module_css_1.default.sliderRow}>
      <div className={styles_module_css_1.default.sliderLabel}>
        <span>{label}</span>
        <span className={styles_module_css_1.default.sliderValue}>
          {value > 0 && min < 0 ? '+' : ''}
          {value.toFixed(step < 1 ? 1 : 0)}
          {unit}
        </span>
      </div>
      <div className={styles_module_css_1.default.sliderTrack}>
        <div className={styles_module_css_1.default.sliderFill} style={{ width: "".concat(pct, "%") }}/>
        <input type="range" min={min} max={max} step={step} value={value} onChange={function (e) { return onChange(parseFloat(e.target.value)); }} className={styles_module_css_1.default.sliderInput} aria-label={label}/>
      </div>
      <div className={styles_module_css_1.default.sliderMinMax}>
        <span>
          {min}
          {unit}
        </span>
        <span>
          {max}
          {unit}
        </span>
      </div>
    </div>);
}
// ─── Main Component ───────────────────────────────────────────────────────────
function WorldModelSimulator() {
    var _a = (0, react_1.useState)('rateHike'), scenarioKey = _a[0], setScenarioKey = _a[1];
    var _b = (0, react_1.useState)(SCENARIOS.rateHike.params), params = _b[0], setParams = _b[1];
    var _c = (0, react_1.useState)('compare'), activeTab = _c[0], setActiveTab = _c[1];
    var setScenario = (0, react_1.useCallback)(function (key) {
        setScenarioKey(key);
        if (key !== 'custom') {
            setParams(SCENARIOS[key].params);
        }
    }, []);
    var updateParam = (0, react_1.useCallback)(function (field) { return function (val) {
        setScenarioKey('custom');
        setParams(function (prev) {
            var _a;
            return (__assign(__assign({}, prev), (_a = {}, _a[field] = val, _a)));
        });
    }; }, []);
    var simulation = (0, react_1.useMemo)(function () { return simulateMarket(params); }, [params]);
    var llmResponse = (0, react_1.useMemo)(function () { return generateLLMResponse(params); }, [params]);
    var metrics = simulation.metrics;
    return (<div className={styles_module_css_1.default.simulator}>
      {/* Header */}
      <div className={styles_module_css_1.default.header}>
        <h2 className={styles_module_css_1.default.headerTitle}>🌍 World Model vs LLM Simulator</h2>
        <p className={styles_module_css_1.default.headerSubtitle}>
          Explore the difference between <strong>descriptive intelligence</strong> (LLM) and{' '}
          <strong>anticipatory intelligence</strong> (World Model). Set economic parameters and see
          how each approach responds.
        </p>
      </div>

      {/* Scenarios */}
      <div className={styles_module_css_1.default.scenarios}>
        <div className={styles_module_css_1.default.scenariosLabel}>Preset Scenarios:</div>
        <div className={styles_module_css_1.default.scenarioPills}>
          {Object.entries(SCENARIOS).map(function (_a) {
            var key = _a[0], s = _a[1];
            return (<button key={key} className={(0, clsx_1.default)(styles_module_css_1.default.scenarioPill, scenarioKey === key && styles_module_css_1.default.scenarioPillActive)} onClick={function () { return setScenario(key); }}>
              {s.label}
            </button>);
        })}
        </div>
        {SCENARIOS[scenarioKey] && (<p className={styles_module_css_1.default.scenarioDesc}>{SCENARIOS[scenarioKey].description}</p>)}
      </div>

      {/* Controls */}
      <div className={styles_module_css_1.default.controls}>
        <div className={styles_module_css_1.default.controlsTitle}>📊 Parameters</div>
        <Slider label="Inflation" value={params.inflation} min={-3} max={12} step={0.5} unit="%" onChange={updateParam('inflation')}/>
        <Slider label="Rate Change (total)" value={params.rateChange} min={-5} max={5} step={0.25} unit="%" onChange={updateParam('rateChange')}/>
        <Slider label="Liquidity Index" value={params.liquidityIndex} min={0} max={100} step={1} unit="" onChange={updateParam('liquidityIndex')}/>
        <Slider label="Time Horizon" value={params.timeHorizon} min={3} max={36} step={1} unit=" mo" onChange={updateParam('timeHorizon')}/>
      </div>

      {/* Tabs */}
      <div className={styles_module_css_1.default.tabs}>
        {[
            { key: 'compare', label: '↔ Side-by-Side' },
            { key: 'worldmodel', label: '🌍 World Model' },
            { key: 'llm', label: '💬 LLM Response' },
        ].map(function (_a) {
            var key = _a.key, label = _a.label;
            return (<button key={key} className={(0, clsx_1.default)(styles_module_css_1.default.tab, activeTab === key && styles_module_css_1.default.tabActive)} onClick={function () { return setActiveTab(key); }}>
            {label}
          </button>);
        })}
      </div>

      {/* Content */}
      <div className={styles_module_css_1.default.content}>
        {/* World Model Panel */}
        {(activeTab === 'worldmodel' || activeTab === 'compare') && (<div className={(0, clsx_1.default)(styles_module_css_1.default.panel, styles_module_css_1.default.wmPanel)}>
            <div className={styles_module_css_1.default.panelHeader}>
              <span className={styles_module_css_1.default.panelBadgeWm}>World Model</span>
              <span className={styles_module_css_1.default.panelHeaderTitle}>Anticipatory Simulation</span>
            </div>

            {/* Metrics */}
            <div className={styles_module_css_1.default.metrics}>
              <MetricCard label="Expected Return" value={metrics.expectedReturn > 0 ? "+".concat(metrics.expectedReturn) : metrics.expectedReturn} unit="%" sentiment={metrics.expectedReturn > 2 ? 'positive' : metrics.expectedReturn < -5 ? 'negative' : 'neutral'}/>
              <MetricCard label="Annualised Vol" value={metrics.annualizedVol} unit="%" sentiment={metrics.annualizedVol > 20 ? 'negative' : metrics.annualizedVol < 12 ? 'positive' : 'neutral'}/>
              <MetricCard label="Max Drawdown" value={"\u2212".concat(metrics.maxDrawdown)} unit="%" sentiment={metrics.maxDrawdown > 15 ? 'negative' : metrics.maxDrawdown < 6 ? 'positive' : 'neutral'}/>
              <MetricCard label="Sharpe Ratio" value={metrics.sharpeRatio} sentiment={metrics.sharpeRatio > 0.5 ? 'positive' : metrics.sharpeRatio < 0 ? 'negative' : 'neutral'}/>
              <MetricCard label="P(Positive)" value={metrics.probabilityPositive} unit="%" sentiment={metrics.probabilityPositive > 60 ? 'positive' : metrics.probabilityPositive < 40 ? 'negative' : 'neutral'}/>
            </div>

            {/* Chart */}
            <SimChart trajectory={simulation.trajectory} upperBand={simulation.upperBand} lowerBand={simulation.lowerBand} timeHorizon={params.timeHorizon}/>

            {/* Insights */}
            <div className={styles_module_css_1.default.insights}>
              <div className={styles_module_css_1.default.insightsTitle}>🔍 Causal Insights</div>
              {simulation.insights.map(function (insight, i) { return (<p key={i} className={styles_module_css_1.default.insightItem}>
                  {insight}
                </p>); })}
            </div>

            <div className={styles_module_css_1.default.wmNote}>
              ✓ Quantified probability distribution · ✓ Causal state propagation · ✓ Actionable metrics
            </div>
          </div>)}

        {/* LLM Panel */}
        {(activeTab === 'llm' || activeTab === 'compare') && (<div className={(0, clsx_1.default)(styles_module_css_1.default.panel, styles_module_css_1.default.llmPanel)}>
            <div className={styles_module_css_1.default.panelHeader}>
              <span className={styles_module_css_1.default.panelBadgeLlm}>LLM</span>
              <span className={styles_module_css_1.default.panelHeaderTitle}>Descriptive Response</span>
            </div>

            <div className={styles_module_css_1.default.llmOutput}>
              <div className={styles_module_css_1.default.llmPrompt}>
                <strong>Prompt:</strong> Inflation is {params.inflation.toFixed(1)}%, rates change by{' '}
                {params.rateChange > 0 ? '+' : ''}
                {params.rateChange.toFixed(2)}%, liquidity index is {params.liquidityIndex}. What
                happens to the equity market over {params.timeHorizon} months?
              </div>
              <div className={styles_module_css_1.default.llmResponse}>
                {llmResponse.map(function (line, i) { return (<p key={i} className={styles_module_css_1.default.llmLine}>
                    {line}
                  </p>); })}
              </div>
            </div>

            <div className={styles_module_css_1.default.llmLimitations}>
              <div className={styles_module_css_1.default.limitationsTitle}>⚠ Limitations of this approach</div>
              <ul className={styles_module_css_1.default.limitationsList}>
                <li>No probability distributions — cannot quantify uncertainty</li>
                <li>No state simulation — reverts to historical averages</li>
                <li>Cannot model interventions or counterfactuals</li>
                <li>No causal chain — only statistical co-occurrence</li>
                <li>Does not adapt to novel initial conditions outside training distribution</li>
              </ul>
            </div>
          </div>)}
      </div>

      {/* Footer note */}
      <div className={styles_module_css_1.default.footerNote}>
        <strong>Educational simulator.</strong> The World Model simulation uses simplified causal
        dynamics for illustration. Real financial world models require large-scale training on
        historical market microstructure data.
      </div>
    </div>);
}

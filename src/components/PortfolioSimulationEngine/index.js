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
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g = Object.create((typeof Iterator === "function" ? Iterator : Object).prototype);
    return g.next = verb(0), g["throw"] = verb(1), g["return"] = verb(2), typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (g && (g = 0, op[0] && (_ = 0)), _) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
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
exports.default = PortfolioSimulationEngine;
var react_1 = require("react");
var clsx_1 = require("clsx");
var styles_module_css_1 = require("./styles.module.css");
// ─── Well-known stock characteristics (demo / fallback) ───────────────────────
var DEMO_ASSETS = {
    AAPL: { symbol: 'AAPL', name: 'Apple Inc.', price: 189.30, annualReturn: 14.2, annualVol: 22.8 },
    MSFT: { symbol: 'MSFT', name: 'Microsoft Corp.', price: 415.20, annualReturn: 18.1, annualVol: 20.4 },
    GOOGL: { symbol: 'GOOGL', name: 'Alphabet Inc.', price: 172.60, annualReturn: 16.3, annualVol: 24.1 },
    AMZN: { symbol: 'AMZN', name: 'Amazon.com Inc.', price: 198.10, annualReturn: 20.5, annualVol: 27.3 },
    NVDA: { symbol: 'NVDA', name: 'NVIDIA Corp.', price: 887.50, annualReturn: 48.3, annualVol: 52.6 },
    META: { symbol: 'META', name: 'Meta Platforms', price: 502.40, annualReturn: 28.7, annualVol: 35.2 },
    TSLA: { symbol: 'TSLA', name: 'Tesla Inc.', price: 175.80, annualReturn: 12.6, annualVol: 58.4 },
    JPM: { symbol: 'JPM', name: 'JPMorgan Chase', price: 198.60, annualReturn: 15.8, annualVol: 19.2 },
    BRK_B: { symbol: 'BRK-B', name: 'Berkshire Hathaway', price: 366.40, annualReturn: 11.4, annualVol: 13.7 },
    SPY: { symbol: 'SPY', name: 'S&P 500 ETF', price: 522.80, annualReturn: 10.8, annualVol: 14.9 },
    QQQ: { symbol: 'QQQ', name: 'Nasdaq 100 ETF', price: 448.30, annualReturn: 15.2, annualVol: 19.3 },
    GLD: { symbol: 'GLD', name: 'Gold ETF', price: 207.50, annualReturn: 6.4, annualVol: 12.8 },
};
// ─── Yahoo Finance fetch ──────────────────────────────────────────────────────
function fetchYahooFinance(symbol) {
    return __awaiter(this, void 0, void 0, function () {
        var proxies, _loop_1, _i, proxies_1, url, state_1;
        var _a, _b, _c, _d, _e, _f, _g, _h, _j, _k, _l, _m, _o;
        return __generator(this, function (_p) {
            switch (_p.label) {
                case 0:
                    proxies = [
                        "https://query1.finance.yahoo.com/v8/finance/chart/".concat(symbol, "?interval=1d&range=1y"),
                        "https://corsproxy.io/?url=".concat(encodeURIComponent("https://query1.finance.yahoo.com/v8/finance/chart/".concat(symbol, "?interval=1d&range=1y"))),
                    ];
                    _loop_1 = function (url) {
                        var res, json, result, closes, price, longName, returns, i, mean_1, variance, dailyVol, annualReturn, annualVol, _q;
                        return __generator(this, function (_r) {
                            switch (_r.label) {
                                case 0:
                                    _r.trys.push([0, 3, , 4]);
                                    return [4 /*yield*/, fetch(url, { signal: AbortSignal.timeout(6000) })];
                                case 1:
                                    res = _r.sent();
                                    if (!res.ok)
                                        return [2 /*return*/, "continue"];
                                    return [4 /*yield*/, res.json()];
                                case 2:
                                    json = _r.sent();
                                    result = (_b = (_a = json === null || json === void 0 ? void 0 : json.chart) === null || _a === void 0 ? void 0 : _a.result) === null || _b === void 0 ? void 0 : _b[0];
                                    if (!result)
                                        return [2 /*return*/, "continue"];
                                    closes = (_f = (_e = (_d = (_c = result.indicators) === null || _c === void 0 ? void 0 : _c.quote) === null || _d === void 0 ? void 0 : _d[0]) === null || _e === void 0 ? void 0 : _e.close) !== null && _f !== void 0 ? _f : [];
                                    price = (_j = (_h = (_g = result.meta) === null || _g === void 0 ? void 0 : _g.regularMarketPrice) !== null && _h !== void 0 ? _h : closes[closes.length - 1]) !== null && _j !== void 0 ? _j : 0;
                                    longName = (_o = (_l = (_k = result.meta) === null || _k === void 0 ? void 0 : _k.longName) !== null && _l !== void 0 ? _l : (_m = result.meta) === null || _m === void 0 ? void 0 : _m.shortName) !== null && _o !== void 0 ? _o : symbol;
                                    returns = [];
                                    for (i = 1; i < closes.length; i++) {
                                        if (closes[i] && closes[i - 1]) {
                                            returns.push(closes[i] / closes[i - 1] - 1);
                                        }
                                    }
                                    if (returns.length < 30)
                                        return [2 /*return*/, "continue"];
                                    mean_1 = returns.reduce(function (s, r) { return s + r; }, 0) / returns.length;
                                    variance = returns.reduce(function (s, r) { return s + Math.pow((r - mean_1), 2); }, 0) / returns.length;
                                    dailyVol = Math.sqrt(variance);
                                    annualReturn = mean_1 * 252 * 100;
                                    annualVol = dailyVol * Math.sqrt(252) * 100;
                                    return [2 /*return*/, { value: { symbol: symbol, name: longName, price: price, annualReturn: annualReturn, annualVol: annualVol, source: 'live' } }];
                                case 3:
                                    _q = _r.sent();
                                    return [3 /*break*/, 4];
                                case 4: return [2 /*return*/];
                            }
                        });
                    };
                    _i = 0, proxies_1 = proxies;
                    _p.label = 1;
                case 1:
                    if (!(_i < proxies_1.length)) return [3 /*break*/, 4];
                    url = proxies_1[_i];
                    return [5 /*yield**/, _loop_1(url)];
                case 2:
                    state_1 = _p.sent();
                    if (typeof state_1 === "object")
                        return [2 /*return*/, state_1.value];
                    _p.label = 3;
                case 3:
                    _i++;
                    return [3 /*break*/, 1];
                case 4: return [2 /*return*/, null];
            }
        });
    });
}
// ─── World-Model Monte Carlo Simulation ──────────────────────────────────────
function runPortfolioSimulation(assets, holdings, timeHorizonMonths, nPaths) {
    if (nPaths === void 0) { nPaths = 5000; }
    // Build weighted portfolio drift and vol
    var portDrift = 0;
    var portVarMonthly = 0;
    var _loop_2 = function (h) {
        var w = h.weight / 100;
        var asset = assets.find(function (a) { return a.symbol === h.symbol; });
        if (!asset || w === 0)
            return "continue";
        var monthlyReturn = asset.annualReturn / 12 / 100;
        var monthlyVol = asset.annualVol / Math.sqrt(12) / 100;
        portDrift += w * monthlyReturn;
        portVarMonthly += w * w * monthlyVol * monthlyVol;
    };
    for (var _i = 0, holdings_1 = holdings; _i < holdings_1.length; _i++) {
        var h = holdings_1[_i];
        _loop_2(h);
    }
    // Add cross-asset correlation benefit (diversification discount ~30%)
    portVarMonthly *= 0.7;
    var portMonthlyVol = Math.sqrt(portVarMonthly);
    // Detect regime from drift vs vol ratio (Sharpe-like)
    var annualDrift = portDrift * 12 * 100;
    var annualVol = portMonthlyVol * Math.sqrt(12) * 100;
    var sharpeApprox = annualVol > 0 ? (annualDrift - 4) / annualVol : 0;
    var regime;
    if (sharpeApprox > 0.8)
        regime = 'Expansion 🟢';
    else if (sharpeApprox > 0.3)
        regime = 'Recovery 🟡';
    else if (sharpeApprox < -0.3)
        regime = 'Contraction 🔴';
    else
        regime = 'Overheating 🟠';
    // Run Monte Carlo paths using GBM with regime-dependent vol multiplier
    var volMultiplier = regime.startsWith('Contraction') ? 1.4 : regime.startsWith('Overheating') ? 1.2 : 1.0;
    var adjVol = portMonthlyVol * volMultiplier;
    // Box-Muller normal random number generator
    function randn() {
        var u = 0, v = 0;
        while (u === 0)
            u = Math.random();
        while (v === 0)
            v = Math.random();
        return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }
    // Store final values and full paths for percentile calculation
    var allPaths = [];
    var maxDrawdowns = [];
    var recoveryMonthsList = [];
    var _loop_3 = function (i) {
        var path = [100];
        var peak = 100;
        var maxDD = 0;
        var recovered = false;
        var recoveryMonth = timeHorizonMonths;
        for (var t = 1; t <= timeHorizonMonths; t++) {
            var shock = randn();
            // GBM: S(t+1) = S(t) * exp((μ - σ²/2)*dt + σ*√dt*Z)
            var monthlyLogReturn = (portDrift - 0.5 * adjVol * adjVol) + adjVol * shock;
            var newVal = path[t - 1] * Math.exp(monthlyLogReturn);
            path.push(newVal);
            if (newVal > peak) {
                peak = newVal;
                if (!recovered && t > 1 && path.some(function (v) { return v < path[0]; }))
                    recovered = true;
            }
            var dd = ((peak - newVal) / peak) * 100;
            if (dd > maxDD)
                maxDD = dd;
        }
        // Recovery time: first time value returns to 100 after going below
        var troughIdx = path.indexOf(Math.min.apply(Math, path));
        if (path[troughIdx] < 100) {
            for (var t = troughIdx + 1; t <= timeHorizonMonths; t++) {
                if (path[t] >= 100) {
                    recoveryMonth = t;
                    break;
                }
            }
        }
        else {
            recoveryMonth = 0;
        }
        allPaths.push(path);
        maxDrawdowns.push(maxDD);
        recoveryMonthsList.push(recoveryMonth);
    };
    for (var i = 0; i < nPaths; i++) {
        _loop_3(i);
    }
    // Compute percentile paths
    function percentilePath(p) {
        return Array.from({ length: timeHorizonMonths + 1 }, function (_, t) {
            var vals = allPaths.map(function (path) { return path[t]; }).sort(function (a, b) { return a - b; });
            var idx = Math.floor((p / 100) * (vals.length - 1));
            return vals[idx];
        });
    }
    var medianPath = percentilePath(50);
    var p10Path = percentilePath(10);
    var p25Path = percentilePath(25);
    var p75Path = percentilePath(75);
    var p90Path = percentilePath(90);
    // Aggregate metrics
    var finalVals = allPaths.map(function (p) { return p[timeHorizonMonths]; });
    finalVals.sort(function (a, b) { return a - b; });
    var medianFinal = finalVals[Math.floor(finalVals.length / 2)];
    var expectedReturn = (medianFinal / 100 - 1) * 100;
    var probPositive = (finalVals.filter(function (v) { return v > 100; }).length / nPaths) * 100;
    var sortedDD = __spreadArray([], maxDrawdowns, true).sort(function (a, b) { return a - b; });
    var p50DD = sortedDD[Math.floor(sortedDD.length / 2)];
    var sortedRecovery = __spreadArray([], recoveryMonthsList, true).sort(function (a, b) { return a - b; });
    var medianRecovery = sortedRecovery[Math.floor(sortedRecovery.length / 2)];
    var riskFreeRate = 4.0;
    var annualizedReturn = (expectedReturn / timeHorizonMonths) * 12;
    var sharpeRatio = annualVol > 0 ? (annualizedReturn - riskFreeRate) / annualVol : 0;
    // Resilience score (0–100): high return, low drawdown, high prob positive
    var resilienceScore = Math.round(Math.min(100, Math.max(0, probPositive * 0.4 +
        Math.max(0, 50 - p50DD) * 0.4 +
        Math.min(100, (sharpeRatio + 0.5) * 50) * 0.2)));
    // Insights
    var insights = [];
    if (annualDrift > 12) {
        insights.push("Strong expected portfolio drift of ".concat(annualDrift.toFixed(1), "%/yr suggests high-growth allocation \u2014 verify concentration risk."));
    }
    else if (annualDrift < 4) {
        insights.push("Low expected drift (".concat(annualDrift.toFixed(1), "%/yr) may lag inflation \u2014 consider increasing growth exposure."));
    }
    if (annualVol > 25) {
        insights.push("High annualised volatility (".concat(annualVol.toFixed(1), "%) \u2014 the portfolio is sensitive to market regime shifts. Consider diversification."));
    }
    else if (annualVol < 12) {
        insights.push("Low volatility (".concat(annualVol.toFixed(1), "%/yr) reflects conservative allocation. Risk-adjusted returns may be strong."));
    }
    if (p50DD > 20) {
        insights.push("Median simulated drawdown of ".concat(p50DD.toFixed(1), "% suggests material downside exposure. Hedging or rebalancing may improve resilience."));
    }
    if (probPositive > 75) {
        insights.push("".concat(probPositive.toFixed(0), "% of simulated paths end above break-even \u2014 the World Model forecasts a favourable probability distribution."));
    }
    else if (probPositive < 50) {
        insights.push("Only ".concat(probPositive.toFixed(0), "% of simulated paths end positive \u2014 reconsider asset allocation or reduce time horizon."));
    }
    if (insights.length === 0) {
        insights.push("Balanced portfolio characteristics. The simulation shows moderate return potential with manageable downside risk.");
    }
    return {
        medianPath: medianPath,
        p10Path: p10Path,
        p25Path: p25Path,
        p75Path: p75Path,
        p90Path: p90Path,
        metrics: {
            expectedReturn: Math.round(expectedReturn * 10) / 10,
            annualizedVol: Math.round(annualVol * 10) / 10,
            maxDrawdown: Math.round(p50DD * 10) / 10,
            sharpeRatio: Math.round(sharpeRatio * 100) / 100,
            probPositive: Math.round(probPositive),
            recoveryMonths: medianRecovery,
            resilienceScore: resilienceScore,
        },
        regime: regime,
        insights: insights,
    };
}
function FanChart(_a) {
    var result = _a.result, timeHorizon = _a.timeHorizon;
    var medianPath = result.medianPath, p10Path = result.p10Path, p25Path = result.p25Path, p75Path = result.p75Path, p90Path = result.p90Path;
    var W = 680;
    var H = 260;
    var PAD = { top: 20, right: 28, bottom: 40, left: 52 };
    var chartW = W - PAD.left - PAD.right;
    var chartH = H - PAD.top - PAD.bottom;
    var allVals = __spreadArray(__spreadArray([], p10Path, true), p90Path, true);
    var yMin = Math.min.apply(Math, allVals) * 0.98;
    var yMax = Math.max.apply(Math, allVals) * 1.02;
    var xs = function (i) { return PAD.left + (i / timeHorizon) * chartW; };
    var ys = function (v) { return PAD.top + chartH - ((v - yMin) / (yMax - yMin)) * chartH; };
    var makePath = function (pts) {
        return pts.map(function (v, i) { return "".concat(i === 0 ? 'M' : 'L').concat(xs(i).toFixed(1), ",").concat(ys(v).toFixed(1)); }).join(' ');
    };
    var makeArea = function (upper, lower) {
        return makePath(upper) +
            ' ' +
            lower.slice().reverse().map(function (v, i) { return "L".concat(xs(timeHorizon - i).toFixed(1), ",").concat(ys(v).toFixed(1)); }).join(' ') +
            ' Z';
    };
    var baselineY = ys(100);
    var yTicks = 5;
    var yStep = (yMax - yMin) / yTicks;
    var gridLines = Array.from({ length: yTicks + 1 }, function (_, i) { return yMin + i * yStep; });
    var xTickInterval = timeHorizon <= 12 ? 2 : timeHorizon <= 18 ? 3 : 6;
    var xTicks = Array.from({ length: Math.floor(timeHorizon / xTickInterval) + 1 }, function (_, i) { return Math.min(i * xTickInterval, timeHorizon); });
    var finalMedian = medianPath[timeHorizon];
    var isPos = finalMedian >= 100;
    return (<svg viewBox={"0 0 ".concat(W, " ").concat(H)} className={styles_module_css_1.default.chart} aria-label="Portfolio simulation fan chart" role="img">
      <rect x="0" y="0" width={W} height={H} fill="#0a1628" rx="8"/>
      <rect x={PAD.left} y={PAD.top} width={chartW} height={chartH} fill="#0d1f3c" rx="4"/>

      {/* Grid */}
      {gridLines.map(function (val, i) { return (<g key={i}>
          <line x1={PAD.left} y1={ys(val)} x2={PAD.left + chartW} y2={ys(val)} stroke="#1e3a5f" strokeWidth="1"/>
          <text x={PAD.left - 6} y={ys(val) + 4} textAnchor="end" fontSize="9" fill="#5b7fa6">{val.toFixed(0)}</text>
        </g>); })}

      {/* Baseline */}
      <line x1={PAD.left} y1={baselineY} x2={PAD.left + chartW} y2={baselineY} stroke="#334d6e" strokeWidth="1.5" strokeDasharray="4 3"/>
      <text x={PAD.left - 6} y={baselineY + 4} textAnchor="end" fontSize="9" fill="#4d90e8">100</text>

      {/* P10–P90 band */}
      <path d={makeArea(p90Path, p10Path)} fill="rgba(77,144,232,0.10)"/>
      {/* P25–P75 band */}
      <path d={makeArea(p75Path, p25Path)} fill="rgba(77,144,232,0.22)"/>

      {/* Band borders */}
      {[p10Path, p90Path].map(function (band, bi) { return (<path key={bi} d={makePath(band)} fill="none" stroke="rgba(77,144,232,0.35)" strokeWidth="1" strokeDasharray="3 2"/>); })}
      {[p25Path, p75Path].map(function (band, bi) { return (<path key={bi} d={makePath(band)} fill="none" stroke="rgba(77,144,232,0.55)" strokeWidth="1"/>); })}

      {/* Median path */}
      <path d={makePath(medianPath)} fill="none" stroke={isPos ? '#22c55e' : '#ef4444'} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>

      {/* Dots */}
      <circle cx={xs(0)} cy={ys(100)} r="4" fill="#4d90e8"/>
      <circle cx={xs(timeHorizon)} cy={ys(finalMedian)} r="5" fill={isPos ? '#22c55e' : '#ef4444'}/>
      <text x={xs(timeHorizon) + 8} y={ys(finalMedian) + 4} fontSize="10" fill={isPos ? '#22c55e' : '#ef4444'} fontWeight="bold">
        {finalMedian.toFixed(1)}
      </text>

      {/* X-axis ticks */}
      {xTicks.map(function (m) { return (<g key={m}>
          <line x1={xs(m)} y1={PAD.top + chartH} x2={xs(m)} y2={PAD.top + chartH + 4} stroke="#5b7fa6" strokeWidth="1"/>
          <text x={xs(m)} y={PAD.top + chartH + 15} textAnchor="middle" fontSize="9" fill="#5b7fa6">{m}m</text>
        </g>); })}

      {/* Axis labels */}
      <text x={PAD.left + chartW / 2} y={H - 4} textAnchor="middle" fontSize="10" fill="#5b7fa6">Time Horizon (months)</text>
      <text x={12} y={PAD.top + chartH / 2} textAnchor="middle" fontSize="10" fill="#5b7fa6" transform={"rotate(-90, 12, ".concat(PAD.top + chartH / 2, ")")}>Portfolio Value</text>

      {/* Legend */}
      <g transform={"translate(".concat(PAD.left + 8, ", ").concat(PAD.top + 6, ")")}>
        <line x1="0" y1="6" x2="18" y2="6" stroke={isPos ? '#22c55e' : '#ef4444'} strokeWidth="2.5"/>
        <text x="22" y="10" fontSize="9" fill="#93a3b8">Median Path</text>
        <rect x="90" y="0" width="18" height="12" fill="rgba(77,144,232,0.22)" stroke="rgba(77,144,232,0.55)" strokeWidth="0.5"/>
        <text x="112" y="10" fontSize="9" fill="#93a3b8">P25–P75</text>
        <rect x="165" y="0" width="18" height="12" fill="rgba(77,144,232,0.10)" stroke="rgba(77,144,232,0.35)" strokeWidth="0.5"/>
        <text x="187" y="10" fontSize="9" fill="#93a3b8">P10–P90</text>
      </g>
    </svg>);
}
// ─── Metric Card ─────────────────────────────────────────────────────────────
function MetricCard(_a) {
    var label = _a.label, value = _a.value, unit = _a.unit, sentiment = _a.sentiment;
    return (<div className={(0, clsx_1.default)(styles_module_css_1.default.metricCard, styles_module_css_1.default["metric_".concat(sentiment)])}>
      <div className={styles_module_css_1.default.metricLabel}>{label}</div>
      <div className={styles_module_css_1.default.metricValue}>
        {value}{unit && <span className={styles_module_css_1.default.metricUnit}>{unit}</span>}
      </div>
    </div>);
}
// ─── Holding Row ──────────────────────────────────────────────────────────────
function HoldingRow(_a) {
    var holding = _a.holding, asset = _a.asset, onWeightChange = _a.onWeightChange, onRemove = _a.onRemove;
    var pct = holding.weight;
    return (<div className={styles_module_css_1.default.holdingRow}>
      <div className={styles_module_css_1.default.holdingSymbol}>
        <span className={styles_module_css_1.default.holdingTicker}>{holding.symbol}</span>
        {asset && <span className={styles_module_css_1.default.holdingName}>{asset.name}</span>}
        {asset && (<span className={(0, clsx_1.default)(styles_module_css_1.default.holdingSource, asset.source === 'live' ? styles_module_css_1.default.sourceLive : styles_module_css_1.default.sourceDemo)}>
            {asset.source === 'live' ? '● Live' : '○ Demo'}
          </span>)}
      </div>
      <div className={styles_module_css_1.default.holdingControls}>
        <div className={styles_module_css_1.default.holdingSlider}>
          <div className={styles_module_css_1.default.sliderTrack}>
            <div className={styles_module_css_1.default.sliderFill} style={{ width: "".concat(pct, "%") }}/>
            <input type="range" min={0} max={100} step={5} value={pct} onChange={function (e) { return onWeightChange(parseInt(e.target.value)); }} className={styles_module_css_1.default.sliderInput} aria-label={"Weight for ".concat(holding.symbol)}/>
          </div>
        </div>
        <span className={styles_module_css_1.default.holdingWeight}>{pct}%</span>
        {asset && (<span className={styles_module_css_1.default.holdingPrice}>
            ${asset.price.toFixed(2)}
          </span>)}
        <button className={styles_module_css_1.default.removeBtn} onClick={onRemove} aria-label={"Remove ".concat(holding.symbol)}>×</button>
      </div>
    </div>);
}
// ─── Main Component ───────────────────────────────────────────────────────────
var DEFAULT_HOLDINGS = [
    { symbol: 'AAPL', weight: 30 },
    { symbol: 'MSFT', weight: 25 },
    { symbol: 'SPY', weight: 45 },
];
var QUICK_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'SPY', 'QQQ', 'GLD', 'JPM', 'TSLA', 'BRK-B'];
function PortfolioSimulationEngine() {
    var _this = this;
    var _a = (0, react_1.useState)(DEFAULT_HOLDINGS), holdings = _a[0], setHoldings = _a[1];
    var _b = (0, react_1.useState)(function () {
        return DEFAULT_HOLDINGS.map(function (h) { var _a; return (__assign(__assign({}, (_a = DEMO_ASSETS[h.symbol]) !== null && _a !== void 0 ? _a : DEMO_ASSETS.SPY), { symbol: h.symbol, source: 'demo' })); });
    }), assets = _b[0], setAssets = _b[1];
    var _c = (0, react_1.useState)(12), timeHorizon = _c[0], setTimeHorizon = _c[1];
    var _d = (0, react_1.useState)(null), result = _d[0], setResult = _d[1];
    var _e = (0, react_1.useState)(false), loading = _e[0], setLoading = _e[1];
    var _f = (0, react_1.useState)(''), symbolInput = _f[0], setSymbolInput = _f[1];
    var _g = (0, react_1.useState)(''), fetchStatus = _g[0], setFetchStatus = _g[1];
    var inputRef = (0, react_1.useRef)(null);
    var totalWeight = holdings.reduce(function (s, h) { return s + h.weight; }, 0);
    // Normalize weights so they sum to 100
    var normalizeWeights = (0, react_1.useCallback)(function (hs) {
        var total = hs.reduce(function (s, h) { return s + h.weight; }, 0);
        if (total === 0)
            return hs.map(function (h) { return (__assign(__assign({}, h), { weight: Math.round(100 / hs.length) })); });
        return hs.map(function (h) { return (__assign(__assign({}, h), { weight: Math.round((h.weight / total) * 100) })); });
    }, []);
    var addSymbol = (0, react_1.useCallback)(function (raw) { return __awaiter(_this, void 0, void 0, function () {
        var symbol, asset, live, demo;
        var _a;
        return __generator(this, function (_b) {
            switch (_b.label) {
                case 0:
                    symbol = raw.trim().toUpperCase().replace(/_/g, '-');
                    if (!symbol || holdings.some(function (h) { return h.symbol === symbol; }))
                        return [2 /*return*/];
                    setFetchStatus("Fetching ".concat(symbol, "\u2026"));
                    return [4 /*yield*/, fetchYahooFinance(symbol)];
                case 1:
                    live = _b.sent();
                    if (live) {
                        asset = live;
                        setFetchStatus("\u2713 Live data loaded for ".concat(symbol));
                    }
                    else {
                        demo = (_a = DEMO_ASSETS[symbol]) !== null && _a !== void 0 ? _a : DEMO_ASSETS[symbol.replace('-', '_')];
                        if (demo) {
                            asset = __assign(__assign({}, demo), { source: 'demo' });
                            setFetchStatus("Using demo data for ".concat(symbol, " (Yahoo Finance unavailable)"));
                        }
                        else {
                            // Unknown symbol: create synthetic asset with market-average characteristics
                            asset = {
                                symbol: symbol,
                                name: symbol, price: 100,
                                annualReturn: 10 + (Math.random() * 8 - 4),
                                annualVol: 20 + (Math.random() * 10 - 5),
                                source: 'demo',
                            };
                            setFetchStatus("Using synthetic data for ".concat(symbol));
                        }
                    }
                    setAssets(function (prev) { return __spreadArray(__spreadArray([], prev.filter(function (a) { return a.symbol !== symbol; }), true), [asset], false); });
                    setHoldings(function (prev) {
                        var newHoldings = __spreadArray(__spreadArray([], prev, true), [{ symbol: symbol, weight: 10 }], false);
                        return normalizeWeights(newHoldings);
                    });
                    setSymbolInput('');
                    setResult(null);
                    return [2 /*return*/];
            }
        });
    }); }, [holdings, normalizeWeights]);
    var handleAddSubmit = (0, react_1.useCallback)(function (e) {
        e.preventDefault();
        if (symbolInput)
            addSymbol(symbolInput);
    }, [symbolInput, addSymbol]);
    var removeHolding = (0, react_1.useCallback)(function (symbol) {
        setHoldings(function (prev) {
            var next = prev.filter(function (h) { return h.symbol !== symbol; });
            return next.length > 0 ? normalizeWeights(next) : next;
        });
        setResult(null);
    }, [normalizeWeights]);
    var updateWeight = (0, react_1.useCallback)(function (symbol, weight) {
        setHoldings(function (prev) { return prev.map(function (h) { return h.symbol === symbol ? __assign(__assign({}, h), { weight: weight }) : h; }); });
        setResult(null);
    }, []);
    var runSimulation = (0, react_1.useCallback)(function () { return __awaiter(_this, void 0, void 0, function () {
        var updatedAssets, _loop_4, _i, holdings_2, h, normHoldings, sim;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    if (holdings.length === 0)
                        return [2 /*return*/];
                    setLoading(true);
                    setFetchStatus('Running World Model simulation…');
                    updatedAssets = __spreadArray([], assets, true);
                    _loop_4 = function (h) {
                        var live, demo;
                        return __generator(this, function (_b) {
                            switch (_b.label) {
                                case 0:
                                    if (!!updatedAssets.find(function (a) { return a.symbol === h.symbol; })) return [3 /*break*/, 2];
                                    return [4 /*yield*/, fetchYahooFinance(h.symbol)];
                                case 1:
                                    live = _b.sent();
                                    if (live) {
                                        updatedAssets.push(live);
                                    }
                                    else {
                                        demo = DEMO_ASSETS[h.symbol];
                                        if (demo)
                                            updatedAssets.push(__assign(__assign({}, demo), { source: 'demo' }));
                                    }
                                    _b.label = 2;
                                case 2: return [2 /*return*/];
                            }
                        });
                    };
                    _i = 0, holdings_2 = holdings;
                    _a.label = 1;
                case 1:
                    if (!(_i < holdings_2.length)) return [3 /*break*/, 4];
                    h = holdings_2[_i];
                    return [5 /*yield**/, _loop_4(h)];
                case 2:
                    _a.sent();
                    _a.label = 3;
                case 3:
                    _i++;
                    return [3 /*break*/, 1];
                case 4:
                    setAssets(updatedAssets);
                    normHoldings = normalizeWeights(holdings.filter(function (h) { return h.weight > 0; }));
                    // Small delay to allow UI to update
                    return [4 /*yield*/, new Promise(function (r) { return setTimeout(r, 40); })];
                case 5:
                    // Small delay to allow UI to update
                    _a.sent();
                    sim = runPortfolioSimulation(updatedAssets, normHoldings, timeHorizon, 4000);
                    setResult(sim);
                    setFetchStatus('');
                    setLoading(false);
                    return [2 /*return*/];
            }
        });
    }); }, [holdings, assets, timeHorizon, normalizeWeights]);
    return (<div className={styles_module_css_1.default.engine}>
      {/* Header */}
      <div className={styles_module_css_1.default.header}>
        <h2 className={styles_module_css_1.default.headerTitle}>📊 Portfolio Simulation Engine</h2>
        <p className={styles_module_css_1.default.headerSubtitle}>
          A <strong>World Model simulation engine</strong> that evaluates a portfolio not by replaying
          history but by generating a <strong>probability distribution over future trajectories</strong>.
          Enter stock symbols, set weights, and run the Monte Carlo world-model forecast.
        </p>
      </div>

      <div className={styles_module_css_1.default.body}>
        {/* Left column: portfolio builder */}
        <div className={styles_module_css_1.default.leftCol}>
          {/* Symbol Search */}
          <div className={styles_module_css_1.default.section}>
            <div className={styles_module_css_1.default.sectionTitle}>🔍 Add Symbols</div>
            <form onSubmit={handleAddSubmit} className={styles_module_css_1.default.addForm}>
              <input ref={inputRef} className={styles_module_css_1.default.symbolInput} type="text" placeholder="e.g. AAPL, NVDA, BRK-B…" value={symbolInput} onChange={function (e) { return setSymbolInput(e.target.value.toUpperCase()); }} aria-label="Stock symbol input"/>
              <button type="submit" className={styles_module_css_1.default.addBtn} disabled={!symbolInput}>Add</button>
            </form>
            <div className={styles_module_css_1.default.quickSymbols}>
              {QUICK_SYMBOLS.map(function (sym) { return (<button key={sym} className={(0, clsx_1.default)(styles_module_css_1.default.quickBtn, holdings.some(function (h) { return h.symbol === sym; }) && styles_module_css_1.default.quickBtnActive)} onClick={function () { return addSymbol(sym); }} disabled={holdings.some(function (h) { return h.symbol === sym; })}>
                  {sym}
                </button>); })}
            </div>
          </div>

          {/* Portfolio Holdings */}
          <div className={styles_module_css_1.default.section}>
            <div className={styles_module_css_1.default.sectionTitle}>
              📁 Portfolio Holdings
              <span className={(0, clsx_1.default)(styles_module_css_1.default.weightTotal, Math.abs(totalWeight - 100) > 5 ? styles_module_css_1.default.weightWarn : styles_module_css_1.default.weightOk)}>
                {totalWeight}% allocated
              </span>
            </div>
            {holdings.length === 0 && (<p className={styles_module_css_1.default.emptyMsg}>Add symbols above to build your portfolio.</p>)}
            {holdings.map(function (h) { return (<HoldingRow key={h.symbol} holding={h} asset={assets.find(function (a) { return a.symbol === h.symbol; })} onWeightChange={function (w) { return updateWeight(h.symbol, w); }} onRemove={function () { return removeHolding(h.symbol); }}/>); })}
          </div>

          {/* Time Horizon */}
          <div className={styles_module_css_1.default.section}>
            <div className={styles_module_css_1.default.sectionTitle}>⏱ Time Horizon</div>
            <div className={styles_module_css_1.default.sliderRow}>
              <div className={styles_module_css_1.default.sliderLabel}>
                <span>Forecast horizon</span>
                <span className={styles_module_css_1.default.sliderValue}>{timeHorizon} months</span>
              </div>
              <div className={styles_module_css_1.default.sliderTrack}>
                <div className={styles_module_css_1.default.sliderFill} style={{ width: "".concat(((timeHorizon - 3) / 33) * 100, "%") }}/>
                <input type="range" min={3} max={36} step={3} value={timeHorizon} onChange={function (e) { setTimeHorizon(parseInt(e.target.value)); setResult(null); }} className={styles_module_css_1.default.sliderInput} aria-label="Time horizon"/>
              </div>
              <div className={styles_module_css_1.default.sliderMinMax}><span>3m</span><span>36m</span></div>
            </div>
          </div>

          <button className={styles_module_css_1.default.runBtn} onClick={runSimulation} disabled={loading || holdings.length === 0}>
            {loading ? '⚙ Simulating…' : '▶ Run World Model Simulation'}
          </button>

          {fetchStatus && <p className={styles_module_css_1.default.fetchStatus}>{fetchStatus}</p>}
        </div>

        {/* Right column: results */}
        <div className={styles_module_css_1.default.rightCol}>
          {!result && !loading && (<div className={styles_module_css_1.default.placeholder}>
              <div className={styles_module_css_1.default.placeholderIcon}>🌍</div>
              <p className={styles_module_css_1.default.placeholderText}>
                Build your portfolio on the left, then click <strong>Run World Model Simulation</strong> to generate a probability
                distribution of future portfolio trajectories.
              </p>
              <p className={styles_module_css_1.default.placeholderSub}>
                The engine uses price data from Yahoo Finance and runs a regime-aware Monte Carlo
                simulation to produce a full fan chart of possible futures.
              </p>
            </div>)}

          {loading && (<div className={styles_module_css_1.default.placeholder}>
              <div className={styles_module_css_1.default.loadingSpinner}/>
              <p className={styles_module_css_1.default.placeholderText}>Running 4,000 simulated futures…</p>
            </div>)}

          {result && !loading && (<>
              {/* Regime Badge */}
              <div className={styles_module_css_1.default.regimeBadge}>
                Detected Regime: <strong>{result.regime}</strong>
              </div>

              {/* Metrics */}
              <div className={styles_module_css_1.default.metrics}>
                <MetricCard label="Expected Return" value={result.metrics.expectedReturn > 0 ? "+".concat(result.metrics.expectedReturn) : result.metrics.expectedReturn} unit="%" sentiment={result.metrics.expectedReturn > 5 ? 'positive' : result.metrics.expectedReturn < 0 ? 'negative' : 'neutral'}/>
                <MetricCard label="Annualised Vol" value={result.metrics.annualizedVol} unit="%" sentiment={result.metrics.annualizedVol > 25 ? 'negative' : result.metrics.annualizedVol < 14 ? 'positive' : 'neutral'}/>
                <MetricCard label="Median Drawdown" value={"\u2212".concat(result.metrics.maxDrawdown)} unit="%" sentiment={result.metrics.maxDrawdown > 20 ? 'negative' : result.metrics.maxDrawdown < 8 ? 'positive' : 'neutral'}/>
                <MetricCard label="Sharpe Ratio" value={result.metrics.sharpeRatio} sentiment={result.metrics.sharpeRatio > 0.6 ? 'positive' : result.metrics.sharpeRatio < 0 ? 'negative' : 'neutral'}/>
                <MetricCard label="P(Positive)" value={result.metrics.probPositive} unit="%" sentiment={result.metrics.probPositive > 65 ? 'positive' : result.metrics.probPositive < 45 ? 'negative' : 'neutral'}/>
                <MetricCard label="Recovery (median)" value={result.metrics.recoveryMonths === 0 ? 'N/A' : "".concat(result.metrics.recoveryMonths, "m")} sentiment="neutral"/>
                <MetricCard label="Resilience Score" value={result.metrics.resilienceScore} unit="/100" sentiment={result.metrics.resilienceScore > 65 ? 'positive' : result.metrics.resilienceScore < 40 ? 'negative' : 'neutral'}/>
              </div>

              {/* Fan Chart */}
              <FanChart result={result} timeHorizon={timeHorizon}/>

              {/* Insights */}
              <div className={styles_module_css_1.default.insights}>
                <div className={styles_module_css_1.default.insightsTitle}>🔍 World Model Insights</div>
                {result.insights.map(function (ins, i) { return (<p key={i} className={styles_module_css_1.default.insightItem}>{ins}</p>); })}
              </div>

              {/* Resilience bar */}
              <div className={styles_module_css_1.default.resilienceBar}>
                <div className={styles_module_css_1.default.resilienceLabel}>Portfolio Resilience Score</div>
                <div className={styles_module_css_1.default.resilienceTrack}>
                  <div className={(0, clsx_1.default)(styles_module_css_1.default.resilienceFill, result.metrics.resilienceScore > 65 ? styles_module_css_1.default.resilienceGood :
                result.metrics.resilienceScore < 40 ? styles_module_css_1.default.resiliencePoor :
                    styles_module_css_1.default.resilienceMed)} style={{ width: "".concat(result.metrics.resilienceScore, "%") }}/>
                </div>
                <div className={styles_module_css_1.default.resilienceValue}>{result.metrics.resilienceScore} / 100</div>
              </div>

              <div className={styles_module_css_1.default.simNote}>
                ✓ 4,000 Monte Carlo paths · ✓ Regime-aware GBM dynamics · ✓ Yahoo Finance price data
              </div>
            </>)}
        </div>
      </div>

      <div className={styles_module_css_1.default.footerNote}>
        <strong>Educational tool.</strong> Live prices sourced from Yahoo Finance where CORS permits;
        demo data used as fallback. Simulation uses simplified geometric Brownian motion with
        regime-dependent volatility scaling. Not investment advice.
      </div>
    </div>);
}

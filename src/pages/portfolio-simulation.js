"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.default = PortfolioSimulationPage;
var Layout_1 = require("@theme/Layout");
var Link_1 = require("@docusaurus/Link");
var PortfolioSimulationEngine_1 = require("@site/src/components/PortfolioSimulationEngine");
function PortfolioSimulationPage() {
    return (<Layout_1.default title="Portfolio Simulation Engine" description="World Model Portfolio Simulation Engine — generate probability distributions over future portfolio trajectories using Yahoo Finance price data.">
      {/* Page header */}
      <div style={{
            background: 'linear-gradient(135deg, #0a1628 0%, #1a4b8c 40%, #0d2d5e 100%)',
            padding: '2rem 1rem 1.5rem',
            textAlign: 'center',
            color: '#fff',
        }}>
        <h1 style={{ fontSize: '2rem', margin: '0 0 0.4rem', color: '#fff' }}>
          📊 Portfolio Simulation Engine
        </h1>
        <p style={{ fontSize: '1.05rem', opacity: 0.88, margin: '0 0 0.75rem', fontStyle: 'italic' }}>
          World Model Monte Carlo Forecasting with Yahoo Finance Price Data
        </p>
        <div style={{ fontSize: '0.9rem', opacity: 0.75 }}>
          Companion to{' '}
          <Link_1.default to="/docs/chapter-09" style={{ color: '#93c5fd' }}>
            Chapter 9 — Scenario Generation and Counterfactual Futures
          </Link_1.default>
          {' '}·{' '}
          <Link_1.default to="/docs/chapter-08" style={{ color: '#93c5fd' }}>
            Chapter 8 — Portfolio Simulation Engines
          </Link_1.default>
        </div>
      </div>

      <main style={{ paddingTop: '1.5rem' }}>
        <PortfolioSimulationEngine_1.default />
      </main>
    </Layout_1.default>);
}

import type { ReactNode } from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import PortfolioSimulationEngine from '@site/src/components/PortfolioSimulationEngine';

export default function PortfolioSimulationPage(): ReactNode {
  return (
    <Layout
      title="Portfolio Simulation Engine"
      description="World Model Portfolio Simulation Engine — generate probability distributions over future portfolio trajectories using Yahoo Finance price data."
    >
      {/* Page header */}
      <div
        style={{
          background: 'linear-gradient(135deg, #0a1628 0%, #1a4b8c 40%, #0d2d5e 100%)',
          padding: '2rem 1rem 1.5rem',
          textAlign: 'center',
          color: '#fff',
        }}
      >
        <h1 style={{ fontSize: '2rem', margin: '0 0 0.4rem', color: '#fff' }}>
          📊 Portfolio Simulation Engine
        </h1>
        <p style={{ fontSize: '1.05rem', opacity: 0.88, margin: '0 0 0.75rem', fontStyle: 'italic' }}>
          World Model Monte Carlo Forecasting with Yahoo Finance Price Data
        </p>
        <div style={{ fontSize: '0.9rem', opacity: 0.75 }}>
          Companion to{' '}
          <Link to="/docs/chapter-09" style={{ color: '#93c5fd' }}>
            Chapter 9 — Scenario Generation and Counterfactual Futures
          </Link>
          {' '}·{' '}
          <Link to="/docs/chapter-08" style={{ color: '#93c5fd' }}>
            Chapter 8 — Portfolio Simulation Engines
          </Link>
        </div>
      </div>

      <main style={{ paddingTop: '1.5rem' }}>
        <PortfolioSimulationEngine />
      </main>
    </Layout>
  );
}

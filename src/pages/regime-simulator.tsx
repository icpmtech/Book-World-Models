import type { ReactNode } from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import RegimeSimulator from '@site/src/components/RegimeSimulator';

export default function RegimeSimulatorPage(): ReactNode {
  return (
    <Layout
      title="Regime Shift Simulator"
      description="Interactive simulator for financial market regime detection — explore hidden states, transition probabilities, and early warning signals from Chapter 7 of Financial World Models."
    >
      <div
        style={{
          background: 'linear-gradient(135deg, #0a1628 0%, #1a4b8c 40%, #0d2d5e 100%)',
          padding: '2rem 1rem 1.5rem',
          textAlign: 'center',
          color: '#fff',
        }}
      >
        <h1 style={{ fontSize: '2rem', margin: '0 0 0.4rem', color: '#fff' }}>
          🔍 Regime Shift Simulator
        </h1>
        <p style={{ fontSize: '1.05rem', opacity: 0.88, margin: '0 0 0.75rem', fontStyle: 'italic' }}>
          Hidden States, Transition Probabilities &amp; Early Warning Signals
        </p>
        <div style={{ fontSize: '0.9rem', opacity: 0.75 }}>
          Companion to{' '}
          <Link to="/docs/chapter-07" style={{ color: '#93c5fd' }}>
            Chapter 7 — Regime Shifts and Hidden States
          </Link>
          {' '}·{' '}
          <Link to="/simulator" style={{ color: '#93c5fd' }}>
            World Model Simulator
          </Link>
          {' '}·{' '}
          <Link to="/docs/intro" style={{ color: '#93c5fd' }}>
            Financial World Models
          </Link>
        </div>
      </div>

      <main style={{ paddingTop: '1.5rem' }}>
        <RegimeSimulator />
      </main>
    </Layout>
  );
}

import type { ReactNode } from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import WorldModelSimulator from '@site/src/components/WorldModelSimulator';

export default function SimulatorPage(): ReactNode {
  return (
    <Layout
      title="World Model Simulator"
      description="Interactive simulator comparing LLM descriptive responses to World Model anticipatory simulations in financial markets."
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
          🌍 World Model Simulator
        </h1>
        <p style={{ fontSize: '1.05rem', opacity: 0.88, margin: '0 0 0.75rem', fontStyle: 'italic' }}>
          Anticipatory Intelligence in Financial Markets
        </p>
        <div style={{ fontSize: '0.9rem', opacity: 0.75 }}>
          Companion to{' '}
          <Link to="/docs/chapter-03" style={{ color: '#93c5fd' }}>
            Chapter 3 — From Words to Worlds
          </Link>
          {' '}·{' '}
          <Link to="/docs/intro" style={{ color: '#93c5fd' }}>
            Financial World Models
          </Link>
        </div>
      </div>

      <main style={{ paddingTop: '1.5rem' }}>
        <WorldModelSimulator />
      </main>
    </Layout>
  );
}

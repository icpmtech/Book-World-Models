"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.default = SimulatorPage;
var Layout_1 = require("@theme/Layout");
var Link_1 = require("@docusaurus/Link");
var WorldModelSimulator_1 = require("@site/src/components/WorldModelSimulator");
function SimulatorPage() {
    return (<Layout_1.default title="World Model Simulator" description="Interactive simulator comparing LLM descriptive responses to World Model anticipatory simulations in financial markets.">
      <div style={{
            background: 'linear-gradient(135deg, #0a1628 0%, #1a4b8c 40%, #0d2d5e 100%)',
            padding: '2rem 1rem 1.5rem',
            textAlign: 'center',
            color: '#fff',
        }}>
        <h1 style={{ fontSize: '2rem', margin: '0 0 0.4rem', color: '#fff' }}>
          🌍 World Model Simulator
        </h1>
        <p style={{ fontSize: '1.05rem', opacity: 0.88, margin: '0 0 0.75rem', fontStyle: 'italic' }}>
          Anticipatory Intelligence in Financial Markets
        </p>
        <div style={{ fontSize: '0.9rem', opacity: 0.75 }}>
          Companion to{' '}
          <Link_1.default to="/docs/chapter-03" style={{ color: '#93c5fd' }}>
            Chapter 3 — From Words to Worlds
          </Link_1.default>
          {' '}·{' '}
          <Link_1.default to="/docs/intro" style={{ color: '#93c5fd' }}>
            Financial World Models
          </Link_1.default>
        </div>
      </div>

      <main style={{ paddingTop: '1.5rem' }}>
        <WorldModelSimulator_1.default />
      </main>
    </Layout_1.default>);
}

"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.default = MarketProPage;
var Layout_1 = require("@theme/Layout");
var Link_1 = require("@docusaurus/Link");
function MarketProPage() {
    return (<Layout_1.default title="Market Pro Digital" description="Market Pro Digital — advanced financial analytics platform for capital markets professionals.">
      {/* Page header */}
      <div style={{
            background: 'linear-gradient(135deg, #0a1628 0%, #1a4b8c 40%, #0d2d5e 100%)',
            padding: '2rem 1rem 1.5rem',
            textAlign: 'center',
            color: '#fff',
        }}>
        <h1 style={{ fontSize: '2rem', margin: '0 0 0.4rem', color: '#fff' }}>
          📈 Market Pro Digital
        </h1>
        <p style={{ fontSize: '1.05rem', opacity: 0.88, margin: '0 0 0.75rem', fontStyle: 'italic' }}>
          Advanced financial analytics platform for capital markets professionals
        </p>
        <Link_1.default href="https://market-pro.digital" target="_blank" rel="noopener noreferrer" style={{
            display: 'inline-block',
            marginTop: '0.5rem',
            padding: '0.5rem 1.25rem',
            background: '#fff',
            color: '#1a4b8c',
            borderRadius: '4px',
            fontWeight: 'bold',
            textDecoration: 'none',
            fontSize: '0.95rem',
        }}>
          🔗 Visit market-pro.digital ↗
        </Link_1.default>
      </div>

      {/* Description section */}
      <div style={{
            maxWidth: '860px',
            margin: '2rem auto',
            padding: '0 1.5rem',
            lineHeight: '1.75',
        }}>
        <h2>About Market Pro Digital</h2>
        <p>
          <strong>Market Pro Digital</strong> is a professional financial analytics platform designed
          for capital markets practitioners, quantitative analysts, and algorithmic traders. It
          provides real-time market data, portfolio analytics, risk modelling tools, and scenario
          simulation capabilities that complement the concepts explored in{' '}
          <Link_1.default to="/docs/intro">Financial World Models</Link_1.default>.
        </p>
        <p>
          The platform bridges the gap between theoretical world-model research and practical
          day-to-day trading workflows, enabling professionals to apply anticipatory intelligence
          directly to portfolio management and risk assessment.
        </p>
        <p>
          Explore the live platform embedded below, or{' '}
          <Link_1.default href="https://market-pro.digital" target="_blank" rel="noopener noreferrer">
            open it in a new tab
          </Link_1.default>
          .
        </p>
      </div>

      {/* Iframe embed */}
      <div style={{
            maxWidth: '1100px',
            margin: '0 auto 3rem',
            padding: '0 1rem',
        }}>
        <div style={{
            position: 'relative',
            width: '100%',
            paddingBottom: '62.5%', /* 16:10 aspect ratio */
            border: '2px solid #1a4b8c',
            borderRadius: '8px',
            overflow: 'hidden',
        }}>
          <iframe src="https://market-pro.digital" title="Market Pro Digital" width="100%" height="100%" style={{
            position: 'absolute',
            top: 0,
            left: 0,
            border: 'none',
            display: 'block',
        }} sandbox="allow-scripts allow-same-origin allow-popups allow-forms" allow="fullscreen" loading="lazy"/>
        </div>
        <p style={{
            textAlign: 'center',
            marginTop: '0.75rem',
            fontSize: '0.875rem',
            color: '#666',
        }}>
          If the embedded view does not load,{' '}
          <Link_1.default href="https://market-pro.digital" target="_blank" rel="noopener noreferrer">
            click here to open market-pro.digital directly
          </Link_1.default>
          .
        </p>
      </div>
    </Layout_1.default>);
}

import type {ReactNode} from 'react';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import styles from './book-pdf.module.css';

type ChapterItem = {
  number: number;
  title: string;
  slug: string;
  description: string;
};

const chapters: ChapterItem[] = [
  {
    number: 1,
    title: 'The Ceiling of Large Language Models',
    slug: 'chapter-01',
    description:
      'Why LLMs predict tokens, not reality — and why markets need something fundamentally different.',
  },
  {
    number: 2,
    title: 'What Is a World Model?',
    slug: 'chapter-02',
    description:
      'From the 2018 formulation by Ha & Schmidhuber to internal simulation engines that predict the next state of the world.',
  },
  {
    number: 3,
    title: 'From Words to Worlds',
    slug: 'chapter-03',
    description:
      'The leap from descriptive intelligence to anticipatory intelligence — simulating causal dynamics, not just language.',
  },
  {
    number: 4,
    title: 'The V-M-C Architecture',
    slug: 'chapter-04',
    description:
      'Vision, Memory, and Controller — the three-component architecture powering modern world models.',
  },
  {
    number: 5,
    title: 'Building a Financial World Model',
    slug: 'chapter-05',
    description:
      'Defining the financial state space and training a probabilistic market simulator.',
  },
  {
    number: 6,
    title: 'Market Dynamics as Physics',
    slug: 'chapter-06',
    description:
      'Treating markets as complex physical systems — liquidity as energy, transaction costs as friction.',
  },
  {
    number: 7,
    title: 'Regime Shifts and Hidden States',
    slug: 'chapter-07',
    description:
      'Detecting expansion, overheating, contraction, and recovery through latent variable inference.',
  },
  {
    number: 8,
    title: 'Portfolio Simulation Engines',
    slug: 'chapter-08',
    description:
      'Generating 10,000 possible futures and evaluating portfolios by resilience, not past performance.',
  },
  {
    number: 9,
    title: 'Scenario Generation and Counterfactual Futures',
    slug: 'chapter-09',
    description:
      'Injecting shocks, simulating propagation, and transforming risk management into experimental science.',
  },
  {
    number: 10,
    title: 'Risk, Ethics, and Market Reflexivity',
    slug: 'chapter-10',
    description:
      'How AI-driven world models can shape the very futures they predict — and how to deploy responsibly.',
  },
  {
    number: 11,
    title: 'Toward Financial AGI',
    slug: 'chapter-11',
    description:
      'Financial markets as one of the most complex dynamic systems — a stepping stone toward general intelligence.',
  },
  {
    number: 12,
    title: 'The Future of Intelligent Capital Allocation',
    slug: 'chapter-12',
    description:
      'Simulate before allocating, stress-test before deploying, optimize across probabilistic futures.',
  },
];

function handlePrint() {
  if (typeof window !== 'undefined') {
    window.print();
  }
}

export default function BookPdf(): ReactNode {
  return (
    <Layout
      title="Download Book as PDF"
      description="Download Financial World Models as a PDF — all chapters in a print-ready format.">
      <div className={styles.pdfPage}>
        {/* Controls — hidden when printing */}
        <div className={styles.controls}>
          <Heading as="h1" className={styles.pageTitle}>
            📄 Download Book as PDF
          </Heading>
          <p className={styles.pageSubtitle}>
            Click the button below to open your browser's print dialog. Choose{' '}
            <strong>Save as PDF</strong> as the destination to download the full
            book.
          </p>
          <div className={styles.buttonRow}>
            <button
              className="button button--primary button--lg"
              onClick={handlePrint}>
              🖨️ Generate PDF
            </button>
            <Link
              className="button button--outline button--secondary button--lg"
              to="/docs/intro">
              📖 Read Online
            </Link>
          </div>
          <p className={styles.tip}>
            <strong>Tip:</strong> For best results, use Chrome or Edge, select{' '}
            <em>More settings → Background graphics</em> and set paper size to{' '}
            <em>A4</em> or <em>Letter</em>.
          </p>
        </div>

        {/* Printable book content */}
        <div className={styles.bookContent}>
          <div className={styles.coverPage}>
            <h1 className={styles.bookTitle}>Financial World Models</h1>
            <p className={styles.bookSubtitle}>
              Simulating the Future of Capital Markets Beyond LLMs
            </p>
            <p className={styles.bookAuthor}>Pedro Martins</p>
            <p className={styles.bookCopyright}>
              © {new Date().getFullYear()} Pedro Martins. All rights reserved.
            </p>
          </div>

          <div className={styles.tocSection}>
            <h2 className={styles.tocTitle}>Table of Contents</h2>
            <ol className={styles.tocList}>
              {chapters.map((chapter) => (
                <li key={chapter.number} className={styles.tocItem}>
                  <span className={styles.tocChapterTitle}>
                    Chapter {chapter.number}: {chapter.title}
                  </span>
                </li>
              ))}
            </ol>
          </div>

          {chapters.map((chapter) => (
            <div key={chapter.number} className={styles.chapterSection}>
              <h2 className={styles.chapterHeading}>
                Chapter {chapter.number}: {chapter.title}
              </h2>
              <p className={styles.chapterDescription}>{chapter.description}</p>
              <p className={styles.readOnlineNote}>
                Read the full chapter online:{' '}
                <Link to={`/docs/${chapter.slug}`}>
                  {chapter.title}
                </Link>
              </p>
            </div>
          ))}

          <div className={styles.backMatter}>
            <h2>About the Author</h2>
            <p>
              Pedro Martins is the author of <em>Financial World Models</em>,
              exploring the application of world-model architectures to
              financial markets, portfolio management, and capital allocation.
            </p>
            <h2>References</h2>
            <p>
              Full references are available at{' '}
              <Link to="/docs/references">/docs/references</Link>.
            </p>
          </div>
        </div>
      </div>
    </Layout>
  );
}

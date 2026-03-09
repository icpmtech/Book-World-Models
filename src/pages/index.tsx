import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import BookChapters from '@site/src/components/BookChapters';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <p className={styles.authorTag}>by Pedro Martins</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/intro">
            📖 Start Reading
          </Link>
          <Link
            className="button button--outline button--secondary button--lg"
            to="/docs/chapter-01">
            Chapter 1 →
          </Link>
          <Link
            className="button button--outline button--secondary button--lg"
            to="/book-pdf">
            📄 Download PDF
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={siteConfig.title}
      description="Simulating the Future of Capital Markets Beyond LLMs — by Pedro Martins">
      <HomepageHeader />
      <main>
        <BookChapters />
      </main>
    </Layout>
  );
}

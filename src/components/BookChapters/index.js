"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.default = BookChapters;
var clsx_1 = require("clsx");
var Link_1 = require("@docusaurus/Link");
var Heading_1 = require("@theme/Heading");
var styles_module_css_1 = require("./styles.module.css");
var chapters = [
    {
        number: 1,
        title: 'The Ceiling of Large Language Models',
        slug: 'chapter-01',
        description: 'Why LLMs predict tokens, not reality — and why markets need something fundamentally different.',
    },
    {
        number: 2,
        title: 'What Is a World Model?',
        slug: 'chapter-02',
        description: 'From the 2018 formulation by Ha & Schmidhuber to internal simulation engines that predict the next state of the world.',
    },
    {
        number: 3,
        title: 'From Words to Worlds',
        slug: 'chapter-03',
        description: 'The leap from descriptive intelligence to anticipatory intelligence — simulating causal dynamics, not just language.',
    },
    {
        number: 4,
        title: 'The V-M-C Architecture',
        slug: 'chapter-04',
        description: 'Vision, Memory, and Controller — the three-component architecture powering modern world models.',
    },
    {
        number: 5,
        title: 'Building a Financial World Model',
        slug: 'chapter-05',
        description: 'Defining the financial state space and training a probabilistic market simulator.',
    },
    {
        number: 6,
        title: 'Market Dynamics as Physics',
        slug: 'chapter-06',
        description: 'Treating markets as complex physical systems — liquidity as energy, transaction costs as friction.',
    },
    {
        number: 7,
        title: 'Regime Shifts and Hidden States',
        slug: 'chapter-07',
        description: 'Detecting expansion, overheating, contraction, and recovery through latent variable inference.',
    },
    {
        number: 8,
        title: 'Portfolio Simulation Engines',
        slug: 'chapter-08',
        description: 'Generating 10,000 possible futures and evaluating portfolios by resilience, not past performance.',
    },
    {
        number: 9,
        title: 'Scenario Generation and Counterfactual Futures',
        slug: 'chapter-09',
        description: 'Injecting shocks, simulating propagation, and transforming risk management into experimental science.',
    },
    {
        number: 10,
        title: 'Risk, Ethics, and Market Reflexivity',
        slug: 'chapter-10',
        description: 'How AI-driven world models can shape the very futures they predict — and how to deploy responsibly.',
    },
    {
        number: 11,
        title: 'Toward Financial AGI',
        slug: 'chapter-11',
        description: 'Financial markets as one of the most complex dynamic systems — a stepping stone toward general intelligence.',
    },
    {
        number: 12,
        title: 'The Future of Intelligent Capital Allocation',
        slug: 'chapter-12',
        description: 'Simulate before allocating, stress-test before deploying, optimize across probabilistic futures.',
    },
];
function ChapterCard(_a) {
    var number = _a.number, title = _a.title, slug = _a.slug, description = _a.description;
    return (<div className={(0, clsx_1.default)('col col--4', styles_module_css_1.default.chapterCard)}>
      <Link_1.default to={"/docs/".concat(slug)} className={styles_module_css_1.default.cardLink}>
        <div className={styles_module_css_1.default.cardInner}>
          <span className={styles_module_css_1.default.chapterNumber}>Chapter {number}</span>
          <Heading_1.default as="h3" className={styles_module_css_1.default.chapterTitle}>
            {title}
          </Heading_1.default>
          <p className={styles_module_css_1.default.chapterDescription}>{description}</p>
        </div>
      </Link_1.default>
    </div>);
}
function BookChapters() {
    return (<section className={styles_module_css_1.default.chaptersSection}>
      <div className="container">
        <Heading_1.default as="h2" className={styles_module_css_1.default.sectionHeading}>
          Table of Contents
        </Heading_1.default>
        <div className="row">
          {chapters.map(function (chapter) { return (<ChapterCard key={chapter.number} {...chapter}/>); })}
        </div>
      </div>
    </section>);
}

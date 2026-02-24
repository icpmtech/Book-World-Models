"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.default = Home;
var clsx_1 = require("clsx");
var Link_1 = require("@docusaurus/Link");
var useDocusaurusContext_1 = require("@docusaurus/useDocusaurusContext");
var Layout_1 = require("@theme/Layout");
var Heading_1 = require("@theme/Heading");
var BookChapters_1 = require("@site/src/components/BookChapters");
var index_module_css_1 = require("./index.module.css");
function HomepageHeader() {
    var siteConfig = (0, useDocusaurusContext_1.default)().siteConfig;
    return (<header className={(0, clsx_1.default)('hero hero--primary', index_module_css_1.default.heroBanner)}>
      <div className="container">
        <Heading_1.default as="h1" className="hero__title">
          {siteConfig.title}
        </Heading_1.default>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <p className={index_module_css_1.default.authorTag}>by Pedro Martins</p>
        <div className={index_module_css_1.default.buttons}>
          <Link_1.default className="button button--secondary button--lg" to="/docs/intro">
            📖 Start Reading
          </Link_1.default>
          <Link_1.default className="button button--outline button--secondary button--lg" to="/docs/chapter-01">
            Chapter 1 →
          </Link_1.default>
        </div>
      </div>
    </header>);
}
function Home() {
    var siteConfig = (0, useDocusaurusContext_1.default)().siteConfig;
    return (<Layout_1.default title={siteConfig.title} description="Simulating the Future of Capital Markets Beyond LLMs — by Pedro Martins">
      <HomepageHeader />
      <main>
        <BookChapters_1.default />
      </main>
    </Layout_1.default>);
}

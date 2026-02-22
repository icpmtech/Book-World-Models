import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'Financial World Models',
  tagline: 'Simulating the Future of Capital Markets Beyond LLMs',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://book-world-models.vercel.app',
  baseUrl: '/',

  organizationName: 'icpmtech',
  projectName: 'Book-World-Models',

  onBrokenLinks: 'throw',
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/icpmtech/Book-World-Models/tree/main/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/docusaurus-social-card.jpg',
    colorMode: {
      defaultMode: 'light',
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'Financial World Models',
      logo: {
        alt: 'Financial World Models Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'bookSidebar',
          position: 'left',
          label: 'Read the Book',
        },
        {
          to: '/simulator',
          position: 'left',
          label: '🌍 Simulator',
        },
        {
          to: '/market-pro',
          position: 'right',
          label: '📈 Market Pro',
        },
        {
          href: 'https://github.com/icpmtech/Book-World-Models',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Book',
          items: [
            {
              label: 'Introduction',
              to: '/docs/intro',
            },
            {
              label: 'Chapter 1 — The Ceiling of LLMs',
              to: '/docs/chapter-01',
            },
            {
              label: 'Chapter 12 — The Future of Capital Allocation',
              to: '/docs/chapter-12',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/icpmtech/Book-World-Models',
            },
            {
              label: 'Market Pro Digital',
              href: 'https://market-pro.digital',
            },
          ],
        },
      ],
      copyright: `© ${new Date().getFullYear()} Pedro Martins. All rights reserved. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;

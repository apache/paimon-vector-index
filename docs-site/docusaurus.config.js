// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied. See the License for the
// specific language governing permissions and limitations
// under the License.

const config = {
  title: 'Apache Paimon Vector Index',
  tagline: 'Native vector indexes for Apache Paimon',
  favicon: 'favicon_blue.svg',

  url: 'https://paimon.apache.org',
  baseUrl: '/docs/vector-index/',
  organizationName: 'apache',
  projectName: 'paimon-vector-index',

  onBrokenLinks: 'throw',
  onBrokenAnchors: 'throw',

  markdown: {
    format: 'detect',
    mdx1Compat: {
      comments: true,
      admonitions: true,
      headingIds: true,
    },
  },

  presets: [
    [
      'classic',
      {
        docs: {
          path: '../docs-src',
          routeBasePath: '/',
          sidebarPath: './sidebars.js',
          editUrl: ({docPath}) =>
            `https://github.com/apache/paimon-vector-index/edit/main/docs-src/${docPath}`,
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      },
    ],
  ],

  themes: [
    [
      require.resolve('@easyops-cn/docusaurus-search-local'),
      {
        hashed: true,
        language: ['en'],
        indexBlog: false,
        docsDir: '../docs-src',
        docsRouteBasePath: '/',
      },
    ],
  ],

  plugins: [
    [
      '@docusaurus/plugin-client-redirects',
      {
        createRedirects(existingPath) {
          if (existingPath === '/' || existingPath.endsWith('.html')) {
            return undefined;
          }
          return `${existingPath}.html`;
        },
      },
    ],
  ],

  themeConfig: {
    navbar: {
      title: 'Apache Paimon Vector Index',
      logo: {
        alt: 'Apache Paimon Logo',
        src: 'favicon_blue.svg',
        srcDark: 'favicon_white.svg',
      },
      items: [
        {
          href: 'https://github.com/apache/paimon-vector-index',
          label: 'GitHub',
          position: 'right',
        },
        {
          href: 'https://paimon.apache.org',
          label: 'Project Home',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Documentation',
          items: [
            {label: 'Index selection', to: '/'},
            {label: 'API and language bindings', to: '/api'},
          ],
        },
        {
          title: 'Community',
          items: [
            {label: 'GitHub', href: 'https://github.com/apache/paimon-vector-index'},
            {label: 'Mailing List', href: 'https://paimon.apache.org/community/mailing-lists'},
          ],
        },
      ],
      copyright:
        'Copyright © The Apache Software Foundation. Apache Paimon, Paimon, and the Paimon logo are trademarks of the Apache Software Foundation.',
    },
    prism: {
      theme: require('prism-react-renderer').themes.github,
      darkTheme: require('prism-react-renderer').themes.dracula,
      additionalLanguages: ['java', 'rust', 'python', 'bash', 'json', 'yaml', 'markup', 'properties'],
    },
    docs: {
      sidebar: {
        hideable: true,
        autoCollapseCategories: true,
      },
    },
  },
};

module.exports = config;

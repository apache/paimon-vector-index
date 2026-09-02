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

const sidebars = {
  docs: [
    {type: 'doc', id: 'index', label: 'Overview'},
    {type: 'doc', id: 'api', label: 'API and language bindings'},
    {
      type: 'category',
      label: 'Indexes',
      collapsed: false,
      items: ['ivf-flat', 'ivf-pq', 'ivf-rq', 'ivf-sq', 'diskann'],
    },
    {type: 'doc', id: 'development', label: 'Development and benchmarks'},
    {
      type: 'category',
      label: 'Releases',
      collapsed: false,
      items: ['releases', 'creating-a-release', 'verifying-a-release-candidate'],
    },
    {
      type: 'link',
      label: 'Storage format',
      href: 'https://github.com/apache/paimon-vector-index/blob/main/core/STORAGE_FORMAT.md',
    },
  ],
};

module.exports = sidebars;

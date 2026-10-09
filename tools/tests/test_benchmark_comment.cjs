// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const workflow = fs.readFileSync(path.join(__dirname, '../../.github/workflows/benchmark-comment.yml'), 'utf8');
const script = workflow.split('          script: |\n')[1].replace(/^            /gm, '');
const execute = new (Object.getPrototypeOf(async function () {}).constructor)(
  'require', 'github', 'context', 'core', script);

async function check({ mutate = () => {}, existing = false, missing = false, posts = 0 } = {}) {
  const run = { id: 1, run_attempt: 1, path: '.github/workflows/benchmark-pr.yml',
    head_repository: { id: 2 }, head_branch: 'feature', head_sha: 'head',
    conclusion: 'success', html_url: 'https://github.com/owner/repo/actions/runs/1' };
  // Distinct repository IDs represent a fork PR; no workflow_run.pull_requests needed.
  const pr = { number: 95, state: 'open', base: { sha: 'base', repo: { full_name: 'owner/repo' } },
    head: { sha: 'head', ref: 'feature', repo: { id: 2 } } };
  const input = { number: 95, head: 'head', base: 'base' };
  const current = { run_attempt: 1 };
  mutate({ run, pr, input, current });
  const calls = [];
  const github = { rest: {
    pulls: { get: async () => ({ data: pr }) },
    actions: { getWorkflowRun: async () => ({ data: current }) },
    issues: { listComments() {},
      updateComment: async value => calls.push(['update', value]),
      createComment: async value => calls.push(['create', value]) },
  }, paginate: async () => existing ? [{ id: 7, user: { login: 'github-actions[bot]' },
    body: '<!-- paimon-pr-benchmark -->old' }] : [] };
  const files = { existsSync: () => !missing, readFileSync: name => {
    if (name === 'report/benchmark-context.json') return JSON.stringify(input);
    assert.equal(name, 'report/results/summary.md');
    return '🟢 Benchmark report';
  } };
  await execute(name => { assert.equal(name, 'fs'); return files; }, github,
    { repo: { owner: 'owner', repo: 'repo' }, payload: { workflow_run: run } }, { info() {} });
  assert.equal(calls.length, posts);
  if (posts) {
    assert.equal(calls[0][0], existing ? 'update' : 'create');
    assert.match(calls[0][1].body, missing ? /did not produce/ : /🟢 Benchmark report/);
  }
}

(async () => {
  await check({ posts: 1 });
  await check({ existing: true, posts: 1 });
  await check({ missing: true, posts: 1 });
  for (const mutate of [
    ({ pr }) => pr.state = 'closed',
    ({ pr }) => pr.head.sha = 'new',
    ({ pr }) => pr.base.sha = 'new',
    ({ pr }) => pr.head.repo.id = 3,
    ({ pr }) => pr.head.ref = 'other',
    ({ pr }) => pr.base.repo.full_name = 'other/repo',
    ({ input }) => input.head = 'other',
    ({ current }) => current.run_attempt = 2,
  ]) await check({ mutate });
  await assert.rejects(check({ mutate: ({ input }) => input.number = '../95' }), /Invalid PR/);
  await assert.rejects(check({ mutate: ({ run }) => run.path = 'other.yml' }), /Unexpected source/);
  console.log('Publisher checks passed: fork routing, create/update, missing report and stale/unrelated runs');
})().catch(error => { console.error(error); process.exitCode = 1; });

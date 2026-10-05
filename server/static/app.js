const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

const state = {
  stats: {},
  token: {},
  agents: [],
  activity: [],
  cases: [],
  assessments: [],
  data: {},
  frame: 0,
  loopNodes: [],
  llm: {},
  schema: {},
  live: {},
  streamConnected: false,
  hoverAgent: null,
  agentRuns: [],
  agentRunsAgent: null,
  agentImplementation: '',
  publisher: null,
  articles: [],
  article: null,
  resultsCount: 0,
  resultsNextOffset: null,
};
let loadVersion = 0;
let progressStream = null;
let tokenRefreshTimer = null;
let tokenRefreshController = null;
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

// FIGlet's 3x5 glyphs, animated as fixed-width columns by Motion.
const wordmark = "##  #    #   ## # # ###  ## # # ### ### #   ##\n# # #   # # #   # # #   #   # #  #  #   #   # #\n##  #   ### # # # # ##   #  ###  #  ##  #   # #\n#   #   # # # # # # #     # # #  #  #   #   # #\n#   ### # #  ## ### ### ##  # # ### ### ### ##";
const wordmarkColumns = Math.max(...wordmark.split('\n').map((line) => line.length));
const wordmarkElements = Array.from({ length: wordmarkColumns }, (_, column) => {
  const element = document.createElement('span');
  element.className = 'ascii-column';
  element.textContent = wordmark.split('\n').map((line) => line[column] || ' ').join('\n');
  $('#ascii-wordmark').append(element);
  return element;
});
let wordmarkAnimations = [];
let wordmarkRevealed = false;
new ResizeObserver(([entry]) => {
  if (entry.contentRect.width) {
    $('#ascii-wordmark').style.fontSize = `${Math.min(22, entry.contentRect.width / (wordmarkColumns * 0.62))}px`;
  }
}).observe($('#ascii-wordmark'));

const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

const ago = (iso) => {
  if (!iso) return 'waiting';
  const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  if (s < 60) return `${s | 0}s ago`;
  if (s < 3600) return `${(s / 60) | 0}m ago`;
  if (s < 86400) return `${(s / 3600) | 0}h ago`;
  return `${(s / 86400) | 0}d ago`;
};

const money = (v) => {
  if (v == null || v === '') return 'pending';
  const n = Number(v);
  if (!Number.isFinite(n)) return v || 'pending';
  if (n >= 1e9) return `$${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `$${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `$${(n / 1e3).toFixed(1)}K`;
  return `$${n.toFixed(0)}`;
};

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
}

async function load() {
  const version = ++loadVersion;
  const path = location.pathname;
  let updates = {};
  if (path === '/') {
    const [stats, agents, catalog] = await Promise.all([
      api('/api/stats'), api('/api/agent-status'), api('/api/agents'),
    ]);
    updates = { stats, agents: agents.agents, activity: agents.activity, llm: catalog.llm };
    if ((agents.live?.revision ?? -1) >= (state.live.revision ?? -1)) updates.live = agents.live;
  } else if (path === '/agents') {
    const catalog = await api('/api/agents');
    updates = { agents: catalog.agents, llm: catalog.llm };
  } else if (path.startsWith('/agents/')) {
    const [catalog, runs] = await Promise.all([
      api('/api/agents'), api(`/api/agents/${encodeURIComponent(path.split('/')[2])}/runs`),
    ]);
    updates = { agents: catalog.agents, llm: catalog.llm, agentRuns: runs.runs,
      agentRunsAgent: runs.agent, agentImplementation: runs.implementation, publisher: runs.publisher || null,
      promptHistory: runs.prompt_history || [], currentPrompt: runs.current_prompt || null };
  } else if (path === '/cases') {
    const [cases, assessments] = await Promise.all([api('/api/cases'), api('/api/assessments')]);
    updates = { cases: cases.cases, assessments: assessments.assessments };
  } else if (path === '/data') {
    updates = { data: await api('/api/public-data') };
  } else if (path === '/docs') {
    updates = { schema: await api('/openapi.json') };
  } else if (path === '/results') {
    const query = new URLSearchParams({ q: $('#results-search').value, origin: $('#results-origin').value });
    const results = await api(`/api/results?${query}`);
    updates = { articles: results.articles, resultsCount: results.count, resultsNextOffset: results.next_offset };
  } else if (path.startsWith('/results/')) {
    updates = { article: await api(`/api/results/${encodeURIComponent(path.split('/')[2])}`) };
  }
  if (version !== loadVersion || path !== location.pathname) return;
  Object.assign(state, updates);
  render();
}

function render() {
  const path = location.pathname;
  if (path === '/agents') return renderAgentDirectory();
  if (path.startsWith('/agents/')) return renderAgentPage(path.split('/')[2]);
  if (path === '/cases') return renderCases();
  if (path === '/data') return renderData();
  if (path === '/docs') return renderDocs();
  if (path === '/results') return renderResults();
  if (path.startsWith('/results/')) return renderPaper();
  if (path !== '/') return;
  $('#research-status').textContent = [
    `[ agents ]  ${state.agents.filter((agent) => !agent.scheduled).length} agents / ${state.activity.length} published updates`,
    `[ cases  ]  ${state.stats.cases_assessed ?? 0}/${state.stats.cases_available ?? 0} assessed`,
    '[ model  ]  GPT-5.5',
    '[ compute]  funded by token fees',
  ].join('\n');
  $('#s-cases').textContent = state.stats.cases_available ?? '-';
  $('#s-assessed').textContent = state.stats.cases_assessed ?? '-';
  $('#s-review').textContent = state.stats.review_required ?? '-';
  $('#s-abstain').textContent = state.stats.abstentions ?? '-';
  renderToken();
  $('#s-llm').textContent = state.llm.configured ? 'GPT-5.5 configured' : 'GPT-5.5 unconfigured';
  renderActivity();
  renderAgents();
  renderLiveProgress();
}

function renderAgentDirectory() {
  $('#agents-directory').innerHTML = state.agents.map((a) => `
    <a class="directory-row" href="/agents/${esc(a.name)}">
      <span class="stage-tag">[${a.stage}]</span>
      <strong>${esc(a.name.replace('_', '-'))}</strong>
      <p>${esc(a.description)}</p>
      <small>${esc(a.execution_kind)}<br>${a.depends_on.length ? `reads: ${esc(a.depends_on.join(', '))}` : 'reads: case record'}</small>
    </a>`).join('');
}

function renderToken() {
  const configured = Boolean(state.token.address);
  const hasMarketCap = configured && state.token.market_cap !== null && state.token.market_cap !== undefined;
  $$('[data-token-address]').forEach((element) => { element.hidden = !configured; });
  $$('[data-token-market-cap]').forEach((element) => { element.hidden = !hasMarketCap; });
  $('#token-details').classList.toggle('token-unconfigured', !configured);
  $('#token-address').textContent = state.token.address || '';
  if ($('#copy-token').textContent !== 'copied') $('#copy-token').textContent = shortAddress(state.token.address);
  ['#token-market-cap', '#mini-mcap'].forEach((selector) => {
    const element = $(selector);
    element.textContent = money(state.token.market_cap);
    element.title = state.token.stale ? 'Last known market cap; refresh temporarily unavailable.'
      : state.token.refreshed_at ? `Market cap fetched ${paperDate(state.token.refreshed_at)}` : '';
  });
}

async function refreshToken() {
  if (location.pathname !== '/' || document.hidden || tokenRefreshController) return;
  const controller = new AbortController();
  tokenRefreshController = controller;
  try {
    const token = await api('/api/token-status', { cache: 'no-store', signal: controller.signal });
    if (location.pathname !== '/' || controller.signal.aborted) return;
    state.token = token;
    renderToken();
  } catch (error) {
    if (error.name !== 'AbortError') {
      state.token.stale = true;
      renderToken();
    }
  } finally {
    if (tokenRefreshController === controller) tokenRefreshController = null;
  }
}

function syncTokenPolling() {
  clearInterval(tokenRefreshTimer);
  tokenRefreshTimer = null;
  if (location.pathname !== '/' || document.hidden) {
    tokenRefreshController?.abort();
    tokenRefreshController = null;
    return;
  }
  refreshToken();
  tokenRefreshTimer = setInterval(refreshToken, 5000);
}
document.addEventListener('visibilitychange', syncTokenPolling);

function renderDocs() {
  const schema = state.schema;
  $('#docs-endpoints').innerHTML = ['get', 'post'].map((method) => {
    const endpoints = Object.entries(schema.paths || {}).filter(([path, methods]) =>
      methods[method] && (path.startsWith('/api/') || path === '/healthz'));
    return `<h3 class="docs-group">${method === 'get' ? 'Read Endpoints' : 'Backend Publishing and Execution'}</h3>` + endpoints.map(([path, methods]) => {
      const operation = methods[method];
      return `<details class="endpoint">
        <summary><span class="method ${method}">${method.toUpperCase()}</span><code>${esc(path)}</code><span>${esc(operation.summary || '')}</span></summary>
        <div class="endpoint-body">
          <p>${esc(operation.description || operation.summary || '')}</p>
          ${(operation.parameters || []).length ? `<h4>Parameters</h4><div class="parameter-list">${operation.parameters.map((p) => `
            <div><code>${esc(p.name)}</code><span>${esc(p.in)} / ${esc(p.schema?.type || 'value')}${p.required ? ' / required' : ' / optional'}</span><p>${esc(p.description || '')}</p></div>`).join('')}</div>` : ''}
          ${operation.requestBody ? `<h4>Request Body</h4><pre class="code-block">${esc(JSON.stringify(operation.requestBody.content, null, 2))}</pre>` : ''}
          <h4>Responses</h4><pre class="code-block">${esc(JSON.stringify(operation.responses, null, 2))}</pre>
        </div>
      </details>`;
    }).join('');
  }).join('') + '<h3 class="docs-group">Data Schemas</h3>' + Object.entries(schema.components?.schemas || {}).map(([name, definition]) => `
    <details class="endpoint"><summary><code>${esc(name)}</code></summary><pre class="code-block">${esc(JSON.stringify(definition, null, 2))}</pre></details>`).join('');
}

function shortAddress(s) {
  if (!s) return 'pending';
  return s.length > 14 ? `${s.slice(0, 5)}...${s.slice(-5)}` : s;
}

function renderActivity() {
  const rows = (state.live.events?.length ? state.live.events : state.activity).slice(0, 18);
  $('#activity-meta').textContent = `${rows.length} recent updates`;
  $('#activity-log').innerHTML = rows.length ? rows.map((a) => `
    <a class="logline" href="/agents/${esc(a.agent)}">
      <time>${esc(ago(a.produced_at))}</time>
      <b>${esc(a.agent)}</b>
      <span>${esc(a.case_id || 'system')}</span>
      <em>${a.status ? `[${esc(a.status)}] ` : ''}${esc(a.headline || a.working_on)}</em>
    </a>`).join('') : '<div class="empty">No published agent activity yet.</div>';
}

function renderAgents() {
  $('#agent-list').innerHTML = state.agents.filter((agent) => !agent.scheduled).map((a) => {
    const recent = state.live.steps?.find((x) => x.agent === a.name) || state.activity.find((x) => x.agent === a.name);
    return `<a class="agent-card" href="/agents/${esc(a.name)}">
      <span class="stage-tag">[${a.stage}]</span>
      <strong>${esc(a.name.replace('_', '-'))}</strong>
      <small>${recent?.status ? `[${esc(recent.status)}] ` : ''}${esc(recent?.headline || a.description)}</small>
    </a>`;
  }).join('');
}

function renderCases() {
  const byId = Object.fromEntries(state.assessments.map((a) => [a.case_id, a]));
  $('#cases-meta').textContent = `${state.assessments.length}/${state.cases.length} assessed`;
  $('#cases-grid').innerHTML = state.cases.map((c) => {
    const a = byId[c.case_id];
    return `<article class="case-card ${a?.human_review_required ? 'hot-card' : ''}">
      <span>${esc(c.origin || 'case')}</span>
      <strong>${esc(c.case_id)}</strong>
      <p>${esc(c.label)}</p>
      <div class="mini-grid">
        <b>${esc(a?.plague_likelihood || 'pending')}</b>
        <b>${esc(a?.confidence || 'waiting')}</b>
        <b>${esc(a?.resistance_anomaly || 'not assessed')}</b>
      </div>
      <div class="case-references">
        <h3>Sources &amp; References</h3>
        <p class="source-note">${esc(c.source_note)}</p>
        <ul>${(c.references || []).map((reference) => {
          let url;
          try { url = new URL(reference.url); } catch { return ''; }
          if (url.protocol !== 'https:') return '';
          return `<li><span>${esc(reference.source)} / ${esc(reference.scope)}</span>
            <a href="${esc(url.href)}" target="_blank" rel="noopener noreferrer">${esc(reference.title)} ↗</a></li>`;
        }).join('')}</ul>
      </div>
    </article>`;
  }).join('');
}

function renderData() {
  $('#data-grid').innerHTML = Object.entries(state.data).map(([name, v]) => `
    <article class="case-card">
      <span>${esc(v.source || name)}</span>
      <strong>${esc(name.replaceAll('_', ' '))}</strong>
      <p>${esc(v.note || (v.ok ? 'snapshot available' : 'not available'))}</p>
      <div class="mini-grid">
        <b>${v.ok ? 'available' : 'unavailable'}</b>
        <b>${esc(v.count ?? 'n/a')}</b>
        <b>${esc((v.fetched_at || '').slice(0, 10) || 'pending')}</b>
      </div>
    </article>`).join('');
}

function route() {
  const path = location.pathname;
  state.hoverAgent = null;
  $('#loop-tooltip').hidden = true;
  syncProgressStream(path === '/');
  syncTokenPolling();
  syncWordmarkAnimation();
  $('.hero').hidden = path !== '/';
  $('#stats-strip').hidden = path !== '/';
  $('.token-mini').hidden = path !== '/';
  const agentMatch = path.match(/^\/agents\/([^/]+)$/);
  const resultMatch = path.match(/^\/results\/([^/]+)$/);
  $('#page-home').hidden = path !== '/';
  $('#page-agents').hidden = path !== '/agents';
  $('#page-agent').hidden = !agentMatch;
  $('#page-cases').hidden = path !== '/cases';
  $('#page-data').hidden = path !== '/data';
  $('#page-docs').hidden = path !== '/docs';
  $('#page-results').hidden = path !== '/results';
  $('#page-result').hidden = !resultMatch;
  if (resultMatch) $('#research-paper').textContent = 'Loading research paper...';
  const known = ['/', '/agents', '/cases', '/data', '/docs', '/results'].includes(path) || !!agentMatch || !!resultMatch;
  $('#page-error').hidden = known;
  if (!known) $('#page-error-message').textContent = 'Page not found.';
  document.title = `${path === '/' ? 'Live' : agentMatch ? agentMatch[1] : path.slice(1)} | PlagueShield`;
  $$('#top-tabs a').forEach((a) => {
    const active = a.dataset.route === path || (agentMatch && a.dataset.route === '/agents') || (resultMatch && a.dataset.route === '/results');
    a.classList.toggle('active', !!active);
  });
  if (agentMatch && state.agents.length) renderAgentPage(agentMatch[1]);
}

function paperDate(value) {
  if (!value) return 'Date not recorded';
  return new Date(value).toLocaleString('en-US', { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', timeZone: 'UTC' }) + ' UTC';
}

function renderResults() {
  $('#results-count').textContent = `${state.resultsCount} research papers`;
  $('#results-list').innerHTML = state.articles.length ? state.articles.map((article) => `
    <article class="paper-index-row">
      <div class="paper-index-meta"><time>${esc(paperDate(article.run_at))}</time><span>${esc(article.case_id)}</span><span>${esc(article.origin.replaceAll('_', ' '))}</span></div>
      <div><a class="paper-title-link" href="/results/${esc(article.id)}"><h3>${esc(article.title)}</h3></a>
      <p>${esc(article.abstract)}</p><a class="paper-read-link" href="/results/${esc(article.id)}">Read research paper ↗</a></div>
    </article>`).join('') : '<div class="empty">No research papers match this view. Completed backend iterations publish here automatically.</div>';
  $('#results-more').hidden = state.resultsNextOffset === null;
}

function paperTable(table) {
  return `<div class="paper-table-wrap" tabindex="0" aria-label="Research data table"><table class="paper-table">
    <thead><tr>${table.columns.map((column) => `<th scope="col">${esc(column)}</th>`).join('')}</tr></thead>
    <tbody>${table.rows.length ? table.rows.map((row) => `<tr>${row.map((value) => `<td>${esc(value ?? 'Not recorded')}</td>`).join('')}</tr>`).join('') : `<tr><td colspan="${table.columns.length}">No experiment results were recorded.</td></tr>`}</tbody>
  </table></div>`;
}

function paperProse(text, references) {
  if (!window.marked || !window.DOMPurify) return `<p>${esc(text)}</p>`;
  const container = document.createElement('div');
  container.innerHTML = window.DOMPurify.sanitize(window.marked.parse(text), {
    ALLOWED_TAGS: ['p', 'h1', 'h2', 'h3', 'h4', 'ul', 'ol', 'li', 'strong', 'em', 'code', 'pre', 'blockquote', 'a', 'br'],
    ALLOWED_ATTR: ['href', 'title'],
  });
  const canonicalURL = (value) => {
    try { const url = new URL(value); return url.protocol === 'https:' ? url.href : null; }
    catch { return null; }
  };
  const allowed = new Set(references.map((reference) => canonicalURL(reference.url)).filter(Boolean));
  container.querySelectorAll('a').forEach((link) => {
    const url = canonicalURL(link.getAttribute('href'));
    if (!url || !allowed.has(url)) { link.replaceWith(document.createTextNode(link.textContent)); return; }
    link.href = url;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
  });
  return `<div class="paper-prose">${container.innerHTML}</div>`;
}

function renderPaper() {
  const article = state.article;
  const references = article.sections.find((section) => section.id === 'references')?.references || [];
  document.title = `${article.title} | PlagueShield Results`;
  $('#research-paper').innerHTML = `
    <div class="paper-breadcrumb"><a href="/results">← All research results</a><span>${esc(article.case_id)}</span></div>
    <div class="paper-layout">
      <aside class="paper-contents"><details${window.innerWidth > 900 ? ' open' : ''}><summary>CONTENTS</summary><nav aria-label="Article sections">${article.sections.map((section, index) => `<a href="#paper-${esc(section.id)}">${String(index + 1).padStart(2, '0')} / ${esc(section.title)}</a>`).join('')}</nav></details></aside>
      <article class="research-paper">
        <header class="paper-header">
          <img src="/static/plagueshield-logo-transparent.png" width="32" height="32" alt="">
          <p class="paper-kicker">PLAGUESHIELD / RESEARCH REPORT</p>
          <h1>${esc(article.title)}</h1>
          <p>By PlagueShield research agents / Automated / Not peer reviewed</p>
          <dl class="paper-metadata"><div><dt>Iteration</dt><dd>${esc(paperDate(article.run_at))}</dd></div><div><dt>Published</dt><dd>${esc(paperDate(article.published_at))}</dd></div><div><dt>Provenance</dt><dd>${esc(article.origin.replaceAll('_', ' '))}</dd></div><div><dt>Pipeline</dt><dd>${esc(article.pipeline_version)}</dd></div></dl>
          <a class="paper-download" href="/api/results/${esc(article.id)}/artifact" download>Download reproducibility JSON ↓</a>
        </header>
        ${article.sections.map((section, index) => `<section class="paper-section" id="paper-${esc(section.id)}">
          <h2><span>${String(index + 1).padStart(2, '0')}</span>${esc(section.title)}</h2>
          ${(section.paragraphs || []).map((paragraph, paragraphIndex) => section.id === 'discussion' && paragraphIndex === 0 ? paperProse(paragraph, references) : `<p>${esc(paragraph)}</p>`).join('')}
          ${section.data ? `<details class="endpoint"><summary>Recorded setup</summary><pre class="code-block">${esc(JSON.stringify(section.data, null, 2))}</pre></details>` : ''}
          ${section.table ? paperTable(section.table) : ''}
          ${(section.agents || []).map((agent) => `<div class="paper-agent">
            <h3><a href="/agents/${esc(agent.name)}">${esc(agent.name.replaceAll('_', '-'))} ↗</a></h3>
            <p class="paper-finding">${esc(agent.headline)}</p>
            ${agent.abstained ? `<p class="run-warning">Abstained: ${esc(agent.abstain_reason)}</p>` : ''}
            <ol>${agent.rationale.map((line) => `<li>${esc(line)}</li>`).join('')}</ol>
            <details class="endpoint"><summary>Structured agent output</summary><pre class="code-block">${esc(JSON.stringify(agent.data, null, 2))}</pre></details>
          </div>`).join('')}
          ${section.references ? `<ol class="paper-references">${sourceLinks(section.references)}</ol>` : ''}
          ${section.id === 'reproducibility' ? `<a class="paper-download" href="/api/results/${esc(article.id)}/artifact" download>Download complete run snapshot ↓</a>` : ''}
        </section>`).join('')}
      </article>
    </div>`;
}

let resultsSearchTimer;
$('#results-search').addEventListener('input', () => {
  clearTimeout(resultsSearchTimer);
  resultsSearchTimer = setTimeout(() => { if (location.pathname === '/results') load().catch(showLoadError); }, 250);
});
$('#results-origin').addEventListener('change', () => load().catch(showLoadError));
$('#results-more').addEventListener('click', async () => {
  const button = $('#results-more');
  const version = ++loadVersion;
  button.disabled = true;
  try {
    const query = new URLSearchParams({ q: $('#results-search').value, origin: $('#results-origin').value, offset: state.resultsNextOffset });
    const results = await api(`/api/results?${query}`);
    if (version !== loadVersion || location.pathname !== '/results') return;
    state.articles.push(...results.articles);
    state.resultsNextOffset = results.next_offset;
    renderResults();
  } catch (error) { if (version === loadVersion) showLoadError(error); }
  finally { button.disabled = false; }
});

function renderAgentPage(name) {
  const agent = state.agents.find((a) => a.name === name);
  $('#agent-nav').innerHTML = state.agents.map((a) => `<a class="${a.name === name ? 'active' : ''}" href="/agents/${esc(a.name)}">${esc(a.name.replace('_', '-'))}</a>`).join('');
  if (!agent) {
    $('#agent-title').textContent = 'Agent not found';
    $('#agent-detail').textContent = '';
    $('#agent-activity').textContent = '';
    return;
  }
  const rows = state.agentRunsAgent === name ? state.agentRuns : [];
  const tracedIndex = rows.findIndex((run) => run.verdict.data?.execution);
  const initialIndex = Math.max(0, tracedIndex);
  $('#agent-num').textContent = `${agent.stage}:agent`;
  $('#agent-title').textContent = agent.name.replace('_', '-');
  $('#agent-meta').textContent = agent.depends_on?.length
    ? `reads ${agent.depends_on.join(', ')}`
    : 'starts the loop';
  $('#agent-detail').innerHTML = `
    <div class="big-num">${agent.stage}</div>
    <h3>${esc(agent.description)}</h3>
    <p>${agent.depends_on?.length ? `Inputs: ${esc(agent.depends_on.join(', '))}.` : 'Inputs: clinical observations and case evidence.'}</p>
    <div class="token-card slim">
      <div><span class="label">agent</span><strong>${esc(agent.name)}</strong></div>
      <div><span class="label">parallel</span><strong>${agent.parallel ? 'yes' : 'no'}</strong></div>
      <div><span class="label">recent updates</span><strong>${rows.length}</strong></div>
    </div>
    <p>${esc(agent.execution_kind)}</p>
    ${name === 'x_publisher' && state.publisher ? `<h3>X Publishing</h3>
      <p>@${esc(state.publisher.account)} / ${esc(state.publisher.status)}<br>
      ${state.publisher.enabled ? 'Enabled' : 'Disabled'} / ${state.publisher.dry_run ? 'Dry run' : 'Live posting'} / ${state.publisher.configured ? 'Credentials configured' : 'Credentials missing'}<br>
      Interval: ${state.publisher.interval_seconds}s${state.publisher.next_check_at ? `<br>Next check: ${esc(state.publisher.next_check_at)}` : ''}</p>
      ${state.publisher.error ? `<p class="run-warning">${esc(state.publisher.error)}</p>` : ''}` : ''}
    ${['analysis', 'meta_review'].includes(name) && state.currentPrompt ? `
      <h3>Research Prompt / v${esc(state.currentPrompt.version)}</h3>
      <p>${esc(state.currentPrompt.reason)}<br>Immutable safety rules / Python methods unchanged</p>
      <details class="endpoint"><summary>Active analysis prompt</summary><pre class="code-block">${esc(state.currentPrompt.instructions)}</pre></details>
      ${(state.promptHistory || []).map(revision => `<details class="endpoint"><summary>v${esc(revision.version)} / ${esc(revision.created_at)}</summary>
        <p>${esc(revision.reason)}</p><p>Source: ${esc(revision.source_iteration)}<br>${esc(revision.status)}</p>
        <h3>Before / v${esc(revision.parent_version)}</h3><pre class="code-block">${esc(revision.previous_instructions)}</pre>
        <h3>After / v${esc(revision.version)}</h3><pre class="code-block">${esc(revision.instructions)}</pre></details>`).join('')}` : ''}
    <details class="endpoint"><summary>Implementation / Python</summary><pre class="code-block">${esc(state.agentRunsAgent === name ? state.agentImplementation : '')}</pre></details>`;
  $('#agent-activity').innerHTML = rows.length ? `
    <label class="run-picker">${name === 'x_publisher' ? 'Research thread' : 'Published case'}
      <select id="agent-run-select">${rows.map((run, index) => `<option value="${index}"${index === initialIndex ? ' selected' : ''}>${esc(run.case_id)} / ${esc(run.case_label)}</option>`).join('')}</select>
    </label>
    <div id="agent-run-output"></div>` : '<div class="empty">No published runs for this agent yet. Its first result will appear after the next backend run.</div>';
  if (rows.length) {
    $('#agent-run-select').addEventListener('change', (event) => renderAgentRun(rows[Number(event.target.value)]));
    renderAgentRun(rows[initialIndex]);
  }
}

function sourceLinks(references) {
  return references.map((reference) => {
    let url;
    try {
      url = new URL(reference.url || (reference.identifier?.startsWith('PMID:')
        ? `https://pubmed.ncbi.nlm.nih.gov/${reference.identifier.slice(5)}/` : ''));
    } catch { return `<li>${esc(reference.title)} / ${esc(reference.identifier || 'No source URL recorded')}</li>`; }
    if (url.protocol !== 'https:') return '';
    return `<li><a href="${esc(url.href)}" target="_blank" rel="noopener noreferrer">${esc(reference.title || url.hostname)} ↗</a><span> / ${esc(reference.source)}</span></li>`;
  }).join('');
}

function renderAgentRun(run) {
  const verdict = run.verdict;
  const data = verdict.data || {};
  const execution = data.execution;
  const request = data.llm_request;
  const calculations = Object.fromEntries(Object.entries(data).filter(([key]) => !['execution', 'llm_request', 'synthesis', 'traceback'].includes(key)));
  $('#agent-run-output').innerHTML = `
    <div class="run-heading"><span>${esc(run.case_id)} / ${esc(run.assessed_at)}</span>
      <h3>${esc(verdict.headline)}</h3>
      <p>${esc(verdict.publication_status?.toUpperCase() || (verdict.abstained ? 'ABSTAINED' : 'PUBLISHED'))} / ${esc(verdict.confidence)}${verdict.publication_status ? '' : ' confidence'}${execution ? ` / ${execution.duration_ms} ms / v${esc(execution.pipeline_version)}` : verdict.publication_status ? '' : ' / legacy run: execution trace not recorded'}</p>
    </div>
    ${verdict.abstain_reason ? `<p class="run-warning">${esc(verdict.abstain_reason)}</p>` : ''}
    <section class="run-section"><h4>Findings &amp; Rationale</h4>
      ${data.synthesis ? `<div class="research-brief">${esc(data.synthesis)}</div>` : `<ol>${(verdict.rationale || []).map((line) => `<li>${esc(line)}</li>`).join('')}</ol>`}
    </section>
    ${(verdict.flags || []).length ? `<section class="run-section"><h4>Flags &amp; Limitations</h4>${verdict.flags.map((flag) => `<p><b>${esc(flag.severity)} / ${esc(flag.code)}</b><br>${esc(flag.message)}<br>${esc(flag.detail || '')}</p>`).join('')}</section>` : ''}
    <section class="run-section"><h4>Sources &amp; References</h4><p>${esc(run.source_note)}</p><ul>${sourceLinks([...(run.citations || []), ...(run.references || [])])}</ul></section>
    <details class="endpoint" open><summary>Structured Results / Calculations</summary><pre class="code-block">${esc(JSON.stringify(calculations, null, 2))}</pre></details>
    ${request ? `<details class="endpoint"><summary>Exact LLM Instructions &amp; Request</summary><pre class="code-block">${esc(JSON.stringify(request, null, 2))}</pre></details>` : ''}
    ${verdict.publication_status ? '' : `<details class="endpoint"><summary>Case Input &amp; Upstream Verdicts</summary><pre class="code-block">${esc(execution ? JSON.stringify(execution, null, 2) : 'Inputs were not recorded for this older run. New runs include the complete execution trace.')}</pre></details>`}`;
}

function syncProgressStream(enabled) {
  if (!enabled) {
    progressStream?.close();
    progressStream = null;
    state.streamConnected = false;
    return;
  }
  if (progressStream) return;
  progressStream = new EventSource('/api/live/stream');
  progressStream.onopen = () => { state.streamConnected = true; renderLiveProgress(); };
  progressStream.addEventListener('progress', (event) => {
    state.live = JSON.parse(event.data);
    state.streamConnected = true;
    renderLiveProgress();
    renderActivity();
    renderAgents();
  });
  progressStream.onerror = () => { state.streamConnected = false; renderLiveProgress(); };
}

function renderLiveProgress() {
  if (location.pathname !== '/') return;
  const live = state.live;
  const steps = live.steps || [];
  const finished = steps.filter((s) => ['completed', 'abstained', 'failed'].includes(s.status)).length;
  const running = steps.filter((s) => s.status === 'running');
  const remaining = Math.max(0, Math.ceil((new Date(live.next_run_at).getTime() - Date.now()) / 1000));
  const status = !state.streamConnected ? 'reconnecting' : !live.alive ? 'worker stopped' : live.status === 'waiting'
    ? `next run in ${remaining}s` : live.status || 'connecting';
  $('#loop-meta').textContent = state.streamConnected ? 'connected' : 'reconnecting';
  $('#loop-run-status').textContent = status;
  $('#loop-progress').textContent = `${finished}/${steps.length || 10} steps`;
  $('#run-progress').max = steps.length || 10;
  $('#run-progress').value = finished;
  $('#loop-case').textContent = live.case_id || 'awaiting run';
  $('#loop-agent').textContent = running.length ? running.map((s) => s.agent.replace('_', '-')).join(' + ') : status;
  $('.hot').classList.toggle('disconnected', !state.streamConnected || !live.alive);
  if (state.hoverAgent) showStepTooltip(state.hoverAgent);
}

function showStepTooltip(name) {
  const tooltip = $('#loop-tooltip');
  const step = state.live.steps?.find((s) => s.agent === name);
  const agent = state.agents.find((a) => a.name === name);
  const node = state.loopNodes.find((n) => n.name === name);
  if (!node) return;
  const elapsed = step?.duration_ms != null ? `${(step.duration_ms / 1000).toFixed(2)}s`
    : step?.started_at ? `${Math.max(0, (Date.now() - new Date(step.started_at).getTime()) / 1000).toFixed(1)}s` : 'not started';
  tooltip.innerHTML = `<div class="tooltip-heading"><strong>${esc(name.replace('_', '-'))}</strong><span>${esc(step?.status || 'queued')}</span></div>
    <p>${esc(step?.headline || agent?.description || 'Awaiting this stage.')}</p>
    <dl><dt>case</dt><dd>${esc(state.live.case_id || 'pending')}</dd><dt>elapsed</dt><dd>${esc(elapsed)}</dd>
    ${step?.confidence ? `<dt>confidence</dt><dd>${esc(step.confidence)}</dd>` : ''}</dl>
    ${step?.reason ? `<p class="tooltip-reason">${esc(step.reason)}</p>` : ''}
    ${step?.rationale?.length ? `<p class="tooltip-rationale">${esc(step.rationale.slice(0, 2).join(' '))}</p>` : ''}`;
  tooltip.hidden = false;
  const wrap = $('.loop-wrap');
  tooltip.style.left = `${Math.max(8, Math.min(wrap.clientWidth - tooltip.offsetWidth - 8, node.x > wrap.clientWidth / 2 ? node.x - tooltip.offsetWidth - 16 : node.x + 16))}px`;
  tooltip.style.top = `${Math.max(8, Math.min(wrap.clientHeight - tooltip.offsetHeight - 8, node.y - tooltip.offsetHeight / 2))}px`;
}

function clearStepTooltip() {
  state.hoverAgent = null;
  $('#loop-tooltip').hidden = true;
}

$('#loop-steps').addEventListener('pointerover', (event) => {
  const step = event.target.closest('.loop-step');
  if (step) { state.hoverAgent = step.dataset.agent; showStepTooltip(state.hoverAgent); }
});
$('#loop-steps').addEventListener('pointerout', (event) => {
  if (!event.relatedTarget?.closest?.('.loop-step')) clearStepTooltip();
});
$('#loop-steps').addEventListener('focusin', (event) => {
  state.hoverAgent = event.target.dataset.agent;
  showStepTooltip(state.hoverAgent);
});
$('#loop-steps').addEventListener('focusout', clearStepTooltip);
document.addEventListener('keydown', (event) => { if (event.key === 'Escape') clearStepTooltip(); });

function syncWordmarkAnimation() {
  if (reducedMotion.matches || !window.Motion) {
    wordmarkAnimations.forEach((animation) => animation.cancel());
    wordmarkAnimations = [];
    wordmarkElements.forEach((element) => {
      element.style.opacity = '1';
      element.style.transform = 'none';
      element.style.color = '';
    });
    wordmarkRevealed = true;
    return;
  }
  const active = location.pathname === '/' && !document.hidden;
  if (!wordmarkAnimations.length && active) {
    const { animate, stagger } = window.Motion;
    const colors = getComputedStyle(document.documentElement);
    if (!wordmarkRevealed) {
      wordmarkAnimations.push(animate(wordmarkElements, { opacity: [0, 1], y: [10, 0] }, {
        duration: 0.65, delay: stagger(0.015), ease: [0.16, 1, 0.3, 1],
      }));
    }
    // Start the repeating ripple after the entrance has fully settled.
    wordmarkAnimations.push(animate(wordmarkElements, {
      y: [0, -2, 0],
      color: [colors.getPropertyValue('--white').trim(), colors.getPropertyValue('--accent').trim(), colors.getPropertyValue('--white').trim()],
    }, {
      duration: 1.4, delay: stagger(0.035, { startDelay: wordmarkRevealed ? 0.5 : 1.5 }),
      repeat: Infinity, repeatDelay: 5, ease: 'easeInOut',
    }));
    wordmarkRevealed = true;
  }
  wordmarkAnimations.forEach((animation) => active ? animation.play() : animation.pause());
}
document.addEventListener('visibilitychange', syncWordmarkAnimation);
reducedMotion.addEventListener('change', syncWordmarkAnimation);

function drawLoop() {
  const canvas = $('#agent-loop');
  const ctx = canvas?.getContext('2d');
  if (!ctx) return requestAnimationFrame(drawLoop);
  const bounds = canvas.getBoundingClientRect();
  const w = bounds.width, h = bounds.height;
  if (!w || !h) return requestAnimationFrame(drawLoop);
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const theme = getComputedStyle(document.documentElement);
  const accent = theme.getPropertyValue('--accent').trim();
  const review = theme.getPropertyValue('--review').trim();
  ctx.fillStyle = '#0d0d0d';
  ctx.fillRect(0, 0, w, h);
  const agents = state.agents.length ? state.agents.filter((agent) => !agent.scheduled) : [
    { name: 'diagnostic' }, { name: 'resistance' }, { name: 'evidence' },
    { name: 'discordance' }, { name: 'uncertainty' }, { name: 'next_test' },
    { name: 'code_analysis' }, { name: 'analysis' }, { name: 'summary' }, { name: 'meta_review' },
  ];
  const cx = w / 2, cy = h / 2, r = Math.min(w * 0.31, h * 0.34);
  const t = state.frame / 70;
  const steps = Object.fromEntries((state.live.steps || []).map((s) => [s.agent, s]));
  ctx.lineWidth = 1;
  state.loopNodes = [];
  for (let i = 0; i < agents.length; i++) {
    const a1 = (i / agents.length) * Math.PI * 2 - Math.PI / 2;
    const a2 = (((i + 1) % agents.length) / agents.length) * Math.PI * 2 - Math.PI / 2;
    const x1 = cx + Math.cos(a1) * r, y1 = cy + Math.sin(a1) * r;
    const x2 = cx + Math.cos(a2) * r, y2 = cy + Math.sin(a2) * r;
    const step = steps[agents[i].name];
    const active = step?.status === 'running' && state.streamConnected;
    const done = ['completed', 'abstained', 'failed'].includes(step?.status);
    const hovered = state.hoverAgent === agents[i].name;
    const color = ['failed', 'abstained'].includes(step?.status) ? review : done || active ? accent : '#64645c';
    ctx.strokeStyle = active ? accent : done ? '#53553a' : '#292925';
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    const pulse = active && !reducedMotion.matches ? (Math.sin(t * 4 + i) + 1) / 2 : 0;
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(x1, y1, 4 + pulse * 3, 0, Math.PI * 2); ctx.fill();
    if (hovered || active) {
      ctx.strokeStyle = color;
      ctx.beginPath(); ctx.arc(x1, y1, 11 + pulse * 2, 0, Math.PI * 2); ctx.stroke();
    }
    state.loopNodes.push({ x: x1, y: y1, name: agents[i].name });
    let link = $(`.loop-step[data-agent="${agents[i].name}"]`);
    if (!link) {
      link = document.createElement('a');
      link.className = 'loop-step';
      link.dataset.agent = agents[i].name;
      link.href = `/agents/${agents[i].name}`;
      link.setAttribute('aria-describedby', 'loop-tooltip');
      $('#loop-steps').append(link);
    }
    link.setAttribute('aria-label', `${agents[i].name}: ${step?.status || 'queued'}`);
    link.style.left = `${x1}px`; link.style.top = `${y1}px`;
    if (active && !reducedMotion.matches) {
      const progress = (t * 0.6) % 1;
      ctx.fillStyle = accent;
      ctx.fillRect(x1 + (x2 - x1) * progress - 2, y1 + (y2 - y1) * progress - 2, 4, 4);
    }
    ctx.fillStyle = hovered || active ? accent : '#b4b4aa';
    ctx.font = `${w < 400 ? 9 : 11}px JetBrains Mono, monospace`;
    const label = agents[i].name.replace('_', '-');
    const labelWidth = ctx.measureText(label).width;
    const lx = x1 < cx - 10 ? x1 - labelWidth - 10 : x1 > cx + 10 ? x1 + 10 : x1 - labelWidth / 2;
    ctx.fillText(label, Math.max(8, Math.min(w - labelWidth - 8, lx)), y1 < cy - r * .8 ? y1 - 14 : y1 > cy + r * .8 ? y1 + 22 : y1 + 4);
  }
  ctx.strokeStyle = '#414139';
  ctx.setLineDash([2, 5]);
  ctx.beginPath(); ctx.arc(cx, cy, r * 0.6, 0, Math.PI * 2); ctx.stroke();
  ctx.setLineDash([]);
  state.frame++;
  requestAnimationFrame(drawLoop);
}

document.addEventListener('click', (event) => {
  const link = event.target.closest('a[href^="/"]');
  if (link && !event.ctrlKey && !event.metaKey && !event.shiftKey && !event.altKey && event.button === 0
      && /^\/(?:$|agents(?:\/[^/]+)?$|results(?:\/[^/]+)?$|cases$|data$|docs$)/.test(link.getAttribute('href'))) {
    event.preventDefault();
    history.pushState({}, '', link.getAttribute('href'));
    route();
    window.scrollTo(0, 0);
    load().catch(showLoadError);
  }
});
window.addEventListener('popstate', () => { route(); load().catch(showLoadError); });

$('#agent-loop').addEventListener('click', (event) => {
  const bounds = event.currentTarget.getBoundingClientRect();
  const node = state.loopNodes.find((n) => Math.hypot(
    n.x - (event.clientX - bounds.left), n.y - (event.clientY - bounds.top),
  ) < 24);
  if (node) {
    history.pushState({}, '', `/agents/${node.name}`);
    route();
    load().catch(showLoadError);
  }
});

$('#copy-token').addEventListener('click', async () => {
  if (!state.token.address) return;
  await navigator.clipboard?.writeText(state.token.address);
  $('#copy-token').textContent = 'copied';
  setTimeout(() => $('#copy-token').textContent = shortAddress(state.token.address), 900);
});

function showLoadError(err) {
  console.error(err);
  $('#page-error').hidden = false;
  $('#page-error-message').textContent = 'Could not load this page. Please refresh to retry.';
}
route();
load().catch(showLoadError);
setInterval(() => { if (location.pathname === '/') load().catch(showLoadError); }, 25000);
setInterval(renderLiveProgress, 1000);
drawLoop();

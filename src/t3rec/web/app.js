const state = { users: [], history: [] };
const $ = (selector) => document.querySelector(selector);

async function request(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}

function escapeText(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  })[character]);
}

function renderRecommendations(payload) {
  $('#results-title').textContent = payload.recommendations.length
    ? `${payload.recommendations.length} stories for ${payload.user_id}` : `No eligible stories for ${payload.user_id}`;
  $('#mode-badge').textContent = payload.mode.replace('_', ' ').toUpperCase();
  $('#candidate-stat').textContent = `${payload.candidate_count} candidates · ${payload.candidate_mode}` +
    (payload.history_copies_removed ? ` · ${payload.history_copies_removed} already-read copies hidden` : '');

  const strip = $('#profile-strip');
  strip.replaceChildren();
  const label = document.createElement('span');
  label.className = 'strip-label';
  label.textContent = 'PROFILE VECTOR';
  strip.append(label);
  if (!payload.profile_terms.length) {
    const empty = document.createElement('span');
    empty.className = 'muted';
    empty.textContent = 'No indexable terms in this history';
    strip.append(empty);
  }
  for (const item of payload.profile_terms) {
    const chip = document.createElement('span');
    chip.className = 'term-chip';
    chip.textContent = item.term;
    const weight = document.createElement('small');
    weight.textContent = item.weight.toFixed(2);
    chip.append(weight);
    strip.append(chip);
  }
  if (payload.interests.length) {
    const interestLabel = document.createElement('span');
    interestLabel.className = 'strip-label';
    interestLabel.textContent = 'INTERESTS';
    strip.append(interestLabel);
    for (const interest of payload.interests) {
      const chip = document.createElement('span');
      chip.className = 'term-chip';
      chip.textContent = interest.label;
      const size = document.createElement('small');
      size.textContent = `${interest.size} read`;
      chip.append(size);
      strip.append(chip);
    }
  }

  const root = $('#recommendation-list');
  root.replaceChildren();
  payload.recommendations.forEach((item, index) => {
    const row = document.createElement('article');
    row.className = 'recommendation';
    row.style.animationDelay = `${index * 55}ms`;
    const terms = item.terms.map((term) => `<span class="contribution">${escapeText(term.term)} +${term.contribution.toFixed(3)}</span>`).join('');
    // Bars show the values that enter the net score: cosine and Jaccard are scaled to the best candidate.
    const values = [['cosine', item.cosine_normalized], ['set', item.jaccard_normalized],
      ['fresh', payload.recency_available ? item.recency : null], ['popular', item.popularity]];
    const bars = values.map(([name, value]) => `<div class="score-line"><span>${name}</span><div class="bar-track"><div class="bar-fill" style="width:${Math.max(0, Math.min(100, (value ?? 0) * 100))}%"></div></div><span>${value === null ? 'n/a' : value.toFixed(2)}</span></div>`).join('');
    row.innerHTML = `<div class="rank">${String(item.rank).padStart(2, '0')}</div>
      <div class="item-copy"><h3 class="item-title">${escapeText(item.title || item.article_id)}</h3>
      <p class="item-meta">${escapeText(item.article_id)} · ${escapeText(item.category || 'UNCATEGORIZED')}${item.subcategory ? ` / ${escapeText(item.subcategory)}` : ''}${item.interest ? ` · interest: ${escapeText(item.interest)}` : ''}</p>
      ${terms ? `<div class="contribution-list">${terms}</div>` : ''}
      <p class="explanation">${escapeText(item.explanation)}</p></div>
      <div class="score-block"><strong class="score-number">${item.final_score.toFixed(3)}</strong><span class="score-caption">FINAL SCORE</span><div class="score-bars">${bars}</div></div>`;
    root.append(row);
  });
  if (!payload.recommendations.length) {
    root.innerHTML = '<div class="welcome-state"><span class="welcome-index">NO MATCHES</span><p>Nothing left to recommend.</p><span>Try another reader profile or widen candidate generation.</span></div>';
  }
}

async function runRecommendations() {
  const button = $('#run-button');
  const error = $('#error-message');
  error.textContent = '';
  const weights = Object.fromEntries([...document.querySelectorAll('[data-weight]')].map((input) => [
    input.dataset.weight, Math.max(0, Number(input.value) || 0)
  ]));
  if (Object.values(weights).every((weight) => weight === 0)) {
    error.textContent = 'At least one score weight must be greater than zero.';
    return;
  }
  const total = Object.values(weights).reduce((sum, weight) => sum + weight, 0);
  for (const key of Object.keys(weights)) weights[key] /= total;
  button.disabled = true;
  button.firstElementChild.textContent = 'Scoring candidates…';
  try {
    const payload = await request('/api/recommend', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: $('#user-id').value.trim() || 'visitor', history: state.history,
        limit: $('#top-k').value, candidate_pool_size: $('#candidate-pool').value,
        candidate_mode: $('#candidate-mode').value, diverse: $('#diversify').checked,
        multi_interest: $('#multi-interest').checked,
        diversity_relevance_weight: Number($('#lambda').value) / 100, weights
      })
    });
    renderRecommendations(payload);
  } catch (failure) {
    error.textContent = failure.message;
  } finally {
    button.disabled = false;
    button.firstElementChild.textContent = 'Run recommendations';
  }
}

async function start() {
  try {
    const data = await request('/api/state');
    state.users = data.users;
    $('#catalog-count').textContent = `${data.article_count.toLocaleString()} ARTICLES`;
    const selector = $('#user-select');
    for (const user of data.users) {
      const option = document.createElement('option');
      option.value = user.user_id;
      option.textContent = `${user.user_id} · ${user.history.length} reads`;
      selector.append(option);
    }
    $('#user-select').addEventListener('change', () => {
      const user = state.users.find((item) => item.user_id === selector.value);
      $('#user-id').value = user?.user_id || 'visitor';
      state.history = user ? [...user.history] : [];
    });
    $('#run-button').addEventListener('click', runRecommendations);
    $('#lambda').addEventListener('input', (event) => {
      $('#lambda-value').value = `${event.target.value}%`;
    });
    $('#diversify').addEventListener('change', (event) => {
      $('#lambda').disabled = !event.target.checked;
    });
  } catch (error) {
    $('#error-message').textContent = error.message;
  }
}

start();
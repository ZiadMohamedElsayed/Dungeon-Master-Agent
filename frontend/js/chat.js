const messagesEl = document.getElementById('messages');

function escHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function formatDMText(text) {
  return escHtml(text)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/\n/g, '<br>');
}

function scrollBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

export function hideEmptyState() {
  const el = document.getElementById('emptyState');
  if (el) el.remove();
}

export function showEmptyState(text = 'Upload your world lore, then speak your first action to begin the adventure...') {
  messagesEl.innerHTML = `
    <div class="empty-state" id="emptyState">
      <div class="big-icon">🎲</div>
      <p>${text}</p>
    </div>`;
}

export function appendPlayerMessage(text) {
  const el = document.createElement('div');
  el.className = 'message player';
  el.innerHTML = `
    <div class="msg-header">
      <span class="role">⚔ Adventurer</span>
    </div>
    <div class="msg-body">${escHtml(text)}</div>
  `;
  messagesEl.appendChild(el);
  scrollBottom();
}

export function appendDMMessage(text, turn, sources) {
  const id = 'src-' + Date.now();
  const hasLore     = sources?.lore?.length > 0;
  const hasCampaign = sources?.campaign?.length > 0;
  const hasSources  = hasLore || hasCampaign;

  const loreSrcs = [...new Map((sources?.lore || []).map(s => [s.source, s])).values()]
  .map(s => `
    <div class="source-chip">
      <span class="chip-file">📜 ${escHtml(s.source)}</span>
      <span class="chip-snippet">${escHtml(s.snippet)}</span>
    </div>`).join('');

  const campSrcs = [...new Map((sources?.campaign || []).map(s => [s.source, s])).values()]
    .map(s => `
      <div class="source-chip">
        <span class="chip-file">⚔ ${escHtml(s.source)}</span>
        <span class="chip-snippet">${escHtml(s.snippet)}</span>
      </div>`).join('');
      
  const el = document.createElement('div');
  el.className = 'message dm';
  el.innerHTML = `
    <div class="msg-header">
      <span class="role">☠ Dungeon Master</span>
      <span class="turn-badge">Turn ${turn}</span>
    </div>
    <div class="msg-body">${formatDMText(text)}</div>
    ${hasSources ? `
      <div class="sources-toggle" data-target="${id}">
        <span class="arrow">▶</span> Sources consulted
      </div>
      <div class="sources-panel" id="${id}">
        ${hasLore     ? `<div class="source-group"><div class="source-group-label">World Lore</div>${loreSrcs}</div>` : ''}
        ${hasCampaign ? `<div class="source-group"><div class="source-group-label">Campaign History</div>${campSrcs}</div>` : ''}
      </div>
    ` : ''}
  `;

  // wire sources toggle
  const toggle = el.querySelector('.sources-toggle');
  if (toggle) {
    toggle.addEventListener('click', () => {
      const panel = document.getElementById(toggle.dataset.target);
      const open  = panel.classList.toggle('open');
      toggle.classList.toggle('open', open);
    });
  }

  messagesEl.appendChild(el);
  scrollBottom();
}

export function appendSystemMessage(text, isError = false) {
  const el = document.createElement('div');
  el.className = 'system-msg' + (isError ? ' error' : '');
  el.textContent = text;
  messagesEl.appendChild(el);
  scrollBottom();
}

export function appendTyping() {
  const el = document.createElement('div');
  el.className = 'typing';
  el.innerHTML = `
    <div class="typing-dots"><span></span><span></span><span></span></div>
    <span class="typing-text">The Dungeon Master weaves the tale...</span>
  `;
  messagesEl.appendChild(el);
  scrollBottom();
  return el;
}
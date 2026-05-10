import { checkHealth, playTurn } from './api.js';
import { hideEmptyState, appendPlayerMessage, appendDMMessage, appendSystemMessage, appendTyping } from './chat.js';
import { setStatus, setTurnCount, refreshLoreList, initSidebar, showToast } from './sidebar.js';

const textarea = document.getElementById('queryInput');
const sendBtn  = document.getElementById('sendBtn');

// ── Init ──────────────────────────────────────────────
async function init() {
  initSidebar();
  await Promise.all([
    ping(),
    refreshLoreList(),
    loadTurnCount(),
  ]);
  textarea.focus();
}

async function ping() {
  try {
    await checkHealth();
    setStatus(true);
  } catch {
    setStatus(false);
  }
}

async function loadTurnCount() {
  try {
    setTurnCount(0);
  } catch {}
}

// ── Input handling ────────────────────────────────────
textarea.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendTurn();
  }
});

textarea.addEventListener('input', () => {
  textarea.style.height = 'auto';
  textarea.style.height = Math.min(textarea.scrollHeight, 140) + 'px';
});

sendBtn.addEventListener('click', sendTurn);

// ── Send turn ─────────────────────────────────────────
async function sendTurn() {
  const query = textarea.value.trim();
  if (!query) return;

  textarea.value = '';
  textarea.style.height = 'auto';
  hideEmptyState();
  appendPlayerMessage(query);

  const typingEl = appendTyping();
  sendBtn.disabled = true;

  try {
    const data = await playTurn(query);
    typingEl.remove();
    appendDMMessage(data.answer, data.turn, data.sources);
    setTurnCount(data.turn);
  } catch (err) {
    typingEl.remove();
    appendSystemMessage(`⚠ ${err.message}`, true);
    showToast(`✗ ${err.message}`, 'error');
  } finally {
    sendBtn.disabled = false;
    textarea.focus();
  }
}

init();
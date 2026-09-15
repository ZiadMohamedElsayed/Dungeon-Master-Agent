import { checkHealth, playTurn, getTurnCount } from './api.js';
import { hideEmptyState, appendPlayerMessage, appendDMMessage, appendSystemMessage, appendTyping } from './chat.js';
import { setStatus, setTurnCount, refreshLoreList, initSidebar, showToast } from './sidebar.js';

const textarea = document.getElementById('queryInput');
const sendBtn  = document.getElementById('sendBtn');
const streamToggle = document.getElementById('streamToggle');
const evalToggle = document.getElementById('evalToggle');

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
    const data = await getTurnCount();
    setTurnCount(data.turns ?? 0);
  } catch {
    setTurnCount(0);
  }
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

// /roll dice command (local, no backend needed)
function tryDice(query) {
  const m = query.trim().match(/^\/roll\s+(\d*)d(\d+)([+-]\d+)?$/i);
  if (!m) return null;
  const n = Math.min(parseInt(m[1] || '1', 10), 100);
  const sides = parseInt(m[2], 10);
  const mod = parseInt(m[3] || '0', 10);
  if (sides < 2 || sides > 1000) return null;
  const rolls = Array.from({ length: n }, () => 1 + Math.floor(Math.random() * sides));
  const total = rolls.reduce((a, b) => a + b, 0) + mod;
  return `🎲 Rolled ${n}d${sides}${m[3] || ''}: [${rolls.join(', ')}]${m[3] || ''} = **${total}** — describe how you use this result...`;
}

// ── Send turn ─────────────────────────────────────────
async function sendTurn() {
  const query = textarea.value.trim();
  if (!query) return;

  textarea.value = '';
  textarea.style.height = 'auto';
  hideEmptyState();

  const dice = tryDice(query);
  if (dice) {
    appendPlayerMessage(query);
    appendDMMessage(dice, '', null);
    return;
  }
  appendPlayerMessage(query);

  const typingEl = appendTyping();
  sendBtn.disabled = true;
  const useStream = streamToggle?.checked ?? false;
  const useEval = evalToggle?.checked ?? false;

  try {
    if (useStream) {
      // live-update a placeholder DM bubble
      typingEl.remove();
      const { appendStreamingDM } = await import('./chat.js');
      const { update, finish } = appendStreamingDM();
      const data = await playTurn(query, {
        stream: true,
        evaluate: useEval,
        onToken: (tok, full) => update(full),
      });
      finish(data.answer, data.turn, data.sources);
      setTurnCount(data.turn);
      if (data.evaluation) appendSystemMessage(`✨ eval: ${JSON.stringify(data.evaluation)}`);
    } else {
      const data = await playTurn(query, { evaluate: useEval });
      typingEl.remove();
      appendDMMessage(data.answer, data.turn, data.sources);
      setTurnCount(data.turn);
      if (data.evaluation) appendSystemMessage(`✨ eval: ${JSON.stringify(data.evaluation)}`);
    }
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

import { uploadLore, deleteLore, getLoreList, resetCampaign, exportCampaign, importCampaign } from './api.js';
import { appendSystemMessage, showEmptyState } from './chat.js';

// ── Toast ─────────────────────────────────────────────
let toastTimer;
export function showToast(msg, type = '') {
  clearTimeout(toastTimer);
  document.querySelectorAll('.toast').forEach(t => t.remove());
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = msg;
  document.body.appendChild(el);
  toastTimer = setTimeout(() => el.remove(), 3500);
}

// ── Server status ─────────────────────────────────────
export function setStatus(online) {
  document.getElementById('statusDot').className = 'status-dot' + (online ? '' : ' offline');
  document.getElementById('statusText').textContent = online ? 'The Tavern is Open' : 'Server Offline';
}

// ── Turn counter ──────────────────────────────────────
export function setTurnCount(n) {
  document.getElementById('turnNum').textContent = n;
}

// ── Lore list ─────────────────────────────────────────
export async function refreshLoreList() {
  try {
    const data = await getLoreList();
    const el   = document.getElementById('loreList');
    const details = data.details || (data.documents || []).map(name => ({ name, chunks: '?' }));
    el.innerHTML = details.map(doc => `
      <div class="lore-item">
        <span title="${doc.name}">📄 ${doc.name}</span>
        <span class="chunks">${doc.chunks} chunks</span>
        <button class="del" data-file="${doc.name}" title="Remove">✕</button>
      </div>
    `).join('');
    el.querySelectorAll('.del').forEach(btn => {
      btn.addEventListener('click', async () => {
        try {
          await deleteLore(btn.dataset.file);
          showToast(`✦ ${btn.dataset.file} removed`, 'success');
          await refreshLoreList();
        } catch (err) { showToast(`✗ ${err.message}`, 'error'); }
      });
    });
  } catch {}
}

// ── Upload ────────────────────────────────────────────
function setProgress(pct) {
  const prog = document.getElementById('uploadProgress');
  const bar  = document.getElementById('uploadBar');
  if (pct === null) {
    setTimeout(() => { prog.classList.remove('active'); bar.style.width = '0%'; }, 600);
  } else {
    prog.classList.add('active');
    bar.style.width = pct + '%';
  }
}

async function handleUpload(file) {
  if (!file) return;
  setProgress(30);
  try {
    setProgress(70);
    const data = await uploadLore(file);
    setProgress(100);
    showToast(`✦ ${file.name} ingested — ${data.chunks_added} scrolls indexed`, 'success');
    await refreshLoreList();
  } catch (err) {
    showToast(`✗ ${err.message}`, 'error');
  } finally {
    setProgress(null);
  }
}

export function initSidebar() {
  // file input
  const fileInput = document.getElementById('loreFile');
  fileInput.addEventListener('change', e => {
    handleUpload(e.target.files[0]);
    e.target.value = '';
  });

  // drag & drop
  const zone = document.getElementById('uploadZone');
  zone.addEventListener('dragover',  e => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('dragover');
    handleUpload(e.dataTransfer.files[0]);
  });

  // reset
  document.getElementById('resetBtn').addEventListener('click', async () => {
    if (!confirm('Reset the campaign? All turn history will be lost. World lore is preserved.')) return;
    try {
      await resetCampaign();
      setTurnCount(0);
      showEmptyState('The campaign has been reset. A new adventure awaits...');
      showToast('✦ Campaign reset. The world endures.', 'success');
    } catch { showToast('Failed to reset', 'error'); }
  });

  // save
  document.getElementById('saveBtn')?.addEventListener('click', async () => {
    try {
      const data = await exportCampaign();
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `campaign-${new Date().toISOString().slice(0, 10)}.json`;
      a.click();
      showToast('✦ Campaign saved', 'success');
    } catch { showToast('Save failed', 'error'); }
  });

  // load
  const campInput = document.getElementById('campaignFile');
  document.getElementById('loadBtn')?.addEventListener('click', () => campInput.click());
  campInput?.addEventListener('change', async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    try {
      const data = await importCampaign(f);
      showToast(`✦ Campaign loaded — ${data.turns_added} turns`, 'success');
      const { getTurnCount } = await import('./api.js');
      setTurnCount((await getTurnCount()).turns ?? 0);
    } catch (err) { showToast(`✗ ${err.message}`, 'error'); }
    finally { e.target.value = ''; }
  });
}

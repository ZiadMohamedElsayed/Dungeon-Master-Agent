const BASE = (() => {
  // Local dev: backend on :8000, frontend on :5000/file://
  if (typeof window !== 'undefined') {
    const h = window.location.hostname;
    if (h === 'localhost' || h === '127.0.0.1' || h === '') return 'http://localhost:8000';
    // codespaces / hosted: same host, backend port forwarded
    if (window.location.port === '5000') return `${window.location.protocol}//${h}:8000`;
  }
  return 'http://localhost:8000';
})();

export async function checkHealth() {
  const r = await fetch(`${BASE}/health`);
  if (!r.ok) throw new Error('offline');
  return r.json();
}

export async function uploadLore(file) {
  const form = new FormData();
  form.append('file', file);
  const r = await fetch(`${BASE}/api/documents/lore/upload`, {
    method: 'POST',
    body: form,
  });
  const data = await r.json();
  if (!r.ok) throw new Error(data.detail || 'Upload failed');
  return data;
}

export async function deleteLore(filename) {
  const r = await fetch(`${BASE}/api/documents/lore/${encodeURIComponent(filename)}`, { method: 'DELETE' });
  const data = await r.json();
  if (!r.ok) throw new Error(data.detail || 'Delete failed');
  return data;
}

export async function getLoreList() {
  const r = await fetch(`${BASE}/api/documents/lore/list`);
  if (!r.ok) throw new Error('Failed to load lore list');
  return r.json();
}

export async function resetCampaign() {
  const r = await fetch(`${BASE}/api/documents/campaign/clear`, { method: 'DELETE' });
  if (!r.ok) throw new Error('Failed to reset campaign');
  return r.json();
}

export async function getTurnCount() {
  const r = await fetch(`${BASE}/api/chat/turns/count`);
  if (!r.ok) throw new Error('Failed to load turn count');
  return r.json();
}

export async function exportCampaign() {
  const r = await fetch(`${BASE}/api/chat/export`);
  if (!r.ok) throw new Error('Export failed');
  return r.json();
}

export async function importCampaign(file) {
  const form = new FormData();
  form.append('file', file);
  const r = await fetch(`${BASE}/api/chat/import`, { method: 'POST', body: form });
  const data = await r.json();
  if (!r.ok) throw new Error(data.detail || 'Import failed');
  return data;
}

export async function playTurn(query, { evaluate = false, reference = null, stream = false, onToken = null } = {}) {
  if (stream) {
    const r = await fetch(`${BASE}/api/chat/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, evaluate, reference, stream: true }),
    });
    if (!r.ok) throw new Error('The DM is silent...');
    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    let result = { answer: '', turn: 0, sources: { lore: [], campaign: [] }, dice: [] };
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const events = buf.split('\n\n');
      buf = events.pop();
      for (const ev of events) {
        const line = ev.trim();
        if (!line.startsWith('data:')) continue;
        const evt = JSON.parse(line.slice(5).trim());
        if (evt.type === 'token') {
          result.answer += evt.token;
          if (onToken) onToken(evt.token, result.answer);
        } else if (evt.type === 'dice') {
          result.dice.push(evt);
          const note = `\n\n🎲 ${evt.result}\n\n`;
          result.answer += note;
          if (onToken) onToken(note, result.answer);
        } else if (evt.type === 'sources') {
          result.sources = evt.sources;
        } else if (evt.type === 'error') {
          throw new Error(evt.message || 'The DM is silent...');
        } else if (evt.type === 'done') {
          result = { ...result, turn: evt.turn, answer: evt.answer, sources: evt.sources };
        }
      }
    }
    return result;
  }
  const r = await fetch(`${BASE}/api/chat/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, evaluate, reference }),
  });
  const data = await r.json();
  if (!r.ok) throw new Error(data.detail || 'The DM is silent...');
  return data;
}

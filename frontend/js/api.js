const BASE = 'https://refactored-train-4j7wv4ppj7vvcjj9q-8000.app.github.dev';

export async function checkHealth() {
  const r = await fetch(`${BASE}/health`);
  if (!r.ok) throw new Error('offline');
  return r.json();
}

export async function uploadLore(file) {
  // Clear existing lore before uploading new file
  await fetch(`${BASE}/api/documents/lore/clear`, { method: 'DELETE' });
 
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

export async function playTurn(query) {
  const r = await fetch(`${BASE}/api/chat/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  });
  const data = await r.json();
  if (!r.ok) throw new Error(data.detail || 'The DM is silent...');
  return data;
}
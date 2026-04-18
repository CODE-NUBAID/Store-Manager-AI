const chatLog = document.getElementById('chat-log');
const queryEl = document.getElementById('query');
const sendBtn = document.getElementById('send-btn');

queryEl.addEventListener('input', () => {
  queryEl.style.height = 'auto';
  queryEl.style.height = Math.min(queryEl.scrollHeight, 120) + 'px';
});

queryEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendQuery(); }
});

function setQuery(text) {
  queryEl.value = text;
  queryEl.focus();
}

function escapeHtml(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\n/g,'<br>');
}

function appendMsg(role, text) {
  const icons = { user: '👤', ai: '🤖', error: '⚠️' };
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  div.innerHTML = `
    <div class="msg-avatar">${icons[role]}</div>
    <div class="msg-body">${escapeHtml(text)}</div>`;
  chatLog.appendChild(div);
  div.scrollIntoView({ behavior: 'smooth', block: 'end' });
}

function addTyping() {
  const div = document.createElement('div');
  div.className = 'msg ai typing';
  div.id = 'typing-indicator';
  div.innerHTML = `
    <div class="msg-avatar">🤖</div>
    <div class="msg-body"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>`;
  chatLog.appendChild(div);
  div.scrollIntoView({ behavior: 'smooth', block: 'end' });
}

function removeTyping() {
  const t = document.getElementById('typing-indicator');
  if (t) t.remove();
}

async function sendQuery() {
  const query = queryEl.value.trim();
  if (!query) return;

  appendMsg('user', query);
  queryEl.value = '';
  queryEl.style.height = '44px';
  sendBtn.disabled = true;
  addTyping();

  try {
    const res = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query })
    });
    const data = await res.json();
    removeTyping();
    if (data.error) appendMsg('error', 'Error: ' + data.error);
    else appendMsg('ai', data.response);
  } catch (err) {
    removeTyping();
    appendMsg('error', 'Network error: ' + err.message);
  } finally {
    sendBtn.disabled = false;
    queryEl.focus();
  }
}

async function trainModel() {
  const btn = document.getElementById('train-btn');
  const status = document.getElementById('train-status');
  btn.disabled = true;
  status.textContent = 'Training...';
  status.style.color = 'var(--muted)';
  try {
    const res = await fetch('/train', { method: 'POST' });
    const data = await res.json();
    if (data.error) {
      status.textContent = '✗ ' + data.error;
      status.style.color = 'var(--error)';
    } else {
      status.textContent = '✓ Model retrained!';
      status.style.color = 'var(--accent2)';
    }
  } catch (e) {
    status.textContent = '✗ Network error';
    status.style.color = 'var(--error)';
  } finally {
    btn.disabled = false;
    setTimeout(() => { status.textContent = ''; }, 4000);
  }
}

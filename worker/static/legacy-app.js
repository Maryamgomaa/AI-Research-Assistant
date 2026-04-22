/**
 * Legacy Bootstrap UI — wired to the same FastAPI backend as the modern SPA.
 */
(function () {
  'use strict';

  const RESEARCH_SYSTEM = `You are Athena Research AI (Groq + LangChain). Answer any topic; you excel at research, NLP, Arabic NLP, ML, and paper analysis.
When session context or ingested-paper excerpts are provided, use them when relevant; otherwise use general knowledge. Be clear and structured.`;

  const ASSISTANT_SYSTEM = `You are Athena Research Assistant (Groq + LangChain). You can discuss any subject; when relevant, give practical tips on arXiv search, prompts, this app, and n8n automation. Be concise unless the user wants depth.`;

  let accessToken = localStorage.getItem('accessToken') || '';
  let currentUser = null;
  try {
    currentUser = JSON.parse(localStorage.getItem('currentUser') || 'null');
  } catch (_) {
    currentUser = null;
  }
  const LS_SESSIONS = 'athenaResearchSessions_v1';
  const LS_CUR_SID = 'athenaCurrentSessionId_v1';

  let discoveredPapers = [];
  let chatbotHistory = [];
  /** Last /report JSON — used for “AI final report” */
  let lastReportData = null;
  /** Active research chat: multi-turn messages (user|assistant) */
  let qaThread = [];
  let currentSessionId = '';

  function apiErrorDetail(data) {
    const d = data && data.detail;
    if (typeof d === 'string') return d;
    if (Array.isArray(d)) {
      return d.map((e) => (e && (e.msg || e.message)) || String(e)).join('; ');
    }
    if (d && typeof d === 'object') return d.msg || JSON.stringify(d);
    return (data && data.message) || 'Request failed';
  }

  /** LLM webhooks require Bearer when signed in, or server ALLOW_ANONYMOUS_LLM_WEBHOOK for dev. */
  function llmWebhookHeaders() {
    const h = { 'Content-Type': 'application/json' };
    if (accessToken) h['Authorization'] = 'Bearer ' + accessToken;
    return h;
  }

  function escapeHtml(s) {
    const d = document.createElement('div');
    d.textContent = s == null ? '' : String(s);
    return d.innerHTML;
  }

  function showAlert(elId, message, type) {
    const el = document.getElementById(elId);
    if (!el) return;
    el.className = 'alert alert-' + type + ' mt-2 mb-0 py-2 small';
    el.textContent = message;
    el.classList.remove('d-none');
  }

  function hideAlert(elId) {
    const el = document.getElementById(elId);
    if (el) {
      el.classList.add('d-none');
      el.textContent = '';
    }
  }

  function showStatus(msg, type) {
    const s = document.getElementById('status');
    if (!s) return;
    const map = { success: 'success', error: 'danger', warning: 'warning', info: 'info' };
    s.className = 'alert alert-' + (map[type] || 'info') + ' mt-3';
    s.textContent = msg;
    s.style.display = 'block';
    window.clearTimeout(showStatus._t);
    showStatus._t = window.setTimeout(function () {
      s.style.display = 'none';
    }, 5500);
  }

  function setButtonLoading(btn, loading, loadingLabel) {
    if (!btn) return;
    if (!btn.dataset.origHtml) btn.dataset.origHtml = btn.innerHTML;
    btn.disabled = loading;
    btn.innerHTML = loading
      ? '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>' + loadingLabel
      : btn.dataset.origHtml;
  }

  function updateAuthUI() {
    var logged = !!(accessToken && currentUser);
    var loginBtn = document.getElementById('loginBtn');
    var regBtn = document.getElementById('registerBtn');
    var bar = document.getElementById('userSessionBar');
    if (loginBtn) loginBtn.classList.toggle('d-none', logged);
    if (regBtn) regBtn.classList.toggle('d-none', logged);
    if (bar) {
      bar.hidden = !logged;
      var nameEl = document.getElementById('sessionUserName');
      var av = document.getElementById('sessionUserAvatar');
      if (nameEl) nameEl.textContent = (currentUser && currentUser.username) || '';
      if (av) av.textContent = ((currentUser && currentUser.username) || 'U').charAt(0).toUpperCase();
    }
    var curS = currentSessionId && getSessionById(currentSessionId);
    renderSessionUploadsList(curS || null);
  }

  function logout() {
    accessToken = '';
    currentUser = null;
    localStorage.removeItem('accessToken');
    localStorage.removeItem('currentUser');
    updateAuthUI();
    initResearchSessionsAfterLoad();
    showStatus('You have been signed out.', 'info');
  }

  async function restoreSession() {
    if (!accessToken) {
      updateAuthUI();
      return;
    }
    try {
      var r = await fetch('/me', { headers: { Authorization: 'Bearer ' + accessToken } });
      if (!r.ok) throw new Error('expired');
      currentUser = await r.json();
      localStorage.setItem(
        'currentUser',
        JSON.stringify({
          username: currentUser.username,
          id: currentUser.id,
          email: currentUser.email,
        })
      );
      updateAuthUI();
    } catch (_) {
      accessToken = '';
      currentUser = null;
      localStorage.removeItem('accessToken');
      localStorage.removeItem('currentUser');
      updateAuthUI();
    }
  }

  function initTheme() {
    var saved = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', saved);
    document.getElementById('themeToggle')?.addEventListener('click', function () {
      var cur = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', cur);
      localStorage.setItem('theme', cur);
    });
  }

  window.switchAuthModal = function (fromId, toId) {
    var fromEl = document.getElementById(fromId);
    var toEl = document.getElementById(toId);
    if (!fromEl || !toEl || typeof bootstrap === 'undefined') return;
    var inst = bootstrap.Modal.getInstance(fromEl);
    if (inst) {
      fromEl.addEventListener(
        'hidden.bs.modal',
        function once() {
          fromEl.removeEventListener('hidden.bs.modal', once);
          bootstrap.Modal.getOrCreateInstance(toEl).show();
        },
        { once: true }
      );
      inst.hide();
    } else {
      bootstrap.Modal.getOrCreateInstance(toEl).show();
    }
  };

  function appendChatbot(role, text) {
    var box = document.getElementById('chatbotMessages');
    if (!box) return;
    var div = document.createElement('div');
    div.className = 'chatbot-message ' + role;
    div.innerHTML = escapeHtml(text).replace(/\n/g, '<br>');
    box.appendChild(div);
    box.scrollTop = box.scrollHeight;
  }

  function clearChatbotUI() {
    chatbotHistory = [];
    var box = document.getElementById('chatbotMessages');
    if (box) box.innerHTML = '';
    var st = document.getElementById('chatbotStatus');
    if (st) st.textContent = '';
  }

  function renderPapers() {
    var list = document.getElementById('papersList');
    if (!list) return;
    list.innerHTML = discoveredPapers
      .map(function (p, i) {
        var id = p.arxiv_id || p.title || String(i);
        var authors = Array.isArray(p.authors) ? p.authors.slice(0, 3).join(', ') : p.authors || '';
        var ab = (p.abstract || '').slice(0, 280).replace(/\s+/g, ' ');
        return (
          '<label class="list-group-item list-group-item-action">' +
          '<div class="d-flex gap-2 align-items-start">' +
          '<input class="form-check-input mt-1 paper-cb" type="checkbox" value="' +
          escapeHtml(String(id)) +
          '" data-idx="' +
          i +
          '"/>' +
          '<div class="flex-grow-1">' +
          '<div class="fw-semibold">' +
          escapeHtml(p.title || 'Untitled') +
          '</div>' +
          '<div class="small text-muted">' +
          escapeHtml(authors) +
          '</div>' +
          '<div class="small mt-1 text-body-secondary">' +
          escapeHtml(ab) +
          '…</div>' +
          '</div></div></label>'
        );
      })
      .join('');
  }

  function renderReportHtml(data) {
    var reports = data.reports || [];
    var sum = data.topic_summary || {};
    var h = '';
    if (sum.overall_summary) {
      h +=
        '<div class="report-card"><h6 class="fw-semibold">Topic overview</h6><p>' +
        escapeHtml(sum.overall_summary) +
        '</p></div>';
    }
    reports.forEach(function (rep, i) {
      h += '<div class="report-card"><h6 class="fw-semibold">' + (i + 1) + '. ' + escapeHtml(rep.title || '') + '</h6>';
      Object.keys(rep).forEach(function (k) {
        var v = rep[k];
        if (['title', 'raw_llm_response', 'pdf_url'].indexOf(k) >= 0 || v == null || v === '') return;
        if (typeof v === 'object') return;
        h +=
          '<p class="mb-1 small"><span class="text-muted text-uppercase" style="font-size:0.7rem">' +
          escapeHtml(k) +
          '</span><br>' +
          escapeHtml(String(v)) +
          '</p>';
      });
      h += '</div>';
    });
    return h || '<p class="text-muted mb-0">No report content returned.</p>';
  }

  async function generateReport(arxivIds, topic) {
    var sec = document.getElementById('analysisReportSection');
    var body = document.getElementById('analysisReportContent');
    if (!sec || !body || !accessToken) return;
    sec.style.display = 'block';
    body.innerHTML = '<p class="text-muted mb-0"><span class="spinner-border spinner-border-sm me-2"></span>Generating analysis report…</p>';
    try {
      var r = await fetch('/report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + accessToken },
        body: JSON.stringify({ arxiv_ids: arxivIds, topic: topic || '' }),
      });
      if (!r.ok) throw new Error(await r.text());
      var data = await r.json();
      body.innerHTML = renderReportHtml(data);
      lastReportData = data;
      var fr = document.getElementById('finalReportSection');
      if (fr) fr.style.display = 'block';
      var frBody = document.getElementById('finalReportBody');
      if (frBody) frBody.value = '';
      showStatus('Analysis report ready.', 'success');
    } catch (e) {
      body.innerHTML = '<div class="alert alert-warning mb-0">' + escapeHtml(e.message) + '</div>';
    }
  }

  function getAllSessions() {
    try {
      return JSON.parse(localStorage.getItem(LS_SESSIONS) || '[]');
    } catch (_) {
      return [];
    }
  }

  function saveAllSessions(arr) {
    localStorage.setItem(LS_SESSIONS, JSON.stringify(arr));
  }

  function getSessionById(id) {
    return getAllSessions().filter(function (s) {
      return s.id === id;
    })[0];
  }

  function sessionHasContent(s) {
    if (!s) return false;
    if (s.messages && s.messages.length > 0) return true;
    if ((s.contextText || '').trim().length > 0) return true;
    if (s.pdfFiles && s.pdfFiles.length > 0) return true;
    if (s.discoveredPapers && s.discoveredPapers.length > 0) return true;
    return false;
  }

  function deleteSession(id) {
    if (!id) return;
    saveAllSessions(
      getAllSessions().filter(function (x) {
        return x.id !== id;
      })
    );
  }

  function pruneEmptySessions() {
    saveAllSessions(getAllSessions().filter(sessionHasContent));
  }

  function upsertSession(sess) {
    var all = getAllSessions();
    var i = -1;
    for (var j = 0; j < all.length; j++) {
      if (all[j].id === sess.id) {
        i = j;
        break;
      }
    }
    if (i >= 0) all[i] = sess;
    else all.unshift(sess);
    all.sort(function (a, b) {
      return new Date(b.updatedAt || b.createdAt) - new Date(a.updatedAt || a.createdAt);
    });
    saveAllSessions(all);
  }

  function loadCurrentSessionIdFromStorage() {
    currentSessionId = localStorage.getItem(LS_CUR_SID) || '';
  }

  function resetDiscoverAndPipeline() {
    discoveredPapers = [];
    lastReportData = null;
    var topic = document.getElementById('topic');
    if (topic) topic.value = '';
    var maxR = document.getElementById('maxResults');
    if (maxR) maxR.value = '5';
    var cat = document.getElementById('categories');
    if (cat) cat.value = '';
    var cit = document.getElementById('citationThreshold');
    if (cit) cit.value = '';
    var sortEl = document.getElementById('sortOrder');
    if (sortEl) sortEl.value = 'submittedDate';
    var sourcesEl = document.getElementById('sources');
    if (sourcesEl) {
      Array.from(sourcesEl.options).forEach(function (o) {
        o.selected = o.value === 'arxiv';
      });
    }
    var sessionPdfInp = document.getElementById('sessionPdfInput');
    if (sessionPdfInp) sessionPdfInp.value = '';
    var repSec = document.getElementById('reportsSection');
    if (repSec) {
      repSec.style.display = 'none';
      var repList = document.getElementById('reportsList');
      if (repList) repList.innerHTML = '';
    }
    var ps = document.getElementById('papersSection');
    if (ps) ps.style.display = 'none';
    var pl = document.getElementById('papersList');
    if (pl) pl.innerHTML = '';
    var ars = document.getElementById('analysisReportSection');
    if (ars) {
      ars.style.display = 'none';
      var ac = document.getElementById('analysisReportContent');
      if (ac) ac.innerHTML = '';
    }
    var frs = document.getElementById('finalReportSection');
    if (frs) {
      frs.style.display = 'none';
      var fb = document.getElementById('finalReportBody');
      if (fb) fb.value = '';
    }
    var fst = document.getElementById('finalReportStatus');
    if (fst) fst.textContent = '';
  }

  function renderResearchThread() {
    var box = document.getElementById('researchThreadScroll');
    if (!box) return;
    if (!qaThread.length) {
      box.innerHTML = '';
      return;
    }
    box.innerHTML = qaThread
      .map(function (m) {
        var cls = m.role === 'user' ? 'research-bubble user' : 'research-bubble assistant';
        return '<div class="' + cls + '">' + escapeHtml(m.content).replace(/\n/g, '<br>') + '</div>';
      })
      .join('');
    box.scrollTop = box.scrollHeight;
  }

  function renderSessionUploadsList(s) {
    var ul = document.getElementById('sessionPdfList');
    if (!ul) return;
    if (!s) {
      ul.innerHTML = '';
      return;
    }
    var parts = [];
    var label = 'PDF';
    (s.pdfFiles || []).forEach(function (f, idx) {
      var name = f.name || f.filename || 'file.pdf';
      var pid = f.paperId || f.paper_id;
      var dl = '';
      if (pid) {
        if (accessToken) {
          dl =
            ' <button type="button" class="btn btn-link btn-sm p-0 align-baseline session-dl" data-paper-id="' +
            escapeHtml(String(pid)) +
            '" data-filename="' +
            escapeHtml(name) +
            '">Download</button>';
        } else {
          dl = ' <span class="text-muted">(sign in to download)</span>';
        }
      }
      var rm =
        ' <button type="button" class="btn btn-link btn-sm text-danger p-0 align-baseline session-rm-pdf" data-pdf-index="' +
        idx +
        '" title="Remove this upload from this chat">Cancel</button>';
      parts.push(
        '<li class="mb-1 text-body-secondary d-flex flex-wrap align-items-baseline gap-1"><span class="badge rounded-pill bg-secondary">' +
          escapeHtml(label) +
          '</span> <span>' +
          escapeHtml(name) +
          '</span><span class="ms-auto d-inline-flex gap-2 flex-shrink-0">' +
          dl +
          rm +
          '</span></li>'
      );
    });
    ul.innerHTML = parts.length ? parts.join('') : '';
  }

  async function removeReferencePdfAtIndex(idx) {
    if (!currentSessionId) return;
    var s = getSessionById(currentSessionId);
    if (!s || !s.pdfFiles || idx < 0 || idx >= s.pdfFiles.length) return;
    var f = s.pdfFiles[idx];
    var pid = f.paperId || f.paper_id;
    if (accessToken && pid) {
      try {
        var r = await fetch('/remove_reference_pdf', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: 'Bearer ' + accessToken,
          },
          body: JSON.stringify({ paper_id: pid }),
        });
        if (!r.ok && r.status !== 404) {
          var errBody = await r.json().catch(function () {
            return {};
          });
          var det = errBody.detail;
          var msg =
            typeof det === 'string'
              ? det
              : Array.isArray(det)
                ? det
                    .map(function (e) {
                      return (e && e.msg) || '';
                    })
                    .filter(Boolean)
                    .join('; ')
                : '';
          throw new Error(msg || 'Remove failed');
        }
      } catch (err) {
        showStatus(err.message || 'Could not remove file on server.', 'error');
        return;
      }
    }
    s = getSessionById(currentSessionId);
    if (!s || !s.pdfFiles) return;
    s.pdfFiles = s.pdfFiles.filter(function (_, i) {
      return i !== idx;
    });
    s.updatedAt = new Date().toISOString();
    upsertSession(s);
    persistActiveSession();
    renderSessionUploadsList(currentSessionId ? getSessionById(currentSessionId) : null);
    renderSessionSidebar();
    refreshNewChatControls();
    showStatus('Upload removed from this chat.', 'info');
  }

  function downloadPaperById(paperId, filename) {
    if (!accessToken || !paperId) return;
    fetch('/paper_file/' + encodeURIComponent(paperId), {
      headers: { Authorization: 'Bearer ' + accessToken },
    })
      .then(function (r) {
        if (!r.ok) throw new Error('Download failed');
        return r.blob();
      })
      .then(function (blob) {
        var u = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = u;
        a.download = filename || 'document.pdf';
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(u);
      })
      .catch(function (e) {
        showStatus(e.message || 'Could not download file.', 'error');
      });
  }

  function defaultNewSessionShell() {
    var ctxEl2 = document.getElementById('sessionContext');
    return {
      id: currentSessionId,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      title: 'New chat',
      contextText: ctxEl2 ? ctxEl2.value : '',
      messages: [],
      pdfFiles: [],
      discoveredPapers: [],
    };
  }

  async function processReferencePdfUploads(files, inputEl) {
    if (!files || !files.length) return;
    if (!accessToken) {
      showStatus('Sign in to upload PDFs to your workspace.', 'warning');
      if (inputEl) inputEl.value = '';
      return;
    }
    ensureResearchSessionId();
    for (var i = 0; i < files.length; i++) {
      var file = files[i];
      if (!file.name.toLowerCase().endsWith('.pdf')) continue;
      var fd = new FormData();
      fd.append('file', file);
      try {
        var r = await fetch('/upload_pdf', {
          method: 'POST',
          headers: { Authorization: 'Bearer ' + accessToken },
          body: fd,
        });
        if (!r.ok) throw new Error(await r.text());
        var j = await r.json().catch(function () {
          return {};
        });
        var pid = j.paper_id || j.paperId;
        var sess = getSessionById(currentSessionId);
        if (!sess) sess = defaultNewSessionShell();
        sess.pdfFiles = sess.pdfFiles || [];
        sess.pdfFiles.push({
          name: file.name,
          at: new Date().toISOString(),
          paperId: pid,
        });
        sess.updatedAt = new Date().toISOString();
        upsertSession(sess);
        showStatus('Uploaded: ' + file.name, 'success');
      } catch (err) {
        showStatus('Upload failed (' + file.name + '): ' + err.message, 'error');
      }
    }
    renderSessionUploadsList(getSessionById(currentSessionId));
    if (inputEl) inputEl.value = '';
    renderSessionSidebar();
    refreshNewChatControls();
  }

  function isCurrentResearchEmpty() {
    if (qaThread.length) return false;
    if (discoveredPapers.length) return false;
    var ctx = document.getElementById('sessionContext');
    if (ctx && ctx.value.trim()) return false;
    if (currentSessionId) {
      var s = getSessionById(currentSessionId);
      if (s && sessionHasContent(s)) return false;
    }
    return true;
  }

  function refreshNewChatControls() {
    var empty = isCurrentResearchEmpty();
    var el = document.getElementById('sidebarNewChatBtn');
    if (!el) return;
    el.disabled = empty;
    el.title = empty ? 'Send a message, add context, or upload a PDF in this chat first' : '';
  }

  function renderSessionSidebar() {
    var all = getAllSessions().filter(sessionHasContent);
    var list = document.getElementById('chatHistorySidebarList');
    if (!list) return;
    if (!all.length) {
      list.innerHTML = '';
      refreshNewChatControls();
      return;
    }
    list.innerHTML = all
      .map(function (s) {
        var title = (s.title || 'Chat').replace(/\s+/g, ' ').trim().slice(0, 52) || 'Chat';
        var n = (s.messages && s.messages.length) || 0;
        var active = s.id === currentSessionId ? ' active' : '';
        return (
          '<button type="button" class="chat-sidebar-item' +
          active +
          '" data-session-id="' +
          escapeHtml(s.id) +
          '">' +
          '<div class="chat-item-title">' +
          escapeHtml(title) +
          (s.title && s.title.length > 52 ? '…' : '') +
          '</div>' +
          '<div class="chat-item-meta">' +
          escapeHtml(new Date(s.updatedAt || s.createdAt).toLocaleString()) +
          ' · ' +
          n +
          ' msgs</div></button>'
        );
      })
      .join('');
    refreshNewChatControls();
  }

  function persistActiveSession() {
    if (!currentSessionId) return;
    var ctxEl = document.getElementById('sessionContext');
    var contextText = ctxEl ? ctxEl.value : '';
    var existing = getSessionById(currentSessionId);
    var s = existing
      ? Object.assign({}, existing)
      : {
          id: currentSessionId,
          createdAt: new Date().toISOString(),
          pdfFiles: [],
          discoveredPapers: [],
          messages: [],
          title: 'New chat',
        };
    if (!s.pdfFiles) s.pdfFiles = [];
    delete s.reportUploads;
    delete s.foundPaperUploads;
    s.contextText = contextText;
    s.messages = qaThread.map(function (m) {
      return { role: m.role, content: m.content };
    });
    s.discoveredPapers =
      discoveredPapers && discoveredPapers.length
        ? JSON.parse(JSON.stringify(discoveredPapers))
        : s.discoveredPapers && s.discoveredPapers.length
          ? s.discoveredPapers
          : [];
    s.updatedAt = new Date().toISOString();
    var firstU = null;
    for (var i = 0; i < qaThread.length; i++) {
      if (qaThread[i].role === 'user') {
        firstU = qaThread[i].content;
        break;
      }
    }
    var t = (firstU || contextText || 'New chat').replace(/\s+/g, ' ').trim().slice(0, 48);
    s.title = t || 'New chat';
    if (!sessionHasContent(s)) {
      deleteSession(currentSessionId);
      currentSessionId = '';
      localStorage.removeItem(LS_CUR_SID);
      renderSessionSidebar();
      refreshNewChatControls();
      return;
    }
    upsertSession(s);
    renderSessionSidebar();
  }

  function selectSession(id) {
    var s = getSessionById(id);
    if (!s) return;
    currentSessionId = id;
    localStorage.setItem(LS_CUR_SID, id);
    qaThread = (s.messages || []).map(function (m) {
      return { role: m.role, content: m.content };
    });
    var ctx = document.getElementById('sessionContext');
    if (ctx) ctx.value = s.contextText || '';
    discoveredPapers =
      Array.isArray(s.discoveredPapers) && s.discoveredPapers.length
        ? JSON.parse(JSON.stringify(s.discoveredPapers))
        : [];
    renderPapers();
    var ps0 = document.getElementById('papersSection');
    if (ps0) ps0.style.display = discoveredPapers.length ? 'block' : 'none';
    renderSessionUploadsList(s);
    renderResearchThread();
    var qEl = document.getElementById('question');
    if (qEl) qEl.value = '';
    renderSessionSidebar();
    refreshNewChatControls();
  }

  function ensureResearchSessionId() {
    if (currentSessionId) return;
    currentSessionId = 's-' + Date.now() + '-' + Math.random().toString(36).slice(2, 10);
    localStorage.setItem(LS_CUR_SID, currentSessionId);
  }

  function beginNewChat() {
    if (currentSessionId) {
      persistActiveSession();
      var cur = getSessionById(currentSessionId);
      if (cur && !sessionHasContent(cur)) deleteSession(currentSessionId);
    }
    currentSessionId = '';
    localStorage.removeItem(LS_CUR_SID);
    qaThread = [];
    resetDiscoverAndPipeline();
    var ctx = document.getElementById('sessionContext');
    if (ctx) ctx.value = '';
    var qEl = document.getElementById('question');
    if (qEl) qEl.value = '';
    renderSessionUploadsList(null);
    renderResearchThread();
    pruneEmptySessions();
    renderSessionSidebar();
    refreshNewChatControls();
    var mainScroll = document.querySelector('.cg-main');
    if (mainScroll) mainScroll.scrollTop = 0;
  }

  function initResearchSessionsAfterLoad() {
    pruneEmptySessions();
    loadCurrentSessionIdFromStorage();
    if (currentSessionId && !getSessionById(currentSessionId)) {
      currentSessionId = '';
      localStorage.removeItem(LS_CUR_SID);
    }
    if (currentSessionId && getSessionById(currentSessionId)) {
      selectSession(currentSessionId);
      refreshNewChatControls();
      return;
    }
    var all = getAllSessions().filter(sessionHasContent);
    if (all.length) {
      selectSession(all[0].id);
      refreshNewChatControls();
      return;
    }
    beginNewChat();
  }

  function buildResearchSystemPrompt() {
    var ctxEl = document.getElementById('sessionContext');
    var ctx = ctxEl && ctxEl.value ? ctxEl.value.trim() : '';
    var s = currentSessionId ? getSessionById(currentSessionId) : null;
    var blocks = [];

    var disc = discoveredPapers && discoveredPapers.length ? discoveredPapers : (s && s.discoveredPapers) || [];
    if (disc.length) {
      blocks.push(
        '### Papers from discovery (titles and IDs — user may ask about these)\n' +
          disc
            .map(function (p) {
              return '- ' + (p.title || 'Untitled') + (p.arxiv_id ? ' (arXiv:' + p.arxiv_id + ')' : '');
            })
            .join('\n')
      );
    }

    function fileList(title, arr) {
      if (!arr || !arr.length) return;
      blocks.push(
        title +
          '\n' +
          arr
            .map(function (p) {
              return '- ' + (p.name || p.filename || 'file');
            })
            .join('\n')
      );
    }
    fileList('### Reference PDFs attached to this chat (filenames)', s && s.pdfFiles);

    if (ctx) {
      blocks.push('### Session text context (optional; use when relevant)\n' + ctx.slice(0, 12000));
    }

    var ar = document.getElementById('analysisReportContent');
    if (ar && ar.textContent && ar.textContent.trim()) {
      blocks.push(
        '### On-screen per-paper analysis (excerpt; use if relevant)\n' + ar.textContent.trim().slice(0, 6000)
      );
    }
    var fr = document.getElementById('finalReportBody');
    if (fr && fr.value && fr.value.trim()) {
      blocks.push('### On-screen final synthesized report (excerpt; use if relevant)\n' + fr.value.trim().slice(0, 8000));
    }

    blocks.unshift(
      'Answer using whatever applies: discovery results, optional pasted context, optional reference PDF filenames (user-uploaded only), and AI-produced on-screen analysis / final report excerpts below. If some sections are empty, rely on the others.'
    );

    return RESEARCH_SYSTEM + '\n\n' + blocks.join('\n\n');
  }

  function setAssistantDockOpen(open) {
    var dock = document.getElementById('assistantDock');
    var btn = document.getElementById('chatbotBtn');
    if (!dock || !btn) return;
    dock.classList.toggle('open', !!open);
    dock.setAttribute('aria-hidden', open ? 'false' : 'true');
    btn.setAttribute('aria-expanded', open ? 'true' : 'false');
    document.body.classList.toggle('has-assistant-open', !!open);
  }

  document.addEventListener('DOMContentLoaded', function () {
    initTheme();
    restoreSession().finally(function () {
      initResearchSessionsAfterLoad();
    });

    document.getElementById('loginModal')?.addEventListener('show.bs.modal', function () {
      hideAlert('loginAlert');
    });
    document.getElementById('registerModal')?.addEventListener('show.bs.modal', function () {
      hideAlert('regAlert');
    });

    document.getElementById('sessionSignOut')?.addEventListener('click', logout);

    document.getElementById('loginForm')?.addEventListener('submit', async function (e) {
      e.preventDefault();
      hideAlert('loginAlert');
      var u = document.getElementById('loginUsername').value.trim();
      var p = document.getElementById('loginPassword').value;
      var btn = e.target.querySelector('button[type="submit"]');
      if (!u || !p) {
        showAlert('loginAlert', 'Enter your username and password.', 'danger');
        return;
      }
      setButtonLoading(btn, true, 'Signing in…');
      try {
        var r = await fetch('/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username: u, password: p }),
        });
        var data = await r.json().catch(function () {
          return {};
        });
        if (!r.ok) throw new Error(apiErrorDetail(data) || 'Sign-in failed');
        if (!data.access_token) throw new Error('Server did not return a session token.');
        accessToken = data.access_token;
        currentUser = {
          username: data.username || u,
          id: data.user_id,
          email: data.email,
        };
        localStorage.setItem('accessToken', accessToken);
        localStorage.setItem('currentUser', JSON.stringify(currentUser));
        document.getElementById('loginUsername').value = '';
        document.getElementById('loginPassword').value = '';
        var lm = document.getElementById('loginModal');
        bootstrap.Modal.getInstance(lm)?.hide();
        updateAuthUI();
        showStatus('Signed in successfully as ' + currentUser.username + '.', 'success');
        initResearchSessionsAfterLoad();
      } catch (err) {
        showAlert('loginAlert', err.message, 'danger');
      } finally {
        setButtonLoading(btn, false, 'Sign in');
      }
    });

    document.getElementById('registerForm')?.addEventListener('submit', async function (e) {
      e.preventDefault();
      hideAlert('regAlert');
      var username = document.getElementById('regUsername').value.trim();
      var email = document.getElementById('regEmail').value.trim();
      var password = document.getElementById('regPassword').value;
      var btn = e.target.querySelector('button[type="submit"]');
      if (username.length < 3) {
        showAlert('regAlert', 'Username must be at least 3 characters.', 'danger');
        return;
      }
      if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
        showAlert('regAlert', 'Enter a valid email address.', 'danger');
        return;
      }
      if (password.length < 8) {
        showAlert('regAlert', 'Password must be at least 8 characters.', 'danger');
        return;
      }
      setButtonLoading(btn, true, 'Creating account…');
      try {
        var r = await fetch('/register', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username: username, email: email, password: password }),
        });
        var data = await r.json().catch(function () {
          return {};
        });
        if (!r.ok) throw new Error(apiErrorDetail(data) || 'Registration failed');
        if (!data.access_token) throw new Error('Server did not return a session token.');
        accessToken = data.access_token;
        currentUser = {
          username: data.username || username,
          id: data.user_id,
          email: data.email || email,
        };
        localStorage.setItem('accessToken', accessToken);
        localStorage.setItem('currentUser', JSON.stringify(currentUser));
        document.getElementById('regUsername').value = '';
        document.getElementById('regEmail').value = '';
        document.getElementById('regPassword').value = '';
        bootstrap.Modal.getInstance(document.getElementById('registerModal'))?.hide();
        updateAuthUI();
        showStatus(
          'Account created. Your profile is stored securely. You are now signed in as ' + currentUser.username + '.',
          'success'
        );
        initResearchSessionsAfterLoad();
      } catch (err) {
        showAlert('regAlert', err.message, 'danger');
      } finally {
        setButtonLoading(btn, false, 'Create account');
      }
    });

    document.getElementById('discoverForm')?.addEventListener('submit', async function (e) {
      e.preventDefault();
      ensureResearchSessionId();
      var topic = document.getElementById('topic').value.trim();
      if (!topic) {
        showStatus('Enter a research topic to search arXiv.', 'warning');
        return;
      }
      var maxResults = parseInt(document.getElementById('maxResults').value, 10) || 10;
      var categoriesRaw = document.getElementById('categories').value.trim();
      var categories = categoriesRaw
        ? categoriesRaw.split(',').map(function (c) {
            return c.trim();
          }).filter(Boolean)
        : undefined;
      var cit = document.getElementById('citationThreshold').value.trim();
      var citation_threshold = cit ? parseInt(cit, 10) : null;
      var sort_by = document.getElementById('sortOrder').value;
      var btn = e.target.querySelector('button[type="submit"]');
      setButtonLoading(btn, true, 'Searching…');
      try {
        var hDiscover = { 'Content-Type': 'application/json' };
        if (accessToken) hDiscover.Authorization = 'Bearer ' + accessToken;
        var r = await fetch('/discover', {
          method: 'POST',
          headers: hDiscover,
          body: JSON.stringify({
            topic: topic,
            max_results: maxResults,
            categories: categories,
            citation_threshold: citation_threshold,
            sort_by: sort_by,
          }),
        });
        if (!r.ok) throw new Error(await r.text());
        var data = await r.json();
        discoveredPapers = data.papers || [];
        renderPapers();
        var psec = document.getElementById('papersSection');
        if (psec) psec.style.display = discoveredPapers.length ? 'block' : 'none';
        if (currentSessionId) {
          var sdisc = getSessionById(currentSessionId) || defaultNewSessionShell();
          sdisc.discoveredPapers = JSON.parse(JSON.stringify(discoveredPapers));
          sdisc.updatedAt = new Date().toISOString();
          upsertSession(sdisc);
          renderSessionSidebar();
          refreshNewChatControls();
        }
        showStatus('Found ' + discoveredPapers.length + ' accessible paper(s) on arXiv.', 'success');
      } catch (err) {
        showStatus('Search failed: ' + err.message, 'error');
      } finally {
        setButtonLoading(btn, false, 'Discover Papers');
      }
    });

    document.getElementById('ingestPapersBtn')?.addEventListener('click', async function () {
      if (!accessToken) {
        showStatus('Please sign in to ingest papers into your workspace.', 'warning');
        bootstrap.Modal.getOrCreateInstance(document.getElementById('loginModal')).show();
        return;
      }
      var checked = Array.from(document.querySelectorAll('.paper-cb:checked'));
      if (!checked.length) {
        showStatus('Select at least one paper.', 'warning');
        return;
      }
      var selected = checked
        .map(function (cb) {
          return discoveredPapers[parseInt(cb.getAttribute('data-idx'), 10)];
        })
        .filter(Boolean);
      var btn = document.getElementById('ingestPapersBtn');
      setButtonLoading(btn, true, 'Ingesting…');
      try {
        var r = await fetch('/ingest', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + accessToken },
          body: JSON.stringify({ papers: selected }),
        });
        if (!r.ok) throw new Error(await r.text());
        var data = await r.json();
        showStatus('Successfully ingested ' + (data.ingested || selected.length) + ' paper(s).', 'success');
        var arxivIds = selected.map(function (p) {
          return p.arxiv_id;
        }).filter(Boolean);
        if (arxivIds.length) {
          await generateReport(arxivIds, document.getElementById('topic').value.trim() || '');
        }
      } catch (err) {
        showStatus('Ingest failed: ' + err.message, 'error');
      } finally {
        setButtonLoading(btn, false, 'Ingest Selected Papers');
      }
    });

    document.getElementById('qaForm')?.addEventListener('submit', async function (e) {
      e.preventDefault();
      var q = document.getElementById('question').value.trim();
      if (!q) return;
      ensureResearchSessionId();
      var btn = e.target.querySelector('button[type="submit"]');
      var priorForApi = qaThread.map(function (m) {
        return { role: m.role, content: m.content };
      });
      setButtonLoading(btn, true, 'Generating answer…');
      try {
        var r = await fetch('/webhook/research', {
          method: 'POST',
          headers: llmWebhookHeaders(),
          body: JSON.stringify({
            question: q,
            history: priorForApi,
            mode: 'research',
            systemPrompt: buildResearchSystemPrompt(),
          }),
        });
        var raw = await r.text();
        var data = {};
        try {
          data = JSON.parse(raw);
        } catch (_) {}
        if (!r.ok) throw new Error(apiErrorDetail(data) || raw || 'Request failed');
        var out = (data.output || '').trim();
        if (!out) throw new Error('Empty response. Configure GROQ_API_KEY on the server.');
        qaThread.push({ role: 'user', content: q });
        qaThread.push({ role: 'assistant', content: out });
        document.getElementById('question').value = '';
        persistActiveSession();
        renderResearchThread();

        if (accessToken) {
          fetch('/chat_history', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + accessToken },
            body: JSON.stringify({
              question: '[Chat ' + currentSessionId + '] ' + q,
              answer: out,
            }),
          }).catch(function () {});
        }
      } catch (err) {
        showStatus('Could not get an answer: ' + err.message, 'error');
      } finally {
        setButtonLoading(btn, false, 'Send');
      }
    });

    document.getElementById('sidebarNewChatBtn')?.addEventListener('click', function () {
      if (isCurrentResearchEmpty()) return;
      beginNewChat();
    });

    var _ctxPersistTimer;
    document.getElementById('sessionContext')?.addEventListener('input', function () {
      window.clearTimeout(_ctxPersistTimer);
      _ctxPersistTimer = window.setTimeout(function () {
        var ctxEl = document.getElementById('sessionContext');
        var t = ctxEl ? ctxEl.value.trim() : '';
        if (t) ensureResearchSessionId();
        if (currentSessionId) persistActiveSession();
        refreshNewChatControls();
      }, 450);
    });

    document.getElementById('sessionPdfInput')?.addEventListener('change', function (ev) {
      processReferencePdfUploads(ev.target.files, ev.target);
    });

    document.getElementById('sessionPdfList')?.addEventListener('click', function (ev) {
      var rm = ev.target.closest('.session-rm-pdf');
      if (rm) {
        ev.preventDefault();
        var ix = parseInt(rm.getAttribute('data-pdf-index'), 10);
        if (!Number.isNaN(ix)) removeReferencePdfAtIndex(ix);
        return;
      }
      var b = ev.target.closest('.session-dl');
      if (!b) return;
      ev.preventDefault();
      downloadPaperById(b.getAttribute('data-paper-id'), b.getAttribute('data-filename'));
    });

    document.getElementById('chatHistorySidebarList')?.addEventListener('click', function (ev) {
      var row = ev.target.closest('.chat-sidebar-item');
      if (!row) return;
      var sid = row.getAttribute('data-session-id');
      if (sid) selectSession(sid);
    });

    document.getElementById('generateFinalReportBtn')?.addEventListener('click', async function () {
      var statusEl = document.getElementById('finalReportStatus');
      var bodyEl = document.getElementById('finalReportBody');
      var btn = document.getElementById('generateFinalReportBtn');
      if (!lastReportData) {
        showStatus('Generate per-paper analysis first (ingest selected papers).', 'warning');
        return;
      }
      var payload = JSON.stringify(lastReportData);
      if (payload.length > 14000) payload = payload.slice(0, 14000) + '\n…(truncated)';
      var prompt =
        'You are a senior research director. Using ONLY the following JSON (paper analyses and optional topic summary), write a single polished **final report** in plain prose (no JSON). Include:\n' +
        '1) Executive summary (2–3 short paragraphs)\n' +
        '2) Cross-paper themes and how they relate\n' +
        '3) Gaps, risks, or contradictions\n' +
        '4) Concrete next research steps\n\n' +
        'DATA:\n' +
        payload;
      setButtonLoading(btn, true, 'Generating…');
      if (statusEl) statusEl.textContent = 'Synthesizing with AI…';
      if (bodyEl) bodyEl.value = '';
      try {
        var r = await fetch('/webhook/research', {
          method: 'POST',
          headers: llmWebhookHeaders(),
          body: JSON.stringify({
            question: prompt,
            history: [],
            mode: 'research',
            systemPrompt: RESEARCH_SYSTEM,
          }),
        });
        var raw = await r.text();
        var data = {};
        try {
          data = JSON.parse(raw);
        } catch (_) {}
        if (!r.ok) throw new Error(apiErrorDetail(data) || raw || 'Request failed');
        var out = (data.output || '').trim();
        if (!out) throw new Error('Empty final report — check GROQ_API_KEY.');
        if (bodyEl) bodyEl.value = out;
        if (statusEl) statusEl.textContent = 'Done.';
        showStatus('Final report generated.', 'success');
      } catch (err) {
        if (statusEl) statusEl.textContent = '';
        showStatus('Final report failed: ' + err.message, 'error');
      } finally {
        setButtonLoading(btn, false, 'Generate final report');
      }
    });

    document.getElementById('chatbotBtn')?.addEventListener('click', function () {
      var dock = document.getElementById('assistantDock');
      var open = dock && !dock.classList.contains('open');
      setAssistantDockOpen(open);
    });

    document.getElementById('assistantDockClose')?.addEventListener('click', function () {
      setAssistantDockOpen(false);
    });

    document.getElementById('chatbotClearBtn')?.addEventListener('click', function () {
      clearChatbotUI();
    });

    document.querySelectorAll('.chatbot-quick').forEach(function (b) {
      b.addEventListener('click', function () {
        var inp = document.getElementById('chatbotQuestion');
        if (inp) inp.value = b.getAttribute('data-prompt') || b.textContent.trim();
      });
    });

    document.getElementById('chatbotForm')?.addEventListener('submit', async function (e) {
      e.preventDefault();
      var input = document.getElementById('chatbotQuestion');
      var text = input.value.trim();
      if (!text) return;
      input.value = '';
      var prior = chatbotHistory.length ? chatbotHistory.slice(0, -1).slice(-10) : [];
      chatbotHistory.push({ role: 'user', content: text });
      appendChatbot('user', text);
      var btn = e.target.querySelector('button[type="submit"]');
      var statusEl = document.getElementById('chatbotStatus');
      setButtonLoading(btn, true, 'Sending…');
      if (statusEl) statusEl.textContent = 'Contacting assistant (Groq / n8n)…';
      try {
        var r = await fetch('/webhook/chatbot', {
          method: 'POST',
          headers: llmWebhookHeaders(),
          body: JSON.stringify({
            message: text,
            history: prior,
            mode: 'assistant',
            systemPrompt: ASSISTANT_SYSTEM,
          }),
        });
        var data = await r.json().catch(function () {
          return {};
        });
        if (!r.ok) throw new Error('HTTP ' + r.status);
        var out = (data.output || '').trim();
        if (!out) throw new Error('Empty response — set GROQ_API_KEY in your server .env file.');
        appendChatbot('assistant', out);
        chatbotHistory.push({ role: 'assistant', content: out });
      } catch (err) {
        appendChatbot('assistant', 'Something went wrong: ' + err.message);
      } finally {
        setButtonLoading(btn, false, 'Send');
        if (statusEl) statusEl.textContent = '';
      }
    });

  });
})();

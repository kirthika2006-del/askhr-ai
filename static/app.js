(() => {
  "use strict";

  const el = (id) => document.getElementById(id);

  const dropzone = el("dropzone");
  const fileInput = el("file-input");
  const uploadProgressArea = el("upload-progress-area");
  const docList = el("doc-list");
  const chatForm = el("chat-form");
  const chatInput = el("chat-input");
  const sendBtn = el("send-btn");
  const messagesEl = el("messages");
  const chatScroll = el("chat-scroll");
  const chatEmpty = el("chat-empty");
  const clearChatBtn = el("clear-chat-btn");
  const sidebar = el("sidebar");
  const sidebarToggle = el("sidebar-toggle");

  const tplUserMsg = el("tpl-user-message");
  const tplAssistantMsg = el("tpl-assistant-message");
  const tplDocItem = el("tpl-doc-item");
  const tplSourceCard = el("tpl-source-card");

  let history = []; // {role, content}
  let lastQuestion = null;

  // ---------------- Utilities ----------------

  function fmtTime(date = new Date()) {
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  function fmtSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  // Minimal markdown: bold, inline code, fenced code blocks, line breaks
  function renderMarkdown(text) {
    let safe = escapeHtml(text);
    safe = safe.replace(/```([\s\S]*?)```/g, (_, code) => `<pre><code>${code}</code></pre>`);
    safe = safe.replace(/`([^`]+)`/g, "<code>$1</code>");
    safe = safe.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    return safe;
  }

  function scrollToBottom() {
    chatScroll.scrollTop = chatScroll.scrollHeight;
  }

  function toggleEmptyState() {
    chatEmpty.style.display = messagesEl.children.length ? "none" : "flex";
  }

  async function api(path, options = {}) {
    const res = await fetch(path, options);
    let body;
    try {
      body = await res.json();
    } catch {
      body = { success: false, error: { code: "PARSE_ERROR", message: "Invalid server response." } };
    }
    if (!res.ok || !body.success) {
      const message = body?.error?.message || `Request failed (${res.status})`;
      throw new Error(message);
    }
    return body.data;
  }

  // ---------------- Status polling ----------------

  async function refreshStatus() {
    try {
      const data = await api("/api/health");
      setDot("dot-qdrant", data.qdrant_connected);
      setDot("dot-gemini", data.gemini_configured);
      el("status-qdrant").textContent = data.qdrant_connected ? "Connected" : "Offline";
      el("status-gemini").textContent = data.gemini_configured ? "Configured" : "Not set";
      el("stat-docs").textContent = data.document_count ?? 0;
      el("stat-chunks").textContent = data.chunk_count ?? 0;
    } catch (e) {
      setDot("dot-qdrant", false);
      setDot("dot-gemini", false);
    }
  }

  function setDot(id, ok) {
    const dot = el(id);
    dot.classList.remove("online", "offline");
    dot.classList.add(ok ? "online" : "offline");
  }

  // ---------------- Document list ----------------

  async function refreshDocuments() {
    try {
      const data = await api("/api/documents");
      renderDocuments(data.documents || []);
    } catch (e) {
      // silent — status card already reflects backend issues
    }
  }

  function renderDocuments(documents) {
    docList.innerHTML = "";
    if (!documents.length) {
      docList.innerHTML = '<p class="empty-hint">No documents yet. Upload one to get started.</p>';
      return;
    }
    documents.forEach((doc) => {
      const node = tplDocItem.content.cloneNode(true);
      node.querySelector(".doc-name").textContent = doc.filename;
      node.querySelector(".doc-meta").textContent =
        `${fmtSize(doc.size_bytes)} · ${doc.chunk_count} chunks`;
      const delBtn = node.querySelector(".doc-delete");
      delBtn.addEventListener("click", () => deleteDocument(doc.id, doc.filename));
      docList.appendChild(node);
    });
  }

  async function deleteDocument(id, filename) {
    if (!confirm(`Delete "${filename}" from the knowledge base?`)) return;
    try {
      await api(`/api/documents/${id}`, { method: "DELETE" });
      await Promise.all([refreshDocuments(), refreshStatus()]);
    } catch (e) {
      alert(`Could not delete document: ${e.message}`);
    }
  }

  // ---------------- Upload flow ----------------

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) uploadFile(fileInput.files[0]);
    fileInput.value = "";
  });

  function createUploadItem(filename) {
    const item = document.createElement("div");
    item.className = "upload-item";
    item.innerHTML = `
      <div class="upload-item-row">
        <span class="upload-item-name">${escapeHtml(filename)}</span>
        <span class="upload-item-stage">Uploading</span>
      </div>
      <div class="upload-bar-track"><div class="upload-bar-fill"></div></div>
    `;
    uploadProgressArea.prepend(item);
    return item;
  }

  function setStage(item, stage, fillPct) {
    item.querySelector(".upload-item-stage").textContent = stage;
    item.querySelector(".upload-bar-fill").style.width = `${fillPct}%`;
  }

  async function uploadFile(file) {
    const item = createUploadItem(file.name);
    const stages = ["Uploading", "Extracting", "Chunking", "Embedding", "Indexing"];
    let stageIdx = 0;
    setStage(item, stages[0], 15);

    // Simulated progression through stages while the server processes the
    // request (the backend performs these steps synchronously as one call).
    const progressTimer = setInterval(() => {
      if (stageIdx < stages.length - 1) {
        stageIdx += 1;
        setStage(item, stages[stageIdx], 20 + stageIdx * 18);
      }
    }, 500);

    try {
      const formData = new FormData();
      formData.append("file", file);
      const data = await api("/api/documents", { method: "POST", body: formData });
      clearInterval(progressTimer);
      setStage(item, "Completed", 100);
      item.classList.add("done");
      await Promise.all([refreshDocuments(), refreshStatus()]);
      setTimeout(() => item.remove(), 4000);
      void data;
    } catch (e) {
      clearInterval(progressTimer);
      setStage(item, e.message || "Failed", 100);
      item.classList.add("error");
    }
  }

  // ---------------- Chat ----------------

  chatInput.addEventListener("input", () => {
    chatInput.style.height = "auto";
    chatInput.style.height = Math.min(chatInput.scrollHeight, 140) + "px";
  });

  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      chatForm.requestSubmit();
    }
  });

  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const question = chatInput.value.trim();
    if (!question) return;
    chatInput.value = "";
    chatInput.style.height = "auto";
    sendQuestion(question);
  });

  function addUserMessage(text) {
    const node = tplUserMsg.content.cloneNode(true);
    node.querySelector(".msg-bubble").textContent = text;
    node.querySelector(".msg-time").textContent = fmtTime();
    messagesEl.appendChild(node);
    toggleEmptyState();
    scrollToBottom();
  }

  function addLoadingMessage() {
    const wrapper = document.createElement("div");
    wrapper.className = "msg msg-assistant loading";
    wrapper.innerHTML = `
      <div class="msg-avatar">◆</div>
      <div class="msg-body">
        <div class="msg-bubble">
          <span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>
        </div>
      </div>
    `;
    messagesEl.appendChild(wrapper);
    scrollToBottom();
    return wrapper;
  }

  function renderAssistantMessage(container, answer, sources, isError = false) {
    container.classList.remove("loading");
    if (isError) container.classList.add("error");

    const node = tplAssistantMsg.content.cloneNode(true);
    node.querySelector(".msg-bubble").innerHTML = renderMarkdown(answer);
    node.querySelector(".msg-time").textContent = fmtTime();

    const sourcesWrap = node.querySelector(".sources-wrap");
    (sources || []).forEach((src) => {
      const card = tplSourceCard.content.cloneNode(true);
      card.querySelector(".source-file span").textContent = src.filename;
      card.querySelector(".source-page").textContent = src.page_number
        ? `Page ${src.page_number}`
        : `Chunk ${src.chunk_index ?? "-"}`;
      card.querySelector(".source-relevance").textContent = `Relevance: ${src.relevance}%`;
      sourcesWrap.appendChild(card);
    });

    const copyBtn = node.querySelector(".copy-btn");
    copyBtn.addEventListener("click", () => {
      navigator.clipboard.writeText(answer);
      copyBtn.textContent = "✓ Copied";
      setTimeout(() => (copyBtn.textContent = "⧉ Copy"), 1500);
    });

    const regenBtn = node.querySelector(".regen-btn");
    if (isError || !lastQuestion) {
      regenBtn.style.display = "none";
    } else {
      regenBtn.addEventListener("click", () => {
        container.remove();
        history = history.slice(0, -1); // drop the failed/last assistant turn
        sendQuestion(lastQuestion, true);
      });
    }

    container.replaceWith(node);
  }

  async function sendQuestion(question, isRegenerate = false) {
    lastQuestion = question;
    if (!isRegenerate) {
      addUserMessage(question);
      history.push({ role: "user", content: question });
    }

    sendBtn.disabled = true;
    const loadingEl = addLoadingMessage();

    try {
      const data = await api("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, history }),
      });
      history.push({ role: "assistant", content: data.answer });
      renderAssistantMessage(loadingEl, data.answer, data.sources, false);
    } catch (e) {
      renderAssistantMessage(loadingEl, `⚠ ${e.message}`, [], true);
    } finally {
      sendBtn.disabled = false;
      scrollToBottom();
    }
  }

  clearChatBtn.addEventListener("click", () => {
    if (!messagesEl.children.length) return;
    if (!confirm("Clear the current conversation?")) return;
    messagesEl.innerHTML = "";
    history = [];
    lastQuestion = null;
    toggleEmptyState();
  });

  sidebarToggle?.addEventListener("click", () => sidebar.classList.toggle("open"));

  // ---------------- Init ----------------

  refreshStatus();
  refreshDocuments();
  toggleEmptyState();
  setInterval(refreshStatus, 15000);
})();

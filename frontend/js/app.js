/* HR Assistant frontend. Talks to the FastAPI backend with fetch(); shows only what the API returns. */
(() => {
  "use strict";

  const API_BASE = String(window.HR_API_BASE || "").replace(/\/+$/, "");
  const STORE_KEY = "hr_assistant_conversations_v1";
  const STORE_MIGRATED_KEY = "hr_assistant_conversations_sqlite_migrated_v1";
  const ADMIN_KEY_STORE = "hr_assistant_admin_key";

  const $ = (id) => document.getElementById(id);
  const state = {
    view: "chat",
    conversations: [],
    currentId: null,
    busy: false,
    adminKey: "",
    admin: false,
    employeeDataEnabled: false,
    uploadsRequireAdmin: true,
    topK: null,
    healthTimer: null,
  };

  /* ---------------------------------------------------------------- storage */
  const storage = {
    get(store, key) { try { return store.getItem(key); } catch (_) { return null; } },
    set(store, key, value) { try { store.setItem(key, value); } catch (_) { /* private mode: ignore */ } },
    remove(store, key) { try { store.removeItem(key); } catch (_) { /* ignore */ } },
  };

  async function loadConversations() {
    const legacy = storage.get(window.localStorage, STORE_KEY);
    const migrationComplete = storage.get(window.localStorage, STORE_MIGRATED_KEY) === "1";
    if (legacy && !migrationComplete) {
      try {
        const parsed = JSON.parse(legacy);
        if (!Array.isArray(parsed)) throw new Error("Saved conversation data is not a list.");
        if (parsed.length) {
          const migration = await api("/api/conversations/migrate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ conversations: parsed }),
          });
          if (!migration.verified) throw new Error("The server did not verify the imported conversations.");
          const storedConversations = await api("/api/conversations");
          const storedById = new Map(storedConversations.map((conversation) => [conversation.id, conversation]));
          for (const conversation of parsed) {
            if (!storedById.has(conversation.id)) throw new Error("An imported conversation could not be read back.");
            const storedMessages = await api(`/api/conversations/${encodeURIComponent(conversation.id)}/messages`);
            if (storedMessages.length < conversation.messages.length) throw new Error("Imported messages could not be read back.");
            for (let index = 0; index < conversation.messages.length; index++) {
              const expected = conversation.messages[index];
              const actual = storedMessages[index];
              const actualMetadata = Object.assign({}, actual);
              delete actualMetadata.position;
              delete actualMetadata.role;
              delete actualMetadata.content;
              const expectedMetadata = Object.assign({}, expected);
              delete expectedMetadata.role;
              delete expectedMetadata.content;
              if (actual.role !== expected.role || actual.content !== expected.content ||
                  JSON.stringify(sortObject(actualMetadata)) !== JSON.stringify(sortObject(expectedMetadata))) {
                throw new Error("Imported conversation data did not match the saved browser history.");
              }
            }
          }
        }
        storage.set(window.localStorage, STORE_MIGRATED_KEY, "1");
      } catch (err) {
        toast(`Legacy conversations were kept in this browser but could not be migrated: ${err.message}`, 7000);
      }
    }
    const conversations = await api("/api/conversations");
    state.conversations = conversations.map((conversation) => Object.assign({}, conversation, { messages: [] }));
    state.conversations.sort((a, b) => b.updated - a.updated);
  }

  function sortObject(value) {
    if (Array.isArray(value)) return value.map(sortObject);
    if (value && typeof value === "object") {
      return Object.keys(value).sort().reduce((sorted, key) => {
        sorted[key] = sortObject(value[key]);
        return sorted;
      }, {});
    }
    return value;
  }

  const currentConversation = () => state.conversations.find((c) => c.id === state.currentId) || null;

  /* ------------------------------------------------------------------- utils */
  const uid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function escapeHtml(text) {
    return text.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
  function formatBytes(n) {
    if (n == null) return "–";
    if (n < 1024) return n + " B";
    if (n < 1048576) return (n / 1024).toFixed(1) + " KB";
    return (n / 1048576).toFixed(1) + " MB";
  }
  function toast(message, ms = 3200) {
    const node = $("toast");
    node.textContent = message;
    node.classList.remove("hidden");
    clearTimeout(toast._t);
    toast._t = setTimeout(() => node.classList.add("hidden"), ms);
  }

  let purifyHooked = false;
  function renderMarkdown(text) {
    if (window.marked && window.DOMPurify) {
      if (!purifyHooked) {
        window.DOMPurify.addHook("afterSanitizeAttributes", (node) => {
          if (node.tagName === "A") { node.setAttribute("target", "_blank"); node.setAttribute("rel", "noopener noreferrer"); }
        });
        purifyHooked = true;
      }
      const html = window.marked.parse(text, { gfm: true, breaks: true, headerIds: false, mangle: false });
      return window.DOMPurify.sanitize(html);
    }
    return escapeHtml(text).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/\n/g, "<br>");
  }

  /* --------------------------------------------------------------------- API */
  class ApiError extends Error {
    constructor(message, status) { super(message); this.status = status; }
  }
  async function api(path, options = {}) {
    const headers = Object.assign({}, options.headers || {});
    if (state.adminKey) headers["X-Admin-Key"] = state.adminKey;
    let response;
    try {
      response = await fetch(API_BASE + path, Object.assign({}, options, { headers }));
    } catch (_) {
      throw new ApiError("Cannot reach the HR Assistant server. Is it running (and is the API address in config.js correct)?", 0);
    }
    let data = null;
    try { data = await response.json(); } catch (_) { /* non-JSON body */ }
    if (!response.ok) {
      let detail = data && data.detail;
      if (Array.isArray(detail)) detail = detail.map((d) => d.msg || String(d)).join("; ");
      throw new ApiError(detail || `Server error (${response.status})`, response.status);
    }
    return data;
  }

  /* ----------------------------------------------------------------- layout */
  function setSidebar(open) {
    const sidebar = $("sidebar");
    const isDesktop = window.matchMedia("(min-width: 1024px)").matches;
    if (isDesktop) {
      sidebar.classList.toggle("sidebar-collapsed", !open);
    } else {
      sidebar.classList.toggle("-translate-x-[110%]", !open);
      $("sidebar-overlay").classList.toggle("hidden", !open);
    }
  }
  function sidebarOpen() {
    const sidebar = $("sidebar");
    return window.matchMedia("(min-width: 1024px)").matches
      ? !sidebar.classList.contains("sidebar-collapsed")
      : !sidebar.classList.contains("-translate-x-[110%]");
  }

  const VIEW_TITLES = { chat: "HR Assistant", history: "Conversation History", kb: "HR Knowledge Base", faq: "FAQ" };
  function showView(name) {
    state.view = name;
    document.querySelectorAll(".view").forEach((v) => v.classList.add("hidden"));
    $("view-" + name).classList.remove("hidden");
    $("view-title").textContent = VIEW_TITLES[name] || "HR Assistant";
    document.querySelectorAll("[data-nav]").forEach((b) => {
      const active = b.dataset.nav === name || (name === "chat" && b.dataset.nav === "chat");
      b.classList.toggle("active", active);
      if (active) b.setAttribute("aria-current", "page"); else b.removeAttribute("aria-current");
    });
    if (name === "history") renderHistory();
    if (name === "kb") { refreshUploadState(); loadDocuments(); }
    if (!window.matchMedia("(min-width: 1024px)").matches) setSidebar(false);
  }

  /* ------------------------------------------------------------------ banner */
  function showBanner(kind, text) {
    const banner = $("banner");
    if (!text) { banner.classList.add("hidden"); return; }
    const styles = {
      error: "border-red-200 bg-red-50 text-red-800",
      warn: "border-amber-200 bg-amber-50 text-amber-900",
      info: "border-sky-200 bg-sky-50 text-sky-900",
    };
    banner.className = "mb-2 rounded-xl border px-3 py-2 text-[13px] " + styles[kind];
    banner.textContent = text;
  }

  async function checkHealth() {
    try {
      const h = await api("/api/health");
      state.employeeDataEnabled = !!h.employee_data_enabled;
      state.uploadsRequireAdmin = h.uploads_require_admin !== false;
      if (state.view === "kb") refreshUploadState();
      const g = h.groq;
      const e = h.sentence_transformers;
      if (!h.vector_store.available) {
        showBanner("error", "The document index is unavailable: " + (h.vector_store.error || "unknown error"));
      } else if (!e.available) {
        showBanner("error", e.error || "The local embedding model is unavailable.");
      } else if (!g.reachable) {
        showBanner("error", g.error || "GroqCloud is not reachable. Check the server configuration.");
      } else if (!g.model_available) {
        showBanner("error", g.error || `Groq model ${g.model} is unavailable to this account.`);
      } else if (h.vector_store.chunk_count === 0) {
        showBanner("info", "No documents are indexed yet. Open “HR Knowledge Base” to upload your HR policies.");
      } else {
        showBanner("info", "");
      }
      return h;
    } catch (err) {
      showBanner("error", err.message);
      return null;
    }
  }

  /* -------------------------------------------------------------------- chat */
  function scrollToBottom() {
    const panel = $("chat-panel");
    panel.scrollTop = panel.scrollHeight;
  }

  function logoAvatar() {
    const img = el("img", "mt-1 h-7 w-7 shrink-0");
    img.src = "logo.svg"; img.alt = "";
    return img;
  }

  function renderUserMessage(text) {
    const wrap = el("div", "flex justify-end");
    const bubble = el("div", "bubble-user max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-md px-4 py-2.5 text-[13px] leading-relaxed shadow-sm", text);
    wrap.appendChild(bubble);
    return wrap;
  }

  function finalAnswerText(content) {
    if (typeof content !== "string") return "";

    const answerHeading = /^\s*(?:\*\*)?Answer\s*:\s*(?:\*\*)?\s*/im;
    const otherSectionHeading = /^\s*(?:\*\*)?(?:Evidence from the documents|Interpretation and limitations)\s*:\s*(?:\*\*)?/im;
    const answerMatch = answerHeading.exec(content);
    let answer = content;

    if (answerMatch) {
      const remainder = content.slice(answerMatch.index + answerMatch[0].length);
      const sectionMatch = otherSectionHeading.exec(remainder);
      answer = sectionMatch ? remainder.slice(0, sectionMatch.index) : remainder;
    } else {
      const sectionMatch = otherSectionHeading.exec(content);
      if (sectionMatch) answer = content.slice(0, sectionMatch.index);
    }

    return answer
      .replace(/\s*\[[^\]\r\n]+,\s*(?:p\.\s*\d+|record\s+[^\]\r\n]+)\]/gi, "")
      .trim();
  }

  function renderSources(sources) {
    if (!Array.isArray(sources)) return null;
    const filenames = [];
    const seen = new Set();
    sources.forEach((source) => {
      if (!source || typeof source.source !== "string") return;
      const filename = source.source.trim().split(/[\\/]/).pop();
      const key = filename.toLowerCase();
      if (!filename || seen.has(key)) return;
      seen.add(key);
      filenames.push(filename);
    });
    if (!filenames.length) return null;
    return el("p", "mt-2 text-[11px] text-slate-500", "Source: " + filenames.join(", "));
  }

  function renderAssistantMessage(msg, options = {}) {
    const row = el("div", "flex gap-2.5");
    row.appendChild(logoAvatar());
    const bubble = el("div", "bubble-bot min-w-0 max-w-[92%] rounded-2xl rounded-tl-md px-4 py-3 shadow-sm");
    if (msg.error) {
      bubble.className = "min-w-0 max-w-[92%] rounded-2xl rounded-tl-md border border-red-200 bg-red-50 px-4 py-3 text-[13px] text-red-800";
      bubble.appendChild(el("p", "font-medium", "Something went wrong"));
      bubble.appendChild(el("p", "mt-1", msg.content));
      if (options.onRetry) {
        const retry = el("button", "mt-2 rounded-full border border-red-300 bg-white px-3 py-1 text-[12px] font-medium text-red-700 hover:bg-red-100", "Retry");
        retry.addEventListener("click", options.onRetry);
        bubble.appendChild(retry);
      }
    } else {
      const body = el("div", "md");
      body.innerHTML = renderMarkdown(finalAnswerText(msg.content));
      bubble.appendChild(body);
      const sourceLine = renderSources(msg.sources);
      if (sourceLine) bubble.appendChild(sourceLine);
    }
    row.appendChild(bubble);
    return row;
  }

  function renderLoading() {
    const row = el("div", "flex gap-2.5");
    row.id = "loading-row";
    row.appendChild(logoAvatar());
    const bubble = el("div", "bubble-bot flex items-center gap-3 rounded-2xl rounded-tl-md px-4 py-3 shadow-sm");
    const dots = el("div", "flex items-center gap-1.5");
    dots.setAttribute("aria-hidden", "true");
    for (let i = 0; i < 3; i++) dots.appendChild(el("span", "dot"));
    bubble.appendChild(dots);
    bubble.appendChild(el("span", "text-[12px] text-slate-500", "Searching the HR documents…"));
    row.appendChild(bubble);
    return row;
  }

  function showMessages(show) {
    $("welcome").classList.toggle("hidden", show);
    $("messages").classList.toggle("hidden", !show);
  }

  function renderConversation() {
    const container = $("messages");
    container.replaceChildren();
    const conv = currentConversation();
    if (!conv || !conv.messages || conv.messages.length === 0) { showMessages(false); return; }
    showMessages(true);
    conv.messages.forEach((m) => container.appendChild(m.role === "user" ? renderUserMessage(m.content) : renderAssistantMessage(m)));
    scrollToBottom();
  }

  function setBusy(busy) {
    state.busy = busy;
    $("btn-send").disabled = busy;
    $("question").disabled = busy;
    if (!busy) $("question").focus();
  }

  async function sendQuestion(question) {
    if (state.busy || !question.trim()) return;
    let conv = currentConversation();
    let conversationCreated = !conv || !conv.historyUnavailable;
    if (!conv) {
      const created = {
        id: uid(),
        title: question.trim().slice(0, 60),
        updated: Date.now(),
      };
      try {
        conv = await api("/api/conversations", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(created),
        });
      } catch (err) {
        conv = Object.assign({}, created, { messages: [], historyUnavailable: true });
        conversationCreated = false;
        toast(`Could not create conversation history: ${err.message}`);
      }
      if (!conv.messages) conv.messages = [];
      state.conversations.unshift(conv);
      state.currentId = conv.id;
    }
    const content = question.trim();
    const userPosition = conv.messages.length;
    conv.messages.push({ role: "user", content });
    conv.updated = Date.now();
    conv.message_count = conv.messages.length;
    conv.question_count = (conv.question_count || 0) + 1;
    state.conversations.sort((a, b) => b.updated - a.updated);
    showMessages(true);
    const container = $("messages");
    container.appendChild(renderUserMessage(content));
    const loading = renderLoading();
    container.appendChild(loading);
    scrollToBottom();
    setBusy(true);

    const payload = { question: question.trim(), conversation_id: conv.id };
    if (state.topK) payload.top_k = state.topK;
    let userMessageSaved = conversationCreated;
    if (userMessageSaved) {
      try {
        await api(`/api/conversations/${encodeURIComponent(conv.id)}/messages`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            position: userPosition,
            role: "user",
            content,
            metadata: {},
            updated: conv.updated,
          }),
        });
      } catch (err) {
        userMessageSaved = false;
        toast(`Question history could not be saved: ${err.message}`);
      }
    }
    try {
      const data = await api("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const msg = { role: "assistant", content: data.answer, sources: data.sources, citations: data.citations, all_verified: data.all_verified, grounded: data.grounded, elapsed_ms: data.elapsed_ms, searched_for: data.standalone_question || null };
      const assistantPosition = conv.messages.length;
      if (userMessageSaved) {
        try {
          await api(`/api/conversations/${encodeURIComponent(conv.id)}/messages`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              position: assistantPosition,
              role: "assistant",
              content: msg.content,
              metadata: {
                sources: msg.sources,
                citations: msg.citations,
                all_verified: msg.all_verified,
                grounded: msg.grounded,
                elapsed_ms: msg.elapsed_ms,
                searched_for: msg.searched_for,
              },
              updated: Date.now(),
            }),
          });
        } catch (err) {
          toast(`Answer history could not be saved: ${err.message}`);
        }
      }
      conv.messages.push(msg);
      conv.updated = Math.max(conv.updated, Date.now());
      conv.message_count = conv.messages.length;
      loading.replaceWith(renderAssistantMessage(msg));
    } catch (err) {
      const errorMsg = { role: "assistant", content: err.message, error: true };
      loading.replaceWith(renderAssistantMessage(errorMsg, {
        onRetry: () => {
          conv.messages.splice(userPosition);
          conv.message_count = conv.messages.length;
          conv.question_count = Math.max(0, conv.question_count - 1);
          renderConversation();
          sendQuestion(question);
        },
      }));
      checkHealth();
    } finally {
      setBusy(false);
      scrollToBottom();
    }
  }

  function newChat() {
    state.currentId = null;
    renderConversation();
    showView("chat");
    $("question").value = "";
    autoGrow();
    $("question").focus();
  }

  function autoGrow() {
    const q = $("question");
    q.style.height = "auto";
    q.style.height = Math.min(q.scrollHeight, 128) + "px";
  }

  /* ----------------------------------------------------------------- history */
  function renderHistory() {
    const list = $("history-list");
    list.replaceChildren();
    if (!state.conversations.length) {
      list.appendChild(el("li", "rounded-xl border border-dashed border-slate-300 bg-white p-6 text-center text-[13px] text-slate-500", "No conversations yet. Ask a question to start one."));
      return;
    }
    state.conversations.forEach((conv) => {
      const li = el("li", "flex items-center gap-2 rounded-xl border border-slate-200 bg-white p-3");
      const open = el("button", "min-w-0 flex-1 text-left");
      open.appendChild(el("p", "truncate text-[13px] font-medium text-slate-800", conv.title));
      const count = conv.question_count || 0;
      open.appendChild(el("p", "text-[11.5px] text-slate-400", `${new Date(conv.updated).toLocaleString()} · ${count} question${count === 1 ? "" : "s"}`));
      open.addEventListener("click", () => openConversation(conv));
      const del = el("button", "rounded-full px-2.5 py-1 text-[12px] text-red-600 hover:bg-red-50", "Delete");
      del.addEventListener("click", async () => {
        try {
          await api(`/api/conversations/${encodeURIComponent(conv.id)}`, { method: "DELETE" });
        } catch (err) {
          toast(`Could not delete conversation: ${err.message}`);
          return;
        }
        state.conversations = state.conversations.filter((c) => c.id !== conv.id);
        if (state.currentId === conv.id) state.currentId = null;
        renderHistory(); renderConversation();
      });
      li.append(open, del);
      list.appendChild(li);
    });
  }

  async function openConversation(conv) {
    try {
      const messages = await api(`/api/conversations/${encodeURIComponent(conv.id)}/messages`);
      conv.messages = messages.map(({ position, ...message }) => message);
      state.currentId = conv.id;
      renderConversation();
      showView("chat");
    } catch (err) {
      toast(`Could not load conversation: ${err.message}`);
    }
  }

  /* --------------------------------------------------------- knowledge base */
  const canUpload = () => state.admin || !state.uploadsRequireAdmin;

  function refreshAcceptedTypes() {
    const types = state.admin ? ".pdf,.docx,.csv" : ".pdf,.docx";
    $("file-input").accept = types;
    $("accepted-types").textContent = state.admin ? "PDF, DOCX and CSV (employee data, restricted)" : "PDF, DOCX";
  }

  function refreshUploadState() {
    refreshAcceptedTypes();
    const allowed = canUpload();
    const zone = $("dropzone");
    zone.classList.toggle("opacity-50", !allowed);
    zone.classList.toggle("pointer-events-none", !allowed);
    $("file-input").disabled = !allowed;
    const lock = $("upload-lock");
    lock.classList.toggle("hidden", allowed);
    if (!allowed) {
      lock.textContent = state.employeeDataEnabled
        ? "Uploading changes what the assistant knows, so it needs the administrator key. Open Settings and unlock it."
        : "Uploading is disabled because no administrator key is configured. Set ADMIN_API_KEY in the server's .env file, restart the server, then unlock it in Settings.";
    }
  }

  function statusBadge(doc) {
    const map = {
      indexed: ["Indexed", "bg-emerald-100 text-emerald-800"],
      not_indexed: ["Not indexed", "bg-amber-100 text-amber-800"],
      index_only: ["Index only (file missing)", "bg-slate-200 text-slate-700"],
    };
    const [label, cls] = map[doc.status] || [doc.status, "bg-slate-100 text-slate-700"];
    const span = el("span", "rounded-full px-2 py-0.5 text-[10.5px] font-medium " + cls, label);
    return span;
  }

  async function loadDocuments() {
    const rows = $("doc-rows");
    try {
      const data = await api("/api/documents");
      rows.replaceChildren();
      $("doc-empty").classList.toggle("hidden", data.documents.length > 0);
      data.documents.forEach((doc) => {
        const tr = el("tr");
        const nameCell = el("td", "px-4 py-2.5 font-medium text-slate-800");
        nameCell.appendChild(el("span", "break-all", doc.name));
        if (doc.sensitive) nameCell.appendChild(el("span", "ml-2 rounded-full bg-red-100 px-2 py-0.5 text-[10px] font-medium text-red-700", "Restricted"));
        tr.appendChild(nameCell);
        tr.appendChild(el("td", "px-3 py-2.5 uppercase text-slate-500", doc.doc_type));
        tr.appendChild(el("td", "px-3 py-2.5 text-slate-500", formatBytes(doc.size_bytes)));
        tr.appendChild(el("td", "px-3 py-2.5 text-slate-700", String(doc.chunks)));
        const statusCell = el("td", "px-3 py-2.5"); statusCell.appendChild(statusBadge(doc)); tr.appendChild(statusCell);
        tr.appendChild(el("td", "px-3 py-2.5 text-slate-500", doc.modified ? doc.modified.replace("T", " ").replace("+00:00", "") : "–"));
        rows.appendChild(tr);
      });
      $("kb-summary").textContent = `${data.documents.length} document${data.documents.length === 1 ? "" : "s"} · ${data.total_chunks} chunks`;
    } catch (err) {
      rows.replaceChildren();
      $("kb-summary").textContent = "";
      toast(err.message, 5000);
    }
  }

  function addUploadResult(kind, title, detail) {
    const styles = { ok: "border-emerald-200 bg-emerald-50 text-emerald-900", err: "border-red-200 bg-red-50 text-red-800", busy: "border-slate-200 bg-white text-slate-700" };
    const li = el("li", "rounded-xl border px-3 py-2 text-[12.5px] " + styles[kind]);
    li.appendChild(el("p", "font-medium", title));
    if (detail) li.appendChild(el("p", "mt-0.5 opacity-90", detail));
    $("upload-results").prepend(li);
    return li;
  }

  async function uploadFiles(fileList) {
    if (!canUpload()) { toast("Uploading needs the administrator key (see Settings)."); return; }
    const files = Array.from(fileList);
    for (const file of files) {
      const ext = (file.name.split(".").pop() || "").toLowerCase();
      const allowed = state.admin ? ["pdf", "docx", "csv"] : ["pdf", "docx"];
      if (!allowed.includes(ext)) {
        addUploadResult("err", file.name, ext === "csv"
          ? "CSV files hold employee data and can only be uploaded after unlocking employee data in Settings."
          : "Unsupported file type. Use PDF or DOCX.");
        continue;
      }
      const pending = addUploadResult("busy", file.name, "Processing and indexing… (large files can take a minute)");
      const form = new FormData();
      form.append("file", file);
      try {
        const r = await api("/api/documents/upload", { method: "POST", body: form });
        pending.remove();
        addUploadResult("ok", `${r.filename}: indexed`,
          `${r.sections} section(s) read · ${r.chunks_created} chunks created · ${r.chunks_added} added` +
          (r.duplicates_skipped ? ` · ${r.duplicates_skipped} unchanged skipped` : "") +
          (r.stale_removed ? ` · ${r.stale_removed} outdated removed` : "") + (r.replaced ? " · replaced existing file" : ""));
      } catch (err) {
        pending.remove();
        addUploadResult("err", file.name, err.message);
      }
    }
    await loadDocuments();
    checkHealth();
  }

  /* --------------------------------------------------------------------- FAQ */
  const FAQ = [
    ["Where do the answers come from?", "Only from the documents indexed in the HR Knowledge Base. Each answer lists the retrieved excerpts with file name and page, so you can check it yourself."],
    ["Why does the assistant sometimes say the documents do not contain enough information?", "When no excerpt is relevant enough, the assistant says so instead of guessing. Try rephrasing the question, or check that the right documents are uploaded."],
    ["Can the assistant make mistakes?", "Yes. It is a small local language model. It can misread a table or a rule. Always open the cited excerpt for decisions that matter, and confirm with HR."],
    ["Does it remember my previous question?", "Yes, within the same conversation: the last few questions are used to understand follow-ups such as “and for juniors?”. The assistant shows how it understood your follow-up. This memory is kept in the server's memory only and is lost when the server restarts; “New Chat” starts a fresh one."],
    ["Is my data sent to the internet?", "Embeddings and the Chroma index stay on the server. For answer generation, the question and retrieved excerpts are sent to GroqCloud. Follow your organization’s data policy before using sensitive HR information."],
    ["Who can see employee records or change the documents?", "Nobody by default. Employee data (CSV files) is only searchable and listable, and any upload is only possible, after an administrator key is configured on the server and entered in Settings."],
    ["Where are my conversations stored?", "Conversations are stored in the FastAPI server's SQLite database. The previous browser history is kept in local storage after migration; Logout does not delete conversation history."],
  ];
  function renderFaq() {
    const root = $("faq-list");
    FAQ.forEach(([q, a]) => {
      const d = el("details", "rounded-xl border border-slate-200 bg-white");
      const s = el("summary", "flex items-center justify-between px-4 py-3 text-[13px] font-medium text-slate-800");
      s.append(el("span", "", q), el("span", "text-slate-400", "+"));
      d.append(s, el("p", "px-4 pb-3 text-[12.5px] leading-relaxed text-slate-600", a));
      root.appendChild(d);
    });
  }

  /* ------------------------------------------------------------------ modals */
  function openModal(title, buildBody) {
    $("modal-title").textContent = title;
    const body = $("modal-body");
    body.replaceChildren();
    buildBody(body);
    $("modal-root").classList.remove("hidden");
    $("modal-close").focus();
  }
  function closeModal() { $("modal-root").classList.add("hidden"); }

  function openHelp() {
    openModal("Help & Support", (body) => {
      const items = [
        "Ask a question in plain language, for example “How many days of annual leave can be carried over?”.",
        "Open “Sources” under an answer to see the file, page and the exact excerpt that was used.",
        "Upload new policies in HR Knowledge Base. Re-uploading a file with the same name updates it.",
        "If you see a red banner, check GroqCloud connectivity and the configured model, the local embedding model, and the vector index.",
        "For a question about a specific employee or a policy exception, contact your HR department.",
      ];
      const ul = el("ul", "list-disc space-y-2 pl-5");
      items.forEach((t) => ul.appendChild(el("li", "", t)));
      body.appendChild(ul);
    });
  }

  function openSettings() {
    openModal("Settings", (body) => {
      const wrap = el("div", "space-y-5");

      const k = el("div");
      k.appendChild(el("label", "mb-1 block text-[12px] font-medium text-slate-700", "Excerpts per question (top-k)"));
      const select = el("select", "w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-[13px]");
      [["", "Server default"], ["3", "3"], ["5", "5"], ["8", "8"], ["10", "10"]].forEach(([v, t]) => {
        const o = el("option", "", t); o.value = v; if (String(state.topK || "") === v) o.selected = true; select.appendChild(o);
      });
      select.addEventListener("change", () => { state.topK = select.value ? Number(select.value) : null; });
      k.appendChild(select);
      k.appendChild(el("p", "mt-1 text-[11.5px] text-slate-400", "More excerpts can help broad questions but slow down the answer."));
      wrap.appendChild(k);

      const a = el("div", "rounded-xl border border-slate-200 bg-slate-50 p-3");
      a.appendChild(el("p", "text-[12px] font-semibold text-slate-800", "Administrator access"));
      const status = el("p", "mt-1 text-[12px] " + (state.admin ? "text-emerald-700" : "text-slate-500"),
        state.admin ? "Unlocked for this browser session: uploads and restricted employee data are available." : state.employeeDataEnabled
          ? "Locked. Enter the administrator key to upload documents and to search employee records."
          : "Disabled: no administrator key is configured on the server (ADMIN_API_KEY), so uploads and employee data are switched off.");
      a.appendChild(status);
      if (state.employeeDataEnabled && !state.admin) {
        const input = el("input", "mt-2 w-full rounded-lg border border-slate-300 px-3 py-2 text-[13px]");
        input.type = "password"; input.placeholder = "Administrator key"; input.autocomplete = "off";
        const btn = el("button", "mt-2 rounded-lg bg-slate-900 px-3 py-2 text-[12.5px] font-medium text-white hover:bg-slate-700", "Unlock");
        btn.addEventListener("click", async () => {
          state.adminKey = input.value.trim();
          try {
            const s = await api("/api/auth/status");
            if (s.admin) {
              state.admin = true;
              storage.set(window.sessionStorage, ADMIN_KEY_STORE, state.adminKey);
              toast("Administrator access unlocked for this session");
              closeModal(); refreshUploadState(); if (state.view === "kb") loadDocuments();
            } else { state.adminKey = ""; status.textContent = "That key is not valid."; status.className = "mt-1 text-[12px] text-red-600"; }
          } catch (err) { state.adminKey = ""; status.textContent = err.message; status.className = "mt-1 text-[12px] text-red-600"; }
        });
        a.append(input, btn);
      }
      wrap.appendChild(a);

      const h = el("div");
      h.appendChild(el("p", "text-[12px] font-semibold text-slate-800", "Server status"));
      const hs = el("p", "mt-1 text-[12px] text-slate-500", "Checking…");
      h.appendChild(hs);
      wrap.appendChild(h);
      api("/api/health").then((r) => {
        hs.textContent = `Overall: ${r.status} · Groq ${r.groq.reachable && r.groq.model_available ? "ok" : "unavailable"} (${r.groq.model}) · ` +
          `embedding ${r.sentence_transformers.available ? "ok" : "unavailable"} (${r.sentence_transformers.model}) · ` +
          `${r.vector_store.chunk_count ?? "?"} chunks indexed`;
      }).catch((e) => { hs.textContent = e.message; });

      const info = el("p", "text-[11.5px] text-slate-400", "API: " + (API_BASE || window.location.origin));
      wrap.appendChild(info);
      body.appendChild(wrap);
    });
  }

  function logout() {
    if (!window.confirm("Log out? This locks administrator access. Saved conversations will remain available.")) return;
    state.adminKey = ""; state.admin = false;
    storage.remove(window.sessionStorage, ADMIN_KEY_STORE);
    state.currentId = null;
    renderConversation(); showView("chat");
    toast("Logged out. Conversation history was kept.", 5000);
  }

  /* -------------------------------------------------------------------- init */
  async function restoreAdminSession() {
    const saved = storage.get(window.sessionStorage, ADMIN_KEY_STORE);
    if (!saved) return;
    state.adminKey = saved;
    try {
      const s = await api("/api/auth/status");
      state.admin = !!s.admin;
      if (!state.admin) { state.adminKey = ""; storage.remove(window.sessionStorage, ADMIN_KEY_STORE); }
    } catch (_) { state.adminKey = ""; }
  }

  function bindEvents() {
    document.querySelectorAll("[data-nav]").forEach((b) => b.addEventListener("click", () => {
      showView(b.dataset.nav);
      if (b.dataset.nav === "chat" && !currentConversation()) renderConversation();
    }));
    $("btn-menu").addEventListener("click", () => setSidebar(!sidebarOpen()));
    $("sidebar-overlay").addEventListener("click", () => setSidebar(false));
    $("btn-new-chat").addEventListener("click", newChat);
    $("btn-help").addEventListener("click", openHelp);
    $("btn-settings").addEventListener("click", openSettings);
    $("btn-logout").addEventListener("click", logout);
    $("modal-close").addEventListener("click", closeModal);
    $("modal-root").addEventListener("click", (e) => { if (e.target === $("modal-root")) closeModal(); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeModal(); });

    $("chat-form").addEventListener("submit", (e) => {
      e.preventDefault();
      const q = $("question").value;
      if (!q.trim()) return;
      $("question").value = ""; autoGrow();
      sendQuestion(q);
    });
    $("question").addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); $("chat-form").requestSubmit(); }
    });
    $("question").addEventListener("input", autoGrow);

    $("btn-clear-history").addEventListener("click", async () => {
      if (!state.conversations.length || !window.confirm("Delete all conversations saved on this server? This affects every browser using it.")) return;
      try {
        await api("/api/conversations", { method: "DELETE" });
      } catch (err) {
        toast(`Could not delete conversation history: ${err.message}`);
        return;
      }
      state.conversations = []; state.currentId = null;
      renderHistory(); renderConversation();
    });
    $("btn-kb-refresh").addEventListener("click", () => { loadDocuments(); checkHealth(); });
    $("file-input").addEventListener("change", (e) => { if (e.target.files.length) uploadFiles(e.target.files); e.target.value = ""; });
    const dz = $("dropzone");
    ["dragenter", "dragover"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.add("border-brand-blue"); }));
    ["dragleave", "drop"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.remove("border-brand-blue"); }));
    dz.addEventListener("drop", (e) => { if (e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files); });
  }

  async function init() {
    renderFaq();
    bindEvents();
    setSidebar(window.matchMedia("(min-width: 1024px)").matches);
    showView("chat");
    await restoreAdminSession();
    try {
      await loadConversations();
    } catch (err) {
      state.conversations = [];
      toast(`Could not load conversation history: ${err.message}`, 7000);
    }
    renderConversation();
    renderHistory();
    await checkHealth();
    state.healthTimer = setInterval(checkHealth, 30000);
  }

  // Exposed for debugging and automated checks only (no secrets).
  window.__hrAssistant = { sendQuestion, showView };
  document.addEventListener("DOMContentLoaded", init);
})();

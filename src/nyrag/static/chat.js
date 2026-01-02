const chatEl = document.getElementById("chat");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("send");
const statsEl = document.getElementById("corpus-stats");

// Settings Modal
const settingsBtn = document.getElementById("settings-btn");
const modal = document.getElementById("settings-modal");
const closeBtn = modal.querySelector(".close-btn");
const saveBtn = document.getElementById("save-settings");

settingsBtn.onclick = () => modal.style.display = "block";
closeBtn.onclick = () => modal.style.display = "none";
saveBtn.onclick = () => modal.style.display = "none";

// Crawl Modal
const crawlBtn = document.getElementById("crawl-btn");
const crawlModal = document.getElementById("crawl-modal");
const closeCrawlBtn = crawlModal.querySelector(".close-crawl-btn");
const startCrawlBtn = document.getElementById("start-crawl-btn");
const stopCrawlBtn = document.getElementById("stop-crawl-btn");
const terminalLogs = document.getElementById("terminal-logs");
const terminalStatus = document.getElementById("terminal-status");
const yamlContainer = document.getElementById("interactive-yaml-container");
const exampleSelect = document.getElementById("example-select");
const projectSelector = document.getElementById("project-selector");

// Track current event source for stopping
let currentEventSource = null;

// Internal Config State
let currentConfig = {};
let configOptions = {}; // Schema loaded from API
let exampleConfigs = {}; // Example configs from API

// Fallback Default if file is empty
const FALLBACK_CONFIG = {
  name: "new-project",
  mode: "web",
  start_loc: "https://example.com",
  exclude: [],
  crawl_params: {
    respect_robots_txt: true,
    follow_subdomains: true,
    user_agent_type: "chrome",
    aggressive: false,
    strict_mode: false,
    custom_user_agent: "",
    allowed_domains: []
  },
  doc_params: {
    recursive: true,
    include_hidden: false,
    follow_symlinks: false,
    max_file_size_mb: 10,
    file_extensions: [".pdf", ".docx", ".txt", ".md"]
  },
  rag_params: {
    embedding_model: "sentence-transformers/all-MiniLM-L6-v2",
    embedding_dim: 384,
    chunk_size: 1024,
    chunk_overlap: 50,
    distance_metric: "angular"
  },
  llm_config: {
    base_url: "https://openrouter.ai/api/v1",
    model: "gpt-4",
    api_key: ""
  }
};

// =========================================================================
// RENDERER: The Core "Magic" Function
// =========================================================================
function renderConfigEditor() {
  yamlContainer.innerHTML = '';
  if (!configOptions || Object.keys(configOptions).length === 0) {
    yamlContainer.innerHTML = '<div style="color: #666; padding: 1rem;">Loading options...</div>';
    return;
  }

  // Recursive render function
  function renderField(key, schemaItem, value, indentLevel, parentObj) {
    const line = document.createElement('div');
    line.className = 'yaml-line';

    // Indentation
    const indentSpan = document.createElement('span');
    indentSpan.className = 'yaml-indent';
    indentSpan.innerHTML = '&nbsp;'.repeat(indentLevel * 2);
    line.appendChild(indentSpan);

    // Key
    const keySpan = document.createElement('span');
    keySpan.className = 'yaml-key';
    keySpan.textContent = key + ':';
    line.appendChild(keySpan);

    // Value Control
    if (schemaItem.type === 'nested') {
      yamlContainer.appendChild(line);

      // Ensure object exists
      if (value === undefined || value === null) {
        // If it's a nested object but missing in config, maybe init it?
        // For now, let's just skip unless we want to force it.
        // But usually deepMerge handles init.
        return;
      }

      const fields = schemaItem.fields || {};
      Object.keys(fields).forEach(subKey => {
        const subSchema = fields[subKey];
        renderField(subKey, subSchema, value[subKey], indentLevel + 1, value);
      });
      return;
    }

    if (schemaItem.type === 'string' || schemaItem.type === 'number') {
      const input = document.createElement('input');
      input.type = schemaItem.type === 'number' ? 'number' : (schemaItem.masked ? 'password' : 'text');
      input.value = value !== undefined ? value : '';
      input.className = 'yaml-input';

      // Dynamic Placeholder
      if (key === 'start_location') {
        input.placeholder = currentConfig.mode === 'web' ? 'https://example.com' : '/path/to/docs';
      } else {
        input.placeholder = 'null';
      }

      input.onchange = (e) => {
        let val = e.target.value;
        if (schemaItem.type === 'number') val = parseFloat(val);
        parentObj[key] = val;
      };
      line.appendChild(input);
    } else if (schemaItem.type === 'boolean') {
      const select = document.createElement('select');
      select.className = 'yaml-select ' + (value ? 'bool-true' : 'bool-false');

      const optTrue = new Option('true', 'true');
      const optFalse = new Option('false', 'false');
      select.add(optTrue);
      select.add(optFalse);
      select.value = !!value;

      select.onchange = (e) => {
        const val = e.target.value === 'true';
        parentObj[key] = val;
        select.className = 'yaml-select ' + (val ? 'bool-true' : 'bool-false');
      };
      line.appendChild(select);
    } else if (schemaItem.type === 'select') {
      const select = document.createElement('select');
      select.className = 'yaml-select';
      (schemaItem.options || []).forEach(optVal => {
        select.add(new Option(optVal, optVal));
      });
      select.value = value || schemaItem.options[0];
      select.onchange = async (e) => {
        parentObj[key] = e.target.value;
        // TRIGGER RE-FETCH if mode changes
        if (key === 'mode') {
          await loadSchema(e.target.value);
          renderConfigEditor();
        }
      };
      line.appendChild(select);
    } else if (schemaItem.type === 'list') {
      yamlContainer.appendChild(line);

      const list = Array.isArray(value) ? value : [];
      parentObj[key] = list; // ensure array reference

      const listContainer = document.createElement('div');

      const renderList = () => {
        listContainer.innerHTML = '';
        list.forEach((item, idx) => {
          const itemLine = document.createElement('div');
          itemLine.className = 'yaml-line';

          const iSpan = document.createElement('span');
          iSpan.className = 'yaml-indent';
          iSpan.innerHTML = '&nbsp;'.repeat((indentLevel + 1) * 2);

          const dash = document.createElement('span');
          dash.className = 'yaml-dash';
          dash.textContent = '- ';

          const input = document.createElement('input');
          input.className = 'yaml-input';
          input.value = item;
          // width handled by flex
          input.onchange = (e) => {
            list[idx] = e.target.value;
          };

          const delBtn = document.createElement('span');
          delBtn.innerHTML = '&times;';
          delBtn.style.color = '#ff5555';
          delBtn.style.cursor = 'pointer';
          delBtn.style.marginLeft = '8px';
          delBtn.onclick = () => {
            list.splice(idx, 1);
            renderList();
          };

          itemLine.appendChild(iSpan);
          itemLine.appendChild(dash);
          itemLine.appendChild(input);
          itemLine.appendChild(delBtn);
          listContainer.appendChild(itemLine);
        });

        const newLine = document.createElement('div');
        newLine.className = 'yaml-line';
        const niSpan = document.createElement('span');
        niSpan.className = 'yaml-indent';
        niSpan.innerHTML = '&nbsp;'.repeat((indentLevel + 1) * 2);

        const nDash = document.createElement('span');
        nDash.className = 'yaml-dash';
        nDash.textContent = '+ ';
        nDash.style.opacity = '0.5';
        nDash.style.cursor = 'pointer';

        const nInput = document.createElement('input');
        nInput.className = 'yaml-input';
        nInput.placeholder = '(add item)';
        nInput.onchange = (e) => {
          if (e.target.value) {
            list.push(e.target.value);
            renderList();
            nInput.focus();
          }
        };

        newLine.appendChild(niSpan);
        newLine.appendChild(nDash);
        newLine.appendChild(nInput);
        listContainer.appendChild(newLine);
      };

      renderList();
      yamlContainer.appendChild(listContainer);
      return;
    }

    yamlContainer.appendChild(line);
  }

  Object.keys(configOptions).forEach(key => {
    // Top-level render
    renderField(key, configOptions[key], currentConfig[key], 0, currentConfig);
  });
}

// API Interactions
async function loadSchema(mode) {
  try {
    const res = await fetch(`/config/options?mode=${mode}`);
    configOptions = await res.json();
  } catch (e) {
    console.error("Failed to load schema options", e);
    terminalLogs.textContent = "Error loading schema: " + e.message;
  }
}

async function loadProjectConfig() {
  try {
    const projectName = projectSelector.value;
    const url = projectName
      ? `/config?project_name=${encodeURIComponent(projectName)}`
      : "/config";
    const res = await fetch(url);
    const data = await res.json();

    let parsed = {};
    if (data.content) {
      parsed = jsyaml.load(data.content);
    }

    // Merge with fallback to ensure structure
    currentConfig = deepMerge(JSON.parse(JSON.stringify(FALLBACK_CONFIG)), parsed);

    // Determine mode to load correct schema
    const mode = currentConfig.mode || "web";
    await loadSchema(mode);

    renderConfigEditor();
    terminalStatus.textContent = "Ready";
    terminalStatus.style.color = "var(--accent-color)";

  } catch (e) {
    console.error("Failed to load project config", e);
    terminalStatus.textContent = "Load Error";
    terminalStatus.style.color = "#ef4444";
  }
}

async function saveProjectConfig() {
  const yamlStr = jsyaml.dump(currentConfig);
  await fetch("/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content: yamlStr })
  });
}

// Load and populate example configs
async function loadExamples() {
  try {
    const res = await fetch("/config/examples");
    exampleConfigs = await res.json();

    // Populate dropdown
    exampleSelect.innerHTML = '<option value="">-- Select an example --</option>';
    Object.keys(exampleConfigs).forEach(name => {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      exampleSelect.appendChild(opt);
    });
  } catch (e) {
    console.error("Failed to load examples", e);
  }
}

// Handle example selection
exampleSelect.onchange = async () => {
  const name = exampleSelect.value;
  if (!name || !exampleConfigs[name]) return;

  try {
    const parsed = jsyaml.load(exampleConfigs[name]);
    // Merge: start with defaults, then override with example values
    currentConfig = deepMerge(JSON.parse(JSON.stringify(FALLBACK_CONFIG)), parsed);

    const mode = currentConfig.mode || "web";
    await loadSchema(mode);
    renderConfigEditor();

    terminalStatus.textContent = `Loaded: ${name}`;
    terminalStatus.style.color = "#10b981";
  } catch (e) {
    console.error("Failed to parse example", e);
  }
};

// Crawl Modal Logic
crawlBtn.onclick = () => {
  crawlModal.style.display = "block";
  loadExamples();
  loadProjectConfig();
};
closeCrawlBtn.onclick = () => crawlModal.style.display = "none";
window.onclick = (e) => {
  if (e.target == modal) modal.style.display = "none";
  if (e.target == crawlModal) crawlModal.style.display = "none";
};

startCrawlBtn.onclick = async () => {
  terminalLogs.textContent = "";
  terminalStatus.textContent = "Saving...";
  terminalStatus.style.color = "#fbbf24"; // Amber

  try {
    // 1. Save Config First
    await saveProjectConfig();

    terminalStatus.textContent = "Starting...";
    terminalStatus.style.color = "var(--accent-color)";

    // 2. Start Crawl with the YAML content
    const yamlStr = jsyaml.dump(currentConfig);

    const startRes = await fetch("/crawl/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config_yaml: yamlStr })
    });

    if (!startRes.ok) {
      terminalLogs.textContent += `Error starting crawl: ${startRes.statusText}\n`;
      terminalStatus.textContent = "Failed";
      terminalStatus.style.color = "#ef4444";
      return;
    }

    terminalStatus.textContent = "Running...";

    // 3. Connect to log stream using EventSource (SSE)
    currentEventSource = new EventSource("/crawl/logs");

    currentEventSource.onmessage = (event) => {
      if (event.data === "[PROCESS COMPLETED]") {
        currentEventSource.close();
        currentEventSource = null;
        terminalStatus.textContent = "Completed";
        terminalStatus.style.color = "#10b981";
        return;
      }
      terminalLogs.textContent += event.data + "\n";
      terminalLogs.scrollTop = terminalLogs.scrollHeight;
    };

    currentEventSource.onerror = () => {
      currentEventSource.close();
      currentEventSource = null;
      if (terminalStatus.textContent === "Running...") {
        terminalStatus.textContent = "Completed";
        terminalStatus.style.color = "#10b981";
      }
    };

  } catch (e) {
    terminalLogs.textContent += `Connection error: ${e.message}\n`;
    terminalStatus.textContent = "Error";
    terminalStatus.style.color = "#ef4444";
  }
};

// Stop Crawl Button
stopCrawlBtn.onclick = async () => {
  try {
    await fetch("/crawl/stop", { method: "POST" });
    if (currentEventSource) {
      currentEventSource.close();
      currentEventSource = null;
    }
    terminalLogs.textContent += "\n[Crawl stopped by user]\n";
    terminalStatus.textContent = "Stopped";
    terminalStatus.style.color = "#f59e0b"; // Orange
  } catch (e) {
    console.error("Failed to stop crawl", e);
  }
};

// Simple Deep Merge Helper
// (Same as before)
function deepMerge(target, source) {
  if (!source) return target;
  for (const key in source) {
    if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
      if (!target[key]) Object.assign(target, { [key]: {} });
      deepMerge(target[key], source[key]);
    } else {
      Object.assign(target, { [key]: source[key] });
    }
  }
  return target;
}

// ... Chat Message Logic
let chatHistory = [];

async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text) return;

  addMessage("user", text);
  inputEl.value = "";
  inputEl.style.height = 'auto'; // Reset height

  const hits = parseInt(document.getElementById("hits").value) || 5;
  const k = parseInt(document.getElementById("k").value) || 3;
  const query_k = parseInt(document.getElementById("query_k").value) || 3;

  // Create assistant bubble with structure
  const assistantId = addMessage("assistant", "");
  const bubble = document.getElementById(assistantId).querySelector(".bubble");

  // Add typing indicator
  const textEl = bubble.querySelector(".assistant-text");
  textEl.innerHTML = '<div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>';

  const metaEl = bubble.querySelector(".assistant-meta");

  // State for streaming
  let fullResponse = "";
  let thinkingContent = "";
  let thinkingEl = null;
  let thinkingBody = null;
  let statusEl = null;
  let isAnswerPhase = false;
  let hasStartedAnswering = false;
  let chunksCache = [];

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        history: chatHistory,
        hits: hits,
        k: k,
        query_k: query_k
      })
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Parse SSE
      while (true) {
        const doubleNewline = buffer.indexOf("\n\n");
        if (doubleNewline === -1) break;

        const raw = buffer.slice(0, doubleNewline);
        buffer = buffer.slice(doubleNewline + 2);

        const lines = raw.split("\n");
        let event = null;
        let dataStr = "";

        for (const line of lines) {
          if (line.startsWith("event: ")) {
            event = line.substring(7).trim();
          } else if (line.startsWith("data: ")) {
            dataStr += line.substring(6);
          }
        }

        if (event && dataStr) {
          try {
            const data = JSON.parse(dataStr);

            if (event === "status") {
              if (statusEl) statusEl.remove();
              statusEl = document.createElement("div");
              statusEl.className = "status-line";
              statusEl.textContent = data;
              metaEl.appendChild(statusEl);
              if (data.includes("Generating answer")) {
                isAnswerPhase = true;
              }
            } else if (event === "thinking") {
              // Only show thinking during answer phase (not query generation)
              if (!isAnswerPhase) continue;

              if (!thinkingEl) {
                thinkingEl = document.createElement("div");
                thinkingEl.className = "thinking-section";

                const header = document.createElement("div");
                header.className = "thinking-header";
                header.innerHTML = `
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
                  Thinking Process
                `;
                header.onclick = () => {
                  thinkingBody.classList.toggle("collapsed");
                };

                thinkingBody = document.createElement("div");
                thinkingBody.className = "thinking-content";

                thinkingEl.appendChild(header);
                thinkingEl.appendChild(thinkingBody);
                bubble.insertBefore(thinkingEl, textEl);
              }
              thinkingContent += data;
              thinkingBody.textContent = thinkingContent;
            } else if (event === "queries") {
              const details = document.createElement("details");
              details.className = "meta-details";
              const summary = document.createElement("summary");
              summary.textContent = `Queries (${data.length})`;
              details.appendChild(summary);

              const ul = document.createElement("ul");
              data.forEach((q) => {
                const li = document.createElement("li");
                li.textContent = q;
                ul.appendChild(li);
              });
              details.appendChild(ul);
              metaEl.appendChild(details);

              // Reset thinking for next phase
              thinkingEl = null;
              thinkingContent = "";
            } else if (event === "sources") {
              chunksCache = data;
              const locs = [...new Set(data.map(c => c.loc))];

              const details = document.createElement("details");
              details.className = "meta-details";
              const summary = document.createElement("summary");
              summary.textContent = `Sources (${locs.length})`;
              details.appendChild(summary);

              const ul = document.createElement("ul");
              locs.forEach((loc) => {
                const li = document.createElement("li");
                li.textContent = loc;
                ul.appendChild(li);
              });
              details.appendChild(ul);
              metaEl.appendChild(details);

              // Reset thinking for next phase
              thinkingEl = null;
              thinkingContent = "";
            } else if (event === "answer") {
              if (!hasStartedAnswering) {
                textEl.innerHTML = ""; // Remove typing indicator
                hasStartedAnswering = true;
              }
              fullResponse += data;
              textEl.innerHTML = DOMPurify.sanitize(marked.parse(fullResponse));
            } else if (event === "done") {
              if (statusEl) statusEl.remove();
              // Append collapsible references after the response
              if (chunksCache.length) {
                const wrap = document.createElement("details");
                wrap.className = "chunks";
                wrap.open = false;

                const listHtml = chunksCache
                  .map(
                    (c) =>
                      `<details class="chunk-item">
                        <summary>${c.loc} <span class="score">(${c.score ? c.score.toFixed(2) : '0.00'})</span></summary>
                        <div class="chunk-content">${c.chunk}</div>
                      </details>`
                  )
                  .join("");

                wrap.innerHTML = `<summary>Relevant sources (${chunksCache.length})</summary><div class="chunk-list">${listHtml}</div>`;
                bubble.appendChild(wrap);
              }
            } else if (event === "error") {
              if (statusEl) statusEl.remove();
              textEl.textContent += "\nError: " + data;
            }

            chatEl.scrollTop = chatEl.scrollHeight;

          } catch (e) {
            console.error("JSON parse error", e);
          }
        }
      }
    }

    chatHistory.push({ role: "user", content: text });
    chatHistory.push({ role: "assistant", content: fullResponse });

  } catch (e) {
    textEl.textContent = "Error: " + e.message;
  }
}

sendBtn.onclick = sendMessage;
inputEl.onkeydown = (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
};

function addMessage(role, text) {
  // Remove welcome message if it exists
  const welcome = document.querySelector('.welcome-message');
  if (welcome) welcome.remove();

  const id = "msg-" + Date.now();
  const div = document.createElement("div");
  div.className = `msg ${role}-msg`;
  div.id = id;

  if (role === 'user') {
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;
    div.appendChild(bubble);
  } else {
    div.innerHTML = `
      <div class="bubble">
        <div class="assistant-meta"></div>
        <div class="assistant-text"></div>
      </div>
    `;
  }
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
  return id;
}

async function fetchStats() {
  try {
    const res = await fetch("/stats");
    const data = await res.json();
    if (data.documents) {
      statsEl.textContent = `${data.documents} documents indexed`;
    } else {
      statsEl.textContent = "No documents indexed";
    }
  } catch (e) {
    console.error("Failed to fetch stats", e);
    statsEl.textContent = "Error loading stats";
  }
}

async function loadProjects() {
  try {
    const res = await fetch("/projects");
    const projects = await res.json();

    // Maintain default option
    projectSelector.innerHTML = '<option value="">-- Select Project --</option>';

    projects.forEach(p => {
      const opt = document.createElement("option");
      opt.value = p;
      opt.textContent = p;
      projectSelector.appendChild(opt);
    });

    // Auto-select if only one project
    if (projects.length === 1) {
      projectSelector.value = projects[0];
      await selectProject(projects[0]);
    }
  } catch (e) {
    console.error("Failed to load projects", e);
  }
}

async function selectProject(projectName) {
  if (!projectName) return;

  try {
    const res = await fetch("/projects/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project_name: projectName })
    });

    const data = await res.json();
    if (data.status === "success") {
      // Update active project indicator
      const indicator = document.getElementById("active-project-indicator");
      if (indicator) {
        indicator.textContent = `Project: ${projectName}`;
      }
      // Refresh stats
      fetchStats();
      chatHistory = [];
    }
  } catch (e) {
    console.error("Failed to select project", e);
  }
}

projectSelector.onchange = async () => {
  await selectProject(projectSelector.value);
};

// Auto-resize textarea
inputEl.addEventListener('input', function () {
  this.style.height = 'auto';
  this.style.height = (this.scrollHeight) + 'px';
  if (this.value === '') {
    this.style.height = 'auto';
  }
});

// Initial calls
loadProjects();
fetchStats();

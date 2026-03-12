
const qs = (selector, scope = document) => scope.querySelector(selector);
const qsa = (selector, scope = document) => Array.from(scope.querySelectorAll(selector));

const api = {
  async get(url) {
    const res = await fetch(url);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.error || "Request failed");
    }
    return data;
  },
  async post(url, payload, isForm = false) {
    const options = {
      method: "POST",
      headers: {},
    };
    if (isForm) {
      options.body = payload;
    } else {
      options.headers["Content-Type"] = "application/json";
      options.body = JSON.stringify(payload || {});
    }
    const res = await fetch(url, options);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.error || "Request failed");
    }
    return data;
  },
  async put(url, payload) {
    const res = await fetch(url, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload || {}),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.error || "Request failed");
    }
    return data;
  },
  async del(url) {
    const res = await fetch(url, { method: "DELETE" });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.error || "Request failed");
    }
    return data;
  },
};

function showToast(message) {
  const toast = qs("#toast");
  if (!toast) return;
  toast.textContent = message;
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 2600);
}

function setActiveNav() {
  const page = document.body.dataset.page;
  if (!page) return;
  const link = qs(`[data-nav="${page}"]`);
  if (link) link.classList.add("active");
}

function openModal(name) {
  const modal = qs(`.modal[data-modal="${name}"]`);
  if (modal) modal.classList.add("open");
}

function closeModal(modal) {
  if (modal) modal.classList.remove("open");
}

function bindModals() {
  document.addEventListener("click", (event) => {
    const openTrigger = event.target.closest("[data-modal-open]");
    if (openTrigger) {
      openModal(openTrigger.dataset.modalOpen);
      return;
    }
    if (event.target.matches("[data-modal-close]") || event.target.closest("[data-modal-close]")) {
      const modal = event.target.closest(".modal");
      if (modal) closeModal(modal);
    }
    if (event.target.classList.contains("modal")) {
      closeModal(event.target);
    }
  });
}

function renderSelectOptions(select, items, placeholder = "Select") {
  if (!select) return;
  select.innerHTML = "";
  const placeholderOption = document.createElement("option");
  placeholderOption.value = "";
  placeholderOption.textContent = placeholder;
  select.appendChild(placeholderOption);
  items.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.value;
    option.textContent = item.label;
    select.appendChild(option);
  });
}

function formatDate(iso) {
  if (!iso) return "";
  const date = new Date(iso);
  return date.toLocaleDateString();
}

async function pollJob(jobId, onUpdate, onDone) {
  let stopped = false;
  async function tick() {
    if (stopped) return;
    try {
      const data = await api.get(`/api/jobs/${jobId}`);
      onUpdate(data);
      if (data.status === "done") {
        stopped = true;
        onDone(data);
        return;
      }
      if (data.status === "error") {
        stopped = true;
      }
    } catch (err) {
      stopped = true;
    }
  }
  tick();
  const interval = setInterval(() => {
    if (stopped) {
      clearInterval(interval);
    } else {
      tick();
    }
  }, 1200);
}

setActiveNav();
bindModals();
async function loadCourses() {
  const data = await api.get("/api/courses");
  return data.courses || [];
}

async function loadConfigs(courseId) {
  if (!courseId) return [];
  const data = await api.get(`/api/configs?course_id=${courseId}`);
  return data.configs || [];
}

async function loadTopics(courseId) {
  if (!courseId) return [];
  const data = await api.get(`/api/courses/${courseId}/topics`);
  return data.topics || [];
}

async function loadSettings() {
  const data = await api.get("/api/settings");
  return data.settings || {};
}
async function initUploadPage() {
  const courseSelect = qs("#upload-course");
  const configSelect = qs("#upload-config");
  const dropzone = qs("#upload-drop");
  const fileInput = qs("#upload-file");
  const startBtn = qs("#upload-start");
  const statusEl = qs("#upload-status");
  const stepEl = qs("#upload-step");
  const percentEl = qs("#upload-percent");
  const progressEl = qs("#upload-progress");
  const debugEl = qs("#upload-debug");
  const courseSaveBtn = qs("#course-save");
  const courseHelper = qs("#upload-course-helper");
  const configHelper = qs("#upload-config-helper");
  const fileMeta = qs("#upload-file-meta");

  if (!courseSelect || !configSelect) return;

  let courses = [];

  function updateStartState() {
    const ready = !!courseSelect.value && fileInput.files.length > 0;
    startBtn.disabled = !ready;
  }

  function updateFileMeta() {
    if (!fileMeta) return;
    const file = fileInput.files[0];
    if (!file) {
      fileMeta.textContent = "No file selected.";
      return;
    }
    const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
    fileMeta.textContent = `${file.name} · ${sizeMb} MB`;
  }

  function updateConfigHelper(configs) {
    if (!configHelper) return;
    if (!configs || !configs.length) {
      configHelper.textContent = "No saved configs yet. Defaults will be used.";
      return;
    }
    if (configSelect.value) {
      const match = configs.find((c) => String(c.id) === String(configSelect.value));
      configHelper.textContent = match ? `Using ${match.name}.` : "Using selected configuration.";
      return;
    }
    configHelper.textContent = "Optional. Leave blank to use defaults.";
  }

  async function refreshCourses() {
    courses = await loadCourses();
    renderSelectOptions(courseSelect, courses.map((c) => ({ value: c.id, label: c.name })), "Select course");
    if (courses[0]) {
      courseSelect.value = courses[0].id;
      await refreshConfigs(courses[0].id);
      if (courseHelper) courseHelper.textContent = "";
    } else {
      renderSelectOptions(configSelect, [], "Select config");
      if (courseHelper) courseHelper.textContent = "Create a course to get started.";
      if (configHelper) configHelper.textContent = "";
    }
    updateStartState();
  }

  async function refreshConfigs(courseId) {
    const configs = await loadConfigs(courseId);
    renderSelectOptions(configSelect, configs.map((c) => ({ value: c.id, label: c.name })), "Select config");
    configSelect.disabled = configs.length === 0;
    updateConfigHelper(configs);
  }

  courseSelect.addEventListener("change", async () => {
    await refreshConfigs(courseSelect.value);
    updateStartState();
  });

  configSelect.addEventListener("change", async () => {
    const configs = await loadConfigs(courseSelect.value);
    updateConfigHelper(configs);
  });

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropzone.classList.add("hover");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("hover"));
  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropzone.classList.remove("hover");
    const file = event.dataTransfer.files[0];
    if (file) fileInput.files = event.dataTransfer.files;
    updateFileMeta();
    updateStartState();
  });
  fileInput.addEventListener("change", () => {
    updateFileMeta();
    updateStartState();
  });

  courseSaveBtn?.addEventListener("click", async () => {
    const name = qs("#course-name").value.trim();
    const department = qs("#course-dept").value.trim();
    const semester = qs("#course-semester").value.trim();
    if (!name) {
      showToast("Course name is required");
      return;
    }
    try {
      await api.post("/api/courses", { name, department, semester });
      closeModal(qs(".modal[data-modal='course']"));
      qs("#course-name").value = "";
      qs("#course-dept").value = "";
      qs("#course-semester").value = "";
      await refreshCourses();
      showToast("Course created");
    } catch (err) {
      showToast(err.message);
    }
  });

  startBtn.addEventListener("click", async () => {
    const courseId = courseSelect.value;
    const configId = configSelect.value;
    const file = fileInput.files[0];
    if (!courseId) {
      showToast("Select a course first");
      return;
    }
    if (!file) {
      showToast("Upload a syllabus PDF");
      return;
    }
    const formData = new FormData();
    formData.append("course_id", courseId);
    if (configId) formData.append("config_id", configId);
    formData.append("syllabus_pdf", file);

    try {
      statusEl.textContent = "Queued";
      stepEl.textContent = "Starting";
      percentEl.textContent = "0%";
      progressEl.style.width = "0%";
      const response = await api.post("/api/generate", formData, true);
      const jobId = response.job_id;
      pollJob(
        jobId,
        (job) => {
          statusEl.textContent = job.status;
          stepEl.textContent = job.current_step || "Working";
          const progress = job.progress || 0;
          percentEl.textContent = `${progress}%`;
          progressEl.style.width = `${progress}%`;
          if (job.status === "error") {
            showToast(job.error || "Generation failed");
          }
          if (job.debug) {
            const details = [];
            if (job.debug.chunks_created) details.push(`chunks: ${job.debug.chunks_created}`);
            if (job.debug.accepted_questions) details.push(`accepted: ${job.debug.accepted_questions}`);
            if (job.debug.bloom_api_calls_count) details.push(`bloom calls: ${job.debug.bloom_api_calls_count}`);
            debugEl.textContent = details.join(" · ");
          }
        },
        (job) => {
          if (job.result && job.result.generated_paper_id) {
            window.location.href = `/review/${job.result.generated_paper_id}`;
          }
        }
      );
    } catch (err) {
      showToast(err.message);
    }
  });

  await refreshCourses();
}
async function initGeneratePage() {
  const courseSelect = qs("#gen-course");
  const configSelect = qs("#gen-config");
  const configName = qs("#config-name");
  const configTotal = qs("#config-total");
  const configDuration = qs("#config-duration");
  const configDifficulty = qs("#config-difficulty");
  const configRandomize = qs("#config-randomize");
  const bloomGrid = qs("#bloom-grid");
  const bloomPreset = qs("#bloom-preset");
  const bloomSum = qs("#bloom-sum");
  const markTable = qs("#mark-table");
  const addSectionBtn = qs("#add-section");
  const saveBtn = qs("#config-save");
  const saveAsBtn = qs("#config-save-as");
  const generateBtn = qs("#config-generate");
  const statusEl = qs("#gen-status");
  const stepEl = qs("#gen-step");
  const percentEl = qs("#gen-percent");
  const progressEl = qs("#gen-progress");
  const debugEl = qs("#gen-debug");
  const courseHelper = qs("#gen-course-helper");
  const configHelper = qs("#gen-config-helper");
  const summaryBloom = qs("#summary-bloom");
  const summaryMarks = qs("#summary-marks");
  const summaryTarget = qs("#summary-target");
  const summaryHint = qs("#summary-hint");
  const summaryBloomTile = qs("#summary-bloom-tile");
  const summaryMarksTile = qs("#summary-marks-tile");

  if (!courseSelect || !configSelect) return;

  let settings = await loadSettings();
  let currentConfigId = null;

  const bloomLevels = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"];

  function renderBloomGrid(values) {
    bloomGrid.innerHTML = "";
    bloomLevels.forEach((level) => {
      const card = document.createElement("div");
      card.className = "bloom-card";
      card.innerHTML = `
        <label>${level}</label>
        <input type="number" min="0" step="1" data-bloom="${level}" value="${values[level] ?? 0}" />
      `;
      bloomGrid.appendChild(card);
    });
    updateBloomSum();
  }

  function collectBloomDistribution() {
    const values = {};
    qsa("input[data-bloom]", bloomGrid).forEach((input) => {
      values[input.dataset.bloom] = Number(input.value || 0);
    });
    return values;
  }

  function updateBloomSum() {
    const total = Object.values(collectBloomDistribution()).reduce((sum, val) => sum + val, 0);
    bloomSum.textContent = `Bloom total: ${total}%`;
    updateConfigSummary();
  }

  function calculateMarkTotal() {
    let total = 0;
    qsa(".mark-row", markTable).forEach((row) => {
      const inputs = row.querySelectorAll("input");
      const marks = Number(inputs[1].value || 0);
      const count = Number(inputs[2].value || 0);
      total += marks * count;
    });
    return total;
  }

  function updateConfigSummary() {
    const bloomTotal = Object.values(collectBloomDistribution()).reduce((sum, val) => sum + val, 0);
    const markTotal = calculateMarkTotal();
    const target = Number(configTotal.value || 0);

    if (summaryBloom) summaryBloom.textContent = `${bloomTotal}%`;
    if (summaryMarks) summaryMarks.textContent = `${markTotal} marks`;
    if (summaryTarget) summaryTarget.textContent = `${target} marks`;

    if (summaryBloomTile) {
      summaryBloomTile.classList.toggle("warning", bloomTotal !== 100);
      summaryBloomTile.classList.toggle("ok", bloomTotal === 100);
    }
    if (summaryMarksTile) {
      const mismatch = target > 0 && markTotal !== target;
      summaryMarksTile.classList.toggle("warning", mismatch);
      summaryMarksTile.classList.toggle("ok", !mismatch && markTotal > 0);
    }
    if (summaryHint) {
      const hints = [];
      if (bloomTotal !== 100) hints.push("Bloom distribution should sum to 100%.");
      if (target > 0 && markTotal !== target) hints.push("Section totals should match the target marks.");
      summaryHint.textContent = hints.join(" ");
    }
  }

  function renderMarkTable(rows) {
    markTable.innerHTML = "";
    const head = document.createElement("div");
    head.className = "table-row table-head";
    head.innerHTML = "<div>Section</div><div>Marks</div><div>Count</div><div>Total</div><div></div>";
    markTable.appendChild(head);

    rows.forEach((row) => {
      const rowEl = document.createElement("div");
      rowEl.className = "table-row mark-row";
      rowEl.innerHTML = `
        <input type="text" value="${row.section || ""}" />
        <input type="number" min="1" step="1" value="${row.marks_per_question || 0}" />
        <input type="number" min="1" step="1" value="${row.count || 0}" />
        <div class="row-total">0</div>
        <button class="icon-btn" type="button">&times;</button>
      `;
      const inputs = rowEl.querySelectorAll("input");
      const totalEl = rowEl.querySelector(".row-total");

      const updateRowTotal = () => {
        const marks = Number(inputs[1].value || 0);
        const count = Number(inputs[2].value || 0);
        totalEl.textContent = `${marks * count}`;
        updateConfigSummary();
      };

      inputs.forEach((input) => input.addEventListener("input", updateRowTotal));
      updateRowTotal();

      rowEl.querySelector("button").addEventListener("click", () => {
        rowEl.remove();
        updateConfigSummary();
      });
      markTable.appendChild(rowEl);
    });
  }

  function collectMarkDistribution() {
    const rows = [];
    qsa(".mark-row", markTable).forEach((row) => {
      const inputs = row.querySelectorAll("input");
      rows.push({
        section: inputs[0].value,
        marks_per_question: Number(inputs[1].value || 0),
        count: Number(inputs[2].value || 0),
      });
    });
    return rows;
  }

  function applyConfig(config) {
    currentConfigId = config?.id || null;
    configName.value = config?.name || "";
    configTotal.value = config?.total_marks || 50;
    configDuration.value = config?.duration_minutes || 90;
    configDifficulty.value = config?.difficulty || "mixed";
    configRandomize.checked = !!config?.randomize;
    renderBloomGrid(config?.bloom_distribution || settings.default_bloom_distribution || {});
    renderMarkTable(config?.mark_distribution || settings.default_mark_distribution || []);
    updateConfigSummary();
  }

  function buildConfigPayload(nameOverride) {
    return {
      course_id: Number(courseSelect.value),
      name: nameOverride || configName.value.trim() || "New Config",
      total_marks: Number(configTotal.value || 50),
      duration_minutes: Number(configDuration.value || 90),
      difficulty: configDifficulty.value,
      randomize: configRandomize.checked,
      bloom_distribution: collectBloomDistribution(),
      mark_distribution: collectMarkDistribution(),
    };
  }

  async function refreshCourses() {
    const courses = await loadCourses();
    renderSelectOptions(courseSelect, courses.map((c) => ({ value: c.id, label: c.name })), "Select course");
    if (courses[0]) {
      courseSelect.value = courses[0].id;
      await refreshConfigs(courses[0].id);
      if (courseHelper) courseHelper.textContent = "";
    } else {
      renderSelectOptions(configSelect, [], "Select config");
      if (courseHelper) courseHelper.textContent = "Create a course to continue.";
      if (configHelper) configHelper.textContent = "";
    }
  }

  async function refreshConfigs(courseId) {
    const configs = await loadConfigs(courseId);
    renderSelectOptions(configSelect, configs.map((c) => ({ value: c.id, label: c.name })), "Select config");
    configSelect.disabled = configs.length === 0;
    if (configs[0]) {
      configSelect.value = configs[0].id;
      applyConfig(configs[0]);
    } else {
      applyConfig(null);
    }
    if (configHelper) {
      if (!configs.length) {
        configHelper.textContent = "No saved configs yet. Adjust the settings below and save.";
      } else {
        configHelper.textContent = "Select a config to edit, or create a new one below.";
      }
    }
  }

  bloomPreset.innerHTML = "";
  Object.keys(settings.bloom_presets || {}).forEach((name) => {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    bloomPreset.appendChild(option);
  });
  bloomPreset.value = settings.default_bloom_preset || "";
  if (settings.bloom_presets && settings.default_bloom_preset) {
    renderBloomGrid(settings.bloom_presets[settings.default_bloom_preset]);
  }

  bloomPreset.addEventListener("change", () => {
    const preset = settings.bloom_presets?.[bloomPreset.value];
    if (preset) renderBloomGrid(preset);
  });

  bloomGrid.addEventListener("input", updateBloomSum);
  configTotal.addEventListener("input", updateConfigSummary);

  addSectionBtn.addEventListener("click", () => {
    const current = collectMarkDistribution();
    current.push({ section: "Section", marks_per_question: 2, count: 1 });
    renderMarkTable(current);
    updateConfigSummary();
  });

  configSelect.addEventListener("change", async () => {
    const configId = configSelect.value;
    if (!configId) {
      applyConfig(null);
      if (configHelper) configHelper.textContent = "Adjust the settings below and save.";
      return;
    }
    const data = await api.get(`/api/configs/${configId}`);
    applyConfig(data.config);
    if (configHelper) configHelper.textContent = `Editing ${data.config.name}.`;
  });

  courseSelect.addEventListener("change", async () => {
    await refreshConfigs(courseSelect.value);
  });

  saveBtn.addEventListener("click", async () => {
    const payload = buildConfigPayload();
    try {
      if (currentConfigId) {
        await api.put(`/api/configs/${currentConfigId}`, payload);
      } else {
        const data = await api.post("/api/configs", payload);
        currentConfigId = data.config.id;
      }
      await refreshConfigs(courseSelect.value);
      showToast("Config saved");
    } catch (err) {
      showToast(err.message);
    }
  });

  saveAsBtn.addEventListener("click", async () => {
    const payload = buildConfigPayload(`${configName.value.trim() || "New Config"} Copy`);
    try {
      const data = await api.post("/api/configs", payload);
      currentConfigId = data.config.id;
      await refreshConfigs(courseSelect.value);
      showToast("Config duplicated");
    } catch (err) {
      showToast(err.message);
    }
  });

  generateBtn.addEventListener("click", async () => {
    const courseId = courseSelect.value;
    if (!courseId) {
      showToast("Select a course first");
      return;
    }
    try {
      let response;
      if (currentConfigId) {
        response = await api.post("/api/generate", {
          course_id: Number(courseId),
          config_id: Number(currentConfigId),
        });
      } else {
        const inlineConfig = buildConfigPayload();
        delete inlineConfig.course_id;
        response = await api.post("/api/generate", {
          course_id: Number(courseId),
          config: inlineConfig,
        });
        showToast("Generating with unsaved config");
      }
      pollJob(
        response.job_id,
        (job) => {
          statusEl.textContent = job.status;
          stepEl.textContent = job.current_step || "Working";
          const progress = job.progress || 0;
          percentEl.textContent = `${progress}%`;
          progressEl.style.width = `${progress}%`;
          if (job.status === "error") {
            showToast(job.error || "Generation failed");
          }
          if (job.debug) {
            const details = [];
            if (job.debug.chunks_created) details.push(`chunks: ${job.debug.chunks_created}`);
            if (job.debug.accepted_questions) details.push(`accepted: ${job.debug.accepted_questions}`);
            debugEl.textContent = details.join(" · ");
          }
        },
        (job) => {
          if (job.result?.generated_paper_id) {
            window.location.href = `/review/${job.result.generated_paper_id}`;
          }
        }
      );
    } catch (err) {
      showToast(err.message);
    }
  });

  await refreshCourses();
  if (!configTotal.value) configTotal.value = 50;
  if (!configDuration.value) configDuration.value = 90;
}
async function initQuestionBank() {
  const courseSelect = qs("#qb-course");
  const topicSelect = qs("#qb-topic");
  const bloomSelect = qs("#qb-bloom");
  const difficultySelect = qs("#qb-difficulty");
  const searchInput = qs("#qb-search");
  const table = qs("#qb-table");
  const refreshBtn = qs("#qb-refresh");
  const addBtn = qs("#qb-add");
  const saveBtn = qs("#qb-save");

  if (!courseSelect || !table) return;

  let courses = [];
  let topics = [];
  let editingId = null;

  async function refreshCourses() {
    courses = await loadCourses();
    renderSelectOptions(courseSelect, courses.map((c) => ({ value: c.id, label: c.name })), "All courses");
    renderSelectOptions(qs("#qb-form-course"), courses.map((c) => ({ value: c.id, label: c.name })), "Select course");
  }

  async function refreshTopics(courseId) {
    if (!courseId) {
      renderSelectOptions(topicSelect, [], "All topics");
      renderSelectOptions(qs("#qb-form-topic"), [], "Select topic");
      return;
    }
    topics = await loadTopics(courseId);
    renderSelectOptions(topicSelect, topics.map((t) => ({ value: t.id, label: t.name })), "All topics");
    renderSelectOptions(qs("#qb-form-topic"), topics.map((t) => ({ value: t.id, label: t.name })), "Select topic");
  }

  async function refreshQuestions() {
    const params = new URLSearchParams();
    if (courseSelect.value) params.append("course_id", courseSelect.value);
    if (topicSelect.value) params.append("topic_id", topicSelect.value);
    if (bloomSelect.value) params.append("bloom", bloomSelect.value);
    if (difficultySelect.value) params.append("difficulty", difficultySelect.value);
    if (searchInput.value) params.append("q", searchInput.value);

    const data = await api.get(`/api/questions?${params.toString()}`);
    const questions = data.questions || [];

    table.innerHTML = "";
    questions.forEach((question) => {
      const row = document.createElement("div");
      row.className = "question-card";
      row.innerHTML = `
        <div class="list-title">${question.text}</div>
        <div class="question-meta">
          <span class="badge">${question.marks} marks</span>
          <span class="badge">${question.bloom_level || ""}</span>
          <span class="badge">${question.difficulty || ""}</span>
        </div>
        <div class="row-actions">
          <button class="btn ghost" data-edit="${question.id}">Edit</button>
          <button class="btn secondary" data-delete="${question.id}">Delete</button>
        </div>
      `;
      table.appendChild(row);
    });

    qsa("[data-edit]", table).forEach((btn) => {
      btn.addEventListener("click", () => openEdit(btn.dataset.edit, questions));
    });
    qsa("[data-delete]", table).forEach((btn) => {
      btn.addEventListener("click", async () => {
        try {
          await api.del(`/api/questions/${btn.dataset.delete}`);
          showToast("Question deleted");
          refreshQuestions();
        } catch (err) {
          showToast(err.message);
        }
      });
    });
  }

  function openEdit(id, questions) {
    const q = questions.find((item) => String(item.id) === String(id));
    if (!q) return;
    editingId = q.id;
    qs("#qb-modal-title").textContent = "Edit Question";
    qs("#qb-form-course").value = q.course_id;
    refreshTopics(q.course_id).then(() => {
      qs("#qb-form-topic").value = q.topic_id || "";
    });
    qs("#qb-form-text").value = q.text;
    qs("#qb-form-marks").value = q.marks;
    qs("#qb-form-difficulty").value = q.difficulty || "medium";
    qs("#qb-form-bloom").value = q.bloom_level || "Remember";
    qs("#qb-form-verb").value = q.bloom_verb || "";
    openModal("question");
  }

  addBtn.addEventListener("click", () => {
    editingId = null;
    qs("#qb-modal-title").textContent = "Add Question";
    qs("#qb-form-text").value = "";
    qs("#qb-form-marks").value = "";
    qs("#qb-form-verb").value = "";
    if (courseSelect.value) {
      qs("#qb-form-course").value = courseSelect.value;
      refreshTopics(courseSelect.value);
    }
    openModal("question");
  });

  saveBtn.addEventListener("click", async () => {
    const payload = {
      course_id: Number(qs("#qb-form-course").value || courseSelect.value),
      topic_id: Number(qs("#qb-form-topic").value || 0) || null,
      text: qs("#qb-form-text").value.trim(),
      marks: Number(qs("#qb-form-marks").value || 0),
      difficulty: qs("#qb-form-difficulty").value,
      bloom_level: qs("#qb-form-bloom").value,
      bloom_verb: qs("#qb-form-verb").value,
    };
    if (!payload.text || !payload.course_id) {
      showToast("Course and question text required");
      return;
    }
    try {
      if (editingId) {
        await api.put(`/api/questions/${editingId}`, payload);
      } else {
        await api.post("/api/questions", payload);
      }
      closeModal(qs(".modal[data-modal='question']"));
      refreshQuestions();
      showToast("Question saved");
    } catch (err) {
      showToast(err.message);
    }
  });

  courseSelect.addEventListener("change", async () => {
    await refreshTopics(courseSelect.value);
    refreshQuestions();
  });

  refreshBtn.addEventListener("click", refreshQuestions);
  [topicSelect, bloomSelect, difficultySelect].forEach((el) => {
    el.addEventListener("change", refreshQuestions);
  });
  searchInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      refreshQuestions();
    }
  });

  await refreshCourses();
  await refreshTopics(courseSelect.value);
  refreshQuestions();
}
async function initCoursesPage() {
  const courseList = qs("#course-list");
  const topicList = qs("#topic-list");
  const addCourseBtn = qs("#course-add");
  const addTopicBtn = qs("#topic-add");
  const courseSaveBtn = qs("#course-modal-save");
  const topicSaveBtn = qs("#topic-modal-save");

  if (!courseList || !topicList) return;

  let courses = [];
  let selectedCourseId = null;
  let editingCourseId = null;
  let editingTopicId = null;

  async function refreshCourses() {
    courses = await loadCourses();
    courseList.innerHTML = "";
    courses.forEach((course) => {
      const card = document.createElement("div");
      card.className = "list-item";
      card.dataset.id = course.id;
      card.innerHTML = `
        <div class="list-title">${course.name}</div>
        <div class="muted">${course.department || ""} ${course.semester || ""}</div>
        <div class="question-meta">
          <span class="badge">${course.topics_count} topics</span>
          <span class="badge">${course.questions_count} questions</span>
          <span class="badge">${course.has_syllabus ? "Syllabus" : "No syllabus"}</span>
        </div>
        <div class="row-actions">
          <button class="btn ghost" data-edit="${course.id}">Edit</button>
          <button class="btn secondary" data-delete="${course.id}">Delete</button>
        </div>
      `;
      card.addEventListener("click", (event) => {
        if (event.target.closest("button")) return;
        selectedCourseId = course.id;
        refreshTopics();
        highlightCourse();
      });
      courseList.appendChild(card);
    });

    qsa("[data-edit]", courseList).forEach((btn) => {
      btn.addEventListener("click", () => openCourseModal(btn.dataset.edit));
    });
    qsa("[data-delete]", courseList).forEach((btn) => {
      btn.addEventListener("click", async () => {
        try {
          await api.del(`/api/courses/${btn.dataset.delete}`);
          showToast("Course deleted");
          if (selectedCourseId === Number(btn.dataset.delete)) {
            selectedCourseId = null;
            topicList.innerHTML = "";
          }
          refreshCourses();
        } catch (err) {
          showToast(err.message);
        }
      });
    });

    if (!selectedCourseId && courses[0]) {
      selectedCourseId = courses[0].id;
      refreshTopics();
      highlightCourse();
    }
  }

  function highlightCourse() {
    qsa(".list-item", courseList).forEach((card) => {
      card.classList.toggle("active", Number(card.dataset.id) === Number(selectedCourseId));
    });
  }

  async function refreshTopics() {
    if (!selectedCourseId) {
      topicList.innerHTML = "";
      return;
    }
    const data = await api.get(`/api/courses/${selectedCourseId}/topics`);
    topicList.innerHTML = "";
    (data.topics || []).forEach((topic) => {
      const row = document.createElement("div");
      row.className = "list-item";
      row.innerHTML = `
        <div class="list-title">${topic.name}</div>
        <div class="muted">Unit ${topic.unit_number || "-"}</div>
        <div class="row-actions">
          <button class="btn ghost" data-edit-topic="${topic.id}">Edit</button>
          <button class="btn secondary" data-delete-topic="${topic.id}">Delete</button>
        </div>
      `;
      topicList.appendChild(row);
    });

    qsa("[data-edit-topic]", topicList).forEach((btn) => {
      btn.addEventListener("click", () => openTopicModal(btn.dataset.editTopic));
    });
    qsa("[data-delete-topic]", topicList).forEach((btn) => {
      btn.addEventListener("click", async () => {
        try {
          await api.del(`/api/topics/${btn.dataset.deleteTopic}`);
          showToast("Topic deleted");
          refreshTopics();
        } catch (err) {
          showToast(err.message);
        }
      });
    });
  }

  function openCourseModal(id) {
    editingCourseId = id ? Number(id) : null;
    const course = courses.find((c) => c.id === editingCourseId);
    qs("#course-modal-title").textContent = editingCourseId ? "Edit Course" : "Add Course";
    qs("#course-modal-name").value = course?.name || "";
    qs("#course-modal-dept").value = course?.department || "";
    qs("#course-modal-sem").value = course?.semester || "";
    openModal("course");
  }

  function openTopicModal(id) {
    editingTopicId = id ? Number(id) : null;
    if (!selectedCourseId) {
      showToast("Select a course first");
      return;
    }
    if (editingTopicId) {
      api.get(`/api/courses/${selectedCourseId}/topics`).then((data) => {
        const match = (data.topics || []).find((t) => t.id === editingTopicId);
        qs("#topic-modal-name").value = match?.name || "";
        qs("#topic-modal-unit").value = match?.unit_number || "";
      });
    } else {
      qs("#topic-modal-name").value = "";
      qs("#topic-modal-unit").value = "";
    }
    qs("#topic-modal-title").textContent = editingTopicId ? "Edit Topic" : "Add Topic";
    openModal("topic");
  }

  addCourseBtn.addEventListener("click", () => openCourseModal(null));
  addTopicBtn.addEventListener("click", () => openTopicModal(null));

  courseSaveBtn.addEventListener("click", async () => {
    const payload = {
      name: qs("#course-modal-name").value.trim(),
      department: qs("#course-modal-dept").value.trim(),
      semester: qs("#course-modal-sem").value.trim(),
    };
    if (!payload.name) {
      showToast("Course name is required");
      return;
    }
    try {
      if (editingCourseId) {
        await api.put(`/api/courses/${editingCourseId}`, payload);
      } else {
        await api.post("/api/courses", payload);
      }
      closeModal(qs(".modal[data-modal='course']"));
      refreshCourses();
      showToast("Course saved");
    } catch (err) {
      showToast(err.message);
    }
  });

  topicSaveBtn.addEventListener("click", async () => {
    if (!selectedCourseId) return;
    const payload = {
      name: qs("#topic-modal-name").value.trim(),
      unit_number: Number(qs("#topic-modal-unit").value || 0),
    };
    if (!payload.name) {
      showToast("Topic name is required");
      return;
    }
    try {
      if (editingTopicId) {
        await api.put(`/api/topics/${editingTopicId}`, payload);
      } else {
        await api.post(`/api/courses/${selectedCourseId}/topics`, payload);
      }
      closeModal(qs(".modal[data-modal='topic']"));
      refreshTopics();
      showToast("Topic saved");
    } catch (err) {
      showToast(err.message);
    }
  });

  await refreshCourses();
}
async function initPapersPage() {
  const list = qs("#papers-list");
  if (!list) return;
  const data = await api.get("/api/papers");
  list.innerHTML = "";
  (data.papers || []).forEach((paper) => {
    const item = document.createElement("div");
    item.className = "list-item";
    item.innerHTML = `
      <div class="list-title">${paper.title}</div>
      <div class="muted">Created ${formatDate(paper.created_at)} · ${paper.total_marks} marks</div>
      <div class="row-actions">
        <a class="btn ghost" href="/review/${paper.id}">Open Review</a>
      </div>
    `;
    list.appendChild(item);
  });
}
async function initReviewPage() {
  const panel = qs("#review-panel");
  if (!panel) return;
  const paperId = panel.dataset.paperId;
  const summaryEl = qs("#paper-summary");
  const questionsEl = qs("#review-questions");
  const exportBtn = qs("#review-export");
  const swapList = qs("#swap-list");
  const swapContext = qs("#swap-context");
  const swapConfirm = qs("#swap-confirm");

  let currentPaper = null;
  let swapIndex = null;
  let selectedReplacement = null;

  async function loadPaper() {
    const data = await api.get(`/api/papers/${paperId}`);
    currentPaper = data.paper;
    renderSummary();
    renderQuestions();
  }

  function renderSummary() {
    if (!currentPaper) return;
    summaryEl.innerHTML = "";
    const tiles = [
      { label: "Title", value: currentPaper.title },
      { label: "Total Marks", value: currentPaper.total_marks },
      { label: "Duration", value: `${currentPaper.duration_minutes} mins` },
      { label: "Questions", value: currentPaper.questions.length },
    ];
    tiles.forEach((tile) => {
      const div = document.createElement("div");
      div.className = "summary-tile";
      div.innerHTML = `<div class="muted">${tile.label}</div><div class="summary-value">${tile.value}</div>`;
      summaryEl.appendChild(div);
    });
  }

  function renderQuestions() {
    questionsEl.innerHTML = "";
    currentPaper.questions.forEach((q, index) => {
      const card = document.createElement("div");
      card.className = "question-card";
      card.innerHTML = `
        <div class="list-title">Q${index + 1}. ${q.text}</div>
        <div class="question-meta">
          <span class="badge">${q.marks} marks</span>
          <span class="badge">${q.bloom_level || ""}</span>
          <span class="badge">${q.difficulty || ""}</span>
        </div>
        <div class="row-actions">
          <button class="btn ghost" data-swap="${index}">Swap</button>
        </div>
      `;
      questionsEl.appendChild(card);
    });

    qsa("[data-swap]", questionsEl).forEach((btn) => {
      btn.addEventListener("click", () => openSwap(Number(btn.dataset.swap)));
    });
  }

  async function openSwap(index) {
    swapIndex = index;
    selectedReplacement = null;
    const question = currentPaper.questions[index];
    swapContext.textContent = `Looking for ${question.marks} marks · ${question.bloom_level}`;
    const params = new URLSearchParams({
      course_id: currentPaper.course_id,
      bloom: question.bloom_level,
      marks: question.marks,
    });
    const data = await api.get(`/api/questions?${params.toString()}`);
    const options = (data.questions || []).filter((q) => q.id !== question.id);
    swapList.innerHTML = "";
    options.forEach((opt) => {
      const row = document.createElement("label");
      row.className = "swap-option";
      row.innerHTML = `
        <input type="radio" name="swap-choice" value="${opt.id}" />
        <div>
          <div class="list-title">${opt.text}</div>
          <div class="muted">${opt.marks} marks · ${opt.bloom_level}</div>
        </div>
      `;
      row.querySelector("input").addEventListener("change", () => {
        selectedReplacement = opt;
      });
      swapList.appendChild(row);
    });
    if (!options.length) {
      const empty = document.createElement("div");
      empty.className = "muted";
      empty.textContent = "No matching replacements found.";
      swapList.appendChild(empty);
    }
    openModal("swap");
  }

  swapConfirm.addEventListener("click", async () => {
    if (!selectedReplacement || swapIndex === null) {
      showToast("Select a replacement question");
      return;
    }
    const updated = [...currentPaper.questions];
    updated[swapIndex] = {
      id: selectedReplacement.id,
      text: selectedReplacement.text,
      marks: selectedReplacement.marks,
      bloom_level: selectedReplacement.bloom_level,
      bloom_verb: selectedReplacement.bloom_verb,
      difficulty: selectedReplacement.difficulty,
      source_chunk_id: selectedReplacement.source_chunk_id,
    };
    try {
      const data = await api.post(`/api/papers/${paperId}/revise`, {
        questions: updated,
        total_marks: currentPaper.total_marks,
        duration_minutes: currentPaper.duration_minutes,
      });
      closeModal(qs(".modal[data-modal='swap']"));
      window.location.href = `/review/${data.paper.id}`;
    } catch (err) {
      showToast(err.message);
    }
  });

  exportBtn.addEventListener("click", async () => {
    try {
      await api.post(`/api/papers/${paperId}/export/pdf`, {});
    } catch (err) {
      showToast(err.message || "Export not available");
    }
  });

  await loadPaper();
}
async function initSettingsPage() {
  const modelInput = qs("#settings-model");
  const difficultySelect = qs("#settings-difficulty");
  const presetSelect = qs("#settings-preset");
  const preview = qs("#settings-preview");
  const saveBtn = qs("#settings-save");

  if (!modelInput) return;

  const settings = await loadSettings();
  modelInput.value = settings.default_model_name || "";
  difficultySelect.value = settings.default_difficulty || "mixed";

  presetSelect.innerHTML = "";
  Object.keys(settings.bloom_presets || {}).forEach((name) => {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    presetSelect.appendChild(option);
  });
  presetSelect.value = settings.default_bloom_preset || presetSelect.value;

  function renderPreview() {
    const preset = settings.bloom_presets?.[presetSelect.value] || {};
    preview.innerHTML = "";
    Object.entries(preset).forEach(([level, value]) => {
      const card = document.createElement("div");
      card.className = "summary-tile";
      card.innerHTML = `<div class="muted">${level}</div><div class="summary-value">${value}%</div>`;
      preview.appendChild(card);
    });
  }

  presetSelect.addEventListener("change", renderPreview);
  renderPreview();

  saveBtn.addEventListener("click", async () => {
    const presetName = presetSelect.value;
    const payload = {
      default_model_name: modelInput.value.trim(),
      default_difficulty: difficultySelect.value,
      default_bloom_preset: presetName,
      default_bloom_distribution: settings.bloom_presets?.[presetName] || settings.default_bloom_distribution,
    };
    try {
      await api.put("/api/settings", payload);
      showToast("Settings updated");
    } catch (err) {
      showToast(err.message);
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  const page = document.body.dataset.page;
  if (page === "upload") initUploadPage();
  if (page === "generate") initGeneratePage();
  if (page === "question_bank") initQuestionBank();
  if (page === "review") initReviewPage();
  if (page === "settings") initSettingsPage();
  if (page === "courses") initCoursesPage();
  if (page === "papers") initPapersPage();
});

const $ = (id) => document.getElementById(id);

const els = {
  feed: $("feed"),
  feedPlaceholder: $("feedPlaceholder"),
  statusBadge: $("statusBadge"),
  fps: $("fps"),
  detectorMode: $("detectorMode"),
  detCount: $("detCount"),
  deerHits: $("deerHits"),
  deerCount: $("deerCount"),
  streamState: $("streamState"),
  demoFlag: $("demoFlag"),
  sens: $("sens"),
  sensVal: $("sensVal"),
  muteBtn: $("muteBtn"),
  snapshotBtn: $("snapshotBtn"),
  reconnectBtn: $("reconnectBtn"),
  scopeStatus: $("scopeStatus"),
  zoom: $("zoom"),
  zoomVal: $("zoomVal"),
  palette: $("palette"),
  paletteVal: $("paletteVal"),
  bright: $("bright"),
  brightVal: $("brightVal"),
  contrast: $("contrast"),
  contrastVal: $("contrastVal"),
  alertOverlay: $("alertOverlay"),
  llmBackend: $("llmBackend"),
  chatLog: $("chatLog"),
  chatInput: $("chatInput"),
  chatSendBtn: $("chatSendBtn"),
  analyzeBtn: $("analyzeBtn"),
  trainPhase: $("trainPhase"),
  trainProgress: $("trainProgress"),
  trainMessage: $("trainMessage"),
  trainVisual: $("trainVisual"),
  trainThermal: $("trainThermal"),
  trainTotal: $("trainTotal"),
  trainStartBtn: $("trainStartBtn"),
  detectionToggle: $("detectionToggle"),
};

let muted = false;
let useYolo = false;
let syncingControls = false;
let userAdjustingSens = false;
let userAdjustingScope = false;
let lastScopeChange = 0;
let lastAlertAt = 0;
let alertAudio = null;
let audioCtx = null;

function sensFromSlider(v) {
  return useYolo ? 0.15 + (v / 100) * 0.70 : 0.35 + (v / 100) * 2.65;
}

function sliderFromSens(s) {
  return useYolo
    ? Math.round(((s - 0.15) / 0.70) * 100)
    : Math.round(((s - 0.35) / 2.65) * 100);
}

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) throw new Error(`${path} ${res.status}`);
  return res.json();
}

function playAlertBeep() {
  if (muted) return;
  try {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = "square";
    osc.frequency.value = 880;
    gain.gain.value = 0.08;
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.25);
    osc.stop(audioCtx.currentTime + 0.25);
  } catch (_) {}
}

function ensureAlertAudio() {
  if (!alertAudio) {
    alertAudio = new Audio("/audio/deer_deer.wav");
    alertAudio.preload = "auto";
    alertAudio.volume = 1.0;
  }
  return alertAudio;
}

function playDeerAlert() {
  if (muted) return;
  playAlertBeep();
  setTimeout(() => {
    if (muted) return;
    const audio = ensureAlertAudio();
    audio.currentTime = 0;
    audio.play().catch(() => {});
  }, 280);
}

els.sens.addEventListener("pointerdown", () => { userAdjustingSens = true; });
els.sens.addEventListener("pointerup", () => { userAdjustingSens = false; });
els.sens.addEventListener("input", () => {
  if (syncingControls) return;
  const v = sensFromSlider(+els.sens.value);
  els.sensVal.textContent = useYolo ? v.toFixed(2) : v.toFixed(1);
  api("/api/sensitivity", { method: "POST", body: JSON.stringify({ value: v }) });
});

els.detectionToggle.addEventListener("change", async () => {
  const enabled = els.detectionToggle.checked;
  await api("/api/detection", { method: "POST", body: JSON.stringify({ enabled }) });
});

els.muteBtn.addEventListener("click", async () => {
  muted = !muted;
  els.muteBtn.textContent = muted ? "Unmute audio" : "Mute audio";
  els.muteBtn.classList.toggle("active", muted);
  if (!muted) ensureAlertAudio();
  await api("/api/mute", { method: "POST", body: JSON.stringify({ muted }) });
});

els.reconnectBtn.addEventListener("click", () => api("/api/reconnect", { method: "POST" }));

els.snapshotBtn.addEventListener("click", async () => {
  try {
    const j = await api("/api/snapshot", { method: "POST" });
    if (j.ok && j.url) window.open(j.url, "_blank");
  } catch (_) {}
});

els.zoom.addEventListener("input", () => {
  if (syncingControls) return;
  lastScopeChange = Date.now();
  const idx = +els.zoom.value;
  els.zoomVal.textContent = `${[1, 2, 4, 8][idx] || 1}x`;
  api("/api/scope/zoom", { method: "POST", body: JSON.stringify({ index: idx }) });
});

els.palette.addEventListener("change", () => {
  if (syncingControls) return;
  lastScopeChange = Date.now();
  const idx = +els.palette.value;
  els.paletteVal.textContent = els.palette.options[idx]?.text || "—";
  api("/api/scope/palette", { method: "POST", body: JSON.stringify({ index: idx }) })
    .catch(() => {});
});

function sendImageSettings() {
  if (syncingControls) return;
  api("/api/scope/image", {
    method: "POST",
    body: JSON.stringify({
      brightness: +els.bright.value,
      contrast: +els.contrast.value,
    }),
  });
}

els.bright.addEventListener("input", () => {
  els.brightVal.textContent = els.bright.value;
  sendImageSettings();
});

els.contrast.addEventListener("input", () => {
  els.contrastVal.textContent = els.contrast.value;
  sendImageSettings();
});

els.feed.addEventListener("load", () => {
  els.feedPlaceholder.classList.add("hidden");
});

els.feed.addEventListener("error", () => {
  els.feedPlaceholder.classList.remove("hidden");
});

function applyScope(scope, forceSync = false) {
  if (!scope) return;
  const scopeLocked = !forceSync && (Date.now() - lastScopeChange < 3000);
  syncingControls = true;
  if (scope.palettes?.length && els.palette.options.length === 0) {
    scope.palettes.forEach((name, i) => {
      const opt = document.createElement("option");
      opt.value = String(i);
      opt.textContent = name;
      els.palette.appendChild(opt);
    });
  }
  if (!scopeLocked) {
    if (scope.zoom_index != null) {
      els.zoom.value = scope.zoom_index;
      els.zoomVal.textContent = scope.zoom_label || "1x";
    }
    if (scope.palette_index != null) {
      els.palette.value = String(scope.palette_index);
      els.paletteVal.textContent = scope.palette_name || "—";
    }
    if (scope.brightness != null) {
      els.bright.value = scope.brightness;
      els.brightVal.textContent = String(scope.brightness);
    }
    if (scope.contrast != null) {
      els.contrast.value = scope.contrast;
      els.contrastVal.textContent = String(scope.contrast);
    }
  }
  els.scopeStatus.textContent = scope.connected
    ? "Scope: ISAPI connected"
    : scope.enabled
      ? "Scope: offline (connect hotspot)"
      : "Scope: unavailable";
  syncingControls = false;
}

async function poll() {
  try {
    const j = await api("/api/status");
    useYolo = j.use_yolo;
    els.fps.textContent = `${j.fps.toFixed(1)} FPS`;
    els.detCount.textContent = j.detections;
    els.deerHits.textContent = j.deer_hits;
    els.deerCount.textContent = `${j.deer_hits} deer tracked`;
    els.streamState.textContent = j.connected ? "Stream: connected" : "Stream: reconnecting…";
    els.demoFlag.textContent = j.demo ? "Demo" : "Live";
    els.detectorMode.textContent = j.use_yolo ? "CV: YOLO" : "CV: thermal heuristic";

    els.statusBadge.textContent = j.status;
    els.statusBadge.className = "badge " + (j.armed ? "alert" : j.detection_enabled === false ? "off" : "scan");

    if (j.detection_enabled === false) {
      els.alertOverlay.classList.add("hidden");
    } else if (j.armed && Date.now() - lastAlertAt > 3500) {
      lastAlertAt = Date.now();
      playDeerAlert();
      els.alertOverlay.classList.toggle("hidden", false);
    } else {
      els.alertOverlay.classList.toggle("hidden", !j.armed);
    }

    if (els.detectionToggle.checked !== (j.detection_enabled !== false)) {
      syncingControls = true;
      els.detectionToggle.checked = j.detection_enabled !== false;
      syncingControls = false;
    }

    if (!userAdjustingSens) {
      syncingControls = true;
      els.sens.value = sliderFromSens(j.sensitivity);
      els.sensVal.textContent = j.use_yolo
        ? j.sensitivity.toFixed(2)
        : j.sensitivity.toFixed(1);
      syncingControls = false;
    }

    applyScope(j.scope);
  } catch (_) {}
}

setInterval(poll, 800);
poll();

function addChat(role, text) {
  const div = document.createElement("div");
  div.className = `chat-msg ${role}`;
  div.textContent = text;
  els.chatLog.appendChild(div);
  els.chatLog.scrollTop = els.chatLog.scrollHeight;
}

async function sendChat() {
  const msg = els.chatInput.value.trim();
  if (!msg) return;
  els.chatInput.value = "";
  addChat("user", msg);
  try {
    const j = await api("/api/llm/chat", { method: "POST", body: JSON.stringify({ message: msg }) });
    addChat("bot", j.reply || "(no response)");
  } catch (e) {
    addChat("system", "Chat failed — using local assistant only.");
  }
}

els.chatSendBtn.addEventListener("click", sendChat);
els.chatInput.addEventListener("keydown", (e) => { if (e.key === "Enter") sendChat(); });

els.analyzeBtn.addEventListener("click", async () => {
  addChat("system", "Analyzing current frame…");
  try {
    const j = await api("/api/llm/analyze", { method: "POST", body: "{}" });
    addChat("bot", j.reply || "(no analysis)");
  } catch (_) {
    addChat("system", "No frame available yet.");
  }
});

async function pollTraining() {
  try {
    const t = await api("/api/training/status");
    els.trainPhase.textContent = t.running ? t.phase.toUpperCase() : t.phase;
    els.trainProgress.style.width = `${t.progress || 0}%`;
    els.trainMessage.textContent = t.last_log || t.message || "—";
    els.trainVisual.textContent = t.visual_images || 0;
    els.trainThermal.textContent = t.thermal_images || 0;
    els.trainTotal.textContent = t.train_images || 0;
    els.trainStartBtn.disabled = t.running;
    els.trainStartBtn.textContent = t.running ? "Training in progress…" : "Start training pipeline";
  } catch (_) {}
}

async function pollLlm() {
  try {
    const s = await api("/api/llm/status");
    els.llmBackend.textContent = s.backend;
  } catch (_) {}
}

els.trainStartBtn.addEventListener("click", async () => {
  try {
    const j = await api("/api/training/start", { method: "POST", body: JSON.stringify({ epochs: 30 }) });
    if (j.ok) addChat("system", "Training pipeline started — fetching roadside deer images.");
    else addChat("system", j.error || "Could not start training.");
  } catch (_) {}
});

setInterval(pollTraining, 2000);
setInterval(pollLlm, 10000);
pollTraining();
pollLlm();
addChat("system", "AI assistant ready. Ask about detections or training, or analyze the live frame.");

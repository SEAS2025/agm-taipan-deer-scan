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
};

let muted = false;
let useYolo = false;
let syncingControls = false;
let lastAlertAt = 0;
let audioCtx = null;

function sensFromSlider(v) {
  return useYolo ? 0.15 + (v / 100) * 0.70 : 0.5 + (v / 100) * 1.5;
}

function sliderFromSens(s) {
  return useYolo
    ? Math.round(((s - 0.15) / 0.70) * 100)
    : Math.round(((s - 0.5) / 1.5) * 100);
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

els.sens.addEventListener("input", () => {
  if (syncingControls) return;
  const v = sensFromSlider(+els.sens.value);
  els.sensVal.textContent = useYolo ? v.toFixed(2) : v.toFixed(1);
  api("/api/sensitivity", { method: "POST", body: JSON.stringify({ value: v }) });
});

els.muteBtn.addEventListener("click", async () => {
  muted = !muted;
  els.muteBtn.textContent = muted ? "Unmute audio" : "Mute audio";
  els.muteBtn.classList.toggle("active", muted);
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
  const idx = +els.zoom.value;
  els.zoomVal.textContent = `${[1, 2, 4, 8][idx] || 1}x`;
  api("/api/scope/zoom", { method: "POST", body: JSON.stringify({ index: idx }) });
});

els.palette.addEventListener("change", () => {
  if (syncingControls) return;
  const idx = +els.palette.value;
  els.paletteVal.textContent = els.palette.options[idx]?.text || "—";
  api("/api/scope/palette", { method: "POST", body: JSON.stringify({ index: idx }) });
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

function applyScope(scope) {
  if (!scope) return;
  syncingControls = true;
  if (scope.palettes?.length && els.palette.options.length === 0) {
    scope.palettes.forEach((name, i) => {
      const opt = document.createElement("option");
      opt.value = String(i);
      opt.textContent = name;
      els.palette.appendChild(opt);
    });
  }
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
    els.detectorMode.textContent = j.use_yolo ? "YOLO" : "Thermal heuristic";

    els.statusBadge.textContent = j.status;
    els.statusBadge.className = "badge " + (j.armed ? "alert" : "scan");

    if (j.armed && Date.now() - lastAlertAt > 4000) {
      lastAlertAt = Date.now();
      playAlertBeep();
    }
    els.alertOverlay.classList.toggle("hidden", !j.armed);

    syncingControls = true;
    els.sens.value = sliderFromSens(j.sensitivity);
    els.sensVal.textContent = j.use_yolo
      ? j.sensitivity.toFixed(2)
      : j.sensitivity.toFixed(1);
    syncingControls = false;

    applyScope(j.scope);
  } catch (_) {}
}

setInterval(poll, 800);
poll();

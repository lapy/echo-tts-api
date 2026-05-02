const textEl = document.querySelector("#text");
const voiceEl = document.querySelector("#voice");
const streamEl = document.querySelector("#stream");
const formatEl = document.querySelector("#format");
const seedEl = document.querySelector("#seed");
const extraEl = document.querySelector("#extra");
const statusEl = document.querySelector("#status");
const playerEl = document.querySelector("#player");
const genBtn = document.querySelector("#generate");
const reloadVoicesBtn = document.querySelector("#reload-voices");
const saveSettingsBtn = document.querySelector("#save-settings");
const resetSettingsBtn = document.querySelector("#reset-settings");
const presetEl = document.querySelector("#sampler-preset");
const cfgTextEl = document.querySelector("#cfg-text");
const cfgSpeakerEl = document.querySelector("#cfg-speaker");
const cfgMinTEl = document.querySelector("#cfg-min-t");
const cfgMaxTEl = document.querySelector("#cfg-max-t");
const chunkEnabledEl = document.querySelector("#chunk-enabled");
const chunkTargetEl = document.querySelector("#chunk-target");
const chunkMinEl = document.querySelector("#chunk-min");
const chunkMaxEl = document.querySelector("#chunk-max");
const stepsNonstreamEl = document.querySelector("#steps-nonstream");
const nonstreamRow = document.querySelector("#nonstream-row");
const nonstreamHint = document.querySelector("#nonstream-hint");

const DRAFT_KEY = "echo_tts_draft_text_v1";

/** @type {any} */
let echoMeta = null;

function apiUrl(path) {
  const p = path.startsWith("/") ? path : `/${path}`;
  return new URL(p, window.location.origin).href;
}

function debounce(fn, ms) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

function parseExpert() {
  const raw = (extraEl.value || "").trim();
  if (!raw) return {};
  return JSON.parse(raw);
}

function syncNonStreamUi() {
  const streaming = streamEl.checked;
  nonstreamRow.classList.toggle("hidden", streaming);
  nonstreamHint.classList.toggle("hidden", streaming);
}

/** Format numbers for placeholder / title hints from server meta. */
function fmtDef(v) {
  if (v === undefined || v === null) return "—";
  if (typeof v === "number" && Math.abs(v % 1) > 1e-9) return String(v);
  if (typeof v === "number") return String(v);
  return String(v);
}

/** Placeholders and title tooltips from GET /echo/ui/meta (placeholder shows when the field is empty). */
function applyDefaultPlaceholders() {
  seedEl.placeholder = "Optional integer · random if empty";
  seedEl.title = "extra_body.seed — leave blank for random";

  extraEl.placeholder = "{ }\n· optional overrides merged after form fields";
  extraEl.title = "extra_body JSON — merged last; invalid JSON blocks Generate";

  if (!echoMeta?.defaults) return;
  const d = echoMeta.defaults;

  cfgTextEl.placeholder = `Server default: ${fmtDef(d.cfg_scale_text)} · cfg_scale_text`;
  cfgSpeakerEl.placeholder = `Server default: ${fmtDef(d.cfg_scale_speaker)} · cfg_scale_speaker`;
  cfgMinTEl.placeholder = `Server default: ${fmtDef(d.cfg_min_t)} · cfg_min_t`;
  cfgMaxTEl.placeholder = `Server default: ${fmtDef(d.cfg_max_t)} · cfg_max_t`;
  cfgTextEl.title = `Default cfg_scale_text from server: ${fmtDef(d.cfg_scale_text)}`;
  cfgSpeakerEl.title = `Default cfg_scale_speaker: ${fmtDef(d.cfg_scale_speaker)}`;
  cfgMinTEl.title = `Default cfg_min_t: ${fmtDef(d.cfg_min_t)}`;
  cfgMaxTEl.title = `Default cfg_max_t: ${fmtDef(d.cfg_max_t)}`;

  chunkTargetEl.placeholder = `Default ${fmtDef(d.chunk_target_seconds)} sec · chunk_target_seconds`;
  chunkMinEl.placeholder = `Default ${fmtDef(d.chunk_min_seconds)} sec · chunk_min_seconds`;
  chunkMaxEl.placeholder = `Default ${fmtDef(d.chunk_max_seconds)} sec · chunk_max_seconds`;
  chunkTargetEl.title = `Default chunk_target_seconds: ${fmtDef(d.chunk_target_seconds)}`;
  chunkMinEl.title = `Default chunk_min_seconds: ${fmtDef(d.chunk_min_seconds)}`;
  chunkMaxEl.title = `Default chunk_max_seconds: ${fmtDef(d.chunk_max_seconds)}`;
  chunkEnabledEl.title = `Server default chunking_enabled: ${d.chunking_enabled ? "true" : "false"}`;

  stepsNonstreamEl.placeholder = `Default ${fmtDef(d.num_steps_nonstream)} · num_steps (non-stream)`;
  stepsNonstreamEl.title = `Default num_steps when stream is off (ECHO_NUM_STEPS_NONSTREAM): ${fmtDef(d.num_steps_nonstream)}`;

  const active = echoMeta.server_active_preset;
  const presetHint =
    active && echoMeta.sampler_presets?.[active]
      ? `Server active preset: ${active}`
      : "Pick a preset or leave Server default";
  presetEl.title = presetHint;
}

function applyMetaDefaults() {
  if (!echoMeta?.defaults) return;
  const d = echoMeta.defaults;
  cfgTextEl.value = d.cfg_scale_text ?? "";
  cfgSpeakerEl.value = d.cfg_scale_speaker ?? "";
  cfgMinTEl.value = d.cfg_min_t ?? "";
  cfgMaxTEl.value = d.cfg_max_t ?? "";
  chunkEnabledEl.checked = !!d.chunking_enabled;
  chunkTargetEl.value = d.chunk_target_seconds ?? 30;
  chunkMinEl.value = d.chunk_min_seconds ?? 20;
  chunkMaxEl.value = d.chunk_max_seconds ?? 40;
  stepsNonstreamEl.value = d.num_steps_nonstream ?? 20;
  const active = echoMeta.server_active_preset;
  if (active && presetEl.querySelector(`option[value="${active}"]`)) {
    presetEl.value = active;
  }
  applyDefaultPlaceholders();
}

function collectUiPrefs() {
  return {
    stream: streamEl.checked,
    format: formatEl.value,
    seed: seedEl.value.trim(),
    voice_id: voiceEl.value,
    sampler_preset: presetEl.value,
    cfg_scale_text: cfgTextEl.value.trim(),
    cfg_scale_speaker: cfgSpeakerEl.value.trim(),
    cfg_min_t: cfgMinTEl.value.trim(),
    cfg_max_t: cfgMaxTEl.value.trim(),
    chunking_enabled: chunkEnabledEl.checked,
    chunk_target_seconds: chunkTargetEl.value,
    chunk_min_seconds: chunkMinEl.value,
    chunk_max_seconds: chunkMaxEl.value,
    steps_nonstream: stepsNonstreamEl.value,
  };
}

function applyUiPrefs(ui) {
  if (!ui || typeof ui !== "object") return;
  if (typeof ui.stream === "boolean") streamEl.checked = ui.stream;
  if (typeof ui.format === "string") formatEl.value = ui.format;
  if (typeof ui.seed === "string") seedEl.value = ui.seed;
  if (typeof ui.sampler_preset === "string") presetEl.value = ui.sampler_preset;
  if (typeof ui.cfg_scale_text === "string") cfgTextEl.value = ui.cfg_scale_text;
  if (typeof ui.cfg_scale_speaker === "string") cfgSpeakerEl.value = ui.cfg_scale_speaker;
  if (typeof ui.cfg_min_t === "string") cfgMinTEl.value = ui.cfg_min_t;
  if (typeof ui.cfg_max_t === "string") cfgMaxTEl.value = ui.cfg_max_t;
  if (typeof ui.chunking_enabled === "boolean") chunkEnabledEl.checked = ui.chunking_enabled;
  if (typeof ui.chunk_target_seconds === "string" || typeof ui.chunk_target_seconds === "number") {
    chunkTargetEl.value = String(ui.chunk_target_seconds);
  }
  if (typeof ui.chunk_min_seconds === "string" || typeof ui.chunk_min_seconds === "number") {
    chunkMinEl.value = String(ui.chunk_min_seconds);
  }
  if (typeof ui.chunk_max_seconds === "string" || typeof ui.chunk_max_seconds === "number") {
    chunkMaxEl.value = String(ui.chunk_max_seconds);
  }
  if (typeof ui.steps_nonstream === "string" || typeof ui.steps_nonstream === "number") {
    stepsNonstreamEl.value = String(ui.steps_nonstream);
  }
}

function loadDraftText() {
  try {
    const s = localStorage.getItem(DRAFT_KEY);
    if (s) textEl.value = s;
  } catch {
    /* ignore */
  }
}

const persistDraftText = debounce(() => {
  try {
    localStorage.setItem(DRAFT_KEY, textEl.value);
  } catch {
    /* ignore */
  }
}, 400);

function applyPresetToExtra(extra, presetKey) {
  if (!presetKey || !echoMeta?.sampler_presets?.[presetKey]) return;
  const p = echoMeta.sampler_presets[presetKey];
  extra.block_sizes = p.block_sizes;
  extra.num_steps = p.num_steps;
}

function buildExtraBody() {
  const expert = parseExpert();
  const extra = {};

  const cfgT = parseFloat(cfgTextEl.value);
  const cfgS = parseFloat(cfgSpeakerEl.value);
  const cfgMin = parseFloat(cfgMinTEl.value);
  const cfgMax = parseFloat(cfgMaxTEl.value);
  if (!Number.isNaN(cfgT)) extra.cfg_scale_text = cfgT;
  if (!Number.isNaN(cfgS)) extra.cfg_scale_speaker = cfgS;
  if (!Number.isNaN(cfgMin)) extra.cfg_min_t = cfgMin;
  if (!Number.isNaN(cfgMax)) extra.cfg_max_t = cfgMax;

  extra.chunking_enabled = chunkEnabledEl.checked;
  const ct = parseFloat(chunkTargetEl.value);
  const cmin = parseFloat(chunkMinEl.value);
  const cmax = parseFloat(chunkMaxEl.value);
  if (!Number.isNaN(ct)) extra.chunk_target_seconds = ct;
  if (!Number.isNaN(cmin)) extra.chunk_min_seconds = cmin;
  if (!Number.isNaN(cmax)) extra.chunk_max_seconds = cmax;

  if (streamEl.checked && presetEl.value) {
    applyPresetToExtra(extra, presetEl.value);
  }

  if (!streamEl.checked) {
    const ns = parseInt(stepsNonstreamEl.value, 10);
    if (!Number.isNaN(ns) && ns > 0) {
      extra.num_steps = ns;
    }
  }

  return { ...extra, ...expert };
}

async function fetchMeta() {
  const res = await fetch(apiUrl("/echo/ui/meta"));
  if (!res.ok) throw new Error(await res.text());
  echoMeta = await res.json();
}

async function fetchPrefs() {
  const res = await fetch(apiUrl("/echo/ui/preferences"));
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function savePrefsToServer() {
  const body = {
    version: 1,
    ui: collectUiPrefs(),
    expert_json: extraEl.value,
  };
  const res = await fetch(apiUrl("/echo/ui/preferences"), {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text());
}

async function loadVoices(preferredId) {
  statusEl.textContent = "Loading voices…";
  const res = await fetch(apiUrl("/v1/voices"));
  if (!res.ok) {
    throw new Error(await res.text());
  }
  const body = await res.json();
  const items = body.data ?? [];
  voiceEl.innerHTML = "";
  for (const v of items) {
    const opt = document.createElement("option");
    opt.value = v.id;
    opt.textContent = v.name ?? v.id;
    voiceEl.appendChild(opt);
  }
  if (preferredId && [...voiceEl.options].some((o) => o.value === preferredId)) {
    voiceEl.value = preferredId;
  }
  statusEl.textContent = items.length ? `${items.length} voices loaded` : "No voices found — add audio under audio_prompts/";
}

streamEl.addEventListener("change", syncNonStreamUi);
textEl.addEventListener("input", persistDraftText);

reloadVoicesBtn.addEventListener("click", () => {
  loadVoices(voiceEl.value).catch((e) => {
    statusEl.textContent = String(e.message || e);
  });
});

saveSettingsBtn.addEventListener("click", async () => {
  statusEl.textContent = "Saving…";
  try {
    await savePrefsToServer();
    statusEl.textContent = "Settings saved on server.";
  } catch (e) {
    statusEl.textContent = String(e.message || e);
  }
});

resetSettingsBtn.addEventListener("click", async () => {
  statusEl.textContent = "Resetting…";
  try {
    await fetchMeta();
    applyMetaDefaults();
    presetEl.value = echoMeta.server_active_preset || "";
    extraEl.value = "";
    seedEl.value = "";
    streamEl.checked = true;
    formatEl.value = "wav";
    syncNonStreamUi();
    statusEl.textContent = "Form reset to server defaults.";
  } catch (e) {
    statusEl.textContent = String(e.message || e);
  }
});

genBtn.addEventListener("click", async () => {
  statusEl.textContent = "";
  playerEl.hidden = true;
  playerEl.removeAttribute("src");

  const input = (textEl.value || "").trim();
  if (!input) {
    statusEl.textContent = "Enter some text.";
    return;
  }

  const voice = (voiceEl.value || "").trim();
  if (!voice || voiceEl.options.length === 0) {
    statusEl.textContent = "Select a voice (reload the list if empty).";
    return;
  }

  let extra_body;
  try {
    extra_body = buildExtraBody();
  } catch {
    statusEl.textContent = "Invalid JSON in Expert — fix the JSON or clear the box.";
    return;
  }

  const seedRaw = (seedEl.value || "").trim();
  if (seedRaw !== "") {
    const n = Number(seedRaw);
    if (!Number.isInteger(n)) {
      statusEl.textContent = "Seed must be an integer.";
      return;
    }
    extra_body.seed = n;
  }

  const stream = streamEl.checked;
  let response_format = formatEl.value;

  if (!stream && response_format === "pcm") {
    statusEl.textContent = "Non-streaming: choose wav, mp3, or opus for in-browser playback.";
    return;
  }

  genBtn.disabled = true;
  statusEl.textContent = stream ? "Streaming…" : "Generating…";

  try {
    const res = await fetch(apiUrl("/v1/audio/speech"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        input,
        voice,
        stream,
        response_format,
        extra_body,
      }),
    });

    if (!res.ok) {
      statusEl.textContent = await res.text();
      return;
    }

    if (stream && response_format === "pcm") {
      const buf = await res.arrayBuffer();
      statusEl.textContent = `PCM ${buf.byteLength} bytes (raw s16le @ 44.1 kHz). Save to file or use wav for preview.`;
      return;
    }

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    playerEl.src = url;
    playerEl.load();
    playerEl.hidden = false;
    statusEl.textContent = "Done — press play if needed.";
    playerEl.play().catch(() => {});
  } catch (e) {
    statusEl.textContent = String(e.message || e);
  } finally {
    genBtn.disabled = false;
  }
});

(async function init() {
  statusEl.textContent = "Loading…";
  try {
    await fetchMeta();
    applyMetaDefaults();
    let preferredVoice = null;
    try {
      const prefs = await fetchPrefs();
      if (prefs.expert_json != null) extraEl.value = prefs.expert_json;
      applyUiPrefs(prefs.ui);
      preferredVoice = prefs.ui?.voice_id ?? null;
    } catch {
      /* no prefs file yet */
    }
    applyDefaultPlaceholders();
    loadDraftText();
    await loadVoices(preferredVoice);
    syncNonStreamUi();
    statusEl.textContent = "Ready.";
  } catch (e) {
    statusEl.textContent = String(e.message || e);
  }
})();

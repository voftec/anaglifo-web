import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6';

env.allowLocalModels = false;

const $ = id => document.getElementById(id);
const status = s => { $('status').textContent = s; };
const setBar = f => {
  $('bar').style.display = f == null ? 'none' : 'block';
  if (f != null) $('bar').firstElementChild.style.width = (f * 100).toFixed(0) + '%';
};

let imgBitmap = null;   // source photo (ImageBitmap, display res)
let depth = null;       // Float32Array w*h, normalized 0..1 (near=1)
let photoW = 0, photoH = 0;
let estimator = null;
let modelPromise = null;
let focus = null; // {fx, fy} fracción 0..1 del punto de foco elegido por el usuario

// ---------- model ----------
async function loadModel() {
  if (estimator) return estimator;
  if (modelPromise) return modelPromise;
  modelPromise = (async () => {
  status('Descargando el modelo de profundidad (solo la primera vez)…');
  let last = -1;
  const opts = {
    dtype: 'q8',
    progress_callback: p => {
      if (p.status === 'progress' && p.progress != null) {
        const f = p.progress / 100;
        if (f !== last) { last = f; setBar(f); status(`Descargando modelo… ${(f * 100).toFixed(0)}%`); }
      }
    },
  };
  let hasGPU = false;
  try { hasGPU = !!(navigator.gpu && await navigator.gpu.requestAdapter()); } catch {}
  const device = hasGPU ? 'webgpu' : 'wasm';
  estimator = await pipeline('depth-estimation', 'onnx-community/depth-anything-v2-small', {...opts, device});
  setBar(null);
  status('Modelo listo');
  return estimator;
  })();
  return modelPromise;
}

// ---------- depth ----------
async function estimateDepth(bitmap, w, h) {
  const est = await loadModel();
  status('Analizando profundidad…');
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  c.getContext('2d').drawImage(bitmap, 0, 0, w, h);
  const out = await est(c.toDataURL('image/jpeg', 0.92));
  // RawImage -> grayscale, near = bright
  const raw = await out.depth.resize(w, h);
  const px = raw.data; // Uint8ClampedArray, raw.channels can be 1 or 4
  const ch = raw.channels || 1;
  const d = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) d[i] = px[i * ch] / 255;
  return d;
}

// ---------- helpers ----------
function median3(d, w, h) {
  const o = new Float32Array(w * h);
  const v = new Float32Array(9);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let n = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const yy = Math.min(h - 1, Math.max(0, y + dy));
      const xx = Math.min(w - 1, Math.max(0, x + dx));
      v[n++] = d[yy * w + xx];
    }
    v.subarray(0, n).sort();
    o[y * w + x] = n === 9 ? v[4] : v[n >> 1];
  }
  return o;
}

function gaussBlur(d, w, h, sigma) {
  if (sigma < 0.5) return d;
  const r = Math.max(1, Math.round(sigma * 3));
  const k = new Float32Array(2 * r + 1);
  let sum = 0;
  for (let i = -r; i <= r; i++) { k[i + r] = Math.exp(-0.5 * (i / sigma) ** 2); sum += k[i + r]; }
  for (let i = 0; i < k.length; i++) k[i] /= sum;
  const t = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let a = 0;
    for (let i = -r; i <= r; i++)
      a += d[y * w + Math.min(w - 1, Math.max(0, x + i))] * k[i + r];
    t[y * w + x] = a;
  }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let a = 0;
    for (let i = -r; i <= r; i++)
      a += t[Math.min(h - 1, Math.max(0, y + i)) * w + x] * k[i + r];
    d[y * w + x] = a;
  }
  return d;
}

function quantile(d, q) {
  const s = Float32Array.from(d).sort();
  return s[Math.min(s.length - 1, Math.floor(q * s.length))];
}

// forward splat: for each row, z-buffer the nearest (max dn) source into target
function renderRight(src, dn, w, h, spx, convShift, span) {
  const disp = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) disp[i] = -spx * (dn[i] - convShift) / span;

  // mark which target pixels get coverage
  const cov = new Float32Array(w * h).fill(-1); // stores dn of winner
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    const xt = Math.round(x + disp[i]);
    if (xt >= 0 && xt < w) {
      const t = y * w + xt;
      if (dn[i] > cov[t]) cov[t] = dn[i];
    }
  }

  // fill holes row-wise from the nearer boundary with the FARTHER dn
  const dispT = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    // left-to-right nearest-covered pass
    const Ld = new Float32Array(w), Ln = new Float32Array(w);
    let bd = 0, bn = -1, dist = 1e9;
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (cov[i] >= 0) { bd = disp[i]; bn = cov[i]; dist = 0; }
      dist++;
      Ld[x] = bd; Ln[x] = bn;
    }
    bd = 0; bn = -1; dist = 1e9;
    for (let x = w - 1; x >= 0; x--) {
      const i = y * w + x;
      if (cov[i] >= 0) { bd = disp[i]; bn = cov[i]; dist = 0; }
      dist++;
      const i2 = y * w + x;
      if (cov[i2] >= 0) { dispT[i2] = disp[i2]; continue; }
      // choose side with smaller dn (farther background fills the hole)
      dispT[i2] = (Ln[x] <= bn) ? Ld[x] : bd;
    }
  }

  // inverse sample src at x - dispT (bilinear, edge clamp)
  const out = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const sx = Math.min(w - 1.001, Math.max(0, x - dispT[y * w + x]));
    const x0 = Math.floor(sx), fx = sx - x0;
    for (let c = 0; c < 3; c++) {
      const a = src[(y * w + x0) * 4 + c], b = src[(y * w + Math.min(w - 1, x0 + 1)) * 4 + c];
      out[(y * w + x) * 4 + c] = a + (b - a) * fx;
    }
    out[(y * w + x) * 4 + 3] = 255;
  }
  return out;
}

// Dubois least-squares anaglyph (red/cyan, CRT glasses)
const DB = [
  [0.4561, 0.500484, 0.176381, -0.0434706, -0.0879388, -0.00155529],
  [-0.0400822, -0.0378246, -0.0157589, 0.378476, 0.73364, -0.0184503],
  [-0.0152161, -0.0205971, -0.00546856, -0.0721527, -0.112961, 1.2264],
];
function dubois(L, R, n) {
  const out = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) {
    const l = [L[i * 4] / 255, L[i * 4 + 1] / 255, L[i * 4 + 2] / 255];
    const r = [R[i * 4] / 255, R[i * 4 + 1] / 255, R[i * 4 + 2] / 255];
    for (let c = 0; c < 3; c++) {
      let v = DB[c][0] * l[0] + DB[c][1] * l[1] + DB[c][2] * l[2]
            + DB[c][3] * r[0] + DB[c][4] * r[1] + DB[c][5] * r[2];
      out[i * 4 + c] = Math.max(0, Math.min(255, v * 255));
    }
    out[i * 4 + 3] = 255;
  }
  return out;
}

// ---------- pipeline ----------
async function convert() {
  const strength = $('strength').value / 100;      // 0..1
  const conv = $('conv').value / 100;              // convergence point 0..1
  const smooth = $('smooth').value / 100;
  if (!depth) {
    status('Analizando profundidad…');
    depth = await estimateDepth(imgBitmap, photoW, photoH);
  }

  let dn = median3(depth, photoW, photoH);
  // robust normalize 0..1
  const lo = quantile(dn, 0.01), hi = quantile(dn, 0.99);
  for (let i = 0; i < dn.length; i++)
    dn[i] = Math.min(1, Math.max(0, (dn[i] - lo) / Math.max(hi - lo, 1e-6)));
  const sigma = photoW * (0.004 + smooth * 0.024);
  gaussBlur(dn, photoW, photoH, sigma);

  const srcC = document.createElement('canvas');
  srcC.width = photoW; srcC.height = photoH;
  const sctx = srcC.getContext('2d');
  sctx.drawImage(imgBitmap, 0, 0, photoW, photoH);
  const src = sctx.getImageData(0, 0, photoW, photoH).data;

  const spx = (0.005 + strength * 0.045) * photoW;
  const p5 = quantile(dn, 0.05), p95 = quantile(dn, 0.95);
  const span = Math.max(p95 - p5, 1e-6);
  // si el usuario picó un punto de foco, su profundidad define el plano cero;
  // si no, el slider de convergencia decide
  const convShift = focus
    ? dn[Math.min(photoH - 1, Math.floor(focus.fy * photoH)) * photoW
        + Math.min(photoW - 1, Math.floor(focus.fx * photoW))]
    : p5 + conv * span;

  status('Renderizando…');
  await new Promise(r => setTimeout(r)); // let UI breathe
  const right = renderRight(src, dn, photoW, photoH, spx, convShift, span);

  const ana = dubois(src, right, photoW * photoH);
  const out = $('out');
  out.width = photoW; out.height = photoH;
  out.getContext('2d').putImageData(new ImageData(ana, photoW, photoH), 0, 0);
  out.style.display = 'block';
  $('glasses').style.display = 'block';
  $('dl').disabled = false;
  status('Listo');
}

// ---------- ui wiring ----------
const drop = $('drop'), file = $('file');
drop.onclick = () => file.click();
drop.ondragover = e => { e.preventDefault(); drop.classList.add('over'); };
drop.ondragleave = () => drop.classList.remove('over');
drop.ondrop = e => { e.preventDefault(); drop.classList.remove('over'); if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]); };
file.onchange = () => file.files[0] && loadFile(file.files[0]);

const MAX_SIDE = 1200;
async function loadFile(f) {
  const raw = await createImageBitmap(f);
  const s = Math.min(1, MAX_SIDE / Math.max(raw.width, raw.height));
  photoW = Math.round(raw.width * s); photoH = Math.round(raw.height * s);
  imgBitmap = await createImageBitmap(raw, { resizeWidth: photoW, resizeHeight: photoH, resizeQuality: 'high' });
  depth = null;
  focus = null;
  $('focus').style.display = 'none';
  drop.innerHTML = `<b>${f.name}</b><br>${photoW}×${photoH} — listo para convertir`;
  $('go').disabled = false;
  $('dl').disabled = true;
  $('sliders').classList.add('on');
  // mostrar la foto para poder picar el punto de foco
  const oc = $('out');
  oc.width = photoW; oc.height = photoH;
  oc.getContext('2d').drawImage(imgBitmap, 0, 0);
  oc.style.display = 'block';
  $('pickhint').style.display = 'block';
  $('glasses').style.display = 'none';
  status('');
  // pre-load model in background
  loadModel().catch(e => status('Error modelo: ' + e.message));
}

let busy = false;
$('go').onclick = async () => {
  if (!imgBitmap || busy) return;
  busy = true; $('go').disabled = true;
  try { await convert(); }
  catch (e) { status('Error: ' + e.message); console.error(e); }
  busy = false; $('go').disabled = false;
};

let rerenderTimer = null;
for (const id of ['strength', 'conv', 'smooth']) {
  $(id).oninput = () => {
    $('o' + id).textContent = $(id).value + (id === 'strength' ? '%' : '');
    if (!depth || busy) return;
    clearTimeout(rerenderTimer);
    rerenderTimer = setTimeout(() => $('go').click(), 250);
  };
}

$('dl').onclick = () => {
  const a = document.createElement('a');
  a.href = $('out').toDataURL('image/png');
  a.download = 'anaglifo.png';
  a.click();
};

// picker de punto de foco: click/tap sobre la imagen define el plano de mayor interés
$('out').addEventListener('click', e => {
  if (!imgBitmap) return;
  const r = $('out').getBoundingClientRect();
  focus = { fx: (e.clientX - r.left) / r.width, fy: (e.clientY - r.top) / r.height };
  const f = $('focus');
  f.style.display = 'block';
  f.style.left = (focus.fx * 100) + '%';
  f.style.top = (focus.fy * 100) + '%';
  status('Punto de foco marcado — regenerando…');
  if (depth && !busy) { clearTimeout(rerenderTimer); rerenderTimer = setTimeout(() => $('go').click(), 150); }
});

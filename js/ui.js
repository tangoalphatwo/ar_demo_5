// ui.js

export function setStatus(text) {
  const statusEl = document.getElementById('status');
  if (statusEl) statusEl.textContent = text;
}

export function computeContainRect(srcW, srcH, dstW, dstH) {
  const sw = Number(srcW);
  const sh = Number(srcH);
  const dw = Number(dstW);
  const dh = Number(dstH);
  if (!isFinite(sw) || !isFinite(sh) || !isFinite(dw) || !isFinite(dh) || sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0) {
    return { x: 0, y: 0, w: dw || 1, h: dh || 1, scale: 1 };
  }
  const scale = Math.min(dw / sw, dh / sh);
  const w = sw * scale;
  const h = sh * scale;
  const x = (dw - w) * 0.5;
  const y = (dh - h) * 0.5;
  return { x, y, w, h, scale };
}

export function applyViewRect({ videoEl, threeCanvas, cvCanvas }) {
  if (!videoEl || !threeCanvas || !cvCanvas) return;
  const vw = videoEl.videoWidth;
  const vh = videoEl.videoHeight;
  const rect = computeContainRect(vw, vh, window.innerWidth, window.innerHeight);

  // Size/position the visible background video and both canvases identically.
  const px = (n) => `${Math.round(n)}px`;
  for (const el of [videoEl, threeCanvas, cvCanvas]) {
    el.style.left = px(rect.x);
    el.style.top = px(rect.y);
    el.style.width = px(rect.w);
    el.style.height = px(rect.h);
  }

  return rect;
}

// ui.js

export function setStatus(text) {
  const statusEl = document.getElementById('status');
  if (statusEl) statusEl.textContent = text;
}

function computeContainRect(srcW, srcH, dstW, dstH) {
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

export function applyViewRect({ videoEl, cvCanvas, threeCanvas }) {
  if (!videoEl || !cvCanvas || !threeCanvas) return;

  // On some mobile browsers, videoWidth/videoHeight can briefly report 0.
  // Fall back to the canvas backing size, which we set from the video metadata.
  const vw = videoEl.videoWidth || cvCanvas.width;
  const vh = videoEl.videoHeight || cvCanvas.height;
  const rect = computeContainRect(vw, vh, window.innerWidth, window.innerHeight);

  const px = (n) => `${Math.round(n)}px`;
  for (const el of [cvCanvas, threeCanvas]) {
    el.style.left = px(rect.x);
    el.style.top = px(rect.y);
    el.style.width = px(rect.w);
    el.style.height = px(rect.h);
  }

  return rect;
}

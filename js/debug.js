// debug.js

export function logOpenCVInfo(cv) {
  console.log('[OpenCV] instance type:', typeof cv);
  console.log('[OpenCV] version:', cv.version || '(no version field)');

  const caps = {
    Mat: !!cv.Mat,
    getBuildInformation: typeof cv.getBuildInformation,
    ORB: typeof cv.ORB,
    ORB_create: typeof cv.ORB_create,
    BFMatcher: typeof cv.BFMatcher,
    findHomography: typeof cv.findHomography,
    decomposeHomographyMat: typeof cv.decomposeHomographyMat,
    solvePnP: typeof cv.solvePnP,
    Rodrigues: typeof cv.Rodrigues,
    recoverPose: typeof cv.recoverPose,
    findEssentialMat: typeof cv.findEssentialMat,
    findFundamentalMat: typeof cv.findFundamentalMat,
    calcOpticalFlowPyrLK: typeof cv.calcOpticalFlowPyrLK,
    goodFeaturesToTrack: typeof cv.goodFeaturesToTrack
  };
  console.table(caps);

  try {
    if (typeof cv.getBuildInformation === 'function') {
      console.log(
        '[OpenCV] build info (first 20 lines):\n' +
          String(cv.getBuildInformation()).split('\n').slice(0, 20).join('\n')
      );
    }
  } catch (e) {
    console.warn('[OpenCV] getBuildInformation failed:', e);
  }

  // Sanity: create and delete a tiny Mat
  try {
    const m = cv.Mat.eye(2, 2, cv.CV_64F);
    const data = m.data64F
      ? Array.from(m.data64F)
      : (m.data32F ? Array.from(m.data32F) : null);
    console.log('[OpenCV] Mat.eye(2) ok:', data);
    m.delete();
  } catch (e) {
    console.warn('[OpenCV] Mat sanity check failed:', e);
  }
}

export function drawGreenDots(ctx, points, { radius = 2 } = {}) {
  if (!points || points.length === 0) return;
  ctx.save();
  ctx.fillStyle = 'rgba(0, 255, 0, 0.9)';
  for (const p of points) {
    const x = p.x | 0;
    const y = p.y | 0;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

export function drawQuad(ctx, corners, { color = 'rgba(255, 255, 0, 0.9)', width = 4 } = {}) {
  if (!corners || corners.length !== 4) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(corners[0].x, corners[0].y);
  ctx.lineTo(corners[1].x, corners[1].y);
  ctx.lineTo(corners[2].x, corners[2].y);
  ctx.lineTo(corners[3].x, corners[3].y);
  ctx.closePath();
  ctx.stroke();
  ctx.restore();
}

export function createThrottle(intervalMs) {
  let lastMs = 0;
  return (nowMs = performance.now()) => {
    if (nowMs - lastMs > intervalMs) {
      lastMs = nowMs;
      return true;
    }
    return false;
  };
}

export function createDebugUI({ debugToggleEl, debugInfoEl } = {}) {
  let enabled = false;
  let cvCanvas = null;

  const apply = () => {
    if (debugInfoEl) {
      debugInfoEl.hidden = !enabled;
      debugInfoEl.setAttribute('aria-hidden', String(!enabled));
    }

    if (debugToggleEl) {
      debugToggleEl.setAttribute('aria-pressed', String(enabled));
      debugToggleEl.textContent = enabled ? 'Hide Debug' : 'Show Debug';
    }

    if (cvCanvas) {
      cvCanvas.style.zIndex = enabled ? '3' : '1';
      cvCanvas.style.display = enabled ? 'block' : 'none';
    }
  };

  const onToggle = () => {
    enabled = !enabled;
    apply();
  };

  if (debugToggleEl) {
    debugToggleEl.hidden = false;
    debugToggleEl.addEventListener('click', onToggle);
  }

  apply();

  return {
    isEnabled: () => enabled,
    setEnabled: (v) => {
      enabled = !!v;
      apply();
    },
    attachCvCanvas: (canvasEl) => {
      cvCanvas = canvasEl || null;
      apply();
    },
    setDebugText: (text) => {
      if (!debugInfoEl) return;
      debugInfoEl.textContent = text;
    }
  };
}

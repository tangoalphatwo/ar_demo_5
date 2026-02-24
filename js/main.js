// main.js
// Start button -> start camera + wait for OpenCV -> run SLAM per frame

import { SlamCore } from './slam_core.js';
import { PlaneDetector } from './plane_detector.js';

function setStatus(text) {
  const statusEl = document.getElementById('status');
  if (statusEl) statusEl.textContent = text;
}

function setDebugInfo(text) {
  const el = document.getElementById('debugInfo');
  if (!el) return;
  el.textContent = text;
}

function makeNonThenable(cvObj) {
  if (!cvObj) return cvObj;
  try {
    if (typeof cvObj.then === 'function') {
      console.warn('[OpenCV] cv is thenable; wrapping to avoid await-adoption');
      return new Proxy(cvObj, {
        get(target, prop, receiver) {
          if (prop === 'then') return undefined;
          return Reflect.get(target, prop, receiver);
        }
      });
    }
  } catch {
    // ignore
  }
  return cvObj;
}

async function waitForOpenCV({ timeoutMs = 20000 } = {}) {
  const start = performance.now();

  while (performance.now() - start < timeoutMs) {
    // Modular builds sometimes expose cv as a Promise
    if (window.cv instanceof Promise) {
      const cvResolved = await window.cv;
      return makeNonThenable(cvResolved);
    }

    // Docs build uses Module.onRuntimeInitialized -> __opencvReady
    if (window.__opencvReady && window.cv && window.cv.Mat) {
      return makeNonThenable(window.cv);
    }

    // Fallback if __opencvReady never flips but cv looks usable
    if (window.cv && window.cv.Mat) {
      return makeNonThenable(window.cv);
    }

    await new Promise((r) => setTimeout(r, 50));
  }

  throw new Error('Timed out waiting for OpenCV to initialize');
}

function logOpenCVInfo(cv) {
  console.log('[OpenCV] instance type:', typeof cv);
  console.log('[OpenCV] version:', cv.version || '(no version field)');

  const caps = {
    Mat: !!cv.Mat,
    getBuildInformation: typeof cv.getBuildInformation,
    ORB_create: typeof cv.ORB_create,
    BFMatcher: typeof cv.BFMatcher,
    findHomography: typeof cv.findHomography,
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

async function startCameraPreview() {
  const videoEl = document.getElementById('camera');
  const cvCanvas = document.getElementById('cvCanvas');
  const threeCanvas = document.getElementById('threeCanvas');

  if (!videoEl) throw new Error('Missing #camera element');
  if (!cvCanvas) throw new Error('Missing #cvCanvas element');

  // Ensure preview is visible
  if (threeCanvas) threeCanvas.style.display = 'none';

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: { ideal: 'environment' },
      width: { ideal: 1280 },
      height: { ideal: 720 }
    }
  });

  videoEl.srcObject = stream;
  videoEl.playsInline = true;
  videoEl.muted = true;

  await new Promise((resolve) => {
    if (videoEl.readyState >= 1) return resolve();
    videoEl.onloadedmetadata = () => resolve();
  });

  await videoEl.play();

  const vw = videoEl.videoWidth;
  const vh = videoEl.videoHeight;
  console.log('[Camera] video size:', { vw, vh });

  // Backing buffer matches the real video; CSS scales to viewport.
  cvCanvas.width = vw;
  cvCanvas.height = vh;

  const ctx = cvCanvas.getContext('2d', { alpha: false, desynchronized: true, willReadFrequently: true });

  return {
    videoEl,
    cvCanvas,
    ctx,
    stop() {
      try {
        stream.getTracks().forEach((t) => t.stop());
      } catch {
        // ignore
      }
    }
  };
}

function drawGreenDots(ctx, points, { radius = 2 } = {}) {
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

window.addEventListener('load', () => {
  const startBtn = document.getElementById('startBtn');
  const debugToggle = document.getElementById('debugToggle');
  if (!startBtn) {
    console.warn('Start button not found');
    return;
  }

  let running = false;
  let debugEnabled = false;
  let slam = null;
  let planeDetector = null;
  let cameraHandle = null;

  if (debugToggle) {
    debugToggle.hidden = false;
    debugToggle.addEventListener('click', () => {
      debugEnabled = !debugEnabled;
      debugToggle.setAttribute('aria-pressed', String(debugEnabled));
      debugToggle.textContent = debugEnabled ? 'Hide Debug' : 'Show Debug';
      const debugInfoEl = document.getElementById('debugInfo');
      if (debugInfoEl) {
        debugInfoEl.hidden = !debugEnabled;
        debugInfoEl.setAttribute('aria-hidden', String(!debugEnabled));
      }
    });
  }

  startBtn.addEventListener('click', async () => {
    startBtn.disabled = true;
    setStatus('Starting camera + OpenCV…');

    console.log('[Boot] __opencvReady:', window.__opencvReady);
    console.log('[Boot] window.cv type:', typeof window.cv);

    try {
      const [camHandle, cv] = await Promise.all([
        startCameraPreview(),
        waitForOpenCV({ timeoutMs: 20000 })
      ]);

      cameraHandle = camHandle;

      // Handy for DevTools
      window.__cameraHandle = cameraHandle;
      window.cvInstance = cv;

      slam = new SlamCore(cv);
      planeDetector = new PlaneDetector({
        ransacIters: 250,
        inlierThreshold: 0.08,
        minInliers: 50
      });

      setStatus('Running');
      console.log('[Boot] Camera running');
      console.log('[Boot] OpenCV ready');
      logOpenCVInfo(cv);

      running = true;
      const { videoEl, cvCanvas, ctx } = cameraHandle;
      const tick = () => {
        if (!running) return;

        // Draw current video frame
        ctx.drawImage(videoEl, 0, 0, cvCanvas.width, cvCanvas.height);

        // Run SLAM + optional edge extraction
        const imageData = ctx.getImageData(0, 0, cvCanvas.width, cvCanvas.height);
        const result = slam.processFrame(imageData, {
          detectEdges: debugEnabled,
          maxEdgePoints: 700
        });

        // Debug overlay: green edge dots
        if (debugEnabled && result?.edgePoints2D) {
          drawGreenDots(ctx, result.edgePoints2D, { radius: 2 });
        }

        // Plane detection on triangulated 3D points
        let dbg = '';
        const pts3D = result?.mapPoints3D || [];
        dbg += `2D edges: ${result?.edgePoints2D ? result.edgePoints2D.length : 0}\n`;
        dbg += `3D points: ${pts3D.length}\n`;
        if (pts3D.length >= 80 && planeDetector) {
          const plane = planeDetector.detect(pts3D);
          if (plane) {
            const n = plane.normal;
            dbg += `Plane inliers: ${plane.inliers.length}\n`;
            dbg += `Plane n: (${n.x.toFixed(3)}, ${n.y.toFixed(3)}, ${n.z.toFixed(3)})\n`;
            dbg += `Plane d: ${plane.d.toFixed(3)}\n`;
          } else {
            dbg += 'Plane: (none)\n';
          }
        } else {
          dbg += 'Plane: (need more 3D points)\n';
        }
        if (debugEnabled) setDebugInfo(dbg);

        requestAnimationFrame(tick);
      };

      requestAnimationFrame(tick);
    } catch (e) {
      setStatus('Startup failed (see console)');
      console.error('[Boot] Startup failed:', e);
      startBtn.disabled = false;
    }
  });
});
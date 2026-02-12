// main.js
// Minimal bootstrap: Start button -> start camera preview + wait for OpenCV + log sanity checks

function setStatus(text) {
  const statusEl = document.getElementById('status');
  if (statusEl) statusEl.textContent = text;
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

  const ctx = cvCanvas.getContext('2d', { alpha: false, desynchronized: true });

  let running = true;
  const draw = () => {
    if (!running) return;
    ctx.drawImage(videoEl, 0, 0, cvCanvas.width, cvCanvas.height);
    requestAnimationFrame(draw);
  };
  requestAnimationFrame(draw);

  return {
    stop() {
      running = false;
      try {
        stream.getTracks().forEach((t) => t.stop());
      } catch {
        // ignore
      }
    }
  };
}

window.addEventListener('load', () => {
  const startBtn = document.getElementById('startBtn');
  if (!startBtn) {
    console.warn('Start button not found');
    return;
  }

  startBtn.addEventListener('click', async () => {
    startBtn.disabled = true;
    setStatus('Starting camera + OpenCV…');

    console.log('[Boot] __opencvReady:', window.__opencvReady);
    console.log('[Boot] window.cv type:', typeof window.cv);

    try {
      const [cameraHandle, cv] = await Promise.all([
        startCameraPreview(),
        waitForOpenCV({ timeoutMs: 20000 })
      ]);

      // Handy for DevTools
      window.__cameraHandle = cameraHandle;
      window.cvInstance = cv;

      setStatus('Camera running + OpenCV ready (see console)');
      console.log('[Boot] Camera running');
      console.log('[Boot] OpenCV ready');
      logOpenCVInfo(cv);
    } catch (e) {
      setStatus('Startup failed (see console)');
      console.error('[Boot] Startup failed:', e);
      startBtn.disabled = false;
    }
  });
});
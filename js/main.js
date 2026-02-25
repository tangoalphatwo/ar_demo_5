// main.js
// Start button -> start camera + wait for OpenCV -> run SLAM per frame

import { initPose, estimatePose } from './pose.js';
import { detectMarkerQuad } from './marker_quad.js';
import { ARRenderer } from './renderer.js';
import { applyViewRect, setStatus } from './ui.js';

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


async function startCameraPreview() {
  const videoEl = document.getElementById('camera');
  const cvCanvas = document.getElementById('cvCanvas');

  if (!videoEl) throw new Error('Missing #camera element');
  if (!cvCanvas) throw new Error('Missing #cvCanvas element');

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

function radToDeg(r) {
  return (r * 180) / Math.PI;
}

function createThrottle(intervalMs) {
  let last = 0;
  return (now = performance.now()) => {
    if (now - last >= intervalMs) {
      last = now;
      return true;
    }
    return false;
  };
}


window.addEventListener('load', () => {
  const startBtn = document.getElementById('startBtn');
  if (!startBtn) {
    console.warn('Start button not found');
    return;
  }

  let running = false;
  let cameraHandle = null;
  let ar = null;
  const poseLogThrottle = createThrottle(250);
  const markerLogThrottle = createThrottle(1200);

  // Marker-based bootstrap state
  let worldLocked = false;
  let markerSeen = false;

  const MARKER_SIZE_METERS = 0.1016; // 4 inches
  const MARKER_MIN_AREA_FRAC = 0.005;
  const MODEL_URL = 'model/house.glb';
  const TARGET_HEIGHT_M = 0.0762; // ~3 inches

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

      // Pose init for solvePnP
      initPose(cameraHandle.videoEl, cv);

      // Three overlay setup
      const threeCanvas = document.getElementById('threeCanvas');
      if (!threeCanvas) throw new Error('Missing #threeCanvas element');
      ar = new ARRenderer(threeCanvas);

      // Match Three camera projection to the same simple intrinsics used by solvePnP.
      // pose.js uses focalLengthPx = max(videoWidthPx, videoHeightPx).
      const focalLengthPx = Math.max(cameraHandle.videoEl.videoWidth, cameraHandle.videoEl.videoHeight);
      ar.setProjectionFromVideo({
        videoWidthPx: cameraHandle.videoEl.videoWidth,
        videoHeightPx: cameraHandle.videoEl.videoHeight,
        focalLengthPx
      });

      const doLayout = () => {
        applyViewRect({ videoEl: cameraHandle.videoEl, cvCanvas: cameraHandle.cvCanvas, threeCanvas });
        ar.resize();
      };
      doLayout();
      window.addEventListener('resize', doLayout, { passive: true });
      window.addEventListener('orientationchange', doLayout, { passive: true });

      // Load and spawn model at world zero (marker center)
      setStatus('Loading model…');
      console.log('[Model] Loading', MODEL_URL);
      ar.loadGLB(MODEL_URL)
        .then((gltf) => {
          const info = ar.addModelAtWorldZero(gltf.scene, { targetHeightM: TARGET_HEIGHT_M });
          console.log('[Model] Bounds before scale (m-ish units):', info.sizeBefore);
          console.log('[Model] Scale applied:', info.scaleApplied);
          console.log('[Model] Bounds after scale:', info.sizeAfter);
          console.log('[Model] Spawned at world zero (marker center)');
          setStatus('Running');
        })
        .catch((e) => {
          console.error('[Model] Failed to load', MODEL_URL, e);
          setStatus('Model load failed (see console)');
        });

      setStatus('Running');
      console.log('[Boot] Camera running');
      console.log('[Boot] OpenCV ready');
      console.log('[World] Marker coordinate system: origin is marker center (0,0,0)');

      running = true;
      const { videoEl, cvCanvas, ctx } = cameraHandle;

      const tick = () => {
        if (!running) return;

        let rgba = null;
        let gray = null;

        try {
          // Draw current video frame
          ctx.drawImage(videoEl, 0, 0, cvCanvas.width, cvCanvas.height);

          // Read pixels once per frame
          const imageData = ctx.getImageData(0, 0, cvCanvas.width, cvCanvas.height);

          // Grayscale for marker detection
          rgba = cv.matFromImageData(imageData);
          gray = new cv.Mat();
          cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY);

          // Marker detection → solvePnP
          const quad = detectMarkerQuad(cv, gray, { minAreaFrac: MARKER_MIN_AREA_FRAC });
          if (quad?.corners) {
            if (!markerSeen) {
              markerSeen = true;
              console.log('[Marker] detected');
            }

            const s = MARKER_SIZE_METERS;
            const half = s * 0.5;
            const objectPoints = [
              // World/marker coords: X right, Y up, Z out of the marker plane.
              { x: -half, y: half, z: 0 },
              { x: half, y: half, z: 0 },
              { x: half, y: -half, z: 0 },
              { x: -half, y: -half, z: 0 }
            ];

            const pose = estimatePose(quad.corners, objectPoints, cv);

            // Update Three camera from marker pose (marker is world origin).
            if (pose && ar) {
              ar.setCameraFromMarkerPose(pose);
            }

            // Step 4: world zero is the marker center (object points are centered at origin).
            // We "lock" once we have a valid pose.
            if (pose && !worldLocked) {
              worldLocked = true;
              setStatus('World locked');
              console.log('[World] zero set at marker center');
            } else if (!worldLocked) {
              setStatus('Detecting marker…');
            }

            // Step 6: distance + rotation from pose
            if (pose && worldLocked && poseLogThrottle()) {
              const p = pose.position;
              const d = Math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
              const r = pose.rotation;
              console.log('[Pose] distance(m):', d, 'rotation(deg):', {
                yaw: radToDeg(r.yaw),
                pitch: radToDeg(r.pitch),
                roll: radToDeg(r.roll)
              }, 't(m):', p);
            }
          } else {
            if (markerSeen) {
              markerSeen = false;
              console.log('[Marker] lost');
            }

            setStatus('Point at marker');
          }

          // Render Three overlay (even if marker is momentarily lost)
          ar?.render?.();
        } catch (e) {
          console.error('[Tick] error:', e);
        } finally {
          try {
            rgba?.delete?.();
          } catch {
            // ignore
          }
          try {
            gray?.delete?.();
          } catch {
            // ignore
          }
          requestAnimationFrame(tick);
        }
      };

      requestAnimationFrame(tick);
    } catch (e) {
      setStatus('Startup failed (see console)');
      console.error('[Boot] Startup failed:', e);
      startBtn.disabled = false;
    }
  });
});
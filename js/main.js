// main.js
// Start button -> start camera + wait for OpenCV -> run SLAM per frame

import { initPose, estimatePose } from './pose.js';
import { detectMarkerQuad } from './marker_quad.js';
import { ARRenderer } from './renderer.js';
import { applyViewRect, setStatus, setStatusLines } from './ui.js';

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
  const layoutLogThrottle = createThrottle(1200);

  // Marker-based bootstrap state
  let worldLocked = false;
  let markerSeen = false;

  const MARKER_SIZE_METERS = 0.1016; // 4 inches
  const MARKER_MIN_AREA_FRAC = 0.005;
  const MODEL_URL = 'model/house.glb';
  // Keep it under ~3" tall; tune as needed.
  const TARGET_HEIGHT_M = 0.0508; // ~2 inches

  startBtn.addEventListener('click', async () => {
    startBtn.disabled = true;
    setStatusLines(['Starting…', 'Requesting camera + OpenCV…']);

    console.log('[Boot] __opencvReady:', window.__opencvReady);
    console.log('[Boot] window.cv type:', typeof window.cv);

    try {
      const [camHandle, cv] = await Promise.all([
        startCameraPreview(),
        waitForOpenCV({ timeoutMs: 20000 })
      ]);

      setStatusLines(['Camera + OpenCV ready', 'Initializing pose…']);

      cameraHandle = camHandle;

      // Handy for DevTools
      window.__cameraHandle = cameraHandle;
      window.cvInstance = cv;

      // Pose init for solvePnP
      initPose(cameraHandle.videoEl, cv);

      setStatusLines(['Pose initialized', 'Setting up renderer…']);

      // Three overlay setup
      const threeCanvas = document.getElementById('threeCanvas');
      if (!threeCanvas) throw new Error('Missing #threeCanvas element');
      ar = new ARRenderer(threeCanvas);

      // Match Three camera projection to the same simple intrinsics used by solvePnP.
      // pose.js uses focalLengthPx = videoWidthPx.
      const focalLengthPx = cameraHandle.videoEl.videoWidth;
      ar.setProjectionFromVideo({
        videoWidthPx: cameraHandle.videoEl.videoWidth,
        videoHeightPx: cameraHandle.videoEl.videoHeight,
        focalLengthPx
      });

      const doLayout = () => {
        const rect = applyViewRect({ videoEl: cameraHandle.videoEl, cvCanvas: cameraHandle.cvCanvas, threeCanvas });
        ar.resize();

        if (layoutLogThrottle()) {
          const cvRect = cameraHandle.cvCanvas.getBoundingClientRect();
          const threeRect = threeCanvas.getBoundingClientRect();
          console.log('[Layout] viewRect:', rect, {
            video: { vw: cameraHandle.videoEl.videoWidth, vh: cameraHandle.videoEl.videoHeight },
            cvCanvas: { cssW: cvRect.width, cssH: cvRect.height, left: cvRect.left, top: cvRect.top, w: cameraHandle.cvCanvas.width, h: cameraHandle.cvCanvas.height },
            threeCanvas: { cssW: threeRect.width, cssH: threeRect.height, left: threeRect.left, top: threeRect.top },
            three: ar.getDebugInfo?.()
          });
        }
      };
      doLayout();
      window.addEventListener('resize', doLayout, { passive: true });
      window.addEventListener('orientationchange', doLayout, { passive: true });

      // Load and spawn model at world zero (marker center)
      setStatusLines(['Renderer ready', 'Loading model…']);
      console.log('[Model] Loading', MODEL_URL);
      ar.loadGLB(MODEL_URL)
        .then((gltf) => {
          const model = gltf.scene;

          // TEMP DEBUG: mimic ar_demo_7's simple placement style
          ar.clearWorld();

          model.position.set(0, 0, 0.02);   // 2 cm off marker plane
          model.scale.setScalar(0.001);      // temporary hardcoded test scale

          ar.world.add(model);
          ar.world.visible = !!(worldLocked && markerSeen);

          console.log('[Model] test placement active');
          setStatusLines(['Model loaded', 'Point at marker']);
        });

      setStatusLines(['Running', 'Point at marker']);
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
              setStatusLines(['Marker detected', 'Estimating pose…']);
            }

            const s = MARKER_SIZE_METERS;
            const half = s * 0.5;
            const objectPoints = [
              // Marker/object coordinates for solvePnP (matches ar_demo_7):
              // - Marker lies on X/Y plane, Z = 0
              // - Y is "down" in the marker plane, so TL/TR have negative Y
              // - detectMarkerQuad() corners are ordered [tl,tr,br,bl]
              { x: -half, y: -half, z: 0 },
              { x: half, y: -half, z: 0 },
              { x: half, y: half, z: 0 },
              { x: -half, y: half, z: 0 }
            ];

            const pose = estimatePose(quad.corners, objectPoints, cv);

            if (!pose) {
              // Keep the user-facing status accurate if solvePnP is failing.
              if (poseLogThrottle()) {
                setStatusLines(['Marker detected', 'Pose failed (solvePnP)']);
              }
            }

            // Update Three camera from marker pose (marker is world origin).
            if (pose && ar) {
              // Anchor the world group to the marker pose (camera stays at origin).
              // This matches the approach used in the working ar_demo_7 repo and
              // avoids subtle pose-inversion sign issues that can make models
              // "spawn" but remain off-camera.
              ar.setWorldFromMarkerPose(pose);
              // Show model only when pose is valid.
              ar.world.visible = true;
            }

            if (pose && ar && markerLogThrottle()) {
              console.log('[AR] world:', {
                visible: !!ar.world?.visible,
                children: ar.world?.children?.length ?? null
              });
            }

            // Step 4: world zero is the marker center (object points are centered at origin).
            // We "lock" once we have a valid pose.
            if (pose && !worldLocked) {
              worldLocked = true;
              setStatusLines(['World locked', 'Tracking…']);
              console.log('[World] zero set at marker center');
            } else if (!worldLocked) {
              setStatusLines(['Running', 'Detecting marker…']);
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

              // Extra instrumentation: helps diagnose "model stuck" issues.
              if (ar?.camera) {
                console.log('[Three] camera pos:', {
                  x: ar.camera.position.x,
                  y: ar.camera.position.y,
                  z: ar.camera.position.z
                });
              }
            }
          } else {
            if (markerSeen) {
              markerSeen = false;
              console.log('[Marker] lost');
              setStatusLines(['Point at marker', '(marker lost)']);
            }

            // Hide model when tracking is lost.
            if (ar?.world) ar.world.visible = false;

            if (!markerSeen) setStatusLines(['Point at marker', '']);
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
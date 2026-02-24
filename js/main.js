// main.js
// Start button -> start camera + wait for OpenCV -> run SLAM per frame

import { ARRenderer } from './renderer.js';
import { initPose, estimatePose } from './pose.js';
import { detectMarkerQuad } from './marker_quad.js';
import { SlamCore } from './slam_core.js';

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

function getSlamPoseRecoveryAvailable(cv) {
  const hasEorF = typeof cv.findEssentialMat === 'function' || typeof cv.findFundamentalMat === 'function';
  const hasRecover = typeof cv.recoverPose === 'function';
  return hasEorF && hasRecover;
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

function drawQuad(ctx, corners, { color = 'rgba(255, 255, 0, 0.9)', width = 4 } = {}) {
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

function scaleModelToRoughMarkerSize(ar, object3d, markerMeters) {
  if (!ar || !object3d) return;
  const size = ar.computeBoundingSize(object3d);
  if (!size) return;
  const maxDim = Math.max(size.x, size.y, size.z);
  if (!isFinite(maxDim) || maxDim <= 1e-6) return;
  const s = markerMeters / maxDim;
  object3d.scale.setScalar(s);
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
  let cameraHandle = null;
  let ar = null;

  // Marker-based bootstrap state
  let appState = 'WAIT_FOR_MARKER';
  let markerLockFrames = 0;
  let worldLocked = false;
  let lastMarkerPose = null;
  let houseLoaded = false;
  let houseLoading = false;
  let lastMarkerLogMs = 0;

  const MARKER_SIZE_METERS = 0.1016; // 4 inches
  const MODEL_URL = 'model/Avocado2.glb';
  // 1.0 => roughly marker-sized (~10cm). If you want smaller/larger, tweak this.
  const MODEL_SCALE_FACTOR = 1.0;
  const MARKER_MIN_AREA_FRAC = 0.005;

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

      // If the camera has started, bring the cv canvas above/below Three.
      if (cameraHandle?.cvCanvas) {
        cameraHandle.cvCanvas.style.zIndex = debugEnabled ? '3' : '1';
        cameraHandle.cvCanvas.style.display = debugEnabled ? 'block' : 'none';
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

      // Renderer setup (Three.js)
      const threeCanvas = document.getElementById('threeCanvas');
      if (!threeCanvas) throw new Error('Missing #threeCanvas element');
      threeCanvas.style.display = 'block';

      ar = new ARRenderer(threeCanvas);
      // Camera background is provided by #cvCanvas (drawn each frame).
      // Three is rendered with a transparent clear color over it.

      // Keep Three sized to viewport
      const onResize = () => ar.resize();
      window.addEventListener('resize', onResize, { passive: true });

      // Pose init for solvePnP
      initPose(cameraHandle.videoEl, cv);

      // Match Three camera projection to the same simple intrinsics used by solvePnP.
      // pose.js currently uses focalLengthPx = videoWidthPx.
      ar.setProjectionFromVideo({
        videoWidthPx: cameraHandle.videoEl.videoWidth,
        videoHeightPx: cameraHandle.videoEl.videoHeight,
        focalLengthPx: cameraHandle.videoEl.videoWidth
      });

      slam = new SlamCore(cv);

      setStatus('Running');
      console.log('[Boot] Camera running');
      console.log('[Boot] OpenCV ready');
      logOpenCVInfo(cv);

      const slamPoseAvailable = getSlamPoseRecoveryAvailable(cv);
      if (!slamPoseAvailable) {
        console.warn('[SLAM] Pose recovery not available in this OpenCV.js build; marker bootstrap will still work.');
      }

      running = true;
      const { videoEl, cvCanvas, ctx } = cameraHandle;

      // Debug overlay canvas visibility stacking
      const applyDebugCanvasStacking = () => {
        if (debugEnabled) {
          cvCanvas.style.zIndex = '3';
          cvCanvas.style.display = 'block';
        } else {
          cvCanvas.style.zIndex = '1';
          cvCanvas.style.display = 'none';
        }
      };
      applyDebugCanvasStacking();

      const tick = () => {
        if (!running) return;

        // Draw current video frame
        ctx.drawImage(videoEl, 0, 0, cvCanvas.width, cvCanvas.height);

        // Read pixels once per frame
        const imageData = ctx.getImageData(0, 0, cvCanvas.width, cvCanvas.height);

        // Grayscale for marker detection
        const rgba = cv.matFromImageData(imageData);
        const gray = new cv.Mat();
        cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY);
        rgba.delete();

        let dbg = '';

        // Marker detection → solvePnP → camera pose
        // - Before lock: used to establish world origin and initial camera pose
        // - After lock: may be used to update camera pose ONLY if SLAM pose recovery isn't available
        const quad = detectMarkerQuad(cv, gray, { minAreaFrac: MARKER_MIN_AREA_FRAC });
        if (quad?.corners) {
          const s = MARKER_SIZE_METERS;
          const half = s * 0.5;
          const objectPoints = [
            { x: -half, y: -half, z: 0 },
            { x: half, y: -half, z: 0 },
            { x: half, y: half, z: 0 },
            { x: -half, y: half, z: 0 }
          ];

          const pose = estimatePose(quad.corners, objectPoints, cv);
          if (pose && ar) {
            lastMarkerPose = pose;

            // Update camera pose from marker if we're still bootstrapping,
            // or if SLAM pose recovery isn't available yet.
            if (!worldLocked || !getSlamPoseRecoveryAvailable(cv)) {
              ar.setCameraFromMarkerPose(pose);
            }

            if (!worldLocked) {
              markerLockFrames++;
              if (markerLockFrames >= 8) {
                worldLocked = true;
                appState = 'WORLD_LOCKED';
                setStatus('World locked');
                console.log('[Marker] World locked');
              } else {
                setStatus('Detecting marker…');
              }
            }
          }

          if (debugEnabled) drawQuad(ctx, quad.corners);
        } else {
          markerLockFrames = 0;
          if (!worldLocked) setStatus('Point at marker');
        }

        // Throttled marker debug log (helps diagnose “model never loads”)
        const nowMs = performance.now();
        if (nowMs - lastMarkerLogMs > 1200) {
          lastMarkerLogMs = nowMs;
          console.log('[Marker] quad:', quad ? 'yes' : 'no', {
            state: appState,
            worldLocked,
            markerLockFrames,
            minAreaFrac: MARKER_MIN_AREA_FRAC,
            area: quad?.area
          });
        }

        // Once world is locked, spawn the house at world origin (only once)
        if (worldLocked && ar && !houseLoaded && !houseLoading) {
          houseLoading = true;
          setStatus('Loading model…');
          console.log('[Model] Loading', MODEL_URL, {
            markerMeters: MARKER_SIZE_METERS,
            modelScaleFactor: MODEL_SCALE_FACTOR,
            targetMeters: MARKER_SIZE_METERS * MODEL_SCALE_FACTOR
          });
          ar.loadGLB(MODEL_URL)
            .then((gltf) => {
              // Normalize model transform so it's more likely to be visible:
              // - Center it at origin (some GLBs have far-away origins)
              // - Scale to marker size
              // - Place it on the marker plane (y=0)
              ar.centerObject(gltf.scene);
              scaleModelToRoughMarkerSize(ar, gltf.scene, MARKER_SIZE_METERS * MODEL_SCALE_FACTOR);
              ar.placeOnGround(gltf.scene, { y: 0 });

              const sizeAfter = ar.computeBoundingSize(gltf.scene);
              console.log('[Model] Bounds after normalize:', sizeAfter, {
                markerMeters: MARKER_SIZE_METERS,
                modelScaleFactor: MODEL_SCALE_FACTOR
              });

              // Add model to anchor (world origin)
              ar.anchor.add(gltf.scene);

              setStatus('Running');
              houseLoaded = true;
              houseLoading = false;
              console.log('[Model] Loaded and added to scene');
            })
            .catch((err) => {
              console.error('[Model] loadGLB failed:', err);
              setStatus('Model load failed (see console)');
              houseLoading = false;
              houseLoaded = false;
            });
        }

        // If marker is lost after lock and SLAM isn't ready, hold last pose.
        if (worldLocked && !getSlamPoseRecoveryAvailable(cv) && lastMarkerPose && ar) {
          // ar already has last pose; nothing to do.
          dbg += 'Tracking: marker-based only (SLAM pose recovery missing)\n';
        }

        // Optional SLAM edge dots debug (independent of marker)
        const slamResult = slam.processFrame(imageData, {
          detectEdges: debugEnabled,
          maxEdgePoints: 700
        });
        if (debugEnabled && slamResult?.edgePoints2D) {
          drawGreenDots(ctx, slamResult.edgePoints2D, { radius: 2 });
        }

        if (debugEnabled) {
          dbg += `State: ${appState}\n`;
          dbg += `World locked: ${worldLocked}\n`;
          dbg += `Marker lock frames: ${markerLockFrames}\n`;
          dbg += `SLAM pose recovery: ${getSlamPoseRecoveryAvailable(cv)}\n`;
          setDebugInfo(dbg);
        }

        gray.delete();

        // Render Three
        if (ar) ar.render();

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
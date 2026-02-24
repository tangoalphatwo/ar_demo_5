// main.js
// Start button -> start camera + wait for OpenCV -> run SLAM per frame

import { ARRenderer } from './renderer.js';
import { initPose, estimatePose } from './pose.js';
import { detectMarkerQuad } from './marker_quad.js';
import { SlamCore } from './slam_core.js';
import { createDebugUI, createThrottle, drawGreenDots, drawQuad, logOpenCVInfo } from './debug.js';
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


window.addEventListener('load', () => {
  const startBtn = document.getElementById('startBtn');
  const debugToggle = document.getElementById('debugToggle');
  const debugInfoEl = document.getElementById('debugInfo');
  if (!startBtn) {
    console.warn('Start button not found');
    return;
  }

  let running = false;
  let slam = null;
  let cameraHandle = null;
  let ar = null;

  const debugUI = createDebugUI({ debugToggleEl: debugToggle, debugInfoEl });
  const markerLogThrottle = createThrottle(1200);

  // Marker-based bootstrap state
  let appState = 'WAIT_FOR_MARKER';
  let markerLockFrames = 0;
  let worldLocked = false;
  let lastMarkerPose = null;

  const MARKER_SIZE_METERS = 0.1016; // 4 inches
  const MARKER_MIN_AREA_FRAC = 0.005;

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
      debugUI.attachCvCanvas(cameraHandle.cvCanvas);

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

      // Keep video/canvases aligned (contain-fit) and keep Three sized to that rect.
      const doLayout = () => {
        const rect = applyViewRect({ videoEl: cameraHandle.videoEl, threeCanvas, cvCanvas: cameraHandle.cvCanvas });
        ar.resize();
        return rect;
      };
      doLayout();

      const onResize = () => doLayout();
      window.addEventListener('resize', onResize, { passive: true });
      window.addEventListener('orientationchange', onResize, { passive: true });

      // Pose init for solvePnP
      initPose(cameraHandle.videoEl, cv);

      // Match Three camera projection to the same simple intrinsics used by solvePnP.
      // pose.js currently uses focalLengthPx = max(videoWidthPx, videoHeightPx).
      const focalLengthPx = Math.max(cameraHandle.videoEl.videoWidth, cameraHandle.videoEl.videoHeight);
      ar.setProjectionFromVideo({
        videoWidthPx: cameraHandle.videoEl.videoWidth,
        videoHeightPx: cameraHandle.videoEl.videoHeight,
        focalLengthPx
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

      const tick = () => {
        if (!running) return;

        const debugEnabled = debugUI.isEnabled();

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
            // World/marker coords: X right, Y up, Z out of the marker plane.
            { x: -half, y: half, z: 0 },
            { x: half, y: half, z: 0 },
            { x: half, y: -half, z: 0 },
            { x: -half, y: -half, z: 0 }
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
        if (markerLogThrottle()) {
          console.log('[Marker] quad:', quad ? 'yes' : 'no', {
            state: appState,
            worldLocked,
            markerLockFrames,
            minAreaFrac: MARKER_MIN_AREA_FRAC,
            area: quad?.area
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
          debugUI.setDebugText(dbg);
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
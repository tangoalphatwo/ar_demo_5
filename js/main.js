// main.js
import { CameraManager } from './camera.js';
import { ARRenderer } from './renderer.js';
import { SlamCore } from './slam_core.js';
import { initPose, estimatePose } from "./pose.js";
import { MarkerTracker } from './marker_tracker.js';

window.addEventListener('load', () => {
  const videoEl = document.getElementById('camera');
  const cvCanvas = document.getElementById('cvCanvas');
  const threeCanvas = document.getElementById('threeCanvas');
  const startBtn = document.getElementById('startBtn');
  const statusEl = document.getElementById('status');
  const debugToggle = document.getElementById('debugToggle');
  const debugInfo = document.getElementById('debugInfo');

  const camera = new CameraManager(videoEl, cvCanvas);
  const renderer = new ARRenderer(threeCanvas);

  let debugFeaturePoints = [];
  let showDebug = false;
  let latestPose = null;

  // Wire up debug toggle UI

  if (debugToggle) {
    debugToggle.addEventListener('click', () => {
      showDebug = !showDebug;
      debugToggle.textContent = showDebug ? 'Hide Debug' : 'Show Debug';
      debugToggle.setAttribute('aria-pressed', String(showDebug));
      if (!debugInfo) return;
      debugInfo.hidden = !showDebug;
      debugInfo.setAttribute('aria-hidden', String(!showDebug));
    });
  }

  // Toast helper
  const toastEl = document.getElementById('toast');
  let toastTimer = null;
  function showToast(msg, duration = 1500) {
    if (!toastEl) return;
    toastEl.textContent = msg;
    toastEl.hidden = false;
    toastEl.setAttribute('aria-hidden', 'false');
    toastEl.classList.add('show');
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      toastEl.classList.remove('show');
      toastEl.setAttribute('aria-hidden', 'true');
      toastTimer = setTimeout(() => { toastEl.hidden = true; toastTimer = null; }, 180);
    }, duration);
  }

  let running = false;
  let slam = null;

  let markerTracker = null;
  let avocadoLoaded = false;

  // Throttled logging to avoid spamming the console
  let frameIndex = 0;
  let lastHadMarker = false;
  function logEvery(n, ...args) {
    if (frameIndex % n === 0) console.log(...args);
  }

  // Marker corner tracking for stability
  let markerTracking = false;
  let markerPrevGray = null;
  let markerPrevPts = null; // cv.Mat 4x1 CV_32FC2
  let lastStableMarkerPose = null;

  // SLAM scale estimation using marker PnP while marker is visible
  let slamMetricScale = 0.0;
  let lastMarkerPoseForScale = null;
  let lastSlamDeltaForScale = null;

  function mat3MulVec3(r, v) {
    return {
      x: r[0] * v.x + r[1] * v.y + r[2] * v.z,
      y: r[3] * v.x + r[4] * v.y + r[5] * v.z,
      z: r[6] * v.x + r[7] * v.y + r[8] * v.z
    };
  }

  function mat3MulMat3(a, b) {
    return [
      a[0] * b[0] + a[1] * b[3] + a[2] * b[6],
      a[0] * b[1] + a[1] * b[4] + a[2] * b[7],
      a[0] * b[2] + a[1] * b[5] + a[2] * b[8],

      a[3] * b[0] + a[4] * b[3] + a[5] * b[6],
      a[3] * b[1] + a[4] * b[4] + a[5] * b[7],
      a[3] * b[2] + a[4] * b[5] + a[5] * b[8],

      a[6] * b[0] + a[7] * b[3] + a[8] * b[6],
      a[6] * b[1] + a[7] * b[4] + a[8] * b[7],
      a[6] * b[2] + a[7] * b[5] + a[8] * b[8]
    ];
  }

  function poseJumpTooLarge(prev, next) {
    if (!prev || !next) return false;
    const dx = next.position.x - prev.position.x;
    const dy = next.position.y - prev.position.y;
    const dz = next.position.z - prev.position.z;
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
    // reject very large sudden jumps (usually due to a bad homography / corner mixup)
    return dist > 0.25;
  }

  let cvReady = false;
  let cvInstance = null;

  let prevGray = null;
  let prevPoints = null;

  // Feature tracking / reseeding (for debug dots + pose heuristics)
  const MAX_FEATURES = 300;
  const MIN_FEATURES = 80; // when tracked points drop below this, detect more
  const FEATURE_QUALITY = 0.01;
  const FEATURE_MIN_DIST = 10;
  const FEATURE_EXCLUSION_RADIUS = 12; // pixels; prevents re-detecting on top of existing points

  // Track if loadedmetadata fired before OpenCV became available
  let videoMetadataPending = false;
  videoEl.addEventListener('loadedmetadata', () => {
    if (cvInstance) {
      try {
        initPose(videoEl, cvInstance);
      } catch (e) {
        console.warn('initPose failed from loadedmetadata:', e);
      }
    } else {
      videoMetadataPending = true;
    }
  });
  
  async function initOpenCV() {
    console.log("Initializing OpenCV");
    try {
      console.log('[InitOpenCV] __opencvReady:', window.__opencvReady);
      console.log('[InitOpenCV] window.cv type:', typeof window.cv);
      console.log('[InitOpenCV] window.cv keys sample:', window.cv && typeof window.cv === 'object' ? Object.keys(window.cv).slice(0, 10) : null);
    } catch (e) {
      console.warn('[InitOpenCV] preflight log failed:', e);
    }

    function makeNonThenable(cvObj) {
      if (!cvObj) return cvObj;
      try {
        // Some OpenCV.js builds expose a non-enumerable `then` function.
        // Returning a thenable from an async function causes the outer Promise
        // to adopt it, which can hang forever.
        if (typeof cvObj.then === 'function') {
          console.warn('[InitOpenCV] cv is thenable; wrapping to avoid await hang');
          return new Proxy(cvObj, {
            get(target, prop, receiver) {
              if (prop === 'then') return undefined;
              return Reflect.get(target, prop, receiver);
            }
          });
        }
      } catch (e) {
        console.warn('[InitOpenCV] thenable check failed:', e);
      }
      return cvObj;
    }

    // cv is a Promise in modularized builds
    if (window.cv instanceof Promise) {
      const cvInstance = await window.cv;
      console.log("OpenCV ready (awaited Promise)");
      return makeNonThenable(cvInstance);
    }
  
    // Fallback (non-modularized build)
    if (window.cv && window.cv.Mat) {
      console.log("OpenCV ready (non-modularized)");
      const cvInstance = makeNonThenable(window.cv);
      // Touch a few fields to ensure the object is usable
      try {
        console.log('[InitOpenCV] Mat exists:', !!cvInstance.Mat);
        console.log('[InitOpenCV] version:', cvInstance.version || '(no version field)');
      } catch (e) {
        console.warn('[InitOpenCV] post-ready probe failed:', e);
      }
      return cvInstance;
    }
  
    throw new Error("OpenCV not found");
  }

  startBtn.addEventListener("click", async () => {
    if (running) return; // prevent double-starts
    try {
      // Button UX: show state immediately
      if (startBtn) {
        startBtn.disabled = true;
        startBtn.textContent = 'Starting…';
      }

      statusEl.textContent = "Starting camera...";
      console.log('[Init] Starting camera');
      await camera.start();

      await videoEl.play();
      console.log('[Init] Camera playing', { w: videoEl.videoWidth, h: videoEl.videoHeight });

    const videoW = videoEl.videoWidth;
    const videoH = videoEl.videoHeight;

    // Fit the canvas to the viewport while preserving the video's aspect ratio
    const maxW = window.innerWidth;
    const maxH = window.innerHeight;
    const scale = Math.min(maxW / videoW, maxH / videoH, 1); // don't upscale beyond native
    const displayW = Math.round(videoW * scale);
    const displayH = Math.round(videoH * scale);

    const dpr = window.devicePixelRatio || 1;

    // Backing buffer uses device pixels; CSS size uses CSS pixels
    cvCanvas.width = Math.round(displayW * dpr);
    cvCanvas.height = Math.round(displayH * dpr);
    cvCanvas.style.width = `${displayW}px`;
    cvCanvas.style.height = `${displayH}px`;

    // Ensure 2D context maps CSS pixels correctly onto the backing buffer
    const cvCtx = cvCanvas.getContext('2d');
    cvCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

      statusEl.textContent = "Initializing OpenCV...";
      console.log('[Init] Before initOpenCV await');

      // Watchdog: if init stalls, at least we get a log message.
      const watchdog = setTimeout(() => {
        console.warn('[Init] initOpenCV appears stalled > 5s');
        try { statusEl.textContent = 'OpenCV init stalled (see console)'; } catch {}
      }, 5000);

      cvInstance = await initOpenCV();
      clearTimeout(watchdog);

      console.log('[Init] OpenCV instance acquired');

      // Let the UI paint before continuing into heavier initialization.
      await new Promise(r => setTimeout(r, 0));

      // Log capabilities on the next tick to avoid blocking startup.
      setTimeout(() => {
        try {
          console.log('[OpenCV] mode:', 'hosted');
          console.log('[OpenCV] capabilities:', {
            findEssentialMat: typeof cvInstance.findEssentialMat,
            findFundamentalMat: typeof cvInstance.findFundamentalMat,
            recoverPose: typeof cvInstance.recoverPose
          });
        } catch (e) {
          console.warn('[OpenCV] capability logging failed:', e);
        }
      }, 0);

      statusEl.textContent = "Initializing pose...";
      // If loadedmetadata happened earlier, initialize pose now; otherwise initialize immediately
      try {
        if (videoMetadataPending || videoEl.readyState >= 1) {
          initPose(videoEl, cvInstance);
          console.log('[Init] Pose intrinsics initialized');
        }
      } catch (e) {
        console.warn('initPose failed (OpenCV may not be ready):', e);
      }

      statusEl.textContent = "Checking SLAM...";
      // Only enable SLAM if this OpenCV build has the required epipolar + pose recovery methods.
      const canSlam = !!(cvInstance.recoverPose && (cvInstance.findEssentialMat || cvInstance.findFundamentalMat));
      if (canSlam) {
        slam = new SlamCore(cvInstance);
        console.log('[Init] SLAM enabled');
      } else {
        slam = null;
        console.warn('SLAM disabled: OpenCV build missing findEssentialMat/findFundamentalMat/recoverPose.');
      }

      statusEl.textContent = "Loading marker template...";
      // Initialize marker tracker + load template
      try {
        markerTracker = new MarkerTracker(cvInstance);
        console.log('[Init] Loading marker template');
        await markerTracker.loadTemplate('marker/WorldZeroMarker.png');
        console.log('[Init] Marker template loaded');
        showToast('Marker template loaded');
      } catch (e) {
        console.warn('MarkerTracker init failed:', e);
        showToast('Marker tracker unavailable');
      }

      statusEl.textContent = "Loading model...";
      // Load Avocado model (once)
      try {
        console.log('[Init] Loading model');
        const gltf = await renderer.loadGLB('model/Avocado2.glb');
        renderer.model = gltf.scene;

      const nativeSize = renderer.computeBoundingSize(renderer.model);
      if (nativeSize) console.log('Avocado native bbox (scene units):', nativeSize);

      renderer.model.position.set(0, 0, 0.02); // 2cm above marker plane
      renderer.model.scale.setScalar(0.001); // model is large; scale down to fit marker

      const scaledSize = renderer.computeBoundingSize(renderer.model);
      if (scaledSize) console.log('Avocado scaled bbox (scene units):', scaledSize);

        renderer.anchor.add(renderer.model);
        avocadoLoaded = true;
        console.log('[Init] Model loaded + attached');
        showToast('Avocado loaded');
      } catch (e) {
        console.warn('Failed to load Avocado2.glb:', e);
        showToast('Failed to load model');
      }

      statusEl.textContent = "Running";
      running = true;
      loop(); // START THE FRAME LOOP

      // Reveal the debug controls only after entering the AR experience
      if (debugToggle) {
        debugToggle.hidden = false;
        debugToggle.setAttribute('aria-pressed', 'false');
        debugToggle.textContent = 'Show Debug';
      }
      if (debugInfo) {
        debugInfo.hidden = true;
        debugInfo.setAttribute('aria-hidden', 'true');
      }

      // Hide the Start button after AR begins to avoid accidental re-starts
      if (startBtn) {
        startBtn.hidden = true;
      }
    } catch (e) {
      console.error('[Init] Fatal startup error:', e);
      statusEl.textContent = 'Error starting AR (see console)';
      showToast('Error starting AR');
      running = false;

      if (startBtn) {
        startBtn.disabled = false;
        startBtn.textContent = 'Start AR';
        startBtn.hidden = false;
      }
    }
  });

  function drawFeatures(points) {
    if (!points || points.length === 0) return;

    const ctx = cvCanvas.getContext("2d");

    ctx.save();
    ctx.fillStyle = "lime";

    for (let i = 0; i < points.length; i++) {
      const p = points[i];

      // For now we fake projection: screen-space debug
      const x = (p.x !== undefined) ? p.x : null;
      const y = (p.y !== undefined) ? p.y : null;

      if (x === null || y === null) continue;

      ctx.beginPath();
      ctx.arc(x, y, 2, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.restore();
  }

  function drawVideoPreserveAspect(ctx, video, canvas) {
    const videoAspect = video.videoWidth / video.videoHeight;
    // Use CSS pixels for layout calculations (clientWidth/clientHeight)
    const canvasCssW = canvas.clientWidth;
    const canvasCssH = canvas.clientHeight;
    const canvasAspect = canvasCssW / canvasCssH;

    let drawWidth, drawHeight, offsetX, offsetY;

    if (canvasAspect > videoAspect) {
      // Canvas is wider than video → pillarbox
      drawHeight = canvasCssH;
      drawWidth = drawHeight * videoAspect;
      offsetX = (canvasCssW - drawWidth) / 2;
      offsetY = 0;
    } else {
      // Canvas is taller than video → letterbox
      drawWidth = canvasCssW;
      drawHeight = drawWidth / videoAspect;
      offsetX = 0;
      offsetY = (canvasCssH - drawHeight) / 2;
    }

    // Clear the full backing buffer
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // drawImage coordinates are in CSS pixels because the context transform maps them
    ctx.drawImage(video, offsetX, offsetY, drawWidth, drawHeight);

    return { offsetX, offsetY, drawWidth, drawHeight };
  }

  // Return four image points (in processing coords) to try as correspondences.
  // This is a heuristic placeholder: picks 4 points nearest to extreme corners
  // of the current detected feature set. Replace with marker detection for
  // robust pose estimation.
  function getDetectedPoints() {
    const pts = debugFeaturePoints;
    if (!pts || pts.length < 4) return null;

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const p of pts) {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }

    function closest(targetX, targetY) {
      let best = null;
      let bd = Infinity;
      for (const p of pts) {
        const dx = p.x - targetX;
        const dy = p.y - targetY;
        const d = dx * dx + dy * dy;
        if (d < bd) {
          bd = d;
          best = p;
        }
      }
      return best;
    }

    const tl = closest(minX, minY);
    const tr = closest(maxX, minY);
    const br = closest(maxX, maxY);
    const bl = closest(minX, maxY);

    if (!tl || !tr || !br || !bl) return null;

    return [ { x: tl.x, y: tl.y }, { x: tr.x, y: tr.y }, { x: br.x, y: br.y }, { x: bl.x, y: bl.y } ];
  }


  function loop() {
    if (!running || !cvInstance) return;

    frameIndex++;

    const cv = cvInstance;
    const grabbed = camera.grabFrame();
    if (!grabbed || !grabbed.imageData) {
      requestAnimationFrame(loop);
      return;
    }
    const frame = grabbed.imageData;
    const drawRectForFrame = grabbed.drawRect;
    const frameDpr = grabbed.dpr || 1;

    function procPointToVideo(p) {
      // p is in backing-buffer pixels (ImageData space). drawRect is in CSS pixels.
      if (!drawRectForFrame) return { x: p.x, y: p.y };

      const xCss = p.x / frameDpr;
      const yCss = p.y / frameDpr;

      const xInVideoCss = xCss - drawRectForFrame.offsetX;
      const yInVideoCss = yCss - drawRectForFrame.offsetY;

      const u = xInVideoCss / drawRectForFrame.drawWidth;
      const v = yInVideoCss / drawRectForFrame.drawHeight;

      // Clamp to video bounds to prevent occasional out-of-range points destabilizing PnP
      const uc = Math.min(1, Math.max(0, u));
      const vc = Math.min(1, Math.max(0, v));

      return {
        x: uc * videoEl.videoWidth,
        y: vc * videoEl.videoHeight
      };
    }

    // Feed the raw frame to the SLAM core
    let slamDelta = null;
    try {
      if (slam) {
        const res = slam.processFrame(frame);
        slamDelta = res?.delta || null;
      }
    } catch (err) {
      // SLAM is optional; keep AR running even if it fails
      console.warn('SLAM processing error:', err);
    }

    const rgba = cv.matFromImageData(frame);
    const gray = new cv.Mat();
    cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY);

    // --- Marker-based pose (WorldZeroMarker.png, 4" x 4") ---
    try {
      if (markerTracker) {
        let cornersProc = null; // in processing canvas/backing coords

        if (markerTracking && markerPrevGray && markerPrevPts && markerPrevPts.rows === 4) {
          const currPts = new cv.Mat();
          const status = new cv.Mat();
          const err = new cv.Mat();
          cv.calcOpticalFlowPyrLK(markerPrevGray, gray, markerPrevPts, currPts, status, err);

          let ok = status.rows === 4;
          if (ok) {
            for (let i = 0; i < 4; i++) {
              if (status.data[i] !== 1) { ok = false; break; }
            }
          }

          if (ok) {
            cornersProc = [];
            for (let i = 0; i < 4; i++) {
              cornersProc.push({ x: currPts.data32F[i * 2], y: currPts.data32F[i * 2 + 1] });
            }

            markerPrevGray.delete();
            markerPrevGray = gray.clone();
            markerPrevPts.delete();
            markerPrevPts = currPts;
          } else {
            currPts.delete();
            markerTracking = false;
            markerPrevGray.delete();
            markerPrevGray = null;
            markerPrevPts.delete();
            markerPrevPts = null;
          }

          status.delete();
          err.delete();
        }

        // If not currently tracking, (re)detect the marker using ORB+homography.
        if (!cornersProc) {
          const det = markerTracker.detect(gray);
          if (det && det.corners && det.corners.length === 4) {
            cornersProc = det.corners;

            // Seed tracking state
            markerPrevGray?.delete?.();
            markerPrevGray = gray.clone();
            markerPrevPts?.delete?.();
            markerPrevPts = cv.matFromArray(4, 1, cv.CV_32FC2, [
              cornersProc[0].x, cornersProc[0].y,
              cornersProc[1].x, cornersProc[1].y,
              cornersProc[2].x, cornersProc[2].y,
              cornersProc[3].x, cornersProc[3].y
            ]);
            markerTracking = true;

            det.homography?.delete?.();
          }
        }

        if (cornersProc) {
          // Map from processing canvas coords (letterboxed) to raw video coords (initPose uses video dims)
          const imagePtsScaled = cornersProc.map(procPointToVideo);

          const half = 0.1016 / 2; // 4 inches in meters
          const objectPoints = [
            { x: -half, y: -half, z: 0 },
            { x:  half, y: -half, z: 0 },
            { x:  half, y:  half, z: 0 },
            { x: -half, y:  half, z: 0 }
          ];

          const pose = estimatePose(imagePtsScaled, objectPoints, cvInstance);
          if (pose && !poseJumpTooLarge(lastStableMarkerPose, pose)) {
            lastStableMarkerPose = pose;
            latestPose = pose;
            renderer.setAnchorPose(pose);
            logEvery(30, '[Marker] pose ok', pose.position);

            // Learn SLAM metric scale when both marker pose and SLAM delta are available
            if (slamDelta && slamDelta.R && slamDelta.t && lastMarkerPoseForScale) {
              const dx = pose.position.x - lastMarkerPoseForScale.position.x;
              const dy = pose.position.y - lastMarkerPoseForScale.position.y;
              const dz = pose.position.z - lastMarkerPoseForScale.position.z;
              const dMarker = Math.sqrt(dx * dx + dy * dy + dz * dz);

              const dt = slamDelta.t;
              const dSlam = Math.sqrt(dt[0] * dt[0] + dt[1] * dt[1] + dt[2] * dt[2]);
              if (dMarker > 1e-4 && dSlam > 1e-6) {
                const s = dMarker / dSlam;
                // Exponential moving average
                slamMetricScale = slamMetricScale > 0 ? (0.9 * slamMetricScale + 0.1 * s) : s;
                logEvery(60, '[SLAM] metric scale', slamMetricScale.toFixed(4));
              }
            }

            lastMarkerPoseForScale = pose;
            lastSlamDeltaForScale = slamDelta;
          } else if (lastStableMarkerPose) {
            renderer.setAnchorPose(lastStableMarkerPose);
            if (pose) logEvery(30, '[Marker] pose rejected (jump too large)', { prev: lastStableMarkerPose.position, next: pose.position });
          } else {
            renderer.setAnchorPose(null);
            if (pose === null) logEvery(30, '[Marker] pose null (solvePnP failed)');
          }

          if (!lastHadMarker) console.log('[Marker] acquired');
          lastHadMarker = true;
        } else {
          // no marker in this frame
          // Keep last known pose so content persists after marker loss.
          if (lastStableMarkerPose && slamDelta && slamDelta.R && slamDelta.t) {
            const dR = slamDelta.R;
            const dt = slamDelta.t;

            // Convert SLAM delta translation into the same coordinate convention as pose.js
            // pose.js flips Y, so do the same here.
            const dtPose = { x: dt[0], y: -dt[1], z: dt[2] };

            const scale = slamMetricScale > 0 ? slamMetricScale : 0.0; // if unknown, ignore translation
            const dtScaled = { x: dtPose.x * scale, y: dtPose.y * scale, z: dtPose.z * scale };

            // Propagate marker pose in camera coords: X2 = dR * X1 + dt
            const newPos = mat3MulVec3(dR, lastStableMarkerPose.position);
            newPos.x += dtScaled.x;
            newPos.y += dtScaled.y;
            newPos.z += dtScaled.z;

            const newRot = lastStableMarkerPose.rotationMatrix
              ? mat3MulMat3(dR, lastStableMarkerPose.rotationMatrix)
              : null;

            lastStableMarkerPose = {
              ...lastStableMarkerPose,
              position: newPos,
              rotationMatrix: newRot || lastStableMarkerPose.rotationMatrix
            };
          }

          renderer.setAnchorPose(lastStableMarkerPose);
          if (lastHadMarker) console.log('[Marker] lost');
          lastHadMarker = false;
        }
      }
    } catch (e) {
      console.warn('Marker pose error:', e);
    }

    // Occasional pose debug
    if (showDebug && latestPose && debugInfo) {
      const p = latestPose.position;
      debugInfo.hidden = false;
      debugInfo.setAttribute('aria-hidden', 'false');
      debugInfo.textContent = `${debugInfo.textContent}\npose(m): x=${p.x.toFixed(3)} y=${p.y.toFixed(3)} z=${p.z.toFixed(3)}`;
    }

    if (!prevGray || !prevPoints || prevPoints.rows === 0) {
      // FIRST FRAME (or fully lost): detect features
      prevGray = gray.clone();
      prevPoints = new cv.Mat();
      cv.goodFeaturesToTrack(prevGray, prevPoints, MAX_FEATURES, FEATURE_QUALITY, FEATURE_MIN_DIST);

      debugFeaturePoints = [];
      for (let i = 0; i < prevPoints.rows; i++) {
        debugFeaturePoints.push({
          x: prevPoints.data32F[i * 2],
          y: prevPoints.data32F[i * 2 + 1]
        });
      }
    } else {
      // TRACK features
      const currPoints = new cv.Mat();
      const status = new cv.Mat();
      const err = new cv.Mat();

      cv.calcOpticalFlowPyrLK(prevGray, gray, prevPoints, currPoints, status, err);

      // Keep only successful tracks (status == 1)
      const goodCurrFloats = [];
      for (let i = 0; i < status.rows; i++) {
        if (status.data[i] === 1) {
          goodCurrFloats.push(
            currPoints.data32F[i * 2],
            currPoints.data32F[i * 2 + 1]
          );
        }
      }

      let mergedFloats = goodCurrFloats;
      const goodCount = goodCurrFloats.length / 2;

      // If we’re running low, detect more points on the current frame and merge them in.
      if (goodCount < MIN_FEATURES) {
        const want = Math.max(0, MAX_FEATURES - goodCount);
        if (want > 0) {
          const mask = new cv.Mat(gray.rows, gray.cols, cv.CV_8UC1, new cv.Scalar(255));
          // Exclude areas around existing points so we add *new* features.
          for (let i = 0; i < goodCount; i++) {
            const x = goodCurrFloats[i * 2];
            const y = goodCurrFloats[i * 2 + 1];
            cv.circle(mask, new cv.Point(x, y), FEATURE_EXCLUSION_RADIUS, new cv.Scalar(0), -1);
          }

          const extra = new cv.Mat();
          // mask parameter is optional; OpenCV.js supports it for goodFeaturesToTrack.
          cv.goodFeaturesToTrack(gray, extra, want, FEATURE_QUALITY, FEATURE_MIN_DIST, mask);

          if (extra.rows > 0) {
            mergedFloats = goodCurrFloats.slice();
            for (let i = 0; i < extra.rows; i++) {
              mergedFloats.push(extra.data32F[i * 2], extra.data32F[i * 2 + 1]);
            }
          }

          extra.delete();
          mask.delete();
        }
      }

      debugFeaturePoints = [];
      for (let i = 0; i < mergedFloats.length; i += 2) {
        debugFeaturePoints.push({ x: mergedFloats[i], y: mergedFloats[i + 1] });
      }

      // Update tracking state to only the surviving + newly detected points
      const nextPoints = mergedFloats.length
        ? cv.matFromArray(mergedFloats.length / 2, 1, cv.CV_32FC2, mergedFloats)
        : new cv.Mat(0, 1, cv.CV_32FC2);

      prevGray.delete();
      prevPoints.delete();
      prevGray = gray.clone();
      prevPoints = nextPoints;

      currPoints.delete();
      status.delete();
      err.delete();
    }

    // Try pose estimation when we have a viable 4-point set
    try {
      // Keep the previous feature-based heuristic only as a fallback if the marker tracker is not active.
      if (!markerTracker) {
        const rawImagePts = getDetectedPoints(); // in processing canvas coords
        if (!rawImagePts || !cvInstance) return;
        // scale to the video coordinate space used by initPose (video.videoWidth/height)
        const sx = videoEl.videoWidth / camera.cvCanvas.width;
        const sy = videoEl.videoHeight / camera.cvCanvas.height;
        const imagePtsScaled = rawImagePts.map(p => ({ x: p.x * sx, y: p.y * sy }));

        const objectPoints = [
          { x: -0.05, y: -0.05, z: 0 },
          { x:  0.05, y: -0.05, z: 0 },
          { x:  0.05, y:  0.05, z: 0 },
          { x: -0.05, y:  0.05, z: 0 }
        ];

        const pose = estimatePose(imagePtsScaled, objectPoints, cvInstance);
        if (pose) {
          // pose is now a smoothed poseState { position: {x,y,z}, rotation: {yaw,pitch,roll} }
          latestPose = pose;
          console.log('Position:', pose.position);
          console.log('Yaw:', pose.rotation.yaw);
        } else {
          latestPose = null;
        }
      }
    } catch (e) {
      console.warn('Pose estimation error:', e);
    }

    rgba.delete();
    gray.delete();

    // DRAW
    const ctx = cvCanvas.getContext("2d");
    const drawRect = drawVideoPreserveAspect(ctx, videoEl, cvCanvas);

    if (showDebug) {
      // draw bounding rect for the video content (letterbox/pillarbox area)
      ctx.save();
      ctx.strokeStyle = 'red';
      ctx.lineWidth = 2;
      ctx.strokeRect(drawRect.offsetX + 0.5, drawRect.offsetY + 0.5, drawRect.drawWidth, drawRect.drawHeight);
      ctx.restore();

      // draw mapped feature points into screen-space
      ctx.fillStyle = "lime";

      const srcW = camera.cvCanvas.width;
      const srcH = camera.cvCanvas.height;

      for (const p of debugFeaturePoints) {
        const x = drawRect.offsetX + (p.x / srcW) * drawRect.drawWidth;
        const y = drawRect.offsetY + (p.y / srcH) * drawRect.drawHeight;

        ctx.beginPath();
        ctx.arc(x, y, 2, 0, Math.PI * 2);
        ctx.fill();
      }

      if (debugInfo) {
        debugInfo.hidden = false;
        debugInfo.setAttribute('aria-hidden', 'false');
        debugInfo.textContent = `drawRect:\noffsetX: ${drawRect.offsetX.toFixed(1)}\noffsetY: ${drawRect.offsetY.toFixed(1)}\ndrawW: ${drawRect.drawWidth.toFixed(1)}\ndrawH: ${drawRect.drawHeight.toFixed(1)}\nfeatures: ${debugFeaturePoints.length}`;
      }
    } else {
      if (debugInfo) {
        debugInfo.hidden = true;
        debugInfo.setAttribute('aria-hidden', 'true');
      }
    }

    // Render Three.js overlay
    try {
      renderer.render();
    } catch (e) {
      // ignore transient GL init errors
    }

    requestAnimationFrame(loop); // KEEP GOING
  }

  window.addEventListener('resize', () => renderer.resize());
});

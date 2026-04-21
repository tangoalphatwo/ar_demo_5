// main.js
// Working entrypoint (moved back from main_fixed.js)

import { CameraManager } from './camera.js';
import { createToast } from './ui.js';

const ENABLE_OPENCV_DIAGNOSTICS = window.__opencvDiagnostics === true;
// Step-wise refactor flag: when true, use camera-tracked runtime path.
// (Marker seeds camera pose; SLAM applies deltas to the camera.)
const USE_CAMERA_TRACKED_RUNTIME = true;

window.addEventListener('load', () => {
  const videoEl = document.getElementById('camera');
  const cvCanvas = document.getElementById('cvCanvas');
  const threeCanvas = document.getElementById('threeCanvas');
  const startBtn = document.getElementById('startBtn');
  const statusEl = document.getElementById('status');
  const debugToggle = document.getElementById('debugToggle');
  const setWorldZeroBtn = document.getElementById('setWorldZeroBtn');
  const debugInfo = document.getElementById('debugInfo');

  const camera = new CameraManager(videoEl, cvCanvas);

  let renderer = null;

  let initPoseFn = null;
  let estimatePoseFn = null;
  let getPoseIntrinsicsFn = null;
  let SlamCoreClass = null;
  let MarkerTrackerClass = null;

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
  const showToast = createToast(toastEl);

  if (setWorldZeroBtn) {
    setWorldZeroBtn.addEventListener('click', () => {
      try {
        if (!renderer) {
          showToast('Start AR first');
          return;
        }
        if (!renderer?.worldZeroRoot?.visible) {
          showToast('Show marker first');
          return;
        }
        renderer.setWorldZero?.();
        showToast('World zero set');
      } catch (e) {
        console.warn('[WorldZero] set failed:', e);
        showToast('Failed to set world zero');
      }
    });
  }

  let running = false;
  let slam = null;
  let markerTracker = null;

  // Throttled logging to avoid spamming the console
  let frameIndex = 0;
  let lastHadMarker = false;
  let framesSinceMarker = 0;
  let loggedReacquireRejection = false;
  function logEvery(n, ...args) {
    if (frameIndex % n === 0) console.log(...args);
  }

  // Marker corner tracking for stability
  let markerTracking = false;
  let markerPrevGray = null;
  let markerPrevPts = null; // cv.Mat 4x1 CV_32FC2
  let markerCurrPts = null; // cv.Mat 4x1 CV_32FC2 (reused LK output)
  let markerLkStatus = null;
  let markerLkErr = null;
  let lastStableMarkerPose = null;

  // ORB marker (re)detect is expensive; throttle it when the marker is lost.
  const MARKER_DETECT_EVERY_N_FRAMES = 3;
  let markerDetectCountdown = 0;

  // SLAM scale estimation using marker PnP while marker is visible
  let slamMetricScale = 0.0;
  let lastMarkerPoseForScale = null;

  function poseJumpTooLarge(prev, next) {
    if (!prev || !next) return false;
    const dx = next.position.x - prev.position.x;
    const dy = next.position.y - prev.position.y;
    const dz = next.position.z - prev.position.z;
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
    // reject very large sudden jumps (usually due to a bad homography / corner mixup)
    return dist > 0.25;
  }

  function orderQuadIndicesTLTRBRBL(pts) {
    if (!pts || pts.length !== 4) return null;
    const sums = pts.map(p => p.x + p.y);
    const diffs = pts.map(p => p.x - p.y);
    const tl = sums.indexOf(Math.min(...sums));
    const br = sums.indexOf(Math.max(...sums));
    const tr = diffs.indexOf(Math.max(...diffs));
    const bl = diffs.indexOf(Math.min(...diffs));
    return [tl, tr, br, bl];
  }

  function poseLooksPlausible(pose) {
    if (!pose || !pose.position || !pose.rotationMatrix) return false;
    const { x, y, z } = pose.position;
    if (![x, y, z].every(Number.isFinite)) return false;
    // In this demo the marker is typically within ~5cm..2m.
    if (z < 0.03 || z > 2.5) return false;
    if (Math.abs(x) > 2.5 || Math.abs(y) > 2.5) return false;
    return true;
  }

  let cvInstance = null;

  // Reuse Mats for frame conversion to avoid per-frame allocations.
  let frameRgbaMat = null;
  let frameGrayMat = null;

  let prevGray = null;
  let prevPoints = null;

  // Reused LK output mats for debug/feature tracking
  let featureCurrPoints = null;
  let featureLkStatus = null;
  let featureLkErr = null;

  // Feature tracking / reseeding (for debug dots + pose heuristics)
  const MAX_FEATURES = 300;
  const MIN_FEATURES = 80; // when tracked points drop below this, detect more
  const FEATURE_QUALITY = 0.01;
  const FEATURE_MIN_DIST = 10;
  const FEATURE_EXCLUSION_RADIUS = 12; // pixels; prevents re-detecting on top of existing points

  // Track if loadedmetadata fired before OpenCV became available
  let videoMetadataPending = false;
  videoEl.addEventListener('loadedmetadata', () => {
    if (cvInstance && initPoseFn) {
      try {
        initPoseFn(videoEl, cvInstance);
      } catch (e) {
        console.warn('initPose failed from loadedmetadata:', e);
      }
    } else {
      videoMetadataPending = true;
    }
  });

  async function initOpenCV() {
    if (ENABLE_OPENCV_DIAGNOSTICS) console.log('Initializing OpenCV');

    // Log which OpenCV artifacts the page is actually serving (useful for GitHub Pages caching).
    // This checks HTTP headers (ETag/Last-Modified/Content-Length) without downloading full files.
    if (ENABLE_OPENCV_DIAGNOSTICS) {
      try {
        const assetTag = window.__opencvAssetTag ? String(window.__opencvAssetTag) : '';
        const suffix = assetTag ? `?${encodeURIComponent(assetTag)}` : '';
        const jsUrl = `opencv/opencv.js${suffix}`;
        const wasmUrl = (window.Module && typeof window.Module.locateFile === 'function')
          ? window.Module.locateFile('opencv_js.wasm')
          : `opencv/opencv_js.wasm${suffix}`;

        const head = async (url) => {
          const res = await fetch(url, { method: 'HEAD', cache: 'no-store' });
          const h = res.headers;
          console.log('[OpenCV][HEAD]', url, {
            ok: res.ok,
            status: res.status,
            etag: h.get('etag'),
            lastModified: h.get('last-modified'),
            contentLength: h.get('content-length'),
            contentType: h.get('content-type')
          });
        };

        // Fire and forget: diagnostics only.
        head(jsUrl);
        head(wasmUrl);
      } catch (e) {
        console.warn('[OpenCV] asset HEAD diagnostics failed:', e);
      }
    }

    // If OpenCV is still loading, wait briefly.
    if (!window.__opencvReady && !window.cv) {
      const start = performance.now();
      const timeoutMs = 8000;
      while (!window.__opencvReady && !window.cv && (performance.now() - start) < timeoutMs) {
        await new Promise(r => setTimeout(r, 50));
      }
      if (!window.__opencvReady && !window.cv) {
        console.warn('[InitOpenCV] Timed out waiting for OpenCV to load');
      }
    }

    // Modularized OpenCV.js builds expose `cv` as a factory function.
    if (typeof window.cv === 'function') {
      try {
        if (ENABLE_OPENCV_DIAGNOSTICS) {
          console.log('[InitOpenCV] window.cv is a factory; instantiating with Module');
        }
        window.cv = window.cv(window.Module || {});
      } catch (e) {
        console.warn('[InitOpenCV] Failed to instantiate cv factory:', e);
      }
    }

    function makeNonThenable(cvObj) {
      if (!cvObj) return cvObj;
      try {
        if (typeof cvObj.then === 'function') {
          if (ENABLE_OPENCV_DIAGNOSTICS) {
            console.warn('[InitOpenCV] cv is thenable; wrapping to avoid await hang');
          }
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

    if (window.cv instanceof Promise) {
      const resolved = await window.cv;
      if (ENABLE_OPENCV_DIAGNOSTICS) console.log('OpenCV ready (awaited Promise)');
      return makeNonThenable(resolved);
    }

    if (window.cv && window.cv.Mat) {
      if (ENABLE_OPENCV_DIAGNOSTICS) console.log('OpenCV ready (non-modularized)');
      return makeNonThenable(window.cv);
    }

    throw new Error('OpenCV not found');
  }

  startBtn.addEventListener('click', async () => {
    if (running) return;

    try {
      if (startBtn) {
        startBtn.disabled = true;
        startBtn.textContent = 'Starting…';
      }

      // Lazy-load heavy modules while waiting for camera permission.
      const rendererImport = import('./renderer.js');
      const slamImport = import('./slam_core.js');
      const poseImport = import('./pose.js');
      const markerImport = import('./marker_tracker.js');

      statusEl.textContent = 'Starting camera...';
      await camera.start();
      await videoEl.play();

      const [{ ARRenderer }, { SlamCore }, poseMod, { MarkerTracker }] = await Promise.all([
        rendererImport,
        slamImport,
        poseImport,
        markerImport
      ]);

      initPoseFn = poseMod?.initPose || null;
      estimatePoseFn = poseMod?.estimatePose || null;
      getPoseIntrinsicsFn = poseMod?.getPoseIntrinsics || null;
      SlamCoreClass = SlamCore || null;
      MarkerTrackerClass = MarkerTracker || null;

      if (!renderer) {
        renderer = new ARRenderer(threeCanvas);
        try {
          renderer.restorePersistedHouse?.();
        } catch (e) {
          console.warn('[Persist] restore failed:', e);
        }
      }

      function resizeCvCanvasToViewport() {
        const vv = window.visualViewport;
        const cssW = Math.max(1, Math.round(vv?.width ?? window.innerWidth));
        const cssH = Math.max(1, Math.round(vv?.height ?? window.innerHeight));
        const dpr = window.devicePixelRatio || 1;

        cvCanvas.width = Math.round(cssW * dpr);
        cvCanvas.height = Math.round(cssH * dpr);
        cvCanvas.style.width = `${cssW}px`;
        cvCanvas.style.height = `${cssH}px`;
        cvCanvas.style.left = '0px';
        cvCanvas.style.top = '0px';

        camera.cvCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
      }

      resizeCvCanvasToViewport();

      statusEl.textContent = 'Initializing OpenCV...';
      const watchdog = setTimeout(() => {
        console.warn('[Init] initOpenCV appears stalled > 5s');
        try {
          statusEl.textContent = 'OpenCV init stalled (see console)';
        } catch {}
      }, 5000);

      cvInstance = await initOpenCV();
      clearTimeout(watchdog);

      if (ENABLE_OPENCV_DIAGNOSTICS) {
        try {
          const { runOpenCVStartupDiagnostics } = await import('./debug.js');
          runOpenCVStartupDiagnostics(cvInstance, { source: window.__opencvSource || '(unknown)' });
        } catch (e) {
          console.warn('[OpenCV] diagnostics import/run failed:', e);
        }
      }

      await new Promise(r => setTimeout(r, 0));

      statusEl.textContent = 'Initializing pose...';
      try {
        if (videoMetadataPending || videoEl.readyState >= 1) {
          initPoseFn?.(videoEl, cvInstance);
        }
      } catch (e) {
        console.warn('initPose failed:', e);
      }

      statusEl.textContent = 'Checking SLAM...';
      const canSlam = !!(
        cvInstance.recoverPose &&
        (cvInstance.findEssentialMat || cvInstance.findFundamentalMat) &&
        SlamCoreClass
      );
      if (canSlam) {
        let intrinsics = null;
        try {
          intrinsics = getPoseIntrinsicsFn?.();
        } catch (e) {
          console.warn('getPoseIntrinsics failed; falling back to heuristic intrinsics:', e);
        }
        slam = new SlamCoreClass(cvInstance, intrinsics);
      } else {
        slam = null;
        if (ENABLE_OPENCV_DIAGNOSTICS) {
          console.warn('SLAM disabled: missing findEssentialMat/findFundamentalMat/recoverPose or SlamCore');
        }
      }

      statusEl.textContent = 'Loading marker template...';
      try {
        markerTracker = MarkerTrackerClass ? new MarkerTrackerClass(cvInstance) : null;
        await markerTracker.loadTemplate('marker/WorldZeroMarker.png');
        showToast('Marker template loaded');
      } catch (e) {
        console.warn('MarkerTracker init failed:', e);
        showToast('Marker tracker unavailable');
      }

      statusEl.textContent = 'Loading model...';
      try {
        const gltf = await renderer.loadGLB('model/house.glb');
        renderer.model = gltf.scene;

        renderer.model.position.set(0, 0, -0.02);
        renderer.model.rotation.set(-Math.PI / 2, 0, Math.PI);
        renderer.model.scale.setScalar(0.002);

        renderer.houseRoot.add(renderer.model);
        showToast('House loaded');
      } catch (e) {
        console.warn('Failed to load house.glb:', e);
        showToast('Failed to load model');
      }

      statusEl.textContent = 'Running';
      running = true;
      loop();

      if (debugToggle) {
        debugToggle.hidden = false;
        debugToggle.setAttribute('aria-pressed', 'false');
        debugToggle.textContent = 'Show Debug';
      }
      if (setWorldZeroBtn) {
        setWorldZeroBtn.hidden = false;
      }
      if (debugInfo) {
        debugInfo.hidden = true;
        debugInfo.setAttribute('aria-hidden', 'true');
      }

      if (startBtn) startBtn.hidden = true;
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

  function getDetectedPoints() {
    const pts = debugFeaturePoints;
    if (!pts || pts.length < 4) return null;

    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;

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

    return [
      { x: tl.x, y: tl.y },
      { x: tr.x, y: tr.y },
      { x: br.x, y: br.y },
      { x: bl.x, y: bl.y }
    ];
  }

  function loop() {
    if (!running || !cvInstance || !renderer) return;

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
      if (!drawRectForFrame) return { x: p.x, y: p.y };

      const xCss = p.x / frameDpr;
      const yCss = p.y / frameDpr;

      const xInVideoCss = xCss - drawRectForFrame.offsetX;
      const yInVideoCss = yCss - drawRectForFrame.offsetY;

      const u = xInVideoCss / drawRectForFrame.drawWidth;
      const v = yInVideoCss / drawRectForFrame.drawHeight;

      const oobMargin = 0.02;
      if (u < -oobMargin || u > 1 + oobMargin || v < -oobMargin || v > 1 + oobMargin) return null;

      const edgeGuard = 0.12;
      const nearEdge = u < edgeGuard || u > 1 - edgeGuard || v < edgeGuard || v > 1 - edgeGuard;

      return {
        x: u * videoEl.videoWidth,
        y: v * videoEl.videoHeight,
        nearEdge
      };
    }

    let slamDelta = null;
    try {
      if (slam) {
        const res = slam.processFrame(frame);
        slamDelta = res?.delta || null;
      }
    } catch (err) {
      console.warn('SLAM processing error:', err);
    }

    if (!frameRgbaMat || frameRgbaMat.rows !== frame.height || frameRgbaMat.cols !== frame.width) {
      frameRgbaMat?.delete?.();
      frameGrayMat?.delete?.();
      frameRgbaMat = new cv.Mat(frame.height, frame.width, cv.CV_8UC4);
      frameGrayMat = new cv.Mat(frame.height, frame.width, cv.CV_8UC1);
    }
    frameRgbaMat.data.set(frame.data);
    cv.cvtColor(frameRgbaMat, frameGrayMat, cv.COLOR_RGBA2GRAY);
    const gray = frameGrayMat;

    try {
      if (markerTracker) {
        let cornersProc = null;

        if (markerTracking && markerPrevGray && markerPrevPts && markerPrevPts.rows === 4) {
          if (!markerCurrPts) markerCurrPts = new cv.Mat();
          if (!markerLkStatus) markerLkStatus = new cv.Mat();
          if (!markerLkErr) markerLkErr = new cv.Mat();

          cv.calcOpticalFlowPyrLK(markerPrevGray, gray, markerPrevPts, markerCurrPts, markerLkStatus, markerLkErr);

          let ok = markerLkStatus.rows === 4;
          if (ok) {
            for (let i = 0; i < 4; i++) {
              if (markerLkStatus.data[i] !== 1) {
                ok = false;
                break;
              }
            }
          }

          if (ok) {
            cornersProc = [];
            for (let i = 0; i < 4; i++) {
              cornersProc.push({ x: markerCurrPts.data32F[i * 2], y: markerCurrPts.data32F[i * 2 + 1] });
            }

            // Update previous-gray buffer without allocating a new Mat.
            if (!markerPrevGray || markerPrevGray.rows !== gray.rows || markerPrevGray.cols !== gray.cols) {
              markerPrevGray?.delete?.();
              markerPrevGray = new cv.Mat(gray.rows, gray.cols, cv.CV_8UC1);
            }
            gray.copyTo(markerPrevGray);

            // Swap point mats (reuses both Mats without copy/alloc).
            const tmpPts = markerPrevPts;
            markerPrevPts = markerCurrPts;
            markerCurrPts = tmpPts;
          } else {
            markerTracking = false;
            markerPrevGray.delete();
            markerPrevGray = null;
            markerPrevPts.delete();
            markerPrevPts = null;
            markerCurrPts?.delete?.();
            markerCurrPts = null;
          }
        }

        if (!cornersProc) {
          if (markerDetectCountdown > 0) {
            markerDetectCountdown--;
          } else {
            const det = markerTracker.detect(gray);
            if (det && det.corners && det.corners.length === 4) {
              cornersProc = det.corners;

              markerPrevGray?.delete?.();
              markerPrevGray = new cv.Mat(gray.rows, gray.cols, cv.CV_8UC1);
              gray.copyTo(markerPrevGray);
              markerPrevPts?.delete?.();
              markerPrevPts = cv.matFromArray(4, 1, cv.CV_32FC2, [
                cornersProc[0].x, cornersProc[0].y,
                cornersProc[1].x, cornersProc[1].y,
                cornersProc[2].x, cornersProc[2].y,
                cornersProc[3].x, cornersProc[3].y
              ]);
              markerCurrPts?.delete?.();
              markerCurrPts = new cv.Mat();
              markerTracking = true;
              markerDetectCountdown = 0;

              det.homography?.delete?.();
            } else {
              markerDetectCountdown = MARKER_DETECT_EVERY_N_FRAMES - 1;
              det?.homography?.delete?.();
            }
          }
        }

        if (cornersProc) {
          const imagePtsScaledFullRaw = cornersProc.map(procPointToVideo);
          const markerPtsOk = imagePtsScaledFullRaw.every(p => p && Number.isFinite(p.x) && Number.isFinite(p.y));

          let imagePtsScaledFull = imagePtsScaledFullRaw;
          if (markerPtsOk) {
            const idx = orderQuadIndicesTLTRBRBL(imagePtsScaledFullRaw);
            if (idx) imagePtsScaledFull = idx.map(i => imagePtsScaledFullRaw[i]);
          }

          const markerNearEdge = markerPtsOk && imagePtsScaledFull.some(p => p.nearEdge);

          if (!markerPtsOk) {
            markerTracking = false;
            markerPrevGray?.delete?.();
            markerPrevGray = null;
            markerPrevPts?.delete?.();
            markerPrevPts = null;
            markerCurrPts?.delete?.();
            markerCurrPts = null;

            cornersProc = null;
          } else if (markerNearEdge && slam) {
            cornersProc = null;
          } else {
            const imagePtsScaled = imagePtsScaledFull.map(({ x, y }) => ({ x, y }));

            const justAcquired = !lastHadMarker;
            if (justAcquired) loggedReacquireRejection = false;

            const half = 0.1016 / 2;
            const objectPoints = [
              { x: -half, y: -half, z: 0 },
              { x: half, y: -half, z: 0 },
              { x: half, y: half, z: 0 },
              { x: -half, y: half, z: 0 }
            ];

            const pose = estimatePoseFn ? estimatePoseFn(imagePtsScaled, objectPoints, cvInstance) : null;
            const poseOk = poseLooksPlausible(pose);
            const acceptPose = !!pose && poseOk && (justAcquired || !poseJumpTooLarge(lastStableMarkerPose, pose));

            if (acceptPose) {
              lastStableMarkerPose = pose;
              latestPose = pose;
              if (USE_CAMERA_TRACKED_RUNTIME) {
                renderer.setCameraFromMarkerPose(pose);
              } else {
                renderer.setAnchorPose(pose);
              }
              logEvery(30, '[Marker] pose ok', pose.position);

              if (slamDelta && slamDelta.R && slamDelta.t && lastMarkerPoseForScale) {
                const dx = pose.position.x - lastMarkerPoseForScale.position.x;
                const dy = pose.position.y - lastMarkerPoseForScale.position.y;
                const dz = pose.position.z - lastMarkerPoseForScale.position.z;
                const dMarker = Math.sqrt(dx * dx + dy * dy + dz * dz);

                const dt = slamDelta.t;
                const dSlam = Math.sqrt(dt[0] * dt[0] + dt[1] * dt[1] + dt[2] * dt[2]);
                if (dMarker > 1e-4 && dSlam > 1e-6) {
                  const s = dMarker / dSlam;
                  slamMetricScale = slamMetricScale > 0 ? 0.9 * slamMetricScale + 0.1 * s : s;
                  logEvery(60, '[SLAM] metric scale', slamMetricScale.toFixed(4));
                }
              }

              lastMarkerPoseForScale = pose;
            } else if (lastStableMarkerPose) {
              if (justAcquired || !slam) {
                if (USE_CAMERA_TRACKED_RUNTIME) {
                  renderer.setCameraFromMarkerPose(null);
                } else {
                  renderer.setAnchorPose(null);
                }
              } else {
                if (USE_CAMERA_TRACKED_RUNTIME) {
                  renderer.setCameraFromMarkerPose(lastStableMarkerPose);
                } else {
                  renderer.setAnchorPose(lastStableMarkerPose);
                }
              }

              if (pose && justAcquired && !loggedReacquireRejection) {
                loggedReacquireRejection = true;
                console.warn('[Marker] reacquired but pose rejected; keeping lastStable pose', {
                  prev: lastStableMarkerPose.position,
                  next: pose.position
                });
              }

              if (pose && !poseOk) {
                markerTracking = false;
                markerPrevGray?.delete?.();
                markerPrevGray = null;
                markerPrevPts?.delete?.();
                markerPrevPts = null;
                markerCurrPts?.delete?.();
                markerCurrPts = null;
              }
            } else {
              if (USE_CAMERA_TRACKED_RUNTIME) {
                renderer.setCameraFromMarkerPose(null);
              } else {
                renderer.setAnchorPose(null);
              }
            }

            if (!lastHadMarker) console.log('[Marker] acquired');
            lastHadMarker = true;
            framesSinceMarker = 0;
          }
        }

        if (!cornersProc) {
          framesSinceMarker++;

          if (!slam) {
            if (USE_CAMERA_TRACKED_RUNTIME) {
              renderer.setCameraFromMarkerPose(null);
            } else {
              renderer.setAnchorPose(null);
            }
            if (lastHadMarker) console.log('[Marker] lost');
            lastHadMarker = false;
            return requestAnimationFrame(loop);
          }

          if (slamDelta && slamDelta.R && slamDelta.t) {
            const scale = slamMetricScale > 0 ? slamMetricScale : 0.01;
            if (USE_CAMERA_TRACKED_RUNTIME) {
              renderer.applySlamDelta?.(slamDelta.R, slamDelta.t, scale);
            } else {
              renderer.applySlamDeltaToAnchor?.(slamDelta.R, slamDelta.t, scale);
            }
          }

          if (lastHadMarker) console.log('[Marker] lost');
          lastHadMarker = false;
        }
      }
    } catch (e) {
      console.warn('Marker pose error:', e);
    }

    // Feature tracking is only needed for debug overlay (or as a fallback pose demo when markerTracker is unavailable).
    const needFeatureTracking = showDebug || !markerTracker;
    if (needFeatureTracking) {
      if (!prevGray || !prevPoints || prevPoints.rows === 0) {
        if (!prevGray || prevGray.rows !== gray.rows || prevGray.cols !== gray.cols) {
          prevGray?.delete?.();
          prevGray = new cv.Mat(gray.rows, gray.cols, cv.CV_8UC1);
        }
        gray.copyTo(prevGray);

        prevPoints?.delete?.();
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
        if (!featureCurrPoints) featureCurrPoints = new cv.Mat();
        if (!featureLkStatus) featureLkStatus = new cv.Mat();
        if (!featureLkErr) featureLkErr = new cv.Mat();

        cv.calcOpticalFlowPyrLK(prevGray, gray, prevPoints, featureCurrPoints, featureLkStatus, featureLkErr);

        const goodCurrFloats = [];
        for (let i = 0; i < featureLkStatus.rows; i++) {
          if (featureLkStatus.data[i] === 1) {
            goodCurrFloats.push(featureCurrPoints.data32F[i * 2], featureCurrPoints.data32F[i * 2 + 1]);
          }
        }

        let mergedFloats = goodCurrFloats;
        const goodCount = goodCurrFloats.length / 2;

        if (goodCount < MIN_FEATURES) {
          const want = Math.max(0, MAX_FEATURES - goodCount);
          if (want > 0) {
            const mask = new cv.Mat(gray.rows, gray.cols, cv.CV_8UC1, new cv.Scalar(255));
            for (let i = 0; i < goodCount; i++) {
              const x = goodCurrFloats[i * 2];
              const y = goodCurrFloats[i * 2 + 1];
              cv.circle(mask, new cv.Point(x, y), FEATURE_EXCLUSION_RADIUS, new cv.Scalar(0), -1);
            }

            const extra = new cv.Mat();
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

        const nextRows = mergedFloats.length / 2;
        if (!prevPoints || prevPoints.rows !== nextRows || prevPoints.type?.() !== cv.CV_32FC2) {
          prevPoints?.delete?.();
          prevPoints = new cv.Mat(nextRows, 1, cv.CV_32FC2);
        } else {
          // Ensure the backing buffer exists and is large enough
          // (OpenCV.js allocates on demand for some Mats)
          if (!prevPoints.data32F || prevPoints.data32F.length < mergedFloats.length) {
            prevPoints.delete();
            prevPoints = new cv.Mat(nextRows, 1, cv.CV_32FC2);
          }
        }

        if (mergedFloats.length > 0) {
          prevPoints.data32F.set(mergedFloats);
        }

        if (!prevGray || prevGray.rows !== gray.rows || prevGray.cols !== gray.cols) {
          prevGray?.delete?.();
          prevGray = new cv.Mat(gray.rows, gray.cols, cv.CV_8UC1);
        }
        gray.copyTo(prevGray);
      }
    }

    try {
      if (!markerTracker) {
        const rawImagePts = getDetectedPoints();
        if (rawImagePts && cvInstance) {
          const sx = videoEl.videoWidth / camera.cvCanvas.width;
          const sy = videoEl.videoHeight / camera.cvCanvas.height;
          const imagePtsScaled = rawImagePts.map(p => ({ x: p.x * sx, y: p.y * sy }));

          const objectPoints = [
            { x: -0.05, y: -0.05, z: 0 },
            { x: 0.05, y: -0.05, z: 0 },
            { x: 0.05, y: 0.05, z: 0 },
            { x: -0.05, y: 0.05, z: 0 }
          ];

          const pose = estimatePoseFn ? estimatePoseFn(imagePtsScaled, objectPoints, cvInstance) : null;
          latestPose = pose || null;
        }
      }
    } catch (e) {
      console.warn('Pose estimation error:', e);
    }

    const ctx = camera.cvCtx;
    const drawRect = drawRectForFrame;

    if (showDebug) {
      ctx.save();
      ctx.strokeStyle = 'red';
      ctx.lineWidth = 2;
      ctx.strokeRect(
        drawRect.offsetX + 0.5,
        drawRect.offsetY + 0.5,
        drawRect.drawWidth,
        drawRect.drawHeight
      );
      ctx.restore();

      ctx.fillStyle = 'lime';

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
        const poseLine = latestPose?.position
          ? `\npose(m): x=${latestPose.position.x.toFixed(3)} y=${latestPose.position.y.toFixed(3)} z=${latestPose.position.z.toFixed(3)}`
          : '';
        debugInfo.textContent = `drawRect:\noffsetX: ${drawRect.offsetX.toFixed(1)}\noffsetY: ${drawRect.offsetY.toFixed(1)}\ndrawW: ${drawRect.drawWidth.toFixed(1)}\ndrawH: ${drawRect.drawHeight.toFixed(1)}\nfeatures: ${needFeatureTracking ? debugFeaturePoints.length : 0}${poseLine}`;
      }
    } else {
      if (debugInfo) {
        debugInfo.hidden = true;
        debugInfo.setAttribute('aria-hidden', 'true');
      }
    }

    try {
      renderer.render();
    } catch {
      // ignore transient GL init errors
    }

    requestAnimationFrame(loop);
  }

  window.addEventListener('resize', () => {
    try {
      if (running) {
        const vv = window.visualViewport;
        const dpr = window.devicePixelRatio || 1;
        const cssW = Math.max(1, Math.round(vv?.width ?? window.innerWidth));
        const cssH = Math.max(1, Math.round(vv?.height ?? window.innerHeight));

        cvCanvas.width = Math.round(cssW * dpr);
        cvCanvas.height = Math.round(cssH * dpr);
        cvCanvas.style.width = `${cssW}px`;
        cvCanvas.style.height = `${cssH}px`;

        camera.cvCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
      }
    } catch (e) {
      console.warn('[Resize] cvCanvas resize failed:', e);
    }

    renderer?.resize?.();
  });
});

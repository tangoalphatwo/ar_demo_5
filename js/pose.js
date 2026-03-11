// pose.js
let cameraMatrix = null;
let distCoeffs = null;
let cvModule = null;
let worldOrigin = null;
let rawPose = null;
let lastPose = null;

let lastSolvePnpFailLogMs = 0;
function logSolvePnpFailOncePer(ms, payload) {
  const now = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
  if (now - lastSolvePnpFailLogMs >= ms) {
    lastSolvePnpFailLogMs = now;
    console.warn('[Pose] solvePnP failed', payload);
  }
}

let lastSolvePnpExceptionLogMs = 0;
function logSolvePnpExceptionOncePer(ms, payload) {
  const now = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
  if (now - lastSolvePnpExceptionLogMs >= ms) {
    lastSolvePnpExceptionLogMs = now;
    console.warn('[Pose] solvePnP threw', payload);
  }
}

// Keep last rvec/tvec for solvePnP initial guess (significantly stabilizes pose)
let lastRvecArr = null; // length 3
let lastTvecArr = null; // length 3

const SMOOTHING = 0.85; // closer to 1 = smoother

function smoothValue(prev, next) {
  return prev * SMOOTHING + next * (1 - SMOOTHING);
}

function matFromArray(rows, cols, type, array) {
  const mat = new cvModule.Mat(rows, cols, type);
  // Validate length
  const expected = rows * cols;
  if (array.length !== expected) {
    throw new RangeError(`matFromArray: expected ${expected} elements (rows*cols=${rows}*${cols}), got ${array.length}`);
  }

  // Prefer 64F backing for numeric safety; set underlying buffer
  if (mat.data64F) mat.data64F.set(array);
  else if (mat.data32F) mat.data32F.set(array);
  else if (mat.data) mat.data.set(array);
  else throw new Error('Unsupported matrix backing type');

  return mat;
}

function normalizeTranslation(tvec) {
  return {
    // Keep raw OpenCV camera coordinates:
    // x: right, y: down, z: forward
    // Axis conversion to Three.js happens in renderer.js.
    x: tvec[0],
    y: tvec[1],
    z: tvec[2]
  };
}

function rotationMatrixToEuler(m) {
  const r00 = m[0], r01 = m[1], r02 = m[2];
  const r10 = m[3], r11 = m[4], r12 = m[5];
  const r20 = m[6], r21 = m[7], r22 = m[8];

  const pitch = Math.asin(-r20);
  const roll  = Math.atan2(r21, r22);
  const yaw   = Math.atan2(r10, r00);

  return {
    yaw,
    pitch,
    roll
  };
}

export function initPose(video, cv) {
  // Accept a cv instance (modular builds) or fall back to global window.cv
  cvModule = cv || (typeof window !== 'undefined' ? window.cv : null);
  if (!cvModule || !cvModule.Mat) {
    throw new Error('OpenCV cv object not provided or not initialized');
  }

  const width  = video.videoWidth;
  const height = video.videoHeight;

  // Rough intrinsics estimate (matches the older working pipeline): fx=fy=width.
  // This is not physically perfect, but tends to keep solvePnP stable.
  const focalLength = width;

  cameraMatrix = matFromArray(3, 3, cvModule.CV_64F, [
    focalLength, 0, width / 2,
    0, focalLength, height / 2,
    0, 0, 1
  ]);

  console.log('[Pose] Intrinsics', { width, height, focalLength });

  // Use 5 coefficients (k1,k2,p1,p2,k3). Many OpenCV builds accept 4, but 5 is safer.
  distCoeffs = cvModule.Mat.zeros(5, 1, cvModule.CV_64F);
}

function toCvPoint2f(points) {
  const data = [];
  points.forEach(p => data.push(p.x, p.y));
  // OpenCV.js solvePnP expects Nx1 with 2 channels (Point2f/Point2d)
  // Use cv.matFromArray which understands multi-channel element sizes.
  return cvModule.matFromArray(points.length, 1, cvModule.CV_32FC2, data);
}

function toCvPoint3f(points) {
  const data = [];
  points.forEach(p => data.push(p.x, p.y, p.z));
  // OpenCV.js solvePnP expects Nx1 with 3 channels (Point3f/Point3d)
  return cvModule.matFromArray(points.length, 1, cvModule.CV_32FC3, data);
}

export function setWorldOrigin() {
  if (!rawPose) return;

  worldOrigin = {
    position: { ...rawPose.position },
    rotation: { ...rawPose.rotation }
  };

  console.log("World origin set:", worldOrigin);
}

function applyWorldZero(pose) {
  if (!worldOrigin) return pose;

  return {
    position: {
      x: pose.position.x - worldOrigin.position.x,
      y: pose.position.y - worldOrigin.position.y,
      z: pose.position.z - worldOrigin.position.z
    },
    rotation: {
      yaw:   pose.rotation.yaw   - worldOrigin.rotation.yaw,
      pitch: pose.rotation.pitch - worldOrigin.rotation.pitch,
      roll:  pose.rotation.roll  - worldOrigin.rotation.roll
    }
  };
}

export function estimatePose(imagePoints, objectPoints, cv) {
  const cvLocal = cv || cvModule;
  if (!cvLocal || !cameraMatrix || !distCoeffs) {
    console.warn('Pose system not initialized');
    return null;
  }

  // Ensure cvModule is set for helper functions
  cvModule = cvModule || cvLocal;

  const imgPts = toCvPoint2f(imagePoints);
  const objPts = toCvPoint3f(objectPoints);

  // Pre-allocate outputs as 3x1 CV_64F (some OpenCV.js builds are picky if these are empty).
  const rvec = cvModule.Mat.zeros(3, 1, cvModule.CV_64F);
  const tvec = cvModule.Mat.zeros(3, 1, cvModule.CV_64F);

  // Use previous pose as an initial guess when available.
  let useGuess = false;
  try {
    if (lastRvecArr && lastTvecArr && lastRvecArr.length === 3 && lastTvecArr.length === 3) {
      // rvec/tvec are 3x1 CV_64F in OpenCV.js
      const rInit = cvModule.matFromArray(3, 1, cvModule.CV_64F, lastRvecArr);
      const tInit = cvModule.matFromArray(3, 1, cvModule.CV_64F, lastTvecArr);
      rInit.copyTo(rvec);
      tInit.copyTo(tvec);
      rInit.delete();
      tInit.delete();
      useGuess = true;
    }
  } catch (e) {
    useGuess = false;
  }

  // Try a small set of flags for robustness across devices/frames.
  const flagsToTry = [];
  // Prefer IPPE for planar squares (typically less "float" than ITERATIVE on markers).
  if (typeof cvModule.SOLVEPNP_IPPE_SQUARE === 'number') flagsToTry.push(cvModule.SOLVEPNP_IPPE_SQUARE);
  if (typeof cvModule.SOLVEPNP_ITERATIVE === 'number') flagsToTry.push(cvModule.SOLVEPNP_ITERATIVE);
  if (typeof cvModule.SOLVEPNP_EPNP === 'number') flagsToTry.push(cvModule.SOLVEPNP_EPNP);

  let success = false;
  let usedFlag = null;
  for (const flag of flagsToTry) {
    try {
      success = cvModule.solvePnP(
        objPts,
        imgPts,
        cameraMatrix,
        distCoeffs,
        rvec,
        tvec,
        useGuess,
        flag
      );
      if (success) {
        usedFlag = flag;
        break;
      }
    } catch (e) {
      logSolvePnpExceptionOncePer(750, { flag, useGuess, message: String(e?.message || e) });
      // Continue to next flag.
    }
  }

  imgPts.delete();
  objPts.delete();

  if (!success) {
    // Throttled diagnostics to help debug point ordering / scale / intrinsics mismatch.
    logSolvePnpFailOncePer(750, {
      usedFlag,
      useGuess,
      imagePoints,
      objectPoints,
      intrinsics: cameraMatrix ? {
        w: cameraMatrix.cols,
        h: cameraMatrix.rows,
        fx: cameraMatrix.data64F ? cameraMatrix.data64F[0] : null,
        fy: cameraMatrix.data64F ? cameraMatrix.data64F[4] : null,
        cx: cameraMatrix.data64F ? cameraMatrix.data64F[2] : null,
        cy: cameraMatrix.data64F ? cameraMatrix.data64F[5] : null
      } : null
    });
    rvec.delete();
    tvec.delete();
    return null;
  }

  const rotationMatrix = new cvModule.Mat();
  cvModule.Rodrigues(rvec, rotationMatrix);

  // Extract numeric arrays from mats (prefer 64F, then 32F, then generic data)
  const tArr = (tvec.data64F && Array.from(tvec.data64F)) || (tvec.data32F && Array.from(tvec.data32F)) || (tvec.data && Array.from(tvec.data));
  const rotArr = (rotationMatrix.data64F && Array.from(rotationMatrix.data64F)) || (rotationMatrix.data32F && Array.from(rotationMatrix.data32F)) || (rotationMatrix.data && Array.from(rotationMatrix.data));

  // Normalize translation and convert rotation matrix to Euler
  const t = normalizeTranslation(tArr);
  const r = rotationMatrixToEuler(rotArr);

  // Save raw rvec/tvec for the next iteration's guess
  try {
    const rArr = (rvec.data64F && Array.from(rvec.data64F)) || (rvec.data32F && Array.from(rvec.data32F)) || (rvec.data && Array.from(rvec.data));
    lastRvecArr = rArr ? rArr.slice(0, 3) : null;
    lastTvecArr = tArr ? tArr.slice(0, 3) : null;
  } catch (e) {
    lastRvecArr = null;
    lastTvecArr = null;
  }

  rawPose = {
    position: { ...t },
    rotation: { ...r },
    rotationMatrix: rotArr.slice() // keep numeric rotation matrix around
    };

  // Build pose state and smooth against lastPose
  let poseState = {
    position: { ...t },
    rotation: { ...r },
    rotationMatrix: rotArr.slice()
  };

  // Clean up mats
  rvec.delete();
  tvec.delete();
  rotationMatrix.delete();

  if (lastPose) {
    poseState.position.x = smoothValue(lastPose.position.x, poseState.position.x);
    poseState.position.y = smoothValue(lastPose.position.y, poseState.position.y);
    poseState.position.z = smoothValue(lastPose.position.z, poseState.position.z);

    poseState.rotation.yaw   = smoothValue(lastPose.rotation.yaw,   poseState.rotation.yaw);
    poseState.rotation.pitch = smoothValue(lastPose.rotation.pitch, poseState.rotation.pitch);
    poseState.rotation.roll  = smoothValue(lastPose.rotation.roll,  poseState.rotation.roll);
  }

  lastPose = poseState;
  const worldPose = applyWorldZero(poseState);
  // lastPose = worldPose;
  // Preserve numeric rotation matrix in worldPose if available
  if (poseState.rotationMatrix) {
    worldPose.rotationMatrix = poseState.rotationMatrix.slice();
  }

  return worldPose;
}

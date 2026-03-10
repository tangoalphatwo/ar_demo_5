// marker_quad.js

function orderCorners(pts) {
  // pts: Array<{x,y}> length 4
  // Return [tl, tr, br, bl] in image coordinates (y down).
  // Robust to rotation; avoids sum/diff tie issues that can create duplicates.
  if (!pts || pts.length !== 4) return pts;

  const uniq = new Set(pts.map((p) => `${p.x},${p.y}`));
  if (uniq.size !== 4) return pts;

  const cx = pts.reduce((s, p) => s + p.x, 0) / 4;
  const cy = pts.reduce((s, p) => s + p.y, 0) / 4;

  const byAngle = pts
    .map((p) => ({ p, a: Math.atan2(p.y - cy, p.x - cx) }))
    .sort((a, b) => a.a - b.a)
    .map((o) => o.p);

  // Rotate so TL (min x+y) is first.
  let tlIdx = 0;
  let best = Infinity;
  for (let i = 0; i < 4; i++) {
    const v = byAngle[i].x + byAngle[i].y;
    if (v < best) {
      best = v;
      tlIdx = i;
    }
  }
  const rot = [0, 1, 2, 3].map((k) => byAngle[(tlIdx + k) % 4]);

  // rot is either [tl,tr,br,bl] or [tl,bl,br,tr]. Pick the one where the
  // second point is to the right of the fourth.
  const tl = rot[0];
  const p1 = rot[1];
  const p3 = rot[3];
  if (p1.x < p3.x) {
    const out = [tl, p3, rot[2], p1];
    // Enforce clockwise winding in image coords (y down) to avoid mirrored poses.
    const a = out[0], b = out[1], c = out[2];
    const cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
    if (cross < 0) {
      // swap TR and BL
      return [out[0], out[3], out[2], out[1]];
    }
    return out;
  }
  {
    const out = [tl, p1, rot[2], p3];
    // Enforce clockwise winding in image coords (y down) to avoid mirrored poses.
    const a = out[0], b = out[1], c = out[2];
    const cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
    if (cross < 0) {
      // swap TR and BL
      return [out[0], out[3], out[2], out[1]];
    }
    return out;
  }
}

function dist2(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return dx * dx + dy * dy;
}

function isRoughlySquare(quad, maxAspectSkew = 0.35) {
  // Compare opposite side lengths.
  const [tl, tr, br, bl] = quad;
  const d01 = Math.sqrt(dist2(tl, tr));
  const d12 = Math.sqrt(dist2(tr, br));
  const d23 = Math.sqrt(dist2(br, bl));
  const d30 = Math.sqrt(dist2(bl, tl));

  const w = (d01 + d23) * 0.5;
  const h = (d12 + d30) * 0.5;
  if (w < 1e-6 || h < 1e-6) return false;

  const aspect = w / h;
  return Math.abs(1 - aspect) < maxAspectSkew;
}

/**
 * Detect a square-ish quad in the frame.
 * Returns null or { corners: Array<{x,y}>, area: number }
 */
export function detectMarkerQuad(cv, gray, {
  minAreaFrac = 0.02,
  maxAreaFrac = 0.85,
  polyEpsFrac = 0.02
} = {}) {
  const w = gray.cols;
  const h = gray.rows;
  const imgArea = w * h;

  const blurred = new cv.Mat();
  const bin = new cv.Mat();
  const edges = new cv.Mat();
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();

  try {
    cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0);

    // Binarize; adaptiveThreshold is often more stable on mobile.
    if (typeof cv.adaptiveThreshold === 'function') {
      cv.adaptiveThreshold(
        blurred,
        bin,
        255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY,
        31,
        7
      );
    } else {
      cv.threshold(blurred, bin, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
    }

    // Edges help remove interior texture.
    cv.Canny(blurred, edges, 60, 160);

    // Combine: prefer edges for contours, fallback to bin.
    const srcForContours = edges;

    cv.findContours(srcForContours, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    let best = null;
    let bestArea = 0;

    const minArea = imgArea * minAreaFrac;
    const maxArea = imgArea * maxAreaFrac;

    for (let i = 0; i < contours.size(); i++) {
      const cnt = contours.get(i);
      const area = cv.contourArea(cnt);
      if (area < minArea || area > maxArea) {
        cnt.delete();
        continue;
      }

      const peri = cv.arcLength(cnt, true);
      const approx = new cv.Mat();
      cv.approxPolyDP(cnt, approx, polyEpsFrac * peri, true);

      const isQuad = approx.rows === 4;
      const isConvex = isQuad && cv.isContourConvex(approx);
      if (!isQuad || !isConvex) {
        approx.delete();
        cnt.delete();
        continue;
      }

      // Extract points
      const pts = [];
      for (let k = 0; k < 4; k++) {
        const x = approx.intPtr(k, 0)[0];
        const y = approx.intPtr(k, 0)[1];
        pts.push({ x, y });
      }

      const ordered = orderCorners(pts);
      if (!isRoughlySquare(ordered)) {
        approx.delete();
        cnt.delete();
        continue;
      }

      if (area > bestArea) {
        bestArea = area;
        best = { corners: ordered, area };
      }

      approx.delete();
      cnt.delete();
    }

    return best;
  } finally {
    blurred.delete();
    bin.delete();
    edges.delete();
    contours.delete();
    hierarchy.delete();
  }
}

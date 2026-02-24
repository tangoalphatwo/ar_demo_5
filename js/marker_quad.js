// marker_quad.js

function orderCorners(pts) {
  // pts: Array<{x,y}> length 4
  // Return [tl, tr, br, bl] in image coordinates (y down).
  const sums = pts.map((p) => p.x + p.y);
  const diffs = pts.map((p) => p.x - p.y);

  const tl = pts[sums.indexOf(Math.min(...sums))];
  const br = pts[sums.indexOf(Math.max(...sums))];
  const tr = pts[diffs.indexOf(Math.max(...diffs))];
  const bl = pts[diffs.indexOf(Math.min(...diffs))];

  return [tl, tr, br, bl];
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

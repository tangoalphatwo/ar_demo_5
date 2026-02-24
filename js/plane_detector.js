// plane_detector.js

function sub(a, b) {
  return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}

function cross(a, b) {
  return {
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x
  };
}

function dot(a, b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

function norm(v) {
  return Math.sqrt(dot(v, v));
}

function normalize(v) {
  const n = norm(v);
  if (!isFinite(n) || n < 1e-12) return null;
  return { x: v.x / n, y: v.y / n, z: v.z / n };
}

function planeFrom3Points(p0, p1, p2) {
  const v1 = sub(p1, p0);
  const v2 = sub(p2, p0);
  const n = normalize(cross(v1, v2));
  if (!n) return null;
  const d = -dot(n, p0);
  return { n, d };
}

function pointPlaneDistance(plane, p) {
  // Signed distance: n·p + d, where |n|=1
  return dot(plane.n, p) + plane.d;
}

export class PlaneDetector {
  /**
   * @param {object} [opts]
   * @param {number} [opts.ransacIters]
   * @param {number} [opts.inlierThreshold]
   * @param {number} [opts.minInliers]
   */
  constructor({ ransacIters = 200, inlierThreshold = 0.05, minInliers = 40 } = {}) {
    this.ransacIters = ransacIters;
    this.inlierThreshold = inlierThreshold;
    this.minInliers = minInliers;
  }

  /**
   * @param {Array<{X:number,Y:number,Z:number}>} points
   * @returns {null | { normal: {x:number,y:number,z:number}, d:number, inliers:Array<number> }}
   */
  detect(points) {
    if (!Array.isArray(points) || points.length < 3) return null;

    // Convert to consistent shape
    const pts = points
      .map((p) => ({ x: p.X, y: p.Y, z: p.Z }))
      .filter((p) => [p.x, p.y, p.z].every((v) => Number.isFinite(v)));

    if (pts.length < 3) return null;

    let bestPlane = null;
    let bestInliers = [];

    const nPts = pts.length;
    const randIdx = () => (Math.random() * nPts) | 0;

    for (let iter = 0; iter < this.ransacIters; iter++) {
      const i0 = randIdx();
      let i1 = randIdx();
      let i2 = randIdx();
      if (i1 === i0) i1 = randIdx();
      if (i2 === i0 || i2 === i1) i2 = randIdx();

      const plane = planeFrom3Points(pts[i0], pts[i1], pts[i2]);
      if (!plane) continue;

      const inliers = [];
      for (let i = 0; i < nPts; i++) {
        const dist = Math.abs(pointPlaneDistance(plane, pts[i]));
        if (dist < this.inlierThreshold) inliers.push(i);
      }

      if (inliers.length > bestInliers.length) {
        bestInliers = inliers;
        bestPlane = plane;
      }
    }

    if (!bestPlane || bestInliers.length < this.minInliers) return null;

    return {
      normal: bestPlane.n,
      d: bestPlane.d,
      inliers: bestInliers
    };
  }
}

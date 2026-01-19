// marker_tracker.js
// Detects a planar image marker in the current frame using ORB + BFMatcher + homography.

export class MarkerTracker {
  constructor(cv, { markerUrl, maxMatches = 60, goodMatchPercent = 0.25 } = {}) {
    this.cv = cv;
    this.markerUrl = markerUrl;

    this.maxMatches = maxMatches;
    this.goodMatchPercent = goodMatchPercent;

    this._ready = false;

    this._markerGray = null;
    this._markerKp = null;
    this._markerDesc = null;
    this._markerSize = null; // { w, h }

    this._orb = null;
    this._bf = null;
  }

  get ready() {
    return this._ready;
  }

  async init() {
    const cv = this.cv;

    if (!this.markerUrl) {
      throw new Error('MarkerTracker: markerUrl is required');
    }

    // Capability checks: if these are missing, attempting detection can crash the WASM runtime.
    if (typeof cv.findHomography !== 'function' || typeof cv.perspectiveTransform !== 'function') {
      throw new Error('MarkerTracker: OpenCV build missing findHomography/perspectiveTransform');
    }
    if (typeof cv.BFMatcher !== 'function') {
      throw new Error('MarkerTracker: OpenCV build missing BFMatcher');
    }

    this._orb = this._createORB();
    this._bf = new cv.BFMatcher(cv.NORM_HAMMING, false);

    const imgEl = await this._loadImage(this.markerUrl);

    const rgba = cv.imread(imgEl);
    const gray = new cv.Mat();
    cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY);
    rgba.delete();

    this._markerGray = gray;
    this._markerSize = { w: gray.cols, h: gray.rows };

    this._markerKp = new cv.KeyPointVector();
    this._markerDesc = new cv.Mat();
    this._orb.detectAndCompute(this._markerGray, new cv.Mat(), this._markerKp, this._markerDesc);

    this._ready = true;
  }

  destroy() {
    const safeDelete = (m) => {
      if (m && m.delete) m.delete();
    };

    safeDelete(this._markerGray);
    safeDelete(this._markerKp);
    safeDelete(this._markerDesc);
    safeDelete(this._orb);
    safeDelete(this._bf);

    this._markerGray = null;
    this._markerKp = null;
    this._markerDesc = null;
    this._orb = null;
    this._bf = null;

    this._ready = false;
  }

  // frameGray: cv.Mat (CV_8UC1)
  // Returns { corners: [{x,y}.. TL,TR,BR,BL], homography } or null
  detect(frameGray) {
    const cv = this.cv;
    if (!this._ready) return null;
    if (!frameGray || frameGray.type() !== cv.CV_8UC1) {
      // Accept other depth but must be 1-channel
      if (!frameGray || frameGray.channels() !== 1) return null;
    }

    if (!this._markerDesc || this._markerDesc.empty()) return null;

    const frameKp = new cv.KeyPointVector();
    const frameDesc = new cv.Mat();
    const emptyMask = new cv.Mat();

    let matches = null;
    let srcMat = null;
    let dstMat = null;
    let inlierMask = null;
    let H = null;
    let projected = null;

    try {
      // ORB prefers 8-bit grayscale
      this._orb.detectAndCompute(frameGray, emptyMask, frameKp, frameDesc);

      if (frameDesc.empty() || frameKp.size() < 20) {
        return null;
      }

      matches = new cv.DMatchVector();
      this._bf.match(this._markerDesc, frameDesc, matches);
      if (matches.size() < 12) {
        return null;
      }

      // Sort matches by distance (ascending)
      const matchArray = [];
      for (let i = 0; i < matches.size(); i++) {
        matchArray.push(matches.get(i));
      }
      matchArray.sort((a, b) => a.distance - b.distance);

      const keepN = Math.max(12, Math.min(this.maxMatches, Math.floor(matchArray.length * this.goodMatchPercent)));
      const good = matchArray.slice(0, keepN);

      const src = [];
      const dst = [];

      for (const m of good) {
        const q = this._markerKp.get(m.queryIdx).pt;
        const t = frameKp.get(m.trainIdx).pt;
        src.push(q.x, q.y);
        dst.push(t.x, t.y);
      }

      srcMat = cv.matFromArray(good.length, 1, cv.CV_32FC2, src);
      dstMat = cv.matFromArray(good.length, 1, cv.CV_32FC2, dst);

      inlierMask = new cv.Mat();
      H = cv.findHomography(srcMat, dstMat, cv.RANSAC, 3.0, inlierMask);

      if (!H || H.empty()) {
        return null;
      }

      // Count inliers
      let inliers = 0;
      if (inlierMask && inlierMask.data) {
        for (let i = 0; i < inlierMask.data.length; i++) {
          if (inlierMask.data[i]) inliers++;
        }
      }

      // Require enough inliers to be confident
      if (inliers < 12) {
        return null;
      }

      const { w, h } = this._markerSize;
      const markerCorners = cv.matFromArray(4, 1, cv.CV_32FC2, [
        0, 0,
        w, 0,
        w, h,
        0, h
      ]);

      projected = new cv.Mat();
      cv.perspectiveTransform(markerCorners, projected, H);
      markerCorners.delete();

      const corners = [];
      for (let i = 0; i < 4; i++) {
        corners.push({
          x: projected.data32F[i * 2],
          y: projected.data32F[i * 2 + 1]
        });
      }

      return { corners, homography: H, inliers };
    } catch (e) {
      // Avoid propagating raw numeric OpenCV exceptions
      return null;
    } finally {
      if (projected && projected.delete) projected.delete();
      if (inlierMask && inlierMask.delete) inlierMask.delete();
      if (srcMat && srcMat.delete) srcMat.delete();
      if (dstMat && dstMat.delete) dstMat.delete();
      if (matches && matches.delete) matches.delete();
      if (H && H.delete) H.delete();

      if (emptyMask && emptyMask.delete) emptyMask.delete();
      if (frameKp && frameKp.delete) frameKp.delete();
      if (frameDesc && frameDesc.delete) frameDesc.delete();
    }
  }

  _createORB() {
    const cv = this.cv;
    // OpenCV.js differs by build; try a couple forms.
    try {
      // Many builds support cv.ORB()
      // eslint-disable-next-line new-cap
      return new cv.ORB();
    } catch (_) {
      // ignore
    }
    if (cv.ORB && typeof cv.ORB.create === 'function') {
      return cv.ORB.create();
    }
    throw new Error('MarkerTracker: ORB is not available in this OpenCV.js build');
  }

  _loadImage(url) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => resolve(img);
      img.onerror = (e) => reject(new Error(`Failed to load marker image: ${url}`));
      img.src = url;
    });
  }
}

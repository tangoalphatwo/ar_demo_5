// marker_tracker.js

export class MarkerTracker {
  /**
   * @param {any} cv OpenCV.js instance
   */
  constructor(cv) {
    this.cv = cv;

    this._templateGray = null;
    this._templateKeypoints = null;
    this._templateDescriptors = null;
    this._templateW = 0;
    this._templateH = 0;

    // ORB + brute-force matcher (Hamming)
    this._orb = null;
    this._matcher = null;

    this._initCvObjects();
  }

  _initCvObjects() {
    const cv = this.cv;
    // ORB factory differs across OpenCV.js builds
    if (cv.ORB_create) {
      this._orb = cv.ORB_create(800);
    } else if (cv.ORB) {
      this._orb = new cv.ORB(800);
    } else {
      throw new Error('ORB not available in this OpenCV.js build');
    }

    if (cv.BFMatcher) {
      this._matcher = new cv.BFMatcher(cv.NORM_HAMMING, false);
    } else {
      throw new Error('BFMatcher not available in this OpenCV.js build');
    }
  }

  /**
   * Loads the marker template and precomputes ORB features.
   * @param {string} url
   */
  async loadTemplate(url) {
    const cv = this.cv;

    const imgEl = await new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = (e) => reject(new Error(`Failed to load marker template: ${url}`));
      img.src = url;
    });

    this._templateW = imgEl.naturalWidth || imgEl.width;
    this._templateH = imgEl.naturalHeight || imgEl.height;

    const off = document.createElement('canvas');
    off.width = this._templateW;
    off.height = this._templateH;
    const ctx = off.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(imgEl, 0, 0);

    const imageData = ctx.getImageData(0, 0, off.width, off.height);
    const rgba = cv.matFromImageData(imageData);

    const gray = new cv.Mat();
    cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY);
    rgba.delete();

    const keypoints = new cv.KeyPointVector();
    const descriptors = new cv.Mat();
    this._orb.detectAndCompute(gray, new cv.Mat(), keypoints, descriptors);

    // Replace any existing template mats
    this._disposeTemplate();
    this._templateGray = gray;
    this._templateKeypoints = keypoints;
    this._templateDescriptors = descriptors;
  }

  _disposeTemplate() {
    if (this._templateGray) this._templateGray.delete();
    if (this._templateKeypoints) this._templateKeypoints.delete();
    if (this._templateDescriptors) this._templateDescriptors.delete();
    this._templateGray = null;
    this._templateKeypoints = null;
    this._templateDescriptors = null;
  }

  dispose() {
    this._disposeTemplate();
    if (this._orb) this._orb.delete?.();
    if (this._matcher) this._matcher.delete?.();
    this._orb = null;
    this._matcher = null;
  }

  /**
   * Detect marker corners in the current grayscale frame.
   * @param {any} frameGray cv.Mat (CV_8UC1)
   * @returns {null | { corners: Array<{x:number,y:number}>, homography: any }}
   */
  detect(frameGray) {
    const cv = this.cv;
    if (!this._templateDescriptors || this._templateDescriptors.empty()) return null;

    const frameKeypoints = new cv.KeyPointVector();
    const frameDescriptors = new cv.Mat();
    this._orb.detectAndCompute(frameGray, new cv.Mat(), frameKeypoints, frameDescriptors);

    if (frameDescriptors.empty() || frameKeypoints.size() < 10) {
      frameKeypoints.delete();
      frameDescriptors.delete();
      return null;
    }

    const matches = new cv.DMatchVectorVector();
    this._matcher.knnMatch(frameDescriptors, this._templateDescriptors, matches, 2);

    const good = [];
    const ratio = 0.75;
    for (let i = 0; i < matches.size(); i++) {
      const m = matches.get(i);
      if (m.size() < 2) {
        m.delete();
        continue;
      }
      const m0 = m.get(0);
      const m1 = m.get(1);
      if (m0.distance < ratio * m1.distance) {
        good.push(m0);
      } else {
        m0.delete();
      }
      m1.delete();
      m.delete();
    }
    matches.delete();

    if (good.length < 12) {
      for (const gm of good) gm.delete();
      frameKeypoints.delete();
      frameDescriptors.delete();
      return null;
    }

    // Build point correspondences: template (src) -> frame (dst)
    const src = [];
    const dst = [];

    for (const m of good) {
      const qIdx = m.queryIdx; // in frame
      const tIdx = m.trainIdx; // in template

      const kpFrame = frameKeypoints.get(qIdx).pt;
      const kpTpl = this._templateKeypoints.get(tIdx).pt;

      src.push(kpTpl.x, kpTpl.y);
      dst.push(kpFrame.x, kpFrame.y);

      m.delete();
    }

    const srcMat = cv.matFromArray(src.length / 2, 1, cv.CV_32FC2, src);
    const dstMat = cv.matFromArray(dst.length / 2, 1, cv.CV_32FC2, dst);
    const mask = new cv.Mat();

    const H = cv.findHomography(srcMat, dstMat, cv.RANSAC, 3, mask);

    srcMat.delete();
    dstMat.delete();
    mask.delete();

    frameKeypoints.delete();
    frameDescriptors.delete();

    if (!H || H.empty?.() || H.rows !== 3 || H.cols !== 3) {
      H?.delete?.();
      return null;
    }

    // Project template corners into the current frame
    const w = this._templateW;
    const h = this._templateH;
    const tplCorners = cv.matFromArray(4, 1, cv.CV_32FC2, [
      0, 0,
      w, 0,
      w, h,
      0, h
    ]);

    const proj = new cv.Mat();
    cv.perspectiveTransform(tplCorners, proj, H);
    tplCorners.delete();

    const corners = [];
    for (let i = 0; i < 4; i++) {
      corners.push({
        x: proj.data32F[i * 2],
        y: proj.data32F[i * 2 + 1]
      });
    }
    proj.delete();

    return { corners, homography: H };
  }
}

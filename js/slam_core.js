// slam_core.js
export class SlamCore {
    constructor(cv, intrinsics = null) {
        this.cv = cv;
        this.intrinsics = intrinsics;

        this.pose = {
            R: cv.Mat.eye(3, 3, cv.CV_64F),
            t: cv.Mat.zeros(3, 1, cv.CV_64F)
        };

        this.initialized = false;
        this.mapPoints = [];
        // 3D points triangulated from matched features (in camera coordinates)
        this.mapPoints3D = [];
        this.rgba = null;
        this.gray = null;
        this.currGray = null;
        this.prevGray = null;
        this.prevPoints = null;

        // Per-frame delta from recoverPose (camera1->camera2)
        this.lastDelta = null;

        // Performance knobs
        this.maxTrackFeatures = 200;
        this.maxPosePoints = 120; // downsample correspondences used for essential/recoverPose

        // Triangulation is debug-only (and not available in some OpenCV.js builds)
        this.enableTriangulationDebug = false;
        this._warnedNoTriangulate = false;
    }

    _ensureMats(width, height) {
        const cv = this.cv;
        const needsResize = !this.currGray || this.currGray.rows !== height || this.currGray.cols !== width;
        if (!needsResize) return;

        // Reset tracking state when frame size changes.
        this.initialized = false;
        try { this.prevPoints?.delete?.(); } catch {}
        this.prevPoints = null;

        // Free old buffers (if any)
        try { this.rgba?.delete?.(); } catch {}
        try { this.gray?.delete?.(); } catch {}
        try { this.currGray?.delete?.(); } catch {}
        try { this.prevGray?.delete?.(); } catch {}

        // Persistent per-frame buffers
        this.rgba = new cv.Mat(height, width, cv.CV_8UC4);
        this.gray = new cv.Mat(height, width, cv.CV_8UC1);
        this.currGray = new cv.Mat(height, width, cv.CV_8UC1);
        this.prevGray = new cv.Mat(height, width, cv.CV_8UC1);
    }

    processFrame(imageData) {
        const cv = this.cv;

        // default: no delta this frame
        this.lastDelta = null;

        const width = imageData.width;
        const height = imageData.height;

        this._ensureMats(width, height);

        // Avoid cv.matFromImageData() allocations: reuse a persistent RGBA Mat.
        this.rgba.data.set(imageData.data);
        cv.cvtColor(this.rgba, this.currGray, cv.COLOR_RGBA2GRAY);

        if (!this.initialized) {
            // Swap buffers so prevGray always holds the last-completed frame.
            const tmp = this.prevGray;
            this.prevGray = this.currGray;
            this.currGray = tmp;

            if (this.prevPoints) this.prevPoints.delete();
            this.prevPoints = this._detectFeatures(this.prevGray);
            this.initialized = true;

            return { pose: this.pose, mapPoints: this.mapPoints };
        }

        const { currPoints, status } = this._trackFeatures(
            this.prevGray,
            this.currGray,
            this.prevPoints
        );

        const goodPrevAll = [];
        const goodCurrAll = [];

        for (let i = 0; i < status.rows; i++) {
            if (status.data[i] === 1) {
                goodPrevAll.push(
                    this.prevPoints.data32F[2 * i],
                    this.prevPoints.data32F[2 * i + 1]
                );
                goodCurrAll.push(
                    currPoints.data32F[2 * i],
                    currPoints.data32F[2 * i + 1]
                );
            }
        }

        // Done with LK output mats (we copy data into JS arrays above)
        currPoints.delete();
        status.delete();

        if (goodPrevAll.length < 16) {
            if (this.prevPoints) this.prevPoints.delete();

            const tmp = this.prevGray;
            this.prevGray = this.currGray;
            this.currGray = tmp;

            this.prevPoints = this._detectFeatures(this.prevGray);
            return { pose: this.pose, mapPoints: this.mapPoints };
        }

        // Downsample points used for essential/recoverPose to improve performance.
        // Keep LK tracking state using all surviving points.
        const nGoodAll = goodPrevAll.length / 2;
        let goodPrevPose = goodPrevAll;
        let goodCurrPose = goodCurrAll;

        if (nGoodAll > this.maxPosePoints) {
            const stride = Math.ceil(nGoodAll / this.maxPosePoints);
            const prevTmp = [];
            const currTmp = [];
            for (let i = 0; i < nGoodAll; i += stride) {
                prevTmp.push(goodPrevAll[i * 2], goodPrevAll[i * 2 + 1]);
                currTmp.push(goodCurrAll[i * 2], goodCurrAll[i * 2 + 1]);
            }
            goodPrevPose = prevTmp;
            goodCurrPose = currTmp;
        }

        const nPose = goodPrevPose.length / 2;
        // For essential matrix / recoverPose, use a conservative, widely-supported format: Nx2 CV_64F
        const prevMat = cv.matFromArray(nPose, 2, cv.CV_64F, goodPrevPose);
        const currMat = cv.matFromArray(nPose, 2, cv.CV_64F, goodCurrPose);
        // For LK state into the next frame, preserve the typical Nx1 CV_32FC2 format
        let currMatFlow = cv.matFromArray(nGoodAll, 1, cv.CV_32FC2, goodCurrAll);

        const fx = this.intrinsics?.fx ?? 600;
        const fy = this.intrinsics?.fy ?? 600;
        const cx = this.intrinsics?.cx ?? (width / 2);
        const cy = this.intrinsics?.cy ?? (height / 2);

        const K = cv.matFromArray(3, 3, cv.CV_64F, [
            fx, 0, cx,
            0, fy, cy,
            0,  0,  1
        ]);

        const mask = new cv.Mat();
        let E = null;
        if (cv.findEssentialMat) {
            E = cv.findEssentialMat(prevMat, currMat, K, cv.RANSAC, 0.999, 1.0, mask);
        } else if (cv.findFundamentalMat) {
            // Fallback: compute Fundamental matrix then convert to Essential: E = K^T * F * K
            const F = cv.findFundamentalMat(prevMat, currMat, cv.FM_RANSAC, 3, 0.99, mask);
            try {
                const Kt = new cv.Mat();
                cv.transpose(K, Kt);
                const tmp = new cv.Mat();
                const empty1 = new cv.Mat();
                cv.gemm(Kt, F, 1, empty1, 0, tmp);
                empty1.delete();
                E = new cv.Mat();
                const empty2 = new cv.Mat();
                cv.gemm(tmp, K, 1, empty2, 0, E);
                empty2.delete();
                tmp.delete();
                Kt.delete();
            } catch (err) {
                console.warn('Failed to compute E from F fallback:', err);
                if (F) F.delete();
                throw err;
            }
            if (F) F.delete();
        } else {
            throw new Error('Neither cv.findEssentialMat nor cv.findFundamentalMat available in OpenCV.js build');
        }

        const R = new cv.Mat();
        const t = new cv.Mat();

        if (E) {
            try {
                // OpenCV.js recoverPose overloads vary by build.
                // Many builds expose overload-specific wrappers: recoverPose1, recoverPose2, ...
                // Prefer those when available to avoid argument-type ambiguity.
                const triangulatedPoints = new cv.Mat();

                const matInfo = (m) => (m && typeof m.rows === 'number')
                    ? { rows: m.rows, cols: m.cols, type: (typeof m.type === 'function' ? m.type() : undefined) }
                    : null;

                try {
                    if (typeof cv.recoverPose1 === 'function') {
                        // Expected: (E, points1, points2, K, R, t, mask)
                        cv.recoverPose1(E, prevMat, currMat, K, R, t, mask);
                    } else if (typeof cv.recoverPose2 === 'function') {
                        // Expected: (E, points1, points2, K, R, t, mask, triangulatedPoints)
                        cv.recoverPose2(E, prevMat, currMat, K, R, t, mask, triangulatedPoints);
                    } else {
                        // Fallback: attempt the base symbol with a 9-arg variant.
                        // NOTE: we do not pass a numeric distance threshold here; some builds interpret
                        // that slot as a Mat and throw "Cannot pass '3' as a Mat".
                        const tmp = new cv.Mat();
                        try {
                            cv.recoverPose(E, prevMat, currMat, K, R, t, mask, triangulatedPoints, tmp);
                        } catch (e1) {
                            console.warn('recoverPose wrapper missing; base recoverPose failed:', e1, {
                                E: matInfo(E),
                                points1: matInfo(prevMat),
                                points2: matInfo(currMat),
                                K: matInfo(K),
                                R: matInfo(R),
                                t: matInfo(t),
                                mask: matInfo(mask),
                                triangulatedPoints: matInfo(triangulatedPoints)
                            });
                            throw e1;
                        } finally {
                            tmp.delete();
                        }
                    }
                } catch (e) {
                    console.warn('recoverPose overload threw:', e, {
                        E: matInfo(E),
                        points1: matInfo(prevMat),
                        points2: matInfo(currMat),
                        K: matInfo(K),
                        R: matInfo(R),
                        t: matInfo(t),
                        mask: matInfo(mask),
                        triangulatedPoints: matInfo(triangulatedPoints)
                    });
                    throw e;
                } finally {
                    triangulatedPoints.delete();
                }

                // Capture delta as plain JS arrays for the caller
                const rArr = (R.data64F && Array.from(R.data64F)) || (R.data32F && Array.from(R.data32F)) || (R.data && Array.from(R.data));
                const tArr = (t.data64F && Array.from(t.data64F)) || (t.data32F && Array.from(t.data32F)) || (t.data && Array.from(t.data));
                if (rArr && tArr && rArr.length >= 9 && tArr.length >= 3) {
                    this.lastDelta = { R: rArr.slice(0, 9), t: tArr.slice(0, 3) };
                }

                const newPoseR = R.mul(this.pose.R);
                if (this.pose.R) this.pose.R.delete();
                this.pose.R = newPoseR;

                const newPoseT = this._addTrans(this.pose.t, t);
                if (this.pose.t) this.pose.t.delete();
                this.pose.t = newPoseT;

                // Triangulation is debug-only. Some OpenCV.js builds don't expose triangulatePoints,
                // and throwing/logging every frame tanks performance.
                this.mapPoints3D = [];
                if (this.enableTriangulationDebug) {
                    if (typeof cv.triangulatePoints !== 'function') {
                        if (!this._warnedNoTriangulate) {
                            this._warnedNoTriangulate = true;
                            console.warn('Triangulation disabled: cv.triangulatePoints not available in this OpenCV.js build');
                        }
                    } else {
                        try {
                            if (goodPrevPose.length >= 16) {
                                const n = goodPrevPose.length / 2;

                                // build point arrays: xs then ys (2 x N mats)
                                const xs1 = [], ys1 = [], xs2 = [], ys2 = [];
                                for (let i = 0; i < goodPrevPose.length; i += 2) {
                                    xs1.push(goodPrevPose[i]); ys1.push(goodPrevPose[i + 1]);
                                    xs2.push(goodCurrPose[i]); ys2.push(goodCurrPose[i + 1]);
                                }

                                const pts1 = cv.matFromArray(2, n, cv.CV_64F, xs1.concat(ys1));
                                const pts2 = cv.matFromArray(2, n, cv.CV_64F, xs2.concat(ys2));

                                const fx = this.intrinsics?.fx ?? 600;
                                const fy = this.intrinsics?.fy ?? 600;
                                const cx = this.intrinsics?.cx ?? (width / 2);
                                const cy = this.intrinsics?.cy ?? (height / 2);
                                const Kmat = cv.matFromArray(3, 3, cv.CV_64F, [
                                    fx, 0, cx,
                                    0, fy, cy,
                                    0, 0, 1
                                ]);

                                // P1 = K * [I|0]
                                const P1 = new cv.Mat(3, 4, cv.CV_64F);
                                for (let r = 0; r < 3; r++) {
                                    for (let c = 0; c < 3; c++) {
                                        P1.doublePtr(r, c)[0] = Kmat.doublePtr(r, c)[0];
                                    }
                                    P1.doublePtr(r, 3)[0] = 0;
                                }

                                // Rt = [R|t]
                                const Rt = new cv.Mat(3, 4, cv.CV_64F);
                                for (let r = 0; r < 3; r++) {
                                    for (let c = 0; c < 3; c++) {
                                        Rt.doublePtr(r, c)[0] = R.doublePtr(r, c)[0];
                                    }
                                    Rt.doublePtr(r, 3)[0] = t.doublePtr(r, 0)[0];
                                }

                                const P2 = new cv.Mat();
                                const empty3 = new cv.Mat();
                                cv.gemm(Kmat, Rt, 1, empty3, 0, P2);
                                empty3.delete();

                                const points4D = new cv.Mat();
                                cv.triangulatePoints(P1, P2, pts1, pts2, points4D);

                                for (let i = 0; i < n; i++) {
                                    const w = points4D.doublePtr(3, i)[0];
                                    if (Math.abs(w) < 1e-8) continue;
                                    const X = points4D.doublePtr(0, i)[0] / w;
                                    const Y = points4D.doublePtr(1, i)[0] / w;
                                    const Z = points4D.doublePtr(2, i)[0] / w;
                                    if (!isFinite(X) || !isFinite(Y) || !isFinite(Z)) continue;
                                    if (Math.abs(X) > 100 || Math.abs(Y) > 100 || Math.abs(Z) > 100) continue;
                                    this.mapPoints3D.push({ X, Y, Z });
                                }

                                points4D.delete();
                                P1.delete(); Rt.delete(); P2.delete(); pts1.delete(); pts2.delete(); Kmat.delete();
                            }
                        } catch {
                            // ignore debug triangulation failures
                        }
                    }
                }

                if (this.prevPoints) this.prevPoints.delete();

                const tmp = this.prevGray;
                this.prevGray = this.currGray;
                this.currGray = tmp;

                // Adopt currMatFlow as next-frame LK input (avoid clone)
                this.prevPoints = currMatFlow;
                currMatFlow = null;

                // Free per-frame pose mats after use
                R.delete();
                t.delete();

            } catch (err) {
                console.warn('recoverPose failed:', err);
                // Keep previous pose and skip triangulation for this frame
                this.mapPoints3D = [];
                if (this.prevPoints) this.prevPoints.delete();

                const tmp = this.prevGray;
                this.prevGray = this.currGray;
                this.currGray = tmp;

                this.prevPoints = currMatFlow;
                currMatFlow = null;

                // Free per-frame pose mats on failure too
                R.delete();
                t.delete();
            }

            if (E && E.delete) E.delete();
        } else {
            // No essential matrix computed; skip pose update and triangulation this frame
            console.warn('Skipping pose recovery: no essential/fundamental computation available for this frame');
            this.mapPoints3D = [];
            if (this.prevPoints) this.prevPoints.delete();

            const tmp = this.prevGray;
            this.prevGray = this.currGray;
            this.currGray = tmp;

            this.prevPoints = currMatFlow;
            currMatFlow = null;

            R.delete();
            t.delete();
        }

        if (mask && mask.delete) mask.delete();
        if (prevMat && prevMat.delete) prevMat.delete();
        if (currMat && currMat.delete) currMat.delete();
        if (currMatFlow && currMatFlow.delete) currMatFlow.delete();
        if (K && K.delete) K.delete();

        return { pose: this.pose, mapPoints: this.mapPoints, mapPoints3D: this.mapPoints3D, delta: this.lastDelta };
    }

    _addTrans(t1, t2) {
        const cv = this.cv;
        const out = new cv.Mat(3, 1, cv.CV_64F);

        for (let i = 0; i < 3; i++) {
            out.doublePtr(i, 0)[0] =
                t1.doublePtr(i, 0)[0] + t2.doublePtr(i, 0)[0];
        }
        return out;
    }

    _detectFeatures(gray) {
      const cv = this.cv;
      const corners = new cv.Mat();
    
      cv.goodFeaturesToTrack(gray, corners, this.maxTrackFeatures, 0.01, 10);
    
      // DEBUG: expose 2D points for rendering
      this.mapPoints = [];
      for (let i = 0; i < corners.rows; i++) {
        const x = corners.data32F[i * 2];
        const y = corners.data32F[i * 2 + 1];
        this.mapPoints.push({ x, y });
      }
    
      return corners;
    }

    _trackFeatures(prevGray, currGray, prevPoints) {
        const cv = this.cv;
        const currPoints = new cv.Mat();
        const status = new cv.Mat();
        const err = new cv.Mat();

        cv.calcOpticalFlowPyrLK(
            prevGray,
            currGray,
            prevPoints,
            currPoints,
            status,
            err
        );

        err.delete();
        return { currPoints, status };
    }
}
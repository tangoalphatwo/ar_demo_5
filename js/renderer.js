// renderer.js
import * as THREE from 'https://esm.sh/three@0.160.0';
import { GLTFLoader } from 'https://esm.sh/three@0.160.0/examples/jsm/loaders/GLTFLoader.js';

export class ARRenderer {
  constructor(canvasEl) {
    this.canvas = canvasEl;

    this.scene = new THREE.Scene();

    // World/root group. Keep models parented here at world origin.
    this.world = new THREE.Group();
    this.scene.add(this.world);

    // Fallback FOV; call setProjectionFromVideo() after camera starts for better alignment.
    const fov = 80;
    const w0 = canvasEl.clientWidth || window.innerWidth || 1;
    const h0 = canvasEl.clientHeight || window.innerHeight || 1;
    const aspect = w0 / h0;
    this.camera = new THREE.PerspectiveCamera(fov, aspect, 0.001, 500);
    this.camera.position.set(0, 0, 0);

    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      alpha: true,
      premultipliedAlpha: false
    });
    const pr = Math.max(1, Math.min(3, window.devicePixelRatio || 1));
    this.renderer.setPixelRatio(pr);
    this.renderer.setSize(w0, h0, false);
    if ('SRGBColorSpace' in THREE) this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    // Ensure the WebGL canvas stays transparent so the underlying camera canvas shows through.
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.autoClear = true;

    // Ambient light
    this.scene.add(new THREE.AmbientLight(0xffffff, 1.0));

    this._projectionInfo = null;
  }

  async loadGLB(url) {
    const loader = new GLTFLoader();
    return new Promise((resolve, reject) => {
      loader.load(
        url,
        (gltf) => resolve(gltf),
        undefined,
        (err) => reject(err)
      );
    });
  }

  clearWorld() {
    while (this.world.children.length) {
      this.world.remove(this.world.children[0]);
    }
  }

  _getBounds(object3d) {
    if (!object3d) return null;
    object3d.updateWorldMatrix(true, true);
    const box = new THREE.Box3().setFromObject(object3d);
    const size = new THREE.Vector3();
    const center = new THREE.Vector3();
    box.getSize(size);
    box.getCenter(center);
    return { box, size, center };
  }

  /**
   * Adds a model at world origin (marker center).
   * - Scales so its height is ~targetHeightM (meters), if provided.
   * - Centers in X/Z and places its bottom on y=0.
   */
  addModelAtWorldZero(object3d, { targetHeightM } = {}) {
    if (!object3d) throw new Error('addModelAtWorldZero: object3d is required');

    this.clearWorld();

    const before = this._getBounds(object3d);
    if (!before) throw new Error('addModelAtWorldZero: could not compute bounds');

    // Some GLBs come in with a different "up" axis, so size.y can be tiny.
    // To avoid accidental huge scaling, choose a stable reference dimension.
    const maxDim = Math.max(before.size.x, before.size.y, before.size.z);
    const yDim = before.size.y;
    const refDim = (isFinite(yDim) && yDim > 1e-6 && yDim >= maxDim * 0.2) ? yDim : maxDim;

    let scaleApplied = 1;
    if (isFinite(targetHeightM) && targetHeightM > 0 && isFinite(refDim) && refDim > 1e-6) {
      scaleApplied = targetHeightM / refDim;
      object3d.scale.multiplyScalar(scaleApplied);
    }

    // After scaling, re-evaluate bounds for placement.
    const afterScale = this._getBounds(object3d);
    if (!afterScale) throw new Error('addModelAtWorldZero: could not compute bounds after scaling');

    // Center horizontally (x/z) around origin.
    object3d.position.x -= afterScale.center.x;
    object3d.position.z -= afterScale.center.z;

    // Put base on y=0.
    object3d.updateWorldMatrix(true, true);
    const afterCenter = this._getBounds(object3d);
    if (afterCenter) {
      object3d.position.y -= afterCenter.box.min.y;
    }

    object3d.updateWorldMatrix(true, true);
    const finalBounds = this._getBounds(object3d);

    this.world.add(object3d);

    return {
      sizeBefore: { x: before.size.x, y: before.size.y, z: before.size.z },
      sizeAfter: finalBounds ? { x: finalBounds.size.x, y: finalBounds.size.y, z: finalBounds.size.z } : null,
      scaleApplied
    };
  }

  // Align Three camera FOV with our (approximate) OpenCV intrinsics.
  // In pose.js we use focalLengthPx = videoWidthPx.
  setProjectionFromVideo({ videoWidthPx, videoHeightPx, focalLengthPx } = {}) {
    const w = Number(videoWidthPx);
    const h = Number(videoHeightPx);
    const f = Number(focalLengthPx);
    if (!isFinite(w) || !isFinite(h) || !isFinite(f) || w <= 0 || h <= 0 || f <= 0) return;

    // fovY = 2 * atan( (h/2) / fy )
    const fovRad = 2 * Math.atan((h * 0.5) / f);
    const fovDeg = (fovRad * 180) / Math.PI;

    if (isFinite(fovDeg) && fovDeg > 10 && fovDeg < 170) {
      this.camera.fov = fovDeg;
      this.camera.updateProjectionMatrix();
      this._projectionInfo = { videoWidthPx: w, videoHeightPx: h, focalLengthPx: f, fovDeg };
      console.log('[Three] Projection updated from video:', this._projectionInfo);
    }
  }

  // Preferred AR path: treat the marker as world origin and move the camera.
  // pose is marker->camera (OpenCV solvePnP), so camera->world is its inverse.
  setCameraFromMarkerPose(pose) {
    if (!pose || !pose.position || !pose.rotationMatrix) {
      return;
    }

    const r = pose.rotationMatrix;
    const r00 = r[0], r01 = r[1], r02 = r[2];
    const r10 = r[3], r11 = r[4], r12 = r[5];
    const r20 = r[6], r21 = r[7], r22 = r[8];

    // Build T_cw (world/marker -> camera) in Three coordinate conventions.
    // OpenCV camera coords: x right, y down, z forward.
    // Three camera coords: x right, y up, z backward.
    // Convert by flipping Y and Z axes (S = diag(1,-1,-1)).
    const tcw = new THREE.Vector3(pose.position.x, pose.position.y, -pose.position.z);
    const Tcw = new THREE.Matrix4();
    Tcw.set(
      r00, -r01, -r02, tcw.x,
      -r10, r11, r12, tcw.y,
      -r20, r21, r22, tcw.z,
      0, 0, 0, 1
    );

    // Camera world matrix is the inverse of world->camera.
    const Twc = Tcw.clone().invert();

    this.camera.matrixAutoUpdate = false;
    this.camera.matrix.copy(Twc);
    this.camera.matrix.decompose(this.camera.position, this.camera.quaternion, this.camera.scale);
  }

  render() {
    // Force a full clear + viewport each frame to avoid seam/scanline artifacts on some mobile GPUs.
    const w = this.canvas.clientWidth || window.innerWidth || 1;
    const h = this.canvas.clientHeight || window.innerHeight || 1;
    this.renderer.setViewport(0, 0, w, h);
    this.renderer.clear(true, true, true);
    this.renderer.render(this.scene, this.camera);
  }

  resize() {
    const w = this.canvas.clientWidth || window.innerWidth || 1;
    const h = this.canvas.clientHeight || window.innerHeight || 1;
    const pr = Math.max(1, Math.min(3, window.devicePixelRatio || 1));
    this.renderer.setPixelRatio(pr);
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }
}

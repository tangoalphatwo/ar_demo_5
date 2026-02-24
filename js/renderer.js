// renderer.js
import * as THREE from 'https://esm.sh/three@0.160.0';
import { GLTFLoader } from 'https://esm.sh/three@0.160.0/examples/jsm/loaders/GLTFLoader.js';

export class ARRenderer {
  constructor(canvasEl) {
    this.canvas = canvasEl;

    this.scene = new THREE.Scene();

    const fov = 60;
    const w0 = canvasEl.clientWidth || window.innerWidth || 1;
    const h0 = canvasEl.clientHeight || window.innerHeight || 1;
    const aspect = w0 / h0;
    this.camera = new THREE.PerspectiveCamera(fov, aspect, 0.01, 100);
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

    // Video background quad
    this.videoTexture = null;
    this.bgMesh = null;

    // Anchored AR content
    this.anchor = new THREE.Group();
    this.anchor.visible = false;
    this.scene.add(this.anchor);

    this.model = null;
  }

  setVideoTexture(video) {
    if (!video) return;

    const create = () => {
      // If the video doesn't have a current frame yet, defer.
      if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || video.videoWidth === 0 || video.videoHeight === 0) {
        return false;
      }

      const texture = new THREE.VideoTexture(video);
      texture.minFilter = THREE.LinearFilter;
      texture.magFilter = THREE.LinearFilter;
      texture.generateMipmaps = false;

      // Avoid forcing texture.format (can trigger texImage2D errors on some browsers).
      // Hint correct color handling when supported.
      if ('SRGBColorSpace' in THREE) texture.colorSpace = THREE.SRGBColorSpace;

      // Most robust: Three handles the full-viewport background.
      this.scene.background = texture;

      this.bgMesh = null;
      this.videoTexture = texture;
      return true;
    };

    if (create()) return;

    const onReady = () => {
      if (create()) {
        video.removeEventListener('loadeddata', onReady);
        video.removeEventListener('canplay', onReady);
        video.removeEventListener('playing', onReady);
      }
    };

    video.addEventListener('loadeddata', onReady, { once: false });
    video.addEventListener('canplay', onReady, { once: false });
    video.addEventListener('playing', onReady, { once: false });
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

  computeBoundingSize(object3d) {
    if (!object3d) return null;
    const box = new THREE.Box3().setFromObject(object3d);
    const size = new THREE.Vector3();
    box.getSize(size);
    return { x: size.x, y: size.y, z: size.z };
  }

  setAnchorPose(pose) {
    if (!pose || !pose.position || !pose.rotationMatrix) {
      this.anchor.visible = false;
      return;
    }

    // pose.rotationMatrix is row-major 3x3 from OpenCV
    const r = pose.rotationMatrix;

    // Convert OpenCV camera coords (x right, y down, z forward)
    // to Three camera coords (x right, y up, z backward): S = diag(1,-1,-1)
    const r00 = r[0], r01 = r[1], r02 = r[2];
    const r10 = r[3], r11 = r[4], r12 = r[5];
    const r20 = r[6], r21 = r[7], r22 = r[8];

    const Rthree = new THREE.Matrix4();
    Rthree.set(
      r00, -r01, -r02, 0,
      -r10, r11, r12, 0,
      -r20, r21, r22, 0,
      0, 0, 0, 1
    );

    const q = new THREE.Quaternion();
    q.setFromRotationMatrix(Rthree);

    this.anchor.position.set(
      pose.position.x,
      pose.position.y,
      -pose.position.z
    );
    this.anchor.quaternion.copy(q);
    this.anchor.visible = true;
  }

  // Preferred AR path: treat the marker as world origin and move the camera.
  // pose is marker->camera (OpenCV solvePnP), so camera->world is its inverse.
  setCameraFromMarkerPose(pose) {
    if (!pose || !pose.position || !pose.rotationMatrix) {
      this.anchor.visible = false;
      return;
    }

    const r = pose.rotationMatrix;
    const r00 = r[0], r01 = r[1], r02 = r[2];
    const r10 = r[3], r11 = r[4], r12 = r[5];
    const r20 = r[6], r21 = r[7], r22 = r[8];

    // Convert OpenCV to Three coordinates: R_three = S * R_cv * S, S=diag(1,-1,-1)
    const Rcw = new THREE.Matrix3();
    Rcw.set(
      r00, -r01, -r02,
      -r10, r11, r12,
      -r20, r21, r22
    );

    // t_three (world->camera) from pose.position (which already flipped Y in pose.js)
    const tcw = new THREE.Vector3(pose.position.x, pose.position.y, -pose.position.z);

    // Invert: T_wc = [R_wc | t_wc] where R_wc = R_cw^T, t_wc = -R_cw^T * t_cw
    const Rwc = Rcw.clone().transpose();

    const t_wc = tcw.clone();
    // multiply by Rwc: t' = Rwc * tcw
    const e = Rwc.elements;
    const x = t_wc.x, y = t_wc.y, z = t_wc.z;
    t_wc.set(
      e[0] * x + e[1] * y + e[2] * z,
      e[3] * x + e[4] * y + e[5] * z,
      e[6] * x + e[7] * y + e[8] * z
    );
    t_wc.multiplyScalar(-1);

    // Apply to Three camera (camera is an Object3D transform in world)
    const m = new THREE.Matrix4();
    const re = Rwc.elements;
    m.set(
      re[0], re[1], re[2], t_wc.x,
      re[3], re[4], re[5], t_wc.y,
      re[6], re[7], re[8], t_wc.z,
      0, 0, 0, 1
    );

    this.camera.matrixAutoUpdate = false;
    this.camera.matrix.copy(m);
    this.camera.matrix.decompose(this.camera.position, this.camera.quaternion, this.camera.scale);

    // Keep anchor at marker origin
    this.anchor.position.set(0, 0, 0);
    this.anchor.quaternion.set(0, 0, 0, 1);
    this.anchor.visible = true;
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

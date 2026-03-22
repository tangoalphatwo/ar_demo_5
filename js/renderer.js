// renderer.js
import * as THREE from 'https://esm.sh/three@0.160.0';
import { GLTFLoader } from 'https://esm.sh/three@0.160.0/examples/jsm/loaders/GLTFLoader.js';

export class ARRenderer {
  constructor(canvasEl) {
    this.canvas = canvasEl;

    this.scene = new THREE.Scene();

    const fov = 60;
    const aspect = canvasEl.clientWidth / canvasEl.clientHeight;
    this.camera = new THREE.PerspectiveCamera(fov, aspect, 0.01, 100);
    this.camera.position.set(0, 0, 0);

    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      alpha: true
    });
    this.renderer.setSize(canvasEl.clientWidth, canvasEl.clientHeight, false);

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
    const texture = new THREE.VideoTexture(video);
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;
    texture.format = THREE.RGBFormat;
  
    const geometry = new THREE.PlaneGeometry(2, 2);
    const material = new THREE.MeshBasicMaterial({ map: texture, depthTest: false, depthWrite: false });
    const mesh = new THREE.Mesh(geometry, material);

    // Keep background always behind content
    mesh.renderOrder = -1;
    mesh.frustumCulled = false;
    mesh.material.transparent = true;

    this.bgMesh = mesh;
    this.videoTexture = texture;
  
    this.scene.add(mesh);
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

  applySlamDelta(deltaR, deltaT, scale = 1.0) {
    if (!deltaR || !deltaT) return;

    // OpenCV -> Three conversion
    // S = diag(1, -1, -1)
    const r00 = deltaR[0], r01 = deltaR[1], r02 = deltaR[2];
    const r10 = deltaR[3], r11 = deltaR[4], r12 = deltaR[5];
    const r20 = deltaR[6], r21 = deltaR[7], r22 = deltaR[8];

    const R = new THREE.Matrix3();
    R.set(
      r00, -r01, -r02,
      -r10, r11, r12,
      -r20, r21, r22
    );

    // OpenCV camera coords -> Three camera coords
    const t = new THREE.Vector3(
      deltaT[0] * scale,
      -deltaT[1] * scale,
      -deltaT[2] * scale
    );

    // recoverPose gives relative motion from prev camera to curr camera:
    // X2 = R * X1 + t
    // For a world camera transform, apply the inverse delta:
    // inv(T) = [R^T, -R^T t]
    const Rinv = R.clone().transpose();
    const e = Rinv.elements;

    const tx = t.x, ty = t.y, tz = t.z;
    const tinv = new THREE.Vector3(
      -(e[0] * tx + e[1] * ty + e[2] * tz),
      -(e[3] * tx + e[4] * ty + e[5] * tz),
      -(e[6] * tx + e[7] * ty + e[8] * tz)
    );

    const deltaInv = new THREE.Matrix4();
    deltaInv.set(
      e[0], e[1], e[2], tinv.x,
      e[3], e[4], e[5], tinv.y,
      e[6], e[7], e[8], tinv.z,
      0, 0, 0, 1
    );

    this.camera.matrixAutoUpdate = false;
    this.camera.matrix.multiply(deltaInv);
    this.camera.matrix.decompose(this.camera.position, this.camera.quaternion, this.camera.scale);

    // Keep world origin at the marker anchor
    this.anchor.position.set(0, 0, 0);
    this.anchor.quaternion.set(0, 0, 0, 1);
    this.anchor.visible = true;
  }

  render() {
    this.renderer.render(this.scene, this.camera);
  }

  resize() {
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }
}
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

    // --- World zero + content roots ---
    // Architecture B: keep content in a fixed world frame and move the Three.js camera.
    // We use the marker pose to initialize/refresh the camera world transform, and SLAM
    // deltas to keep updating the camera when the marker is not visible.
    this.worldZeroRoot = new THREE.Group();
    this.worldZeroRoot.name = 'WorldZeroRoot';
    this.worldZeroRoot.visible = true;
    this.scene.add(this.worldZeroRoot);

    this.houseRoot = new THREE.Group();
    this.houseRoot.name = 'HouseRoot';
    this.worldZeroRoot.add(this.houseRoot);

    // Back-compat: existing code references renderer.anchor
    this.anchor = this.worldZeroRoot;

    // Persisted world-zero / house state (camera-centric “world zero”)
    this.worldZeroState = {
      isSet: false,
      cameraMatrixAtZero: new THREE.Matrix4(),
      cameraMatrixAtZeroInverse: new THREE.Matrix4()
    };

    this.model = null;
  }

  // --- Persistence helpers ---
  saveWorldZero() {
    try {
      localStorage.setItem(
        'ar_world_zero',
        JSON.stringify({
          isSet: this.worldZeroState.isSet,
          cameraMatrixAtZero: this.worldZeroState.cameraMatrixAtZero.toArray()
        })
      );
    } catch {
      // ignore storage failures
    }
  }

  loadWorldZero() {
    try {
      const raw = localStorage.getItem('ar_world_zero');
      if (!raw) return false;

      const data = JSON.parse(raw);
      const matrixArr = Array.isArray(data?.cameraMatrixAtZero)
        ? data.cameraMatrixAtZero
        : (Array.isArray(data?.worldZeroMatrixAtZero) ? data.worldZeroMatrixAtZero : null);
      if (!data?.isSet || !matrixArr) return false;

      this.worldZeroState.isSet = true;
      this.worldZeroState.cameraMatrixAtZero.fromArray(matrixArr);
      this.worldZeroState.cameraMatrixAtZeroInverse
        .copy(this.worldZeroState.cameraMatrixAtZero)
        .invert();

      // worldZeroRoot maps world-zero local coords into world coords.
      this.worldZeroRoot.matrixAutoUpdate = false;
      this.worldZeroRoot.matrix.copy(this.worldZeroState.cameraMatrixAtZero);
      this.worldZeroRoot.matrix.decompose(this.worldZeroRoot.position, this.worldZeroRoot.quaternion, this.worldZeroRoot.scale);
      this.worldZeroRoot.updateMatrixWorld(true);
      return true;
    } catch {
      return false;
    }
  }

  saveHousePose() {
    try {
      // Ensure matrix is current
      this.houseRoot.updateMatrix();
      localStorage.setItem(
        'ar_house_pose',
        JSON.stringify({ matrix: this.houseRoot.matrix.toArray() })
      );
    } catch {
      // ignore storage failures
    }
  }

  loadHousePose() {
    try {
      const raw = localStorage.getItem('ar_house_pose');
      if (!raw) return false;
      const data = JSON.parse(raw);
      if (!Array.isArray(data?.matrix)) return false;

      this.houseRoot.matrixAutoUpdate = false;
      this.houseRoot.matrix.fromArray(data.matrix);
      this.houseRoot.matrix.decompose(this.houseRoot.position, this.houseRoot.quaternion, this.houseRoot.scale);
      return true;
    } catch {
      return false;
    }
  }

  restorePersistedHouse() {
    const wz = this.loadWorldZero();
    const hp = this.loadHousePose();
    return { worldZeroLoaded: wz, housePoseLoaded: hp };
  }

  // Save the current tracking frame as "world zero" (camera pose at the moment you press the button).
  setWorldZero() {
    this.camera.updateMatrixWorld(true);

    this.worldZeroState.cameraMatrixAtZero.copy(this.camera.matrixWorld);
    this.worldZeroState.cameraMatrixAtZeroInverse.copy(this.camera.matrixWorld).invert();
    this.worldZeroState.isSet = true;

    // worldZeroRoot maps world-zero local coords into world coords.
    this.worldZeroRoot.matrixAutoUpdate = false;
    this.worldZeroRoot.matrix.copy(this.worldZeroState.cameraMatrixAtZero);
    this.worldZeroRoot.matrix.decompose(this.worldZeroRoot.position, this.worldZeroRoot.quaternion, this.worldZeroRoot.scale);
    this.worldZeroRoot.updateMatrixWorld(true);

    this.saveWorldZero();
  }

  placeHouseAtCurrentWorldPose(worldMatrix) {
    if (!this.worldZeroState.isSet) return;
    if (!worldMatrix) return;

    const localToWorldZero = new THREE.Matrix4()
      .copy(this.worldZeroState.cameraMatrixAtZeroInverse)
      .multiply(worldMatrix);

    this.houseRoot.matrixAutoUpdate = false;
    this.houseRoot.matrix.copy(localToWorldZero);
    this.houseRoot.matrix.decompose(this.houseRoot.position, this.houseRoot.quaternion, this.houseRoot.scale);

    this.saveHousePose();
  }

  placeHouseFromCameraRelativePose(cameraRelativeMatrix) {
    if (!this.worldZeroState.isSet) return;
    if (!cameraRelativeMatrix) return;

    this.camera.updateMatrixWorld(true);

    const worldMatrix = new THREE.Matrix4()
      .copy(this.camera.matrixWorld)
      .multiply(cameraRelativeMatrix);

    this.placeHouseAtCurrentWorldPose(worldMatrix);
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
    // Back-compat alias: older code called setAnchorPose with a marker pose.
    // In Architecture B, that marker pose drives the camera.
    this.setCameraFromMarkerPose(pose);
  }

  // Preferred AR path: treat the marker as world origin and move the camera.
  // pose is marker->camera (OpenCV solvePnP), so camera->world is its inverse.
  setCameraFromMarkerPose(pose) {
    if (!pose || !pose.position || !pose.rotationMatrix) {
      return;
    }

    // pose.rotationMatrix is row-major 3x3 from OpenCV (world/marker -> camera).
    // Convert OpenCV camera coords (x right, y down, z forward)
    // to Three camera coords (x right, y up, z backward): S = diag(1,-1,-1).
    // Build T_cw in Three coords then invert to get T_wc.
    const r = pose.rotationMatrix;
    const r00 = r[0], r01 = r[1], r02 = r[2];
    const r10 = r[3], r11 = r[4], r12 = r[5];
    const r20 = r[6], r21 = r[7], r22 = r[8];

    const tcw = new THREE.Vector3(pose.position.x, pose.position.y, -pose.position.z);

    const Tcw = new THREE.Matrix4();
    Tcw.set(
      r00, -r01, -r02, tcw.x,
      -r10, r11, r12, tcw.y,
      -r20, r21, r22, tcw.z,
      0, 0, 0, 1
    );

    const Twc = Tcw.clone().invert();

    this.camera.matrixAutoUpdate = false;
    this.camera.matrix.copy(Twc);
    this.camera.matrix.decompose(this.camera.position, this.camera.quaternion, this.camera.scale);
  }

  applySlamDelta(deltaR, deltaT, scale = 1.0) {
    if (!deltaR || !deltaT) return;

    // recoverPose gives relative motion from prev camera to curr camera:
    // X2 = R * X1 + t (camera1 -> camera2).
    // For a world camera transform Twc, apply the inverse delta: Twc2 = Twc1 * inv(T21).
    // Build T21 in Three camera coordinates then invert.
    const r00 = deltaR[0], r01 = deltaR[1], r02 = deltaR[2];
    const r10 = deltaR[3], r11 = deltaR[4], r12 = deltaR[5];
    const r20 = deltaR[6], r21 = deltaR[7], r22 = deltaR[8];

    const t = new THREE.Vector3(
      deltaT[0] * scale,
      -deltaT[1] * scale,
      -deltaT[2] * scale
    );

    const T21 = new THREE.Matrix4();
    T21.set(
      r00, -r01, -r02, t.x,
      -r10, r11, r12, t.y,
      -r20, r21, r22, t.z,
      0, 0, 0, 1
    );

    const T12 = T21.clone().invert();

    this.camera.matrixAutoUpdate = false;
    this.camera.matrix.multiply(T12);
    this.camera.matrix.decompose(this.camera.position, this.camera.quaternion, this.camera.scale);
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
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
    // Camera-tracked architecture: the camera moves through a fixed world.
    // solvePnP yields marker→camera; we invert to get camera→world and drive
    // this.camera directly. worldZeroRoot stays at the world origin (0,0,0).
    this.worldZeroRoot = new THREE.Group();
    this.worldZeroRoot.name = 'WorldZeroRoot';
    this.worldZeroRoot.visible = false;
    this.scene.add(this.worldZeroRoot);

    this.houseRoot = new THREE.Group();
    this.houseRoot.name = 'HouseRoot';
    this.worldZeroRoot.add(this.houseRoot);

    // Back-compat: existing code references renderer.anchor
    this.anchor = this.worldZeroRoot;

    // Persisted world-zero / house state
    this.worldZeroState = {
      isSet: false,
      worldZeroMatrixAtZero: new THREE.Matrix4(),
      worldZeroMatrixAtZeroInverse: new THREE.Matrix4()
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
          worldZeroMatrixAtZero: this.worldZeroState.worldZeroMatrixAtZero.toArray()
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
      if (!data?.isSet || !Array.isArray(data?.worldZeroMatrixAtZero)) return false;

      this.worldZeroState.isSet = true;
      this.worldZeroState.worldZeroMatrixAtZero.fromArray(data.worldZeroMatrixAtZero);
      this.worldZeroState.worldZeroMatrixAtZeroInverse
        .copy(this.worldZeroState.worldZeroMatrixAtZero)
        .invert();
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

  // Match the Three.js camera projection to the real camera's intrinsics.
  // Without this, the rendered overlay uses a different FOV than the video,
  // so 3D content will never stay pinned to the correct real-world position.
  setCameraIntrinsics(fx, fy, cx, cy) {
    // Vertical FOV: 2 * atan(half_image_height_px / focal_length_y_px)
    this.camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan(cy / fy));
    this.camera.updateProjectionMatrix();
  }

  // Mark the current frame as "world zero".
  // In camera-tracked mode worldZeroRoot is always at the origin (identity matrix),
  // so the snapshot is always identity. placeHouseAtCurrentWorldPose therefore works
  // directly in world space, which is the correct behaviour.
  setWorldZero() {
    this.worldZeroRoot.updateMatrixWorld(true);

    this.worldZeroState.worldZeroMatrixAtZero.copy(this.worldZeroRoot.matrixWorld);
    this.worldZeroState.worldZeroMatrixAtZeroInverse.copy(this.worldZeroRoot.matrixWorld).invert();
    this.worldZeroState.isSet = true;

    this.saveWorldZero();
  }

  placeHouseAtCurrentWorldPose(worldMatrix) {
    if (!this.worldZeroState.isSet) return;
    if (!worldMatrix) return;

    const localToWorldZero = new THREE.Matrix4()
      .copy(this.worldZeroState.worldZeroMatrixAtZeroInverse)
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

  // Treat the marker as the world origin and drive the camera.
  // pose is marker→camera (OpenCV solvePnP output).
  // We build Tcw (marker→camera in Three.js coords) directly, invert it to
  // get Twc (camera→world), and apply it to this.camera.
  // worldZeroRoot stays at the world origin (0,0,0) the whole time.
  setCameraFromMarkerPose(pose) {
    if (!pose || !pose.position || !pose.rotationMatrix) {
      this.worldZeroRoot.visible = false;
      return;
    }

    // pose.rotationMatrix is a row-major 3×3 from OpenCV (x right, y down, z forward).
    // Convert to Three.js coords (x right, y up, z backward) via S = diag(1,−1,−1):
    //   R_three = S · R_cv · S
    // pose.position has already had y negated by pose.js normalizeTranslation,
    // so only z needs negating here.
    const r = pose.rotationMatrix;
    const r00 = r[0], r01 = r[1], r02 = r[2];
    const r10 = r[3], r11 = r[4], r12 = r[5];
    const r20 = r[6], r21 = r[7], r22 = r[8];

    // Build Tcw (marker→camera) directly — no scene-object manipulation needed.
    const Tcw = new THREE.Matrix4();
    Tcw.set(
       r00, -r01, -r02,  pose.position.x,
      -r10,  r11,  r12,  pose.position.y,
      -r20,  r21,  r22, -pose.position.z,
         0,    0,    0,  1
    );

    // Invert to get Twc (camera→world) and apply to camera.
    const Twc = Tcw.clone().invert();

    this.camera.matrixAutoUpdate = false;
    this.camera.matrix.copy(Twc);
    this.camera.matrix.decompose(this.camera.position, this.camera.quaternion, this.camera.scale);
    this.camera.matrixWorldNeedsUpdate = true;
    this.camera.updateMatrixWorld(true);

    this.worldZeroRoot.visible = true;
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
    this.camera.matrixWorldNeedsUpdate = true;
    this.camera.updateMatrixWorld(true);

    // Keep world root at origin
    this.worldZeroRoot.position.set(0, 0, 0);
    this.worldZeroRoot.quaternion.set(0, 0, 0, 1);
    this.worldZeroRoot.visible = true;
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
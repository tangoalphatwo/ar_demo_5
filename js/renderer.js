// renderer.js
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

    // Anchor for world-locked content
    this.anchor = new THREE.Group();
    this.scene.add(this.anchor);

    this.modelRoot = null;

    // Video background quad
    this.videoTexture = null;
    this.bgMesh = null;
  }

  async loadGLB(url, { scale = 1 } = {}) {
    if (!THREE || !THREE.GLTFLoader) {
      throw new Error('THREE.GLTFLoader is not available. Ensure GLTFLoader is loaded before main.js.');
    }

    const loader = new THREE.GLTFLoader();
    return await new Promise((resolve, reject) => {
      loader.load(
        url,
        (gltf) => {
          const root = gltf.scene || gltf.scenes?.[0];
          if (!root) {
            reject(new Error('GLB loaded but no scene root found'));
            return;
          }
          root.scale.setScalar(scale);
          resolve(root);
        },
        undefined,
        (err) => reject(err)
      );
    });
  }

  async loadAvocado(url) {
    if (this.modelRoot) {
      this.anchor.remove(this.modelRoot);
      this.modelRoot = null;
    }

    const root = await this.loadGLB(url, { scale: 0.01 });
    this.modelRoot = root;
    this.anchor.add(root);
  }

  // Rcw, tcw are OpenCV Mats such that: X_cam(cv) = Rcw * X_world + tcw
  // Converts from OpenCV camera coords (x right, y down, z forward)
  // to Three camera coords (x right, y up, z backward).
  setCameraFromOpenCvPose(Rcw, tcw) {
    if (!Rcw || !tcw) return;

    // S = diag(1, -1, -1)
    const S = new THREE.Matrix3().set(
      1, 0, 0,
      0, -1, 0,
      0, 0, -1
    );

    const r = Rcw.data64F ? Rcw.data64F : Rcw.data32F;
    const t = tcw.data64F ? tcw.data64F : tcw.data32F;
    if (!r || !t) return;

    // R_three = S * Rcw
    const Rcv = new THREE.Matrix3().set(
      r[0], r[1], r[2],
      r[3], r[4], r[5],
      r[6], r[7], r[8]
    );

    const Rthree = new THREE.Matrix3();
    Rthree.multiplyMatrices(S, Rcv);

    const tcv = new THREE.Vector3(t[0], t[1], t[2]);
    const tthree = tcv.clone();
    // t_three = S * t_cv
    tthree.y *= -1;
    tthree.z *= -1;

    // Camera pose in world: Rwc = R_three^T, C = -R_three^T * t_three
    const Rwc = new THREE.Matrix3();
    Rwc.copy(Rthree).transpose();

    const C = tthree.clone();
    C.applyMatrix3(Rwc).multiplyScalar(-1);

    const m4 = new THREE.Matrix4();
    const rot4 = new THREE.Matrix4().set(
      Rwc.elements[0], Rwc.elements[3], Rwc.elements[6], 0,
      Rwc.elements[1], Rwc.elements[4], Rwc.elements[7], 0,
      Rwc.elements[2], Rwc.elements[5], Rwc.elements[8], 0,
      0, 0, 0, 1
    );
    m4.copy(rot4);
    m4.setPosition(C);

    this.camera.matrixAutoUpdate = false;
    this.camera.matrixWorld.copy(m4);
    this.camera.matrixWorldInverse.copy(m4).invert();
  }

  setVideoTexture(video) {
    const texture = new THREE.VideoTexture(video);
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;
    texture.format = THREE.RGBFormat;
  
    const geometry = new THREE.PlaneGeometry(2, 2);
    const material = new THREE.MeshBasicMaterial({ map: texture });
    const mesh = new THREE.Mesh(geometry, material);
  
    this.scene.add(mesh);
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

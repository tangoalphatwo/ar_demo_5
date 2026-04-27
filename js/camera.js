// camera.js
export class CameraManager {
  constructor(videoEl, cvCanvas) {
    this.video = videoEl;
    this.cvCanvas = cvCanvas;
    this.cvCtx = cvCanvas.getContext('2d', { willReadFrequently: true });
    this.ready = false;

    this.lastDrawRect = null;
    this.lastDpr = 1;

    // Dedicated small canvas for OpenCV processing at video-native resolution.
    // Bypasses the DPR-scaled display canvas, keeping all CV ops fast on mobile.
    this.procCanvas = document.createElement('canvas');
    this.procCanvas.width = 1;
    this.procCanvas.height = 1;
    this.procCtx = this.procCanvas.getContext('2d', { willReadFrequently: true });
  }

  async start() {
    // iOS Safari: must be triggered by user gesture, hence Start button 
    const constraints = {
      audio: false,
      video: {
        facingMode: 'environment',
        width: { ideal: 640 },
        height: { ideal: 480 }
      }
    };

    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    this.video.srcObject = stream;

    return new Promise(resolve => {
      this.video.onloadedmetadata = () => {
        this.video.play();

        const w = this.video.videoWidth;
        const h = this.video.videoHeight;

        // Do not change cvCanvas sizing here — main.js will set proper display/backing size
        this.ready = true;
        resolve({ width: w, height: h });
      };
    });
  }

  drawVideoPreserveAspect(video, canvas) {
    const videoAspect = video.videoWidth / video.videoHeight;
    // Use CSS pixels for layout calculations
    const canvasCssW = canvas.clientWidth;
    const canvasCssH = canvas.clientHeight;
    const canvasAspect = canvasCssW / canvasCssH;

    let drawWidth, drawHeight, offsetX, offsetY;

    // COVER: fill the canvas completely (crop excess).
    if (canvasAspect > videoAspect) {
      // Canvas is wider than video → scale by width and crop vertically
      drawWidth = canvasCssW;
      drawHeight = drawWidth / videoAspect;
      offsetX = 0;
      offsetY = (canvasCssH - drawHeight) / 2;
    } else {
      // Canvas is taller than video → scale by height and crop horizontally
      drawHeight = canvasCssH;
      drawWidth = drawHeight * videoAspect;
      offsetX = (canvasCssW - drawWidth) / 2;
      offsetY = 0;
    }

    // Clear full backing buffer
    this.cvCtx.clearRect(0, 0, canvas.width, canvas.height);
    // drawImage uses CSS pixels coordinates; context should already map CSS→backing
    this.cvCtx.drawImage(video, offsetX, offsetY, drawWidth, drawHeight);

    return { offsetX, offsetY, drawWidth, drawHeight };
  }

  grabFrame() {
    if (!this.ready) return null;

    const w = this.cvCanvas.width;
    const h = this.cvCanvas.height;

    // drawRect is in CSS pixels
    const drawRect = this.drawVideoPreserveAspect(this.video, this.cvCanvas);
    this.lastDrawRect = drawRect;

    // Backing buffer pixels per CSS pixel
    const cssW = this.cvCanvas.clientWidth || 1;
    this.lastDpr = w / cssW;

    return {
      imageData: this.cvCtx.getImageData(0, 0, w, h),
      drawRect,
      dpr: this.lastDpr
    };
  }

  // Returns an ImageData captured directly from the video at native resolution.
  // This is used exclusively for OpenCV processing — no letterboxing, no DPR scaling.
  // Points returned by ORB/LK in this space map 1:1 to video pixel coords.
  grabProcFrame() {
    if (!this.ready) return null;
    const vw = this.video.videoWidth;
    const vh = this.video.videoHeight;
    if (!vw || !vh) return null;

    // Cap the long side at 640px so OpenCV stays fast on all devices.
    const maxDim = 640;
    const scale = Math.min(1, maxDim / Math.max(vw, vh));
    const pw = Math.round(vw * scale);
    const ph = Math.round(vh * scale);

    if (this.procCanvas.width !== pw || this.procCanvas.height !== ph) {
      this.procCanvas.width = pw;
      this.procCanvas.height = ph;
    }

    this.procCtx.drawImage(this.video, 0, 0, pw, ph);
    return {
      imageData: this.procCtx.getImageData(0, 0, pw, ph),
      procWidth: pw,
      procHeight: ph
    };
  }
}
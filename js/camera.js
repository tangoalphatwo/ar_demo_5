// camera.js
export class CameraManager {
  constructor(videoEl, cvCanvas) {
    this.video = videoEl;
    this.cvCanvas = cvCanvas;
    this.cvCtx = cvCanvas.getContext('2d');
    this.ready = false;

    // Last draw rect (CSS px) used to draw the video into cvCanvas
    this.lastDrawRect = null;
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

    if (canvasAspect > videoAspect) {
      // Canvas is wider than video → pillarbox
      drawHeight = canvasCssH;
      drawWidth = drawHeight * videoAspect;
      offsetX = (canvasCssW - drawWidth) / 2;
      offsetY = 0;
    } else {
      // Canvas is taller than video → letterbox
      drawWidth = canvasCssW;
      drawHeight = drawWidth / videoAspect;
      offsetX = 0;
      offsetY = (canvasCssH - drawHeight) / 2;
    }

    // Clear full backing buffer
    this.cvCtx.clearRect(0, 0, canvas.width, canvas.height);
    // drawImage uses CSS pixels coordinates; context should already map CSS→backing
    this.cvCtx.drawImage(video, offsetX, offsetY, drawWidth, drawHeight);

    const rect = { offsetX, offsetY, drawWidth, drawHeight };
    this.lastDrawRect = rect;
    return rect;
  }

  // Map a point measured in cvCanvas *backing pixels* (ImageData coords)
  // into the original video pixel coordinates.
  mapCanvasPointToVideo(p) {
    if (!this.lastDrawRect) return null;

    const dpr = this.cvCanvas.clientWidth ? (this.cvCanvas.width / this.cvCanvas.clientWidth) : 1;

    const ox = this.lastDrawRect.offsetX * dpr;
    const oy = this.lastDrawRect.offsetY * dpr;
    const dw = this.lastDrawRect.drawWidth * dpr;
    const dh = this.lastDrawRect.drawHeight * dpr;

    const nx = (p.x - ox) / dw;
    const ny = (p.y - oy) / dh;

    if (!isFinite(nx) || !isFinite(ny)) return null;
    if (nx < 0 || nx > 1 || ny < 0 || ny > 1) return null;

    return {
      x: nx * this.video.videoWidth,
      y: ny * this.video.videoHeight
    };
  }

  grabFrame() {
    if (!this.ready) return null;

    const w = this.cvCanvas.width;
    const h = this.cvCanvas.height;
    this.drawVideoPreserveAspect(this.video, this.cvCanvas);
    return this.cvCtx.getImageData(0, 0, w, h);
  }
}

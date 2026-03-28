/**
 * Webcam video element manager.
 * Creates a hidden <video> element for MediaPipe + WebGL VideoTexture.
 */

export class WebcamManager {
  video: HTMLVideoElement | null = null;
  active = false;
  private stream: MediaStream | null = null;

  async start(): Promise<HTMLVideoElement | null> {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      const video = document.createElement("video");
      video.srcObject = this.stream;
      video.setAttribute("playsinline", "true");
      video.style.cssText = "position:absolute;top:-9999px;left:-9999px;";
      document.body.appendChild(video);
      await video.play();
      this.video = video;
      this.active = true;
      return video;
    } catch {
      this.active = false;
      return null;
    }
  }

  stop(): void {
    this.active = false;
    if (this.stream) {
      this.stream.getTracks().forEach((t) => t.stop());
      this.stream = null;
    }
    if (this.video) {
      this.video.pause();
      this.video.srcObject = null;
      this.video.remove();
      this.video = null;
    }
  }
}

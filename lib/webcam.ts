/**
 * Webcam video element manager.
 * Creates a hidden <video> element for MediaPipe + WebGL VideoTexture.
 */

export class WebcamManager {
  video: HTMLVideoElement | null = null;
  active = false;
  private stream: MediaStream | null = null;

  /** List available video input devices. */
  static async listDevices(): Promise<MediaDeviceInfo[]> {
    try {
      // Must request permission first so device labels are populated
      const probe = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      probe.getTracks().forEach((t) => t.stop());
      const all = await navigator.mediaDevices.enumerateDevices();
      return all.filter((d) => d.kind === "videoinput");
    } catch {
      return [];
    }
  }

  async start(deviceId?: string): Promise<HTMLVideoElement | null> {
    try {
      const videoConstraints: MediaTrackConstraints = deviceId
        ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
        : { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } };
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: videoConstraints,
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

/**
 * Audio reactivity via Web Audio API.
 *
 * Splits mic input into bass / mid / treble energy bands
 * that drive particle forces, colours, and sizes.
 */

export interface AudioEnergy {
  bass: number;   // 0-1  ~0-300 Hz
  mid: number;    // 0-1  ~300-3 kHz
  treble: number; // 0-1  ~3 kHz+
  overall: number;
}

const ZERO: AudioEnergy = { bass: 0, mid: 0, treble: 0, overall: 0 };

export class AudioReactor {
  private audioCtx: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private stream: MediaStream | null = null;
  private freqData: Uint8Array<ArrayBuffer> = new Uint8Array(0);
  active = false;

  // Smoothed values for less jitter
  private smoothBass = 0;
  private smoothMid = 0;
  private smoothTreble = 0;

  async start(): Promise<boolean> {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      this.audioCtx = new AudioContext();
      const source = this.audioCtx.createMediaStreamSource(this.stream);
      this.analyser = this.audioCtx.createAnalyser();
      this.analyser.fftSize = 256;
      this.analyser.smoothingTimeConstant = 0.75;
      source.connect(this.analyser);
      this.freqData = new Uint8Array(this.analyser.frequencyBinCount);
      this.active = true;
      return true;
    } catch {
      this.active = false;
      return false;
    }
  }

  stop(): void {
    this.active = false;
    if (this.stream) {
      this.stream.getTracks().forEach((t) => t.stop());
      this.stream = null;
    }
    if (this.audioCtx) {
      this.audioCtx.close();
      this.audioCtx = null;
    }
    this.analyser = null;
    this.smoothBass = 0;
    this.smoothMid = 0;
    this.smoothTreble = 0;
  }

  getEnergy(): AudioEnergy {
    if (!this.analyser || !this.active) return ZERO;

    this.analyser.getByteFrequencyData(this.freqData);
    const n = this.freqData.length; // 128 bins

    const bassEnd = Math.floor(n * 0.12);
    const midEnd = Math.floor(n * 0.45);

    let bass = 0;
    let mid = 0;
    let treble = 0;
    for (let i = 0; i < bassEnd; i++) bass += this.freqData[i];
    for (let i = bassEnd; i < midEnd; i++) mid += this.freqData[i];
    for (let i = midEnd; i < n; i++) treble += this.freqData[i];

    bass /= bassEnd * 255 || 1;
    mid /= (midEnd - bassEnd) * 255 || 1;
    treble /= (n - midEnd) * 255 || 1;

    // Exponential smoothing (reduces jitter)
    const a = 0.3;
    this.smoothBass = this.smoothBass * (1 - a) + bass * a;
    this.smoothMid = this.smoothMid * (1 - a) + mid * a;
    this.smoothTreble = this.smoothTreble * (1 - a) + treble * a;

    const overall = (this.smoothBass + this.smoothMid + this.smoothTreble) / 3;
    return {
      bass: this.smoothBass,
      mid: this.smoothMid,
      treble: this.smoothTreble,
      overall,
    };
  }
}

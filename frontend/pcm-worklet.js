/**
 * Turns the microphone's float samples into the 16-bit PCM chunks the
 * ElevenLabs realtime endpoint expects, and reports a level alongside each one
 * so the page can drive its meter without a second analyser node.
 *
 * The AudioContext is created at 16 kHz, so no resampling happens here.
 */

const CHUNK_SAMPLES = 1600; // 100 ms at 16 kHz

class PcmChunker extends AudioWorkletProcessor {
  constructor() {
    super();
    this.chunk = new Int16Array(CHUNK_SAMPLES);
    this.filled = 0;
  }

  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!channel) {
      return true;
    }

    for (let i = 0; i < channel.length; i += 1) {
      const sample = Math.max(-1, Math.min(1, channel[i]));
      this.chunk[this.filled] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
      this.filled += 1;

      if (this.filled === CHUNK_SAMPLES) {
        this.flush();
      }
    }

    return true;
  }

  flush() {
    let sumOfSquares = 0;
    for (let i = 0; i < CHUNK_SAMPLES; i += 1) {
      const sample = this.chunk[i] / 0x8000;
      sumOfSquares += sample * sample;
    }

    // Copy: postMessage transfers ownership, and this.chunk keeps filling.
    const pcm = this.chunk.slice();
    this.port.postMessage(
      { pcm: pcm.buffer, level: Math.sqrt(sumOfSquares / CHUNK_SAMPLES) },
      [pcm.buffer],
    );
    this.filled = 0;
  }
}

registerProcessor("pcm-chunker", PcmChunker);

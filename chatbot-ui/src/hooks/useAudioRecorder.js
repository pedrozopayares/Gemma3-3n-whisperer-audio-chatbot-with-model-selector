import { useState, useRef, useCallback, useEffect } from "react";

const MAX_RECORDING_DURATION = 30000;
const SILENCE_THRESHOLD = 0.01;
const SILENCE_DURATION = 1500;
const TARGET_SAMPLE_RATE = 16000;

/**
 * Sinc interpolation resampler for audio data.
 */
function resampleAudio(sourceData, sourceSampleRate, targetSampleRate) {
  if (sourceSampleRate === targetSampleRate) return sourceData;

  const ratio = sourceSampleRate / targetSampleRate;
  const newLength = Math.round(sourceData.length / ratio);
  const result = new Float32Array(newLength);
  const sincKernelSize = 16;

  for (let i = 0; i < newLength; i++) {
    const srcPosition = i * ratio;
    let sample = 0;
    let weightSum = 0;

    for (let j = -sincKernelSize; j <= sincKernelSize; j++) {
      const srcIndex = Math.floor(srcPosition) + j;
      if (srcIndex >= 0 && srcIndex < sourceData.length) {
        const x = srcPosition - srcIndex;
        let weight;
        if (x === 0) {
          weight = 1;
        } else {
          const piX = Math.PI * x;
          const piXa = piX / sincKernelSize;
          weight = (Math.sin(piX) / piX) * (Math.sin(piXa) / piXa);
        }
        sample += sourceData[srcIndex] * weight;
        weightSum += weight;
      }
    }
    result[i] = weightSum > 0 ? sample / weightSum : 0;
  }
  return result;
}

/**
 * Create WAV from resampled Float32Array.
 */
function audioBufferToWavResampled(data, sampleRate) {
  const numChannels = 1;
  const format = 1;
  const bitDepth = 16;
  const bytesPerSample = bitDepth / 8;
  const blockAlign = numChannels * bytesPerSample;
  const dataLength = data.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataLength);
  const view = new DataView(buffer);

  const writeString = (offset, string) => {
    for (let i = 0; i < string.length; i++) view.setUint8(offset + i, string.charCodeAt(i));
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataLength, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, format, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitDepth, true);
  writeString(36, "data");
  view.setUint32(40, dataLength, true);

  const offset = 44;
  for (let i = 0; i < data.length; i++) {
    const s = Math.max(-1, Math.min(1, data[i]));
    view.setInt16(offset + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buffer;
}

/**
 * Convert audio blob to mono 16kHz WAV.
 */
async function convertToWav(audioBlob) {
  const audioContext = new (window.AudioContext || window.webkitAudioContext)();
  const arrayBuffer = await audioBlob.arrayBuffer();
  const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

  let monoData;
  if (audioBuffer.numberOfChannels > 1) {
    const left = audioBuffer.getChannelData(0);
    const right = audioBuffer.getChannelData(1);
    monoData = new Float32Array(left.length);
    for (let i = 0; i < left.length; i++) monoData[i] = (left[i] + right[i]) / 2;
  } else {
    monoData = audioBuffer.getChannelData(0);
  }

  const resampledData = resampleAudio(monoData, audioBuffer.sampleRate, TARGET_SAMPLE_RATE);
  for (let i = 0; i < resampledData.length; i++) {
    resampledData[i] = Math.max(-1, Math.min(1, resampledData[i]));
  }

  const wavBuffer = audioBufferToWavResampled(resampledData, TARGET_SAMPLE_RATE);
  await audioContext.close();
  return new Blob([wavBuffer], { type: "audio/wav" });
}

/**
 * Custom hook for audio recording with silence detection.
 *
 * @returns {{ isRecording, recordingTime, startRecording, stopRecording }}
 */
export default function useAudioRecorder({ onAudioReady, onError }) {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);

  // Keep latest callbacks in refs to avoid stale closures in onstop/setTimeout
  const onAudioReadyRef = useRef(onAudioReady);
  const onErrorRef = useRef(onError);
  useEffect(() => { onAudioReadyRef.current = onAudioReady; }, [onAudioReady]);
  useEffect(() => { onErrorRef.current = onError; }, [onError]);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const silenceCheckRef = useRef(null);
  const maxTimeoutRef = useRef(null);
  const streamRef = useRef(null);
  const lastSoundRef = useRef(null);
  const startTimeRef = useRef(null);
  const timerRef = useRef(null);

  const cleanup = useCallback(() => {
    if (silenceCheckRef.current) { clearInterval(silenceCheckRef.current); silenceCheckRef.current = null; }
    if (maxTimeoutRef.current) { clearTimeout(maxTimeoutRef.current); maxTimeoutRef.current = null; }
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
    if (audioContextRef.current) {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
      analyserRef.current = null;
    }
  }, []);

  const stopRecording = useCallback(() => {
    cleanup();
    if (mediaRecorderRef.current?.state === "recording") {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setRecordingTime(0);
    }
  }, [cleanup]);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];
      mediaRecorderRef.current.ondataavailable = (e) => audioChunksRef.current.push(e.data);
      mediaRecorderRef.current.onstop = async () => {
        try {
          const audioBlob = new Blob(audioChunksRef.current, { type: mediaRecorderRef.current.mimeType });
          const wavBlob = await convertToWav(audioBlob);
          streamRef.current?.getTracks().forEach((t) => t.stop());
          onAudioReadyRef.current?.(wavBlob);
        } catch (err) {
          streamRef.current?.getTracks().forEach((t) => t.stop());
          onErrorRef.current?.(err);
        }
      };

      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      analyserRef.current = audioContextRef.current.createAnalyser();
      analyserRef.current.fftSize = 512;
      analyserRef.current.smoothingTimeConstant = 0.1;
      const source = audioContextRef.current.createMediaStreamSource(stream);
      source.connect(analyserRef.current);

      mediaRecorderRef.current.start();
      setIsRecording(true);
      setRecordingTime(0);
      startTimeRef.current = Date.now();
      lastSoundRef.current = Date.now();

      timerRef.current = setInterval(() => {
        setRecordingTime(Math.floor((Date.now() - startTimeRef.current) / 1000));
      }, 100);

      const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
      silenceCheckRef.current = setInterval(() => {
        if (!analyserRef.current) return;
        analyserRef.current.getByteFrequencyData(dataArray);
        const average = dataArray.reduce((a, b) => a + b, 0) / dataArray.length / 255;
        if (average > SILENCE_THRESHOLD) {
          lastSoundRef.current = Date.now();
        } else if (Date.now() - lastSoundRef.current >= SILENCE_DURATION) {
          stopRecording();
        }
      }, 100);

      maxTimeoutRef.current = setTimeout(() => stopRecording(), MAX_RECORDING_DURATION);
    } catch {
      onErrorRef.current?.(new Error("No se pudo acceder al micrófono."));
    }
  }, [stopRecording]);

  return { isRecording, recordingTime, startRecording, stopRecording };
}

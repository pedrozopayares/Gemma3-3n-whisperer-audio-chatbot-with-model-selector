import React, { useState, useEffect, useRef } from "react";

// ----------------- Icons -----------------
const IconMic = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
    <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
    <line x1="12" y1="19" x2="12" y2="22"></line>
  </svg>
);

const IconLoader = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="12" y1="2" x2="12" y2="6" />
    <line x1="12" y1="18" x2="12" y2="22" />
    <line x1="4.93" y1="4.93" x2="7.76" y2="7.76" />
    <line x1="16.24" y1="16.24" x2="19.07" y2="19.07" />
    <line x1="2" y1="12" x2="6" y2="12" />
    <line x1="18" y1="12" x2="22" y2="12" />
    <line x1="4.93" y1="19.07" x2="7.76" y2="16.24" />
    <line x1="16.24" y1="7.76" x2="19.07" y2="4.93" />
  </svg>
);

// ----------------- Main Component -----------------
export default function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [conversation, setConversation] = useState([]);
  const [error, setError] = useState(null);
  const [isTTSEnabled, setIsTTSEnabled] = useState(true);
  
  // Model selection
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);
  const [isLoadingModels, setIsLoadingModels] = useState(true);

  // Fetch available models on mount
  const fetchModels = async () => {
    try {
      const res = await fetch("http://localhost:8000/models");
      if (res.ok) {
        const data = await res.json();
        setAvailableModels(data.models);
        setSelectedModel(data.current || data.default);
      }
    } catch (e) {
      console.error("Failed to fetch models:", e);
    } finally {
      setIsLoadingModels(false);
    }
  };

  useEffect(() => {
    fetchModels();
  }, []);

  // Clear model cache
  const clearModelCache = async () => {
    if (!confirm("¿Seguro que quieres borrar todos los modelos de la caché? Tendrás que descargarlos de nuevo.")) {
      return;
    }
    try {
      setIsProcessing(true);
      const res = await fetch("http://localhost:8000/models/cache", { method: "DELETE" });
      if (res.ok) {
        const data = await res.json();
        alert(`${data.message}`);
        fetchModels(); // Refresh model list
      } else {
        throw new Error("Failed to clear cache");
      }
    } catch (e) {
      setError(`Error: ${e.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  // image-related state
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [imagePrompt, setImagePrompt] = useState("");

  // audio recording refs
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  // ----------------- AUDIO RECORDING -----------------
  // Target sample rate for Gemma3n (REQUIRED: 16kHz per Google docs)
  const TARGET_SAMPLE_RATE = 16000;

  // Resample audio using sinc interpolation (closest to Fourier method in browser)
  // Per Google docs: "use a Fourier method for best results"
  const resampleAudio = (sourceData, sourceSampleRate, targetSampleRate) => {
    if (sourceSampleRate === targetSampleRate) {
      return sourceData;
    }
    
    const ratio = sourceSampleRate / targetSampleRate;
    const newLength = Math.round(sourceData.length / ratio);
    const result = new Float32Array(newLength);
    
    // Sinc interpolation with windowed kernel (high quality, closer to Fourier)
    const sincKernelSize = 16; // Number of neighboring samples to consider
    
    for (let i = 0; i < newLength; i++) {
      const srcPosition = i * ratio;
      let sample = 0;
      let weightSum = 0;
      
      // Apply sinc kernel
      for (let j = -sincKernelSize; j <= sincKernelSize; j++) {
        const srcIndex = Math.floor(srcPosition) + j;
        if (srcIndex >= 0 && srcIndex < sourceData.length) {
          const x = srcPosition - srcIndex;
          // Lanczos window (sinc * sinc)
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
  };

  // Convert audio to proper WAV format for Gemma3n
  // Per Google docs: mono, 16kHz, float32 in range [-1, 1]
  const convertToWav = async (audioBlob) => {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const arrayBuffer = await audioBlob.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    
    const nativeSampleRate = audioBuffer.sampleRate;
    
    // Step 1: Get mono audio (average channels if stereo)
    let monoData;
    if (audioBuffer.numberOfChannels > 1) {
      const left = audioBuffer.getChannelData(0);
      const right = audioBuffer.getChannelData(1);
      monoData = new Float32Array(left.length);
      for (let i = 0; i < left.length; i++) {
        monoData[i] = (left[i] + right[i]) / 2;
      }
      console.log(`Downmixed stereo to mono`);
    } else {
      monoData = audioBuffer.getChannelData(0);
    }
    
    // Step 2: Resample to 16kHz using sinc interpolation (Fourier-like method)
    const resampledData = resampleAudio(monoData, nativeSampleRate, TARGET_SAMPLE_RATE);
    console.log(`Audio: ${nativeSampleRate}Hz -> ${TARGET_SAMPLE_RATE}Hz, ${monoData.length} -> ${resampledData.length} samples, ${resampledData.length/TARGET_SAMPLE_RATE}s`);
    
    // Step 3: Ensure float32 range [-1, 1] (clamp any outliers)
    for (let i = 0; i < resampledData.length; i++) {
      resampledData[i] = Math.max(-1, Math.min(1, resampledData[i]));
    }
    
    // Convert to WAV at 16kHz
    const wavBuffer = audioBufferToWavResampled(resampledData, TARGET_SAMPLE_RATE);
    return new Blob([wavBuffer], { type: "audio/wav" });
  };

  // Create WAV from resampled Float32Array
  const audioBufferToWavResampled = (data, sampleRate) => {
    const numChannels = 1; // mono
    const format = 1; // PCM
    const bitDepth = 16;
    
    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;
    const dataLength = data.length * bytesPerSample;
    const buffer = new ArrayBuffer(44 + dataLength);
    const view = new DataView(buffer);
    
    // WAV header
    const writeString = (offset, string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
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
    
    // Write audio data
    const offset = 44;
    for (let i = 0; i < data.length; i++) {
      const sample = Math.max(-1, Math.min(1, data[i]));
      const intSample = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
      view.setInt16(offset + i * 2, intSample, true);
    }
    
    return buffer;
  };

  // Legacy function kept for compatibility
  const audioBufferToWav = (audioBuffer) => {
    const numChannels = audioBuffer.numberOfChannels;
    const sampleRate = audioBuffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;
    
    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;
    
    const data = audioBuffer.getChannelData(0);
    const dataLength = data.length * bytesPerSample;
    const buffer = new ArrayBuffer(44 + dataLength);
    const view = new DataView(buffer);
    
    // WAV header
    const writeString = (offset, string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };
    
    writeString(0, "RIFF");
    view.setUint32(4, 36 + dataLength, true);
    writeString(8, "WAVE");
    writeString(12, "fmt ");
    view.setUint32(16, 16, true); // fmt chunk size
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    writeString(36, "data");
    view.setUint32(40, dataLength, true);
    
    // Write audio data
    const offset = 44;
    for (let i = 0; i < data.length; i++) {
      const sample = Math.max(-1, Math.min(1, data[i]));
      const intSample = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
      view.setInt16(offset + i * 2, intSample, true);
    }
    
    return buffer;
  };

  const startRecording = async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];
      mediaRecorderRef.current.ondataavailable = (e) => audioChunksRef.current.push(e.data);
      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: mediaRecorderRef.current.mimeType });
        const wavBlob = await convertToWav(audioBlob);
        await processAudio(wavBlob);
        stream.getTracks().forEach((t) => t.stop());
      };
      mediaRecorderRef.current.start();
      setIsRecording(true);
      setTimeout(stopRecording, 4000);
    } catch (err) {
      setError("Could not access microphone.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current?.state === "recording") {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setIsProcessing(true);
    }
  };

  const blobToBase64 = (blob) =>
    new Promise((res, rej) => {
      const reader = new FileReader();
      reader.onloadend = () => res(reader.result.split(",")[1]);
      reader.onerror = rej;
      reader.readAsDataURL(blob);
    });

  const processAudio = async (audioBlob) => {
    setIsProcessing(true);
    try {
      const b64 = await blobToBase64(audioBlob);
      // Initially show as audio, will be updated with transcription
      setConversation((p) => [...p, { role: "user", parts: [{ type: "audio" }] }]);
      const res = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ data: b64, model: selectedModel }),
      });
      if (!res.ok) {
        const errorText = await res.text();
        console.error("Server error:", errorText);
        throw new Error(errorText || `Server error: ${res.status}`);
      }
      const data = await res.json();
      console.log("Received response:", data);
      const { text, transcription } = data;
      
      // Update the last user message to show transcription instead of "Audio message"
      if (transcription) {
        setConversation((p) => {
          const updated = [...p];
          const lastUserIdx = updated.length - 1;
          if (updated[lastUserIdx]?.role === "user") {
            updated[lastUserIdx] = { 
              role: "user", 
              parts: [{ type: "text", text: `🎤 "${transcription}"` }] 
            };
          }
          return updated;
        });
      }
      
      addReply(text);
    } catch (e) {
      console.error("Error in processAudio:", e);
      setError(`Audio error: ${e.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  // ----------------- IMAGE + PROMPT -----------------
  const submitImageQuestion = async () => {
    if (!imageFile || !imagePrompt.trim()) return;
    setIsProcessing(true);
    setError(null);
    try {
      setConversation((p) => [
        ...p,
        { role: "user", parts: [{ type: "image", url: imagePreview }, { type: "text", text: imagePrompt }] },
      ]);
      const form = new FormData();
      form.append("prompt", imagePrompt);
      form.append("image", imageFile, imageFile.name);

      const res = await fetch("http://localhost:8000/ask_image", { method: "POST", body: form });
      if (!res.ok) {
        const errorText = await res.text();
        console.error("Server error:", errorText);
        throw new Error(errorText || `Server error: ${res.status}`);
      }
      const data = await res.json();
      console.log("Received response:", data);
      const { text } = data;
      addReply(text);
    } catch (e) {
      console.error("Error in submitImageQuestion:", e);
      setError(`Image error: ${e.message}`);
    } finally {
      setIsProcessing(false);
      setImageFile(null);
      setImagePreview(null);
      setImagePrompt("");
    }
  };

  // ----------------- TTS -----------------
  const addReply = (text) => {
    setConversation((p) => [...p, { role: "model", parts: [{ text }] }]);
    if (isTTSEnabled && "speechSynthesis" in window) {
      const utter = new SpeechSynthesisUtterance(text.replace(/[-•▪●◦—–]/g, "-"));
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(utter);
    }
  };

  // ----------------- Scroll to bottom -----------------
  const bottomRef = useRef(null);
  useEffect(() => bottomRef.current?.scrollIntoView({ behavior: "smooth" }), [conversation]);

  // ----------------- RENDER -----------------
  return (
    <div className="flex flex-col w-full h-screen bg-gray-900 text-white">
      {/* header */}
      <header className="p-4 border-b border-gray-700/50 bg-gray-900/80">
        <div className="flex justify-between items-center">
          <h1 className="text-lg font-bold">Gemma Voice Assistant</h1>
          <label className="flex items-center space-x-2">
            <span className="text-sm">TTS</span>
            <input type="checkbox" checked={isTTSEnabled} onChange={() => setIsTTSEnabled(!isTTSEnabled)} />
          </label>
        </div>
        {/* Model selector */}
        <div className="mt-2 flex items-center space-x-2">
          <span className="text-sm text-gray-400">Modelo:</span>
          {isLoadingModels ? (
            <span className="text-sm text-gray-500">Cargando...</span>
          ) : (
            <>
              <select
                value={selectedModel || ""}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="bg-gray-800 text-white text-sm rounded px-2 py-1 border border-gray-600 focus:outline-none focus:border-indigo-500"
                disabled={isProcessing || isRecording}
              >
                {availableModels.map((m) => (
                  <option key={m.key} value={m.key}>
                    {m.name} ({m.size}){m.loaded ? " ✓" : ""}
                  </option>
                ))}
              </select>
              <button
                onClick={clearModelCache}
                disabled={isProcessing || isRecording}
                className="text-xs bg-red-600 hover:bg-red-500 px-2 py-1 rounded disabled:bg-gray-600"
                title="Borrar modelos descargados de la caché"
              >
                🗑️ Limpiar caché
              </button>
            </>
          )}
        </div>
      </header>

      {/* messages */}
      <main className="flex-1 overflow-y-auto p-4 space-y-6">
        {conversation.length === 0 && (
          <div className="text-center text-gray-500 pt-20">
            <IconMic className="w-16 h-16 mx-auto mb-4" />
            <p>Press the record button to start a conversation.</p>
          </div>
        )}

        {conversation.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
            <div className={`max-w-md p-3 rounded-2xl ${msg.role === "user" ? "bg-blue-600" : "bg-gray-700"}`}>
              {msg.parts[0].type === "audio" ? (
                <p className="italic">Audio message</p>
              ) : msg.parts[0].type === "image" ? (
                <>
                  <img src={msg.parts[0].url} alt="upload" className="max-w-xs mb-2 rounded-lg" />
                  <p>{msg.parts[1].text}</p>
                </>
              ) : (
                <p>{msg.parts[0].text}</p>
              )}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </main>

      {/* image upload + question */}
      <section className="px-4 space-y-4">
        <input
          type="file"
          accept="image/*"
          id="img-input"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files[0];
            if (!file) return;
            setImageFile(file);
            setImagePreview(URL.createObjectURL(file));
          }}
        />
        <label
          htmlFor="img-input"
          className="block w-full border-2 border-dashed border-gray-600 text-center p-6 rounded-xl cursor-pointer"
        >
          {imagePreview ? (
            <img src={imagePreview} alt="preview" className="mx-auto max-h-48 rounded-lg" />
          ) : (
            "Click or drag an image here"
          )}
        </label>

        {imagePreview && (
          <>
            <textarea
              value={imagePrompt}
              onChange={(e) => setImagePrompt(e.target.value)}
              placeholder="Ask a question about the image…"
              className="w-full bg-gray-800 p-3 rounded-lg resize-none"
            />
            <button
              onClick={submitImageQuestion}
              disabled={!imagePrompt.trim() || isProcessing}
              className="w-full bg-indigo-600 hover:bg-indigo-500 py-2 rounded-lg disabled:bg-gray-500"
            >
              Ask
            </button>
          </>
        )}
      </section>

      {/* footer mic button */}
      <footer className="p-4 flex flex-col items-center">
        {error && <p className="text-red-400 mb-2">{error}</p>}
        <button
          onClick={startRecording}
          disabled={isRecording || isProcessing}
          className="w-20 h-20 bg-indigo-600 rounded-full flex items-center justify-center disabled:bg-gray-500"
        >
          {isRecording ? <div className="w-8 h-8 bg-red-500 rounded animate-pulse" /> : isProcessing ? <IconLoader className="w-8 h-8 animate-spin" /> : <IconMic className="w-8 h-8" />}
        </button>
        <p className="text-xs text-gray-500 mt-2">
          {isRecording ? "Recording…" : isProcessing ? "Processing…" : "Tap to speak (4 s)"}
        </p>
      </footer>
    </div>
  );
}

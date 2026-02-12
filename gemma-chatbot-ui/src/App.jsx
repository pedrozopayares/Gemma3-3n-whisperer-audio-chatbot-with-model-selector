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

const IconSend = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="22" y1="2" x2="11" y2="13"></line>
    <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
  </svg>
);

// ----------------- Markdown Renderer -----------------
const MarkdownText = ({ text }) => {
  // Parse inline bold: **text** or __text__
  const parseBold = (str) => {
    const parts = [];
    let key = 0;
    
    const regex = /\*\*(.+?)\*\*|__(.+?)__/g;
    let lastIndex = 0;
    let match;
    
    while ((match = regex.exec(str)) !== null) {
      // Add text before the match
      if (match.index > lastIndex) {
        parts.push(str.slice(lastIndex, match.index));
      }
      // Add bold text
      parts.push(<strong key={key++} className="font-semibold">{match[1] || match[2]}</strong>);
      lastIndex = regex.lastIndex;
    }
    // Add remaining text
    if (lastIndex < str.length) {
      parts.push(str.slice(lastIndex));
    }
    
    return parts.length > 0 ? parts : str;
  };

  // Split by lines and process
  const lines = text.split('\n');
  const elements = [];
  let currentList = [];
  let listType = null; // 'ul' or 'ol'
  let key = 0;

  const flushList = () => {
    if (currentList.length > 0) {
      if (listType === 'ol') {
        elements.push(
          <ol key={key++} className="list-decimal list-inside space-y-1 my-2 ml-2">
            {currentList.map((item, i) => (
              <li key={i} className="leading-relaxed">{parseBold(item)}</li>
            ))}
          </ol>
        );
      } else {
        elements.push(
          <ul key={key++} className="list-disc list-inside space-y-1 my-2 ml-2">
            {currentList.map((item, i) => (
              <li key={i} className="leading-relaxed">{parseBold(item)}</li>
            ))}
          </ul>
        );
      }
      currentList = [];
      listType = null;
    }
  };

  for (const line of lines) {
    const trimmed = line.trim();
    
    // Check for numbered list (1. 2. etc)
    const numberedMatch = trimmed.match(/^(\d+)\.\s+(.*)/);
    if (numberedMatch) {
      if (listType !== 'ol') {
        flushList();
        listType = 'ol';
      }
      currentList.push(numberedMatch[2]);
      continue;
    }
    
    // Check for bullet list (* or - or •)
    const bulletMatch = trimmed.match(/^[*\-•]\s+(.*)/);
    if (bulletMatch) {
      if (listType !== 'ul') {
        flushList();
        listType = 'ul';
      }
      currentList.push(bulletMatch[1]);
      continue;
    }
    
    // Not a list item - flush any pending list
    flushList();
    
    // Empty line = paragraph break
    if (!trimmed) {
      elements.push(<div key={key++} className="h-2" />);
      continue;
    }
    
    // Regular paragraph
    elements.push(
      <p key={key++} className="leading-relaxed">
        {parseBold(trimmed)}
      </p>
    );
  }
  
  // Flush any remaining list
  flushList();
  
  return <div className="space-y-1">{elements}</div>;
};

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

  // text prompt state
  const [textPrompt, setTextPrompt] = useState("");

  // context state (added to all queries)
  const [chatContext, setChatContext] = useState("");

  // audio recording refs
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const silenceCheckIntervalRef = useRef(null);
  const maxTimeoutRef = useRef(null);
  const streamRef = useRef(null);
  const lastSoundTimeRef = useRef(null);
  
  // Recording duration state
  const [recordingTime, setRecordingTime] = useState(0);
  const recordingStartTimeRef = useRef(null);
  const recordingTimerRef = useRef(null);
  
  // Recording config
  const MAX_RECORDING_DURATION = 30000; // 30 seconds max
  const SILENCE_THRESHOLD = 0.01; // Audio level threshold (0-1)
  const SILENCE_DURATION = 1500; // 1.5 seconds of silence to stop

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
      streamRef.current = stream;
      
      // Set up MediaRecorder
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];
      mediaRecorderRef.current.ondataavailable = (e) => audioChunksRef.current.push(e.data);
      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: mediaRecorderRef.current.mimeType });
        const wavBlob = await convertToWav(audioBlob);
        await processAudio(wavBlob);
        streamRef.current?.getTracks().forEach((t) => t.stop());
      };
      
      // Set up AudioContext for silence detection
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      analyserRef.current = audioContextRef.current.createAnalyser();
      analyserRef.current.fftSize = 512;
      analyserRef.current.smoothingTimeConstant = 0.1;
      
      const source = audioContextRef.current.createMediaStreamSource(stream);
      source.connect(analyserRef.current);
      
      // Start recording
      mediaRecorderRef.current.start();
      setIsRecording(true);
      setRecordingTime(0);
      recordingStartTimeRef.current = Date.now();
      lastSoundTimeRef.current = Date.now();
      
      // Update recording time display
      recordingTimerRef.current = setInterval(() => {
        const elapsed = Math.floor((Date.now() - recordingStartTimeRef.current) / 1000);
        setRecordingTime(elapsed);
      }, 100);
      
      // Start silence detection
      const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
      
      silenceCheckIntervalRef.current = setInterval(() => {
        if (!analyserRef.current) return;
        
        analyserRef.current.getByteFrequencyData(dataArray);
        
        // Calculate average audio level (0-255 -> 0-1)
        const average = dataArray.reduce((a, b) => a + b, 0) / dataArray.length / 255;
        
        if (average > SILENCE_THRESHOLD) {
          // Sound detected, reset silence timer
          lastSoundTimeRef.current = Date.now();
        } else {
          // Check if silence duration exceeded
          const silenceDuration = Date.now() - lastSoundTimeRef.current;
          if (silenceDuration >= SILENCE_DURATION) {
            console.log(`Silence detected for ${silenceDuration}ms, stopping recording`);
            stopRecording();
          }
        }
      }, 100);
      
      // Max duration timeout (30 seconds)
      maxTimeoutRef.current = setTimeout(() => {
        console.log("Max recording duration reached (30s)");
        stopRecording();
      }, MAX_RECORDING_DURATION);
      
    } catch (err) {
      setError("No se pudo acceder al micrófono.");
    }
  };

  const stopRecording = () => {
    // Clear all timers and intervals
    if (silenceCheckIntervalRef.current) {
      clearInterval(silenceCheckIntervalRef.current);
      silenceCheckIntervalRef.current = null;
    }
    if (maxTimeoutRef.current) {
      clearTimeout(maxTimeoutRef.current);
      maxTimeoutRef.current = null;
    }
    if (recordingTimerRef.current) {
      clearInterval(recordingTimerRef.current);
      recordingTimerRef.current = null;
    }
    
    // Close audio context
    if (audioContextRef.current) {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
      analyserRef.current = null;
    }
    
    // Stop recording
    if (mediaRecorderRef.current?.state === "recording") {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setIsProcessing(true);
      setRecordingTime(0);
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
        body: JSON.stringify({ data: b64, model: selectedModel, context: chatContext.trim() || null }),
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

  // ----------------- TEXT PROMPT (direct to LLM) -----------------
  const submitTextPrompt = async () => {
    if (!textPrompt.trim()) return;
    setIsProcessing(true);
    setError(null);
    const userText = textPrompt.trim();
    setTextPrompt("");
    
    try {
      setConversation((p) => [
        ...p,
        { role: "user", parts: [{ type: "text", text: userText }] },
      ]);
      const res = await fetch("http://localhost:8000/ask_text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: userText, model: selectedModel, context: chatContext.trim() || null }),
      });
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
      console.error("Error in submitTextPrompt:", e);
      setError(`Text error: ${e.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleTextKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submitTextPrompt();
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
      if (chatContext.trim()) {
        form.append("context", chatContext.trim());
      }

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
        
        {/* Context box */}
        <div className="mt-3">
          <div className="flex items-center justify-between mb-1">
            <span className="text-sm text-gray-400">Contexto del chat:</span>
            {chatContext && (
              <button
                onClick={() => setChatContext("")}
                className="text-xs text-red-400 hover:text-red-300"
              >
                Limpiar
              </button>
            )}
          </div>
          <textarea
            value={chatContext}
            onChange={(e) => setChatContext(e.target.value)}
            placeholder="Ej: Eres un cocinero experto que trabaja con cocina tradicional en el restaurante Yotojoro. Responde de forma técnica y concisa..."
            className="w-full bg-gray-800 text-white text-sm px-3 py-2 rounded border border-gray-600 focus:outline-none focus:border-indigo-500 resize-none"
            rows={2}
            disabled={isProcessing || isRecording}
          />
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
            <div className={`max-w-lg p-4 rounded-2xl ${msg.role === "user" ? "bg-blue-600" : "bg-gray-700"}`}>
              {msg.parts[0].type === "audio" ? (
                <p className="italic">Audio message</p>
              ) : msg.parts[0].type === "image" ? (
                <>
                  <img src={msg.parts[0].url} alt="upload" className="max-w-xs mb-2 rounded-lg" />
                  <p>{msg.parts[1].text}</p>
                </>
              ) : msg.role === "model" ? (
                <MarkdownText text={msg.parts[0].text} />
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

      {/* footer with text input and mic button */}
      <footer className="p-4 border-t border-gray-700/50">
        {error && <p className="text-red-400 mb-2 text-center">{error}</p>}
        
        {/* Text input row */}
        <div className="flex items-center gap-2 mb-3">
          <input
            type="text"
            value={textPrompt}
            onChange={(e) => setTextPrompt(e.target.value)}
            onKeyDown={handleTextKeyDown}
            placeholder="Escribe tu mensaje..."
            disabled={isRecording || isProcessing}
            className="flex-1 bg-gray-800 text-white px-4 py-3 rounded-full border border-gray-600 focus:outline-none focus:border-indigo-500 disabled:opacity-50"
          />
          <button
            onClick={submitTextPrompt}
            disabled={!textPrompt.trim() || isRecording || isProcessing}
            className="w-12 h-12 bg-indigo-600 rounded-full flex items-center justify-center disabled:bg-gray-500 hover:bg-indigo-500 transition-colors"
            title="Enviar mensaje"
          >
            <IconSend className="w-5 h-5" />
          </button>
        </div>
        
        {/* Mic button */}
        <div className="flex flex-col items-center">
          <button
            onClick={isRecording ? stopRecording : startRecording}
            disabled={isProcessing}
            className={`w-16 h-16 rounded-full flex items-center justify-center transition-colors ${
              isRecording 
                ? "bg-red-600 hover:bg-red-500 animate-pulse" 
                : isProcessing 
                  ? "bg-gray-500" 
                  : "bg-indigo-600 hover:bg-indigo-500"
            }`}
          >
            {isRecording ? (
              <div className="flex flex-col items-center">
                <div className="w-6 h-6 bg-white rounded" />
              </div>
            ) : isProcessing ? (
              <IconLoader className="w-6 h-6 animate-spin" />
            ) : (
              <IconMic className="w-6 h-6" />
            )}
          </button>
          <p className="text-xs text-gray-500 mt-2">
            {isRecording 
              ? `Grabando… ${recordingTime}s (toca para detener)` 
              : isProcessing 
                ? "Procesando…" 
                : "Toca para hablar (máx 30s)"}
          </p>
        </div>
      </footer>
    </div>
  );
}

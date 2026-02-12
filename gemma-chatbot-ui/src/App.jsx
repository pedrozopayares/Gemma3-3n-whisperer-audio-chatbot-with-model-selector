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

const IconResend = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M1 4v6h6"></path>
    <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"></path>
  </svg>
);

// ----------------- Markdown Renderer -----------------
const MarkdownText = ({ text }) => {
  // Convert LaTeX math to readable text
  const convertLatexToText = (str) => {
    return str
      // Handle display math \[...\] or $$...$$ (multiline - must come first)
      .replace(/\\\[([\s\S]+?)\\\]/g, (_, math) => convertLatexMath(math))
      .replace(/\$\$([\s\S]+?)\$\$/g, (_, math) => convertLatexMath(math))
      // Handle inline math \(...\) or $...$
      .replace(/\\\((.+?)\\\)/g, (_, math) => convertLatexMath(math))
      .replace(/\$([^$\n]+?)\$/g, (_, math) => convertLatexMath(math));
  };

  const convertLatexMath = (math) => {
    return math
      // Clean up whitespace first
      .replace(/\n/g, ' ')
      .replace(/\s+/g, ' ')
      // Operators
      .replace(/\\times/g, '×')
      .replace(/\\div/g, '÷')
      .replace(/\\pm/g, '±')
      .replace(/\\mp/g, '∓')
      .replace(/\\cdot/g, '·')
      .replace(/\\ast/g, '*')
      .replace(/\\star/g, '★')
      // Comparisons
      .replace(/\\leq/g, '≤')
      .replace(/\\geq/g, '≥')
      .replace(/\\neq/g, '≠')
      .replace(/\\approx/g, '≈')
      .replace(/\\equiv/g, '≡')
      .replace(/\\lt/g, '<')
      .replace(/\\gt/g, '>')
      // Fractions: \frac{a}{b} -> a/b (handle nested braces)
      .replace(/\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}/g, '($1 / $2)')
      // Square root: \sqrt{x} -> √x
      .replace(/\\sqrt\s*\{([^}]+)\}/g, '√($1)')
      .replace(/\\sqrt\s*(\d+)/g, '√$1')
      // Superscript: x^{2} or x^2 -> x²
      .replace(/\^{2}/g, '²')
      .replace(/\^2/g, '²')
      .replace(/\^{3}/g, '³')
      .replace(/\^3/g, '³')
      .replace(/\^\{([^}]+)\}/g, '^($1)')
      // Subscript: x_{i} -> x_i (just clean up braces)
      .replace(/_\{([^}]+)\}/g, '_$1')
      // Greek letters
      .replace(/\\alpha/g, 'α')
      .replace(/\\beta/g, 'β')
      .replace(/\\gamma/g, 'γ')
      .replace(/\\delta/g, 'δ')
      .replace(/\\epsilon/g, 'ε')
      .replace(/\\pi/g, 'π')
      .replace(/\\theta/g, 'θ')
      .replace(/\\lambda/g, 'λ')
      .replace(/\\mu/g, 'μ')
      .replace(/\\sigma/g, 'σ')
      .replace(/\\omega/g, 'ω')
      .replace(/\\sum/g, 'Σ')
      .replace(/\\prod/g, 'Π')
      // Arrows
      .replace(/\\rightarrow/g, '→')
      .replace(/\\leftarrow/g, '←')
      .replace(/\\Rightarrow/g, '⇒')
      .replace(/\\Leftarrow/g, '⇐')
      // Misc
      .replace(/\\infty/g, '∞')
      .replace(/\\partial/g, '∂')
      .replace(/\\nabla/g, '∇')
      .replace(/\\therefore/g, '∴')
      .replace(/\\because/g, '∵')
      // Clean up remaining LaTeX commands
      .replace(/\\text\s*\{([^}]+)\}/g, '$1')
      .replace(/\\mathrm\s*\{([^}]+)\}/g, '$1')
      .replace(/\\mathbf\s*\{([^}]+)\}/g, '$1')
      // Remove any remaining backslash commands
      .replace(/\\[a-zA-Z]+/g, '')
      .replace(/\{/g, '')
      .replace(/\}/g, '')
      .replace(/\s+/g, ' ')
      .trim();
  };

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

  // Convert LaTeX to readable text before processing
  const processedText = convertLatexToText(text);
  
  // Split by lines and process
  const lines = processedText.split('\n');
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
  const [ttsEngine, setTtsEngine] = useState("browser"); // "browser" or "native"
  
  // Ref for native TTS audio playback
  const currentAudioRef = useRef(null);
  
  // Toggle TTS and stop any current speech if disabled
  const toggleTTS = () => {
    const newValue = !isTTSEnabled;
    setIsTTSEnabled(newValue);
    if (!newValue) {
      // Stop native audio
      if (currentAudioRef.current) {
        currentAudioRef.current.pause();
        currentAudioRef.current.currentTime = 0;
        currentAudioRef.current = null;
      }
      // Stop browser TTS
      if ("speechSynthesis" in window) {
        window.speechSynthesis.cancel();
      }
    }
  };
  
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

  // Convert frontend conversation to backend history format
  const getHistoryForBackend = () => {
    return conversation
      .filter(msg => {
        // Only include text messages
        const textPart = msg.parts?.find(p => p.type === "text" || p.text);
        return textPart && textPart.text;
      })
      .map(msg => {
        const textPart = msg.parts.find(p => p.type === "text" || p.text);
        return {
          role: msg.role,
          text: textPart.text.replace(/^🎤 "/, '').replace(/"$/, '') // Remove audio emoji prefix
        };
      });
  };

  const processAudio = async (audioBlob) => {
    setIsProcessing(true);
    try {
      const b64 = await blobToBase64(audioBlob);
      // Get history BEFORE adding new message
      const history = getHistoryForBackend();
      // Initially show as audio, will be updated with transcription
      setConversation((p) => [...p, { role: "user", parts: [{ type: "audio" }] }]);
      const res = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          data: b64, 
          model: selectedModel, 
          context: chatContext.trim() || null,
          history: history.length > 0 ? history : null
        }),
      });
      if (!res.ok) {
        const errorText = await res.text();
        console.error("Server error:", errorText);
        throw new Error(errorText || `Server error: ${res.status}`);
      }
      const data = await res.json();
      console.log("Received response:", data);
      const { text, transcription, model, routed_category } = data;
      
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
      
      addReply(text, routed_category ? { model, category: routed_category } : null);
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
      // Get history BEFORE adding new message
      const history = getHistoryForBackend();
      setConversation((p) => [
        ...p,
        { role: "user", parts: [{ type: "text", text: userText }] },
      ]);
      const res = await fetch("http://localhost:8000/ask_text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          text: userText, 
          model: selectedModel, 
          context: chatContext.trim() || null,
          history: history.length > 0 ? history : null
        }),
      });
      if (!res.ok) {
        const errorText = await res.text();
        console.error("Server error:", errorText);
        throw new Error(errorText || `Server error: ${res.status}`);
      }
      const data = await res.json();
      console.log("Received response:", data);
      const { text, model, routed_category } = data;
      addReply(text, routed_category ? { model, category: routed_category } : null);
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

  // ----------------- RESEND MESSAGE -----------------
  const resendMessage = async (messageIndex) => {
    const msg = conversation[messageIndex];
    if (!msg || msg.role !== "user") return;
    
    // Get the text from the message
    const textPart = msg.parts?.find(p => p.type === "text" || p.text);
    if (!textPart?.text) return;
    
    // Clean the text (remove audio emoji prefix if present)
    let userText = textPart.text.replace(/^🎤 "/, '').replace(/"$/, '');
    
    setIsProcessing(true);
    setError(null);
    
    try {
      // Build history from messages BEFORE the one being resent
      const historyBefore = conversation
        .slice(0, messageIndex)
        .filter(m => {
          const tp = m.parts?.find(p => p.type === "text" || p.text);
          return tp && tp.text;
        })
        .map(m => {
          const tp = m.parts.find(p => p.type === "text" || p.text);
          return {
            role: m.role,
            text: tp.text.replace(/^🎤 "/, '').replace(/"$/, '')
          };
        });
      
      // Remove the old response(s) after this message and re-add the user message
      setConversation(prev => {
        const updated = prev.slice(0, messageIndex);
        updated.push({ role: "user", parts: [{ type: "text", text: userText }] });
        return updated;
      });
      
      const res = await fetch("http://localhost:8000/ask_text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          text: userText, 
          model: selectedModel, 
          context: chatContext.trim() || null,
          history: historyBefore.length > 0 ? historyBefore : null
        }),
      });
      
      if (!res.ok) {
        const errorText = await res.text();
        console.error("Server error:", errorText);
        throw new Error(errorText || `Server error: ${res.status}`);
      }
      
      const data = await res.json();
      console.log("Resend response:", data);
      const { text, model, routed_category } = data;
      addReply(text, routed_category ? { model, category: routed_category } : null);
    } catch (e) {
      console.error("Error in resendMessage:", e);
      setError(`Resend error: ${e.message}`);
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

  // Helper function to convert LaTeX math to speakable text
  const cleanLatexForSpeech = (math) => {
    return math
      // Clean up newlines and extra whitespace first
      .replace(/\n/g, ' ')
      .replace(/\s+/g, ' ')
      // Convert \text{} commands to plain text
      .replace(/\\text\s*\{([^}]+)\}/g, '$1')
      // Convert operators to spoken words
      .replace(/\\times/g, ' por ')
      .replace(/\\div/g, ' dividido ')
      .replace(/\\pm/g, ' más o menos ')
      .replace(/\\cdot/g, ' por ')
      .replace(/×/g, ' por ')
      .replace(/÷/g, ' dividido ')
      // Convert fractions to spoken form
      .replace(/\\frac\s*\{([^}]+)\}\s*\{([^}]+)\}/g, '$1 sobre $2')
      // Convert equals and comparisons
      .replace(/=/g, ' igual a ')
      .replace(/\\leq/g, ' menor o igual a ')
      .replace(/\\geq/g, ' mayor o igual a ')
      .replace(/\\neq/g, ' diferente de ')
      .replace(/\\approx/g, ' aproximadamente ')
      .replace(/<=/g, ' menor o igual a ')
      .replace(/>=/g, ' mayor o igual a ')
      .replace(/</g, ' menor que ')
      .replace(/>/g, ' mayor que ')
      // Convert square root
      .replace(/\\sqrt\s*\{([^}]+)\}/g, ' raíz cuadrada de $1')
      .replace(/\\sqrt\s*(\d+)/g, ' raíz cuadrada de $1')
      // Convert exponents
      .replace(/\^{2}/g, ' al cuadrado')
      .replace(/\^2/g, ' al cuadrado')
      .replace(/\^{3}/g, ' al cubo')
      .replace(/\^3/g, ' al cubo')
      .replace(/\^\{([^}]+)\}/g, ' elevado a $1')
      .replace(/\^(\d+)/g, ' elevado a $1')
      // Remove remaining LaTeX commands
      .replace(/\\[a-zA-Z]+/g, ' ')
      // Clean up braces and special chars
      .replace(/[{}]/g, '')
      .replace(/\s+/g, ' ')
      .trim();
  };

  // ----------------- TTS -----------------
  // Clean text for TTS (remove emojis, markdown, LaTeX, etc.)
  const cleanTextForTTS = (text) => {
    // First, convert LaTeX math expressions to speakable text
    let processed = text
      // Handle display math \[...\] or $$...$$ (multiline - must come first)
      .replace(/\\\[([\s\S]+?)\\\]/g, (_, math) => cleanLatexForSpeech(math))
      .replace(/\$\$([\s\S]+?)\$\$/g, (_, math) => cleanLatexForSpeech(math))
      // Handle inline math \(...\) or $...$
      .replace(/\\\((.+?)\\\)/g, (_, math) => cleanLatexForSpeech(math))
      .replace(/\$([^$\n]+?)\$/g, (_, math) => cleanLatexForSpeech(math));
    
    return processed
      // Remove emojis and emoticons
      .replace(/[\u{1F600}-\u{1F64F}]/gu, '')  // Emoticons
      .replace(/[\u{1F300}-\u{1F5FF}]/gu, '')  // Symbols & pictographs
      .replace(/[\u{1F680}-\u{1F6FF}]/gu, '')  // Transport & map symbols
      .replace(/[\u{1F1E0}-\u{1F1FF}]/gu, '')  // Flags
      .replace(/[\u{2600}-\u{26FF}]/gu, '')   // Misc symbols
      .replace(/[\u{2700}-\u{27BF}]/gu, '')   // Dingbats
      .replace(/[\u{1F900}-\u{1F9FF}]/gu, '')  // Supplemental symbols
      .replace(/[\u{1FA00}-\u{1FA6F}]/gu, '')  // Chess symbols
      .replace(/[\u{1FA70}-\u{1FAFF}]/gu, '')  // Symbols extended-A
      .replace(/[\u{231A}-\u{231B}]/gu, '')   // Watch, hourglass
      .replace(/[\u{23E9}-\u{23F3}]/gu, '')   // Media control symbols
      .replace(/[\u{23F8}-\u{23FA}]/gu, '')   // Media control symbols
      .replace(/[\u{25AA}-\u{25AB}]/gu, '')   // Squares
      .replace(/[\u{25B6}]/gu, '')            // Play button
      .replace(/[\u{25C0}]/gu, '')            // Reverse button
      .replace(/[\u{25FB}-\u{25FE}]/gu, '')   // Squares
      .replace(/[\u{2934}-\u{2935}]/gu, '')   // Arrows
      .replace(/[\u{2B05}-\u{2B07}]/gu, '')   // Arrows
      .replace(/[\u{2B1B}-\u{2B1C}]/gu, '')   // Squares
      .replace(/[\u{2B50}]/gu, '')            // Star
      .replace(/[\u{2B55}]/gu, '')            // Circle
      .replace(/[\u{3030}]/gu, '')            // Wavy dash
      .replace(/[\u{303D}]/gu, '')            // Part alternation mark
      .replace(/[\u{3297}]/gu, '')            // Circled ideograph
      .replace(/[\u{3299}]/gu, '')            // Circled ideograph
      // Remove markdown formatting
      .replace(/\*\*(.+?)\*\*/g, '$1')   // **bold** -> bold
      .replace(/__(.+?)__/g, '$1')       // __bold__ -> bold
      .replace(/\*(.+?)\*/g, '$1')       // *italic* -> italic
      .replace(/_(.+?)_/g, '$1')         // _italic_ -> italic
      // Remove bullets and list markers
      .replace(/^[\s]*[-•▪●◦★☆→►▸]\s*/gm, '')  // Remove bullet points at line start
      .replace(/^[\s]*\d+\.\s+/gm, '')   // Remove numbered list markers (1. 2. etc)
      // Clean up symbols that TTS might pronounce
      .replace(/[*_~`#>|]/g, '')         // Remove remaining markdown symbols
      .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')  // [link text](url) -> link text
      .replace(/```[\s\S]*?```/g, '')    // Remove code blocks
      .replace(/`([^`]+)`/g, '$1')       // `code` -> code
      // Convert math division symbols to speakable text
      .replace(/(\d+)\s*\/\s*(\d+)/g, '$1 sobre $2')  // 1/2 -> 1 sobre 2
      .replace(/\(([^)]+)\)\/\(([^)]+)\)/g, '$1 sobre $2')  // (a)/(b) -> a sobre b
      // Improve flow for spoken text
      .replace(/:\s*$/gm, '. ')          // Colons at end of line -> period
      .replace(/\n+/g, '. ')             // Line breaks -> pauses
      .replace(/\s+/g, ' ')              // Multiple spaces -> single
      .replace(/\.+/g, '.')              // Multiple periods -> single
      .replace(/,\s*\./g, '.')           // ", ." -> "."
      .replace(/\.\s*,/g, '.')           // ". ," -> "."
      .trim();
  };

  // Fallback to browser speechSynthesis
  const speakWithBrowserTTS = (cleanText) => {
    if (!("speechSynthesis" in window)) return;
    
    const utter = new SpeechSynthesisUtterance(cleanText);
    utter.lang = "es-MX";
    utter.rate = 1.0;
    utter.pitch = 1.0;
    
    // Try to find a Spanish voice
    const voices = window.speechSynthesis.getVoices();
    const spanishVoice = voices.find(v => v.lang.startsWith("es")) || 
                         voices.find(v => v.lang.includes("ES") || v.lang.includes("MX"));
    if (spanishVoice) {
      utter.voice = spanishVoice;
    }
    
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utter);
  };

  // Native TTS using backend (macOS native voices)
  const speakWithNativeTTS = async (cleanText) => {
    try {
      const res = await fetch("http://localhost:8000/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: cleanText, voice: "Paulina", rate: 180 }),
      });
      
      if (!res.ok) {
        throw new Error("TTS request failed");
      }
      
      const data = await res.json();
      
      // Check if native TTS is available
      if (data.error || !data.audio) {
        console.log("Native TTS unavailable, using browser fallback");
        speakWithBrowserTTS(cleanText);
        return;
      }
      
      // Play the audio
      const audioData = `data:audio/wav;base64,${data.audio}`;
      const audio = new Audio(audioData);
      currentAudioRef.current = audio;
      
      audio.onended = () => {
        currentAudioRef.current = null;
      };
      
      audio.onerror = () => {
        console.log("Audio playback error, using browser fallback");
        currentAudioRef.current = null;
        speakWithBrowserTTS(cleanText);
      };
      
      await audio.play();
      console.log(`Playing TTS with ${data.engine} voice: ${data.voice}`);
      
    } catch (e) {
      console.log("Native TTS error, using browser fallback:", e.message);
      speakWithBrowserTTS(cleanText);
    }
  };

  const addReply = (text, modelInfo = null) => {
    setConversation((p) => [...p, { role: "model", parts: [{ text }], modelInfo }]);
    
    if (isTTSEnabled) {
      const cleanText = cleanTextForTTS(text);
      if (ttsEngine === "native") {
        // System TTS (macOS/Windows/Android native voices via backend)
        speakWithNativeTTS(cleanText);
      } else {
        // Browser TTS (Web Speech API)
        speakWithBrowserTTS(cleanText);
      }
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
          <div className="flex items-center space-x-3">
            <label className="flex items-center space-x-2">
              <span className="text-sm">TTS</span>
              <input type="checkbox" checked={isTTSEnabled} onChange={toggleTTS} />
            </label>
            {isTTSEnabled && (
              <select
                value={ttsEngine}
                onChange={(e) => setTtsEngine(e.target.value)}
                className="bg-gray-800 text-white text-xs rounded px-2 py-1 border border-gray-600 focus:outline-none focus:border-indigo-500"
              >
                <option value="browser">Audio del navegador</option>
                <option value="native">Audio del sistema</option>
              </select>
            )}
          </div>
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
            <div className={`group relative max-w-lg p-4 rounded-2xl ${msg.role === "user" ? "bg-blue-600" : "bg-gray-700"}`}>
              {msg.parts[0].type === "audio" ? (
                <p className="italic">Audio message</p>
              ) : msg.parts[0].type === "image" ? (
                <>
                  <img src={msg.parts[0].url} alt="upload" className="max-w-xs mb-2 rounded-lg" />
                  <p>{msg.parts[1].text}</p>
                </>
              ) : msg.role === "model" ? (
                <>
                  <MarkdownText text={msg.parts[0].text} />
                  {msg.modelInfo && (
                    <p className="text-xs text-gray-400 mt-2 pt-2 border-t border-gray-600">
                      Auto: {msg.modelInfo.model} ({msg.modelInfo.category})
                    </p>
                  )}
                </>
              ) : (
                <>
                  <p>{msg.parts[0].text}</p>
                  {/* Resend button for user text messages */}
                  <button
                    onClick={() => resendMessage(i)}
                    disabled={isProcessing || isRecording}
                    className="absolute -left-8 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity p-1.5 bg-gray-700 hover:bg-gray-600 rounded-full disabled:opacity-30"
                    title="Reenviar mensaje"
                  >
                    <IconResend className="w-4 h-4" />
                  </button>
                </>
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

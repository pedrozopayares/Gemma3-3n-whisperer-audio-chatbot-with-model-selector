import { useRef, useCallback } from "react";
import { requestTTS } from "../services/api";

/**
 * Helper to convert LaTeX to speakable text.
 */
function cleanLatexForSpeech(math) {
  return math
    .replace(/\n/g, " ")
    .replace(/\s+/g, " ")
    .replace(/\\text\s*\{([^}]+)\}/g, "$1")
    .replace(/\\times/g, " por ")
    .replace(/\\div/g, " dividido ")
    .replace(/\\pm/g, " más o menos ")
    .replace(/\\cdot/g, " por ")
    .replace(/×/g, " por ")
    .replace(/÷/g, " dividido ")
    .replace(/\\frac\s*\{([^}]+)\}\s*\{([^}]+)\}/g, "$1 sobre $2")
    .replace(/=/g, " igual a ")
    .replace(/\\leq/g, " menor o igual a ")
    .replace(/\\geq/g, " mayor o igual a ")
    .replace(/\\neq/g, " diferente de ")
    .replace(/\\approx/g, " aproximadamente ")
    .replace(/<=/g, " menor o igual a ")
    .replace(/>=/g, " mayor o igual a ")
    .replace(/</g, " menor que ")
    .replace(/>/g, " mayor que ")
    .replace(/\\sqrt\s*\{([^}]+)\}/g, " raíz cuadrada de $1")
    .replace(/\\sqrt\s*(\d+)/g, " raíz cuadrada de $1")
    .replace(/\^{2}/g, " al cuadrado")
    .replace(/\^2/g, " al cuadrado")
    .replace(/\^{3}/g, " al cubo")
    .replace(/\^3/g, " al cubo")
    .replace(/\^\{([^}]+)\}/g, " elevado a $1")
    .replace(/\^(\d+)/g, " elevado a $1")
    .replace(/\\[a-zA-Z]+/g, " ")
    .replace(/[{}]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

/**
 * Clean text for TTS (remove emojis, markdown, LaTeX).
 */
function cleanTextForTTS(text) {
  let processed = text
    .replace(/\\\[([\s\S]+?)\\\]/g, (_, math) => cleanLatexForSpeech(math))
    .replace(/\$\$([\s\S]+?)\$\$/g, (_, math) => cleanLatexForSpeech(math))
    .replace(/\\\((.+?)\\\)/g, (_, math) => cleanLatexForSpeech(math))
    .replace(/\$([^$\n]+?)\$/g, (_, math) => cleanLatexForSpeech(math));

  return processed
    .replace(/[\u{1F600}-\u{1F64F}]/gu, "")
    .replace(/[\u{1F300}-\u{1F5FF}]/gu, "")
    .replace(/[\u{1F680}-\u{1F6FF}]/gu, "")
    .replace(/[\u{1F1E0}-\u{1F1FF}]/gu, "")
    .replace(/[\u{2600}-\u{26FF}]/gu, "")
    .replace(/[\u{2700}-\u{27BF}]/gu, "")
    .replace(/[\u{1F900}-\u{1F9FF}]/gu, "")
    .replace(/[\u{1FA00}-\u{1FA6F}]/gu, "")
    .replace(/[\u{1FA70}-\u{1FAFF}]/gu, "")
    .replace(/\*\*(.+?)\*\*/g, "$1")
    .replace(/__(.+?)__/g, "$1")
    .replace(/\*(.+?)\*/g, "$1")
    .replace(/_(.+?)_/g, "$1")
    .replace(/^[\s]*[-•▪●◦★☆→►▸]\s*/gm, "")
    .replace(/^[\s]*\d+\.\s+/gm, "")
    .replace(/[*_~`#>|]/g, "")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/```[\s\S]*?```/g, "")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/(\d+)\s*\/\s*(\d+)/g, "$1 sobre $2")
    .replace(/:\s*$/gm, ". ")
    .replace(/\n+/g, ". ")
    .replace(/\s+/g, " ")
    .replace(/\.+/g, ".")
    .replace(/,\s*\./g, ".")
    .replace(/\.\s*,/g, ".")
    .trim();
}

/**
 * Custom hook for TTS (browser + native engine).
 */
export default function useTTS() {
  const currentAudioRef = useRef(null);

  const stopSpeech = useCallback(() => {
    if (currentAudioRef.current) {
      currentAudioRef.current.pause();
      currentAudioRef.current.currentTime = 0;
      currentAudioRef.current = null;
    }
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
  }, []);

  const speakBrowser = useCallback((text) => {
    if (!("speechSynthesis" in window)) return;
    const utter = new SpeechSynthesisUtterance(text);
    utter.lang = "es-MX";
    utter.rate = 1.0;
    utter.pitch = 1.0;
    const voices = window.speechSynthesis.getVoices();
    const spanishVoice =
      voices.find((v) => v.lang.startsWith("es")) ||
      voices.find((v) => v.lang.includes("ES") || v.lang.includes("MX"));
    if (spanishVoice) utter.voice = spanishVoice;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utter);
  }, []);

  const speakNative = useCallback(
    async (text) => {
      try {
        const data = await requestTTS({ text });
        if (data.error || !data.audio) {
          speakBrowser(text);
          return;
        }
        const audio = new Audio(`data:audio/wav;base64,${data.audio}`);
        currentAudioRef.current = audio;
        audio.onended = () => (currentAudioRef.current = null);
        audio.onerror = () => {
          currentAudioRef.current = null;
          speakBrowser(text);
        };
        await audio.play();
      } catch {
        speakBrowser(text);
      }
    },
    [speakBrowser]
  );

  const speak = useCallback(
    (rawText, engine = "browser") => {
      const clean = cleanTextForTTS(rawText);
      if (engine === "native") {
        speakNative(clean);
      } else {
        speakBrowser(clean);
      }
    },
    [speakBrowser, speakNative]
  );

  return { speak, stopSpeech };
}

import React, { useState, useRef } from "react";
import { IconSend, IconMic, IconLoader, IconClose } from "./Icons";

/**
 * ChatGPT-style input area with text, microphone, and image upload.
 *
 * Props:
 * - onSendText(text): send text message
 * - onStartRecording(): start audio recording
 * - onStopRecording(): stop audio recording
 * - isRecording: whether mic is capturing
 * - isProcessing: whether waiting for response
 * - recordingTime: seconds elapsed
 * - onSendImage({ file, prompt }): send image + prompt
 * - contextValue: current chat context
 * - onContextChange(value): update chat context
 */
export default function ChatInput({
  onSendText,
  onStartRecording,
  onStopRecording,
  isRecording,
  isProcessing,
  recordingTime,
  onSendImage,
  contextValue,
  onContextChange,
}) {
  const [textPrompt, setTextPrompt] = useState("");
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [showContext, setShowContext] = useState(false);
  const fileInputRef = useRef(null);
  const textareaRef = useRef(null);

  const handleSend = () => {
    // If an image is attached, always send via image endpoint
    if (imageFile) {
      const prompt = textPrompt.trim() || "Describe esta imagen";
      onSendImage({ file: imageFile, prompt });
      setImageFile(null);
      setImagePreview(null);
      setTextPrompt("");
      if (textareaRef.current) textareaRef.current.style.height = "auto";
      return;
    }
    if (!textPrompt.trim()) return;
    onSendText(textPrompt.trim());
    setTextPrompt("");
    // Reset textarea height
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleImageSelect = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setImageFile(file);
    setImagePreview(URL.createObjectURL(file));
  };

  const removeImage = () => {
    setImageFile(null);
    setImagePreview(null);
  };

  const handleTextareaInput = (e) => {
    setTextPrompt(e.target.value);
    // Auto-resize
    e.target.style.height = "auto";
    e.target.style.height = Math.min(e.target.scrollHeight, 200) + "px";
  };

  const disabled = isRecording || isProcessing;

  return (
    <div className="border-t border-border bg-chat-bg pb-[env(safe-area-inset-bottom)]">
      {/* Context toggle */}
      {showContext && (
        <div className="px-4 pt-3">
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs text-tertiary font-medium">Contexto del chat</span>
            <div className="flex items-center gap-2">
              {contextValue && (
                <button
                  onClick={() => onContextChange("")}
                  className="text-xs text-red-400 hover:text-red-300"
                >
                  Limpiar
                </button>
              )}
              <button
                onClick={() => setShowContext(false)}
                className="text-xs text-tertiary hover:text-primary"
              >
                Ocultar
              </button>
            </div>
          </div>
          <textarea
            value={contextValue}
            onChange={(e) => onContextChange(e.target.value)}
            placeholder="Ej: Eres un cocinero experto que trabaja con cocina tradicional…"
            className="w-full bg-input-bg text-primary text-sm px-3 py-2 rounded-lg border border-border focus:outline-none focus:border-accent resize-none placeholder:text-placeholder"
            rows={2}
            disabled={disabled}
          />
        </div>
      )}

      {/* Image preview */}
      {imagePreview && (
        <div className="px-4 pt-3">
          <div className="relative inline-block">
            <img src={imagePreview} alt="preview" className="max-h-32 rounded-lg border border-border" />
            <button
              onClick={removeImage}
              className="absolute -top-2 -right-2 w-6 h-6 bg-red-500 hover:bg-red-400 rounded-full flex items-center justify-center"
            >
              <IconClose className="w-3 h-3 text-white" />
            </button>
          </div>
        </div>
      )}

      {/* Main input row */}
      <div className="px-4 py-3">
        <div className="flex items-end gap-2 max-w-3xl mx-auto">
          {/* Attachment + context buttons */}
          <div className="flex items-center gap-1 pb-1">
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={handleImageSelect}
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={disabled}
              className="p-2.5 rounded-lg hover:bg-hover text-tertiary hover:text-primary transition-colors disabled:opacity-40 active:scale-95"
              title="Adjuntar imagen"
            >
              <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" />
              </svg>
            </button>
            <button
              onClick={() => setShowContext(!showContext)}
              disabled={disabled}
              className={`p-2.5 rounded-lg hover:bg-hover transition-colors disabled:opacity-40 active:scale-95 ${showContext ? "text-accent bg-accent/10" : "text-tertiary hover:text-primary"}`}
              title="Contexto del chat"
            >
              <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="3" />
                <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
              </svg>
            </button>
          </div>

          {/* Text area */}
          <div className="flex-1 relative">
            <textarea
              ref={textareaRef}
              value={textPrompt}
              onChange={handleTextareaInput}
              onKeyDown={handleKeyDown}
              placeholder={isRecording ? `Grabando… ${recordingTime}s` : imageFile ? "Escribe sobre la imagen o envía para describirla…" : "Envía un mensaje…"}
              disabled={disabled}
              className="w-full bg-input-bg text-primary text-sm px-4 py-3 rounded-2xl border border-border focus:outline-none focus:border-accent resize-none placeholder:text-placeholder disabled:opacity-50 max-h-[200px]"
              rows={1}
              style={{ minHeight: "44px" }}
            />
          </div>

          {/* Action buttons */}
          <div className="flex items-center gap-1 pb-1">
            {/* Mic button */}
            <button
              onClick={isRecording ? onStopRecording : onStartRecording}
              disabled={isProcessing}
              className={`p-2.5 rounded-full transition-colors ${
                isRecording
                  ? "bg-red-500 hover:bg-red-400 text-white animate-pulse"
                  : "hover:bg-hover text-tertiary hover:text-primary"
              } disabled:opacity-40`}
              title={isRecording ? `Grabando… ${recordingTime}s (click para detener)` : "Grabar audio (máx 30s)"}
            >
              {isRecording ? (
                <div className="w-5 h-5 flex items-center justify-center">
                  <div className="w-3 h-3 bg-white rounded-sm" />
                </div>
              ) : (
                <IconMic className="w-5 h-5" />
              )}
            </button>

            {/* Send button */}
            <button
              onClick={handleSend}
              disabled={disabled || (!textPrompt.trim() && !imageFile)}
              className="p-2.5 rounded-full bg-accent hover:bg-accent-hover text-white transition-colors disabled:opacity-40 disabled:bg-accent/50"
              title="Enviar"
            >
              {isProcessing ? (
                <IconLoader className="w-5 h-5 animate-spin" />
              ) : (
                <IconSend className="w-5 h-5" />
              )}
            </button>
          </div>
        </div>

        <p className="text-center text-xs text-tertiary mt-2">
          {isRecording
            ? `Grabando… ${recordingTime}s — toca stop para detener`
            : isProcessing
            ? "Procesando…"
            : "Yotojoro IA puede cometer errores. Verifica la información."}
        </p>
      </div>
    </div>
  );
}

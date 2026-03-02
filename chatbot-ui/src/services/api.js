/**
 * API service for communicating with the backend server.
 */

const API_BASE = "http://localhost:8000";

/**
 * Fetch available models from the backend.
 */
export async function fetchModels() {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) throw new Error("Failed to fetch models");
  return res.json();
}

/**
 * Clear model cache on the backend.
 */
export async function clearModelCache() {
  const res = await fetch(`${API_BASE}/models/cache`, { method: "DELETE" });
  if (!res.ok) throw new Error("Failed to clear cache");
  return res.json();
}

/**
 * Send audio to the backend for transcription + LLM response.
 */
export async function sendAudio({ audioBase64, model, context, history }) {
  const res = await fetch(`${API_BASE}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      data: audioBase64,
      model,
      context: context || null,
      history: history?.length > 0 ? history : null,
    }),
  });
  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(errorText || `Server error: ${res.status}`);
  }
  return res.json();
}

/**
 * Send text to the backend for LLM response.
 */
export async function sendText({ text, model, context, history }) {
  const res = await fetch(`${API_BASE}/ask_text`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text,
      model,
      context: context || null,
      history: history?.length > 0 ? history : null,
    }),
  });
  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(errorText || `Server error: ${res.status}`);
  }
  return res.json();
}

/**
 * Send image + prompt to the backend for vision LLM response.
 */
export async function sendImage({ imageFile, prompt, context }) {
  const form = new FormData();
  form.append("prompt", prompt);
  form.append("image", imageFile, imageFile.name);
  if (context) form.append("context", context);

  const res = await fetch(`${API_BASE}/ask_image`, { method: "POST", body: form });
  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(errorText || `Server error: ${res.status}`);
  }
  return res.json();
}

/**
 * Request TTS audio from the backend.
 */
export async function requestTTS({ text, voice = "Paulina", rate = 180 }) {
  const res = await fetch(`${API_BASE}/tts`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, voice, rate }),
  });
  if (!res.ok) throw new Error("TTS request failed");
  return res.json();
}

/**
 * Fetch the text content of a document from the backend.
 */
export async function fetchDocumentText(filePath) {
  const res = await fetch(`${API_BASE}/documents/${encodeURIComponent(filePath)}/text`);
  if (!res.ok) throw new Error("Failed to fetch document text");
  return res.json();
}

/**
 * Get the URL for downloading/viewing a raw document.
 */
export function getDocumentUrl(filePath) {
  return `${API_BASE}/documents/${encodeURIComponent(filePath)}`;
}

/**
 * Fetch RAG documents list.
 */
export async function fetchRAGDocuments() {
  const res = await fetch(`${API_BASE}/rag/documents`);
  if (!res.ok) throw new Error("Failed to fetch RAG documents");
  return res.json();
}

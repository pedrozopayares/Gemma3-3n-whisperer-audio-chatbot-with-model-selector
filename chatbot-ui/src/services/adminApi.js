/**
 * Admin API service for the administration panel.
 */

const API_BASE = import.meta.env.VITE_API_BASE || "";

/* ── Configuration ─────────────────────────────────────── */

export async function getConfig() {
  const res = await fetch(`${API_BASE}/admin/config`);
  if (!res.ok) throw new Error("Error al obtener configuración");
  return res.json();
}

export async function updateConfig(partial) {
  const res = await fetch(`${API_BASE}/admin/config`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(partial),
  });
  if (!res.ok) throw new Error("Error al actualizar configuración");
  return res.json();
}

/* ── Services Status ───────────────────────────────────── */

export async function getServicesStatus() {
  const res = await fetch(`${API_BASE}/admin/services/status`);
  if (!res.ok) throw new Error("Error al obtener estado de servicios");
  return res.json();
}

/* ── Model Management ──────────────────────────────────── */

export async function pullModel(modelName, onProgress) {
  const res = await fetch(`${API_BASE}/admin/models/pull?model=${encodeURIComponent(modelName)}`, {
    method: "POST",
  });
  if (!res.ok) throw new Error("Error al descargar modelo");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split("\n");
    buffer = lines.pop(); // keep incomplete line

    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const data = JSON.parse(line);
        if (onProgress) onProgress(data);
      } catch {
        // ignore malformed lines
      }
    }
  }
}

export async function deleteModel(modelName) {
  const res = await fetch(`${API_BASE}/admin/models/${encodeURIComponent(modelName)}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error("Error al eliminar modelo");
  return res.json();
}

/* ── Service Tests ─────────────────────────────────────── */

export async function testOllama(model) {
  const res = await fetch(`${API_BASE}/admin/test/ollama?model=${encodeURIComponent(model)}`, {
    method: "POST",
  });
  if (!res.ok) throw new Error("Error al probar Ollama");
  return res.json();
}

export async function testWhisper() {
  const res = await fetch(`${API_BASE}/admin/test/whisper`, { method: "POST" });
  if (!res.ok) throw new Error("Error al probar Whisper");
  return res.json();
}

/* ── RAG ───────────────────────────────────────────────── */

export async function getRagStatus() {
  const res = await fetch(`${API_BASE}/rag/status`);
  if (!res.ok) throw new Error("Error al obtener RAG status");
  return res.json();
}

export async function syncRag() {
  const res = await fetch(`${API_BASE}/rag/sync`, { method: "POST" });
  if (!res.ok) throw new Error("Error al sincronizar RAG");
  return res.json();
}

export async function rebuildRag() {
  const res = await fetch(`${API_BASE}/rag/rebuild`, { method: "POST" });
  if (!res.ok) throw new Error("Error al reconstruir RAG");
  return res.json();
}

export async function scanDocuments() {
  const res = await fetch(`${API_BASE}/rag/scan`);
  if (!res.ok) throw new Error("Error al escanear documentos");
  return res.json();
}

export async function removeRagDocument(filePath) {
  const res = await fetch(`${API_BASE}/rag/documents/${encodeURIComponent(filePath)}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error("Error al eliminar documento del índice");
  return res.json();
}

/* ── Health ────────────────────────────────────────────── */

export async function getHealth() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error("Error al obtener health");
  return res.json();
}

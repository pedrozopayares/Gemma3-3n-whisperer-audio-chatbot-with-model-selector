import React, { useState, useEffect } from "react";
import { IconClose, IconDownload, IconDocument } from "./Icons";
import { fetchDocumentText, getDocumentUrl } from "../services/api";

/**
 * Panel to view RAG document content inline.
 * Supports pdf (iframe), txt/md/csv (text), and shows extracted
 * text for docx/xlsx with download link for originals.
 *
 * Props:
 * - chunk: { source, content, section, relevance, metadata? }
 * - onClose(): close the panel
 */
export default function DocumentViewer({ chunk, onClose }) {
  const [fullText, setFullText] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState("snippet"); // "snippet" | "full" | "preview"

  const source = chunk?.source || chunk?.metadata?.source || "";
  const ext = source.split(".").pop()?.toLowerCase() || "";
  const canPreview = ["pdf", "txt", "md", "html", "htm", "png", "jpg", "jpeg"].includes(ext);
  const docUrl = source ? getDocumentUrl(source) : null;

  // Load full document text when switching to "full" tab
  useEffect(() => {
    if (activeTab !== "full" || fullText !== null || !source) return;

    setLoading(true);
    setError(null);
    fetchDocumentText(source)
      .then((data) => setFullText(data.content || ""))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [activeTab, source, fullText]);

  if (!chunk) return null;

  return (
    <div className="flex flex-col h-full bg-panel border-l border-border w-[420px] shrink-0">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border">
        <div className="flex items-center gap-2 min-w-0">
          <IconDocument className="w-5 h-5 text-accent shrink-0" />
          <div className="min-w-0">
            <h3 className="text-sm font-semibold text-primary truncate">
              {source || "Documento"}
            </h3>
            {chunk.section && (
              <p className="text-xs text-tertiary truncate">{chunk.section}</p>
            )}
          </div>
        </div>
        <div className="flex items-center gap-1 shrink-0">
          {docUrl && (
            <a
              href={docUrl}
              download
              className="p-1.5 rounded-lg hover:bg-hover text-tertiary hover:text-primary transition-colors"
              title="Descargar documento"
            >
              <IconDownload className="w-4 h-4" />
            </a>
          )}
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-hover text-tertiary hover:text-primary transition-colors"
            title="Cerrar"
          >
            <IconClose className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Tab bar */}
      <div className="flex border-b border-border px-2">
        <button
          onClick={() => setActiveTab("snippet")}
          className={`px-3 py-2 text-xs font-medium transition-colors border-b-2 ${
            activeTab === "snippet"
              ? "text-accent border-accent"
              : "text-tertiary border-transparent hover:text-primary"
          }`}
        >
          Fragmento
        </button>
        <button
          onClick={() => setActiveTab("full")}
          className={`px-3 py-2 text-xs font-medium transition-colors border-b-2 ${
            activeTab === "full"
              ? "text-accent border-accent"
              : "text-tertiary border-transparent hover:text-primary"
          }`}
        >
          Texto completo
        </button>
        {canPreview && (
          <button
            onClick={() => setActiveTab("preview")}
            className={`px-3 py-2 text-xs font-medium transition-colors border-b-2 ${
              activeTab === "preview"
                ? "text-accent border-accent"
                : "text-tertiary border-transparent hover:text-primary"
            }`}
          >
            Vista previa
          </button>
        )}
      </div>

      {/* Relevance badge */}
      {chunk.relevance !== undefined && (
        <div className="px-4 py-2 flex items-center gap-2">
          <div className="h-1.5 flex-1 bg-border rounded-full overflow-hidden">
            <div
              className="h-full bg-accent rounded-full transition-all"
              style={{ width: `${(chunk.relevance * 100).toFixed(0)}%` }}
            />
          </div>
          <span className="text-xs text-accent font-medium">
            {(chunk.relevance * 100).toFixed(0)}% relevancia
          </span>
        </div>
      )}

      {/* Content area */}
      <div className="flex-1 overflow-y-auto">
        {activeTab === "snippet" && (
          <div className="p-4">
            <pre className="text-sm text-primary whitespace-pre-wrap leading-relaxed font-sans">
              {chunk.content || "Sin contenido disponible."}
            </pre>
          </div>
        )}

        {activeTab === "full" && (
          <div className="p-4">
            {loading && (
              <div className="flex items-center justify-center py-12">
                <div className="animate-spin w-6 h-6 border-2 border-accent border-t-transparent rounded-full" />
                <span className="ml-3 text-sm text-tertiary">Cargando documento…</span>
              </div>
            )}
            {error && (
              <div className="text-sm text-red-400 bg-red-500/10 p-3 rounded-lg">
                Error al cargar: {error}
              </div>
            )}
            {!loading && !error && fullText !== null && (
              <pre className="text-sm text-primary whitespace-pre-wrap leading-relaxed font-sans">
                {fullText || "Documento vacío."}
              </pre>
            )}
          </div>
        )}

        {activeTab === "preview" && canPreview && (
          <div className="h-full">
            {["pdf"].includes(ext) && (
              <iframe
                src={docUrl}
                className="w-full h-full border-0"
                title="Document preview"
              />
            )}
            {["png", "jpg", "jpeg"].includes(ext) && (
              <div className="p-4 flex items-center justify-center">
                <img src={docUrl} alt={source} className="max-w-full rounded-lg" />
              </div>
            )}
            {["txt", "md", "html", "htm", "csv"].includes(ext) && (
              <iframe
                src={docUrl}
                className="w-full h-full border-0 bg-white"
                title="Document preview"
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
}

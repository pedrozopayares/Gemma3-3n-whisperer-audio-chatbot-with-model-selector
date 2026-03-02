import React from "react";
import MarkdownText from "./MarkdownText";
import { IconResend, IconDocument } from "./Icons";

/**
 * Renders a single chat message bubble.
 *
 * Props:
 * - message: { role, parts, modelInfo?, ragChunks? }
 * - index: message index in conversation
 * - onResend(index): resend a user message
 * - onViewDocument(source): open a RAG source in the document viewer
 * - isProcessing: whether the model is currently processing
 */
export default function ChatMessage({ message, index, onResend, onViewDocument, isProcessing }) {
  const isUser = message.role === "user";
  const part0 = message.parts?.[0];

  return (
    <div className={`flex gap-4 px-4 py-6 ${isUser ? "" : "bg-msg-assistant"}`}>
      {/* Avatar */}
      <div className={`w-8 h-8 rounded-full shrink-0 flex items-center justify-center text-sm font-bold ${
        isUser ? "bg-user-avatar text-white" : "bg-bot-avatar text-white"
      }`}>
        {isUser ? "U" : "Y"}
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0 max-w-3xl">
        {/* User messages */}
        {isUser && (
          <div className="group relative">
            {part0?.type === "audio" ? (
              <p className="italic text-secondary">Mensaje de audio…</p>
            ) : part0?.type === "image" ? (
              <div>
                <img src={part0.url} alt="upload" className="max-w-xs mb-2 rounded-lg" />
                <p className="text-primary">{message.parts[1]?.text}</p>
              </div>
            ) : (
              <p className="text-primary">{part0?.text}</p>
            )}

            {/* Resend button */}
            {part0?.type !== "audio" && (
              <button
                onClick={() => onResend(index)}
                disabled={isProcessing}
                className="absolute -right-10 top-0 opacity-0 group-hover:opacity-100 transition-opacity p-1.5 rounded-lg hover:bg-hover text-tertiary hover:text-primary disabled:opacity-30"
                title="Reenviar mensaje"
              >
                <IconResend className="w-4 h-4" />
              </button>
            )}
          </div>
        )}

        {/* Assistant messages */}
        {!isUser && (
          <div>
            <MarkdownText text={part0?.text || ""} />

            {/* Model info badge */}
            {message.modelInfo && (
              <p className="text-xs text-tertiary mt-3 pt-2 border-t border-border">
                Auto: {message.modelInfo.model} ({message.modelInfo.category})
              </p>
            )}

            {/* RAG Sources */}
            {message.ragChunks?.length > 0 && (
              <div className="mt-3 pt-3 border-t border-border">
                <p className="text-xs text-tertiary mb-2 font-medium">
                  📚 Fuentes consultadas ({message.ragChunks.length})
                </p>
                <div className="flex flex-wrap gap-2">
                  {message.ragChunks.map((chunk, idx) => (
                    <button
                      key={idx}
                      onClick={() => onViewDocument(chunk)}
                      className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-source-chip hover:bg-source-chip-hover text-sm text-source-chip-text transition-colors border border-border"
                      title={`Ver: ${chunk.source || "Documento"} — Relevancia: ${chunk.relevance !== undefined ? (chunk.relevance * 100).toFixed(0) + "%" : "N/A"}`}
                    >
                      <IconDocument className="w-3.5 h-3.5" />
                      <span className="truncate max-w-[180px]">
                        {chunk.source || "Documento"}
                      </span>
                      {chunk.relevance !== undefined && (
                        <span className="text-xs text-accent opacity-80">
                          {(chunk.relevance * 100).toFixed(0)}%
                        </span>
                      )}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

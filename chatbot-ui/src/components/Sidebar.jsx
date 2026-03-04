import React, { useState } from "react";
import { IconPlus, IconTrash, IconEdit, IconChat, IconCheck, IconSidebar, IconSettings } from "./Icons";

/**
 * Sidebar component — ChatGPT-style conversation list.
 *
 * Props:
 * - chats: array of { id, title, updatedAt }
 * - activeChatId: currently selected chat ID
 * - onSelectChat(id): switch to chat
 * - onNewChat(): create a new chat
 * - onDeleteChat(id): delete a chat
 * - onRenameChat(id, newTitle): rename a chat
 * - isCollapsed / onToggleCollapse: sidebar visibility
 * - modelSelector: JSX element for model dropdown
 * - ttsControls: JSX element for TTS toggles
 */
export default function Sidebar({
  chats,
  activeChatId,
  onSelectChat,
  onNewChat,
  onDeleteChat,
  onRenameChat,
  isCollapsed,
  onToggleCollapse,
  modelSelector,
  ttsControls,
}) {
  const [editingId, setEditingId] = useState(null);
  const [editTitle, setEditTitle] = useState("");
  const [confirmDeleteId, setConfirmDeleteId] = useState(null);

  const startRename = (chat) => {
    setEditingId(chat.id);
    setEditTitle(chat.title);
  };

  const commitRename = () => {
    if (editingId && editTitle.trim()) {
      onRenameChat(editingId, editTitle.trim());
    }
    setEditingId(null);
    setEditTitle("");
  };

  const handleDelete = (id) => {
    if (confirmDeleteId === id) {
      onDeleteChat(id);
      setConfirmDeleteId(null);
    } else {
      setConfirmDeleteId(id);
      setTimeout(() => setConfirmDeleteId(null), 3000);
    }
  };

  // Collapsed state — just show toggle button
  if (isCollapsed) {
    return (
      <div className="flex flex-col items-center py-3 px-1 bg-sidebar border-r border-border w-[50px] shrink-0">
        <button
          onClick={onToggleCollapse}
          className="p-2.5 rounded-lg bg-hover hover:bg-border transition-colors text-secondary hover:text-primary active:scale-95"
          title="Abrir sidebar"
        >
          <IconSidebar className="w-5 h-5" />
        </button>
        <button
          onClick={onNewChat}
          className="mt-3 p-2.5 rounded-lg bg-hover hover:bg-border transition-colors text-secondary hover:text-primary active:scale-95"
          title="Nuevo chat"
        >
          <IconPlus className="w-5 h-5" />
        </button>
      </div>
    );
  }

  return (
    <div className="flex flex-col bg-sidebar border-r border-border w-[260px] shrink-0 h-full">
      {/* Header */}
      <div className="flex items-center justify-between p-3 border-b border-border">
        <button
          onClick={onToggleCollapse}
          className="p-2.5 rounded-lg bg-hover hover:bg-border transition-colors text-secondary hover:text-primary active:scale-95"
          title="Cerrar sidebar"
        >
          <IconSidebar className="w-5 h-5" />
        </button>
        <button
          onClick={onNewChat}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-accent hover:bg-accent-hover text-white text-sm font-semibold transition-colors active:scale-95 shadow-sm"
        >
          <IconPlus className="w-4 h-4" />
          Nuevo chat
        </button>
      </div>

      {/* Chat list */}
      <div className="flex-1 overflow-y-auto py-2 scrollbar-thin">
        {chats.length === 0 && (
          <p className="text-center text-tertiary text-sm px-3 py-8">
            No hay conversaciones.
            <br />
            Inicia una nueva.
          </p>
        )}
        {chats.map((chat) => {
          const isActive = chat.id === activeChatId;
          const isEditing = editingId === chat.id;

          return (
            <div
              key={chat.id}
              className={`group flex items-center gap-2 px-3 py-3 mx-2 rounded-lg cursor-pointer transition-colors ${
                isActive
                  ? "bg-active-chat text-primary shadow-sm"
                  : "text-secondary hover:bg-hover hover:text-primary"
              }`}
              onClick={() => !isEditing && onSelectChat(chat.id)}
            >
              <IconChat className="w-4 h-4 shrink-0 opacity-60" />

              {isEditing ? (
                <div className="flex-1 flex items-center gap-1">
                  <input
                    type="text"
                    value={editTitle}
                    onChange={(e) => setEditTitle(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") commitRename();
                      if (e.key === "Escape") setEditingId(null);
                    }}
                    className="flex-1 bg-transparent border border-border rounded px-1 py-0.5 text-sm text-primary outline-none focus:border-accent"
                    autoFocus
                    onClick={(e) => e.stopPropagation()}
                  />
                  <button
                    onClick={(e) => { e.stopPropagation(); commitRename(); }}
                    className="p-0.5 text-green-400 hover:text-green-300"
                  >
                    <IconCheck className="w-3.5 h-3.5" />
                  </button>
                </div>
              ) : (
                <>
                  <span className="flex-1 text-sm truncate">
                    {chat.title}
                  </span>

                  {/* Action buttons — visible on hover */}
                  <div className="hidden group-hover:flex items-center gap-0.5 shrink-0">
                    <button
                      onClick={(e) => { e.stopPropagation(); startRename(chat); }}
                      className="p-1 rounded hover:bg-action-hover text-tertiary hover:text-primary transition-colors"
                      title="Renombrar"
                    >
                      <IconEdit className="w-3.5 h-3.5" />
                    </button>
                    <button
                      onClick={(e) => { e.stopPropagation(); handleDelete(chat.id); }}
                      className={`p-1 rounded hover:bg-action-hover transition-colors ${
                        confirmDeleteId === chat.id
                          ? "text-red-400 hover:text-red-300"
                          : "text-tertiary hover:text-primary"
                      }`}
                      title={confirmDeleteId === chat.id ? "Click de nuevo para confirmar" : "Eliminar"}
                    >
                      <IconTrash className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </>
              )}
            </div>
          );
        })}
      </div>

      {/* Bottom panel — model selector and TTS controls */}
      <div className="border-t border-border p-3 space-y-3">
        {modelSelector}
        {ttsControls}
      </div>
    </div>
  );
}

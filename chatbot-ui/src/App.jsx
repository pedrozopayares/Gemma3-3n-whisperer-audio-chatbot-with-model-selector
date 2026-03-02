import React, { useState, useEffect, useRef, useCallback } from "react";
import Sidebar from "./components/Sidebar";
import ChatMessage from "./components/ChatMessage";
import ChatInput from "./components/ChatInput";
import DocumentViewer from "./components/DocumentViewer";
import ThemeToggle from "./components/ThemeToggle";
import AdminPanel from "./components/AdminPanel";
import { IconMic, IconSidebar, IconSettings } from "./components/Icons";
import useAudioRecorder from "./hooks/useAudioRecorder";
import useTTS from "./hooks/useTTS";
import {
  fetchModels as apiFetchModels,
  sendAudio,
  sendText,
  sendImage,
} from "./services/api";
import {
  getAllChats,
  getChat,
  createChat,
  deleteChat as storageDeleteChat,
  renameChat as storageRenameChat,
  addMessage,
  setMessages,
  getActiveChat,
  setActiveChat,
  updateChat,
} from "./services/chatStorage";

// ─── helpers ──────────────────────────────────────────────
const blobToBase64 = (blob) =>
  new Promise((res, rej) => {
    const reader = new FileReader();
    reader.onloadend = () => res(reader.result.split(",")[1]);
    reader.onerror = rej;
    reader.readAsDataURL(blob);
  });

/** Convert conversation messages to backend history format. */
function buildHistory(messages) {
  return messages
    .filter((m) => {
      const tp = m.parts?.find((p) => p.type === "text" || p.text);
      return tp && tp.text;
    })
    .map((m) => {
      const tp = m.parts.find((p) => p.type === "text" || p.text);
      return {
        role: m.role,
        text: tp.text.replace(/^🎤 "/, "").replace(/"$/, ""),
      };
    });
}

// ─── Main App ─────────────────────────────────────────────
export default function App() {
  // ── Chat state ──
  const [chats, setChats] = useState([]);
  const [activeChatId, setActiveChatIdState] = useState(null);
  const [conversation, setConversation] = useState([]);
  const [chatContext, setChatContext] = useState("");

  // ── UI state ──
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [viewerChunk, setViewerChunk] = useState(null);

  // ── TTS state ──
  const [isTTSEnabled, setIsTTSEnabled] = useState(true);
  const [ttsEngine, setTtsEngine] = useState("browser");
  const { speak, stopSpeech } = useTTS();

  // ── Model state ──
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);
  const [isLoadingModels, setIsLoadingModels] = useState(true);

  // ── Theme state ──
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem("yotojoro_theme") || "dark";
  });

  // ── Admin panel state ──
  const [showAdminPanel, setShowAdminPanel] = useState(false);

  // ── Refs ──
  const bottomRef = useRef(null);

  // ── Apply theme to document ──
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("yotojoro_theme", theme);
  }, [theme]);

  const toggleTheme = useCallback(() => {
    // Add transition class for smooth color change
    document.documentElement.classList.add("theme-transition");
    setTheme((prev) => (prev === "dark" ? "light" : "dark"));
    // Remove transition class after animation completes
    setTimeout(() => {
      document.documentElement.classList.remove("theme-transition");
    }, 350);
  }, []);

  // ── Load chats from storage on mount ──
  useEffect(() => {
    const storedChats = getAllChats();
    setChats(storedChats);

    const activeId = getActiveChat();
    if (activeId) {
      const chat = getChat(activeId);
      if (chat) {
        setActiveChatIdState(activeId);
        setActiveChat(activeId);
        setConversation(chat.messages || []);
        setChatContext(chat.context || "");
      } else if (storedChats.length > 0) {
        const first = getChat(storedChats[0].id);
        if (first) {
          setActiveChatIdState(first.id);
          setActiveChat(first.id);
          setConversation(first.messages || []);
          setChatContext(first.context || "");
        }
      }
    } else if (storedChats.length > 0) {
      const first = getChat(storedChats[0].id);
      if (first) {
        setActiveChatIdState(first.id);
        setActiveChat(first.id);
        setConversation(first.messages || []);
        setChatContext(first.context || "");
      }
    }
  }, []);

  // ── Fetch models ──
  useEffect(() => {
    apiFetchModels()
      .then((data) => {
        setAvailableModels(data.models);
        setSelectedModel(data.current || data.default);
      })
      .catch((e) => console.error("Failed to fetch models:", e))
      .finally(() => setIsLoadingModels(false));
  }, []);

  // ── Scroll to bottom on new messages ──
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [conversation]);

  // ── Persist context changes ──
  useEffect(() => {
    if (activeChatId) {
      updateChat(activeChatId, { context: chatContext });
    }
  }, [chatContext, activeChatId]);

  // ── Chat management ──
  const switchToChat = (id) => {
    // Save current conversation first
    if (activeChatId) {
      updateChat(activeChatId, { messages: conversation, context: chatContext });
    }
    const chat = getChat(id);
    if (chat) {
      setActiveChatIdState(id);
      setActiveChat(id);
      setConversation(chat.messages || []);
      setChatContext(chat.context || "");
      setError(null);
      setViewerChunk(null);
    }
    refreshChatList();
  };

  const handleNewChat = () => {
    if (activeChatId) {
      updateChat(activeChatId, { messages: conversation, context: chatContext });
    }
    const chat = createChat();
    setActiveChatIdState(chat.id);
    setConversation([]);
    setChatContext("");
    setError(null);
    setViewerChunk(null);
    refreshChatList();
  };

  const handleDeleteChat = (id) => {
    storageDeleteChat(id);
    const remaining = getAllChats();
    setChats(remaining);
    if (id === activeChatId) {
      if (remaining.length > 0) {
        switchToChat(remaining[0].id);
      } else {
        const newChat = createChat();
        setActiveChatIdState(newChat.id);
        setConversation([]);
        setChatContext("");
        refreshChatList();
      }
    }
  };

  const handleRenameChat = (id, title) => {
    storageRenameChat(id, title);
    refreshChatList();
  };

  const refreshChatList = () => {
    setChats(getAllChats());
  };

  // ── Ensure there's always an active chat ──
  const activeChatIdRef = useRef(activeChatId);
  activeChatIdRef.current = activeChatId;

  const ensureActiveChat = useCallback(() => {
    if (activeChatIdRef.current) return activeChatIdRef.current;
    const chat = createChat();
    setActiveChatIdState(chat.id);
    setChats(getAllChats());
    return chat.id;
  }, []);

  // ── Add reply (model response) ──
  const addReply = useCallback(
    (text, modelInfo = null, ragChunks = null) => {
      const msg = { role: "model", parts: [{ text }], modelInfo, ragChunks };
      setConversation((prev) => {
        const updated = [...prev, msg];
        // Persist
        if (activeChatId) {
          setMessages(activeChatId, updated);
          refreshChatList();
        }
        return updated;
      });

      if (isTTSEnabled) {
        speak(text, ttsEngine);
      }
    },
    [activeChatId, isTTSEnabled, ttsEngine, speak]
  );

  // ── Audio handling ──
  const handleAudioReady = useCallback(
    async (wavBlob) => {
      setIsProcessing(true);
      const chatId = ensureActiveChat();
      try {
        const b64 = await blobToBase64(wavBlob);
        const history = buildHistory(conversation);

        // Add placeholder user message
        const userMsg = { role: "user", parts: [{ type: "audio" }] };
        setConversation((prev) => [...prev, userMsg]);

        const data = await sendAudio({
          audioBase64: b64,
          model: selectedModel,
          context: chatContext.trim() || null,
          history,
        });

        const { text, transcription, model, routed_category, rag_chunks } = data;

        // Update user message with transcription
        if (transcription) {
          setConversation((prev) => {
            const updated = [...prev];
            const lastIdx = updated.length - 1;
            if (updated[lastIdx]?.role === "user") {
              updated[lastIdx] = {
                role: "user",
                parts: [{ type: "text", text: `🎤 "${transcription}"` }],
              };
            }
            if (chatId) setMessages(chatId, updated);
            return updated;
          });
        }

        addReply(
          text,
          routed_category ? { model, category: routed_category } : null,
          rag_chunks || null
        );
      } catch (e) {
        setError(`Audio error: ${e.message}`);
      } finally {
        setIsProcessing(false);
      }
    },
    [conversation, selectedModel, chatContext, addReply, ensureActiveChat]
  );

  const { isRecording, recordingTime, startRecording, stopRecording } = useAudioRecorder({
    onAudioReady: handleAudioReady,
  });

  // ── Text submit ──
  const handleSendText = useCallback(
    async (userText) => {
      setIsProcessing(true);
      setError(null);
      const chatId = ensureActiveChat();

      const history = buildHistory(conversation);
      const userMsg = { role: "user", parts: [{ type: "text", text: userText }] };
      setConversation((prev) => {
        const updated = [...prev, userMsg];
        if (chatId) {
          addMessage(chatId, userMsg);
          refreshChatList();
        }
        return updated;
      });

      try {
        const data = await sendText({
          text: userText,
          model: selectedModel,
          context: chatContext.trim() || null,
          history,
        });
        const { text, model, routed_category } = data;
        addReply(
          text,
          routed_category ? { model, category: routed_category } : null,
          data.rag_chunks || null
        );
      } catch (e) {
        setError(`Text error: ${e.message}`);
      } finally {
        setIsProcessing(false);
      }
    },
    [conversation, selectedModel, chatContext, addReply, ensureActiveChat]
  );

  // ── Image submit ──
  const handleSendImage = useCallback(
    async ({ file, prompt }) => {
      setIsProcessing(true);
      setError(null);
      const chatId = ensureActiveChat();

      const previewUrl = URL.createObjectURL(file);
      const userMsg = {
        role: "user",
        parts: [
          { type: "image", url: previewUrl },
          { type: "text", text: prompt },
        ],
      };
      setConversation((prev) => {
        const updated = [...prev, userMsg];
        if (chatId) {
          addMessage(chatId, userMsg);
          refreshChatList();
        }
        return updated;
      });

      try {
        const data = await sendImage({
          imageFile: file,
          prompt,
          context: chatContext.trim() || null,
        });
        addReply(data.text);
      } catch (e) {
        setError(`Image error: ${e.message}`);
      } finally {
        setIsProcessing(false);
      }
    },
    [chatContext, addReply, ensureActiveChat]
  );

  // ── Resend message ──
  const handleResend = useCallback(
    async (messageIndex) => {
      const msg = conversation[messageIndex];
      if (!msg || msg.role !== "user") return;

      const textPart = msg.parts?.find((p) => p.type === "text" || p.text);
      if (!textPart?.text) return;

      const userText = textPart.text.replace(/^🎤 "/, "").replace(/"$/, "");
      setIsProcessing(true);
      setError(null);

      // History = messages before the resent one
      const historyBefore = buildHistory(conversation.slice(0, messageIndex));

      // Truncate conversation and re-add user message
      setConversation((prev) => {
        const updated = prev.slice(0, messageIndex);
        updated.push({ role: "user", parts: [{ type: "text", text: userText }] });
        if (activeChatId) setMessages(activeChatId, updated);
        return updated;
      });

      try {
        const data = await sendText({
          text: userText,
          model: selectedModel,
          context: chatContext.trim() || null,
          history: historyBefore,
        });
        const { text, model, routed_category } = data;
        addReply(
          text,
          routed_category ? { model, category: routed_category } : null,
          data.rag_chunks || null
        );
      } catch (e) {
        setError(`Resend error: ${e.message}`);
      } finally {
        setIsProcessing(false);
      }
    },
    [conversation, selectedModel, chatContext, activeChatId, addReply]
  );

  // ── TTS toggle ──
  const toggleTTS = () => {
    const newVal = !isTTSEnabled;
    setIsTTSEnabled(newVal);
    if (!newVal) stopSpeech();
  };

  // ── Model selector JSX (for sidebar) ──
  const modelSelector = (
    <div>
      <label className="text-xs text-tertiary font-medium block mb-1">Modelo</label>
      {isLoadingModels ? (
        <span className="text-xs text-tertiary">Cargando…</span>
      ) : (
        <select
          value={selectedModel || ""}
          onChange={(e) => setSelectedModel(e.target.value)}
          disabled={isProcessing || isRecording}
          className="w-full bg-input-bg text-primary text-xs rounded-lg px-2 py-1.5 border border-border focus:outline-none focus:border-accent disabled:opacity-50"
        >
          {availableModels.map((m) => (
            <option key={m.key} value={m.key}>
              {m.name} ({m.size}){m.loaded ? " ✓" : ""}
            </option>
          ))}
        </select>
      )}
    </div>
  );

  // ── TTS controls JSX (for sidebar) ──
  const ttsControls = (
    <div>
      <div className="flex items-center justify-between">
        <label className="text-xs text-tertiary font-medium">TTS</label>
        <button
          onClick={toggleTTS}
          className={`relative w-9 h-5 rounded-full transition-colors ${
            isTTSEnabled ? "bg-accent" : "bg-border"
          }`}
        >
          <div
            className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-transform ${
              isTTSEnabled ? "left-[18px]" : "left-0.5"
            }`}
          />
        </button>
      </div>
      {isTTSEnabled && (
        <select
          value={ttsEngine}
          onChange={(e) => setTtsEngine(e.target.value)}
          className="mt-1.5 w-full bg-input-bg text-primary text-xs rounded-lg px-2 py-1.5 border border-border focus:outline-none focus:border-accent"
        >
          <option value="browser">Audio del navegador</option>
          <option value="native">Audio del sistema</option>
        </select>
      )}
    </div>
  );

  // ─── RENDER ─────────────────────────────────────────────
  return (
    <div className="flex w-full h-screen bg-chat-bg text-primary overflow-hidden">
      {/* Sidebar */}
      <Sidebar
        chats={chats}
        activeChatId={activeChatId}
        onSelectChat={switchToChat}
        onNewChat={handleNewChat}
        onDeleteChat={handleDeleteChat}
        onRenameChat={handleRenameChat}
        isCollapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
        modelSelector={modelSelector}
        ttsControls={ttsControls}
      />

      {/* Main chat area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <header className="flex items-center justify-between px-4 py-2.5 border-b border-border bg-chat-bg shrink-0">
          <div className="flex items-center gap-3">
            {sidebarCollapsed && (
              <button
                onClick={() => setSidebarCollapsed(false)}
                className="p-1.5 rounded-lg hover:bg-hover text-secondary hover:text-primary transition-colors lg:hidden"
              >
                <IconSidebar className="w-5 h-5" />
              </button>
            )}
            <h1 className="text-sm font-semibold text-primary">
              Yotojoro IA
            </h1>
            {selectedModel && (
              <span className="text-xs text-tertiary bg-hover px-2 py-0.5 rounded-full">
                {availableModels.find((m) => m.key === selectedModel)?.name || selectedModel}
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            {error && (
              <p className="text-xs text-red-400 truncate max-w-md">{error}</p>
            )}
            <ThemeToggle theme={theme} onToggle={toggleTheme} />
            <button
              onClick={() => setShowAdminPanel(true)}
              className="p-1.5 rounded-lg hover:bg-hover text-tertiary hover:text-primary transition-colors"
              title="Panel de administración"
            >
              <IconSettings className="w-5 h-5" />
            </button>
          </div>
        </header>

        {/* Admin Panel Overlay */}
        {showAdminPanel && (
          <AdminPanel onClose={() => setShowAdminPanel(false)} />
        )}

        {/* Messages */}
        <main className="flex-1 overflow-y-auto scrollbar-thin">
          {conversation.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-secondary">
              <IconMic className="w-12 h-12 mb-4 opacity-30" />
              <h2 className="text-xl font-semibold mb-2">Yotojoro IA</h2>
              <p className="text-sm text-tertiary max-w-sm text-center">
                Escribe un mensaje, graba audio o adjunta una imagen para comenzar la conversación.
              </p>
            </div>
          )}

          <div className="max-w-3xl mx-auto">
            {conversation.map((msg, i) => (
              <ChatMessage
                key={i}
                message={msg}
                index={i}
                onResend={handleResend}
                onViewDocument={(chunk) => setViewerChunk(chunk)}
                isProcessing={isProcessing}
              />
            ))}
          </div>
          <div ref={bottomRef} />
        </main>

        {/* Input */}
        <ChatInput
          onSendText={handleSendText}
          onStartRecording={startRecording}
          onStopRecording={stopRecording}
          isRecording={isRecording}
          isProcessing={isProcessing}
          recordingTime={recordingTime}
          onSendImage={handleSendImage}
          contextValue={chatContext}
          onContextChange={setChatContext}
        />
      </div>

      {/* Document viewer panel */}
      {viewerChunk && (
        <DocumentViewer
          chunk={viewerChunk}
          onClose={() => setViewerChunk(null)}
        />
      )}
    </div>
  );
}

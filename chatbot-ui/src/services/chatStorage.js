/**
 * Chat persistence service using localStorage.
 * Manages multiple conversations with CRUD operations.
 */

const STORAGE_KEY = "yotojoro_chats";
const ACTIVE_CHAT_KEY = "yotojoro_active_chat";

/**
 * Generate a unique ID for a new conversation.
 */
function generateId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

/**
 * Get all conversations from localStorage.
 * @returns {Array} List of conversation objects sorted by updatedAt descending.
 */
export function getAllChats() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const chats = raw ? JSON.parse(raw) : [];
    return chats.sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
  } catch {
    return [];
  }
}

/**
 * Get a single conversation by ID.
 */
export function getChat(id) {
  const chats = getAllChats();
  return chats.find((c) => c.id === id) || null;
}

/**
 * Create a new conversation.
 * @param {string} [title] - Optional title. Defaults to "Nuevo chat".
 * @returns {Object} The new conversation object.
 */
export function createChat(title = "Nuevo chat") {
  const chats = getAllChats();
  const newChat = {
    id: generateId(),
    title,
    messages: [],
    context: "",
    createdAt: Date.now(),
    updatedAt: Date.now(),
  };
  chats.push(newChat);
  _persist(chats);
  setActiveChat(newChat.id);
  return newChat;
}

/**
 * Update a conversation (messages, title, context, etc.).
 */
export function updateChat(id, updates) {
  const chats = getAllChats();
  const idx = chats.findIndex((c) => c.id === id);
  if (idx === -1) return null;

  chats[idx] = { ...chats[idx], ...updates, updatedAt: Date.now() };
  _persist(chats);
  return chats[idx];
}

/**
 * Delete a conversation by ID.
 */
export function deleteChat(id) {
  let chats = getAllChats();
  chats = chats.filter((c) => c.id !== id);
  _persist(chats);

  // If the deleted chat was active, clear or switch
  if (getActiveChat() === id) {
    const next = chats[0];
    setActiveChat(next ? next.id : null);
  }
}

/**
 * Add a message to a conversation.
 * Also auto-titles the chat from the first user message.
 */
export function addMessage(chatId, message) {
  const chat = getChat(chatId);
  if (!chat) return null;

  chat.messages.push(message);

  // Auto-title from first user message
  if (
    chat.title === "Nuevo chat" &&
    message.role === "user" &&
    message.parts?.[0]?.text
  ) {
    const text = message.parts[0].text.replace(/^🎤 "/, "").replace(/"$/, "");
    chat.title = text.slice(0, 40) + (text.length > 40 ? "…" : "");
  }

  updateChat(chatId, { messages: chat.messages, title: chat.title });
  return chat;
}

/**
 * Replace conversation messages (e.g., after resend truncation).
 */
export function setMessages(chatId, messages) {
  updateChat(chatId, { messages });
}

/**
 * Get the active chat ID.
 */
export function getActiveChat() {
  return localStorage.getItem(ACTIVE_CHAT_KEY);
}

/**
 * Set the active chat ID.
 */
export function setActiveChat(id) {
  if (id) {
    localStorage.setItem(ACTIVE_CHAT_KEY, id);
  } else {
    localStorage.removeItem(ACTIVE_CHAT_KEY);
  }
}

/**
 * Rename a conversation.
 */
export function renameChat(id, title) {
  return updateChat(id, { title });
}

// Internal persist helper
function _persist(chats) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(chats));
}

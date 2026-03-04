import React, { useState, useEffect, useCallback } from "react";
import { IconClose, IconCheck, IconLoader } from "./Icons";
import {
  getConfig,
  updateConfig,
  getServicesStatus,
  pullModel,
  deleteModel,
  testOllama,
  testWhisper,
  getRagStatus,
  syncRag,
  rebuildRag,
  scanDocuments,
  removeRagDocument,
  uploadDocuments,
} from "../services/adminApi";

// ─── Available models catalog (mirrors backend AVAILABLE_MODELS) ───
const MODEL_CATALOG = {
  "gemma2:2b":       { name: "Gemma 2 2B",        size: "~1.6GB",  vision: false },
  "gemma2:9b":       { name: "Gemma 2 9B",        size: "~5.4GB",  vision: false },
  "gemma3:1b":       { name: "Gemma 3 1B",        size: "~815MB",  vision: false },
  "gemma3:4b":       { name: "Gemma 3 4B",        size: "~3.3GB",  vision: false },
  "gemma3:12b":      { name: "Gemma 3 12B",       size: "~8.9GB",  vision: false },
  "gemma3:27b":      { name: "Gemma 3 27B",       size: "~17GB",   vision: false },
  "llama3.2:1b":     { name: "Llama 3.2 1B",      size: "~1.3GB",  vision: false },
  "llama3.2:3b":     { name: "Llama 3.2 3B",      size: "~2GB",    vision: false },
  "llava:7b":        { name: "LLaVA 7B",          size: "~4.7GB",  vision: true },
  "llava:13b":       { name: "LLaVA 13B",         size: "~8GB",    vision: true },
  "mistral:7b":      { name: "Mistral 7B",        size: "~4.1GB",  vision: false },
  "qwen2.5:0.5b":   { name: "Qwen 2.5 0.5B",     size: "~400MB",  vision: false },
  "qwen2.5:3b":     { name: "Qwen 2.5 3B",       size: "~1.9GB",  vision: false },
  "qwen2.5:7b":     { name: "Qwen 2.5 7B",       size: "~4.7GB",  vision: false },
  "phi3:mini":       { name: "Phi-3 Mini",        size: "~2.3GB",  vision: false },
  "phi4:latest":     { name: "Phi-4 14B",         size: "~9.1GB",  vision: false },
  "deepseek-r1:1.5b":{ name: "DeepSeek R1 1.5B", size: "~1.1GB",  vision: false },
  "deepseek-r1:7b":  { name: "DeepSeek R1 7B",   size: "~4.7GB",  vision: false },
  "deepseek-r1:8b":  { name: "DeepSeek R1 8B",   size: "~4.9GB",  vision: false },
  "deepseek-r1:14b": { name: "DeepSeek R1 14B",  size: "~9GB",    vision: false },
  "deepseek-r1:32b": { name: "DeepSeek R1 32B",  size: "~20GB",   vision: false },
};

const TABS = [
  { key: "general",  label: "General" },
  { key: "models",   label: "Modelos" },
  { key: "services", label: "Servicios" },
  { key: "rag",      label: "RAG" },
];

// ─── Status badge helper ──────────────────────────────────
function StatusBadge({ status }) {
  const colors = {
    ok:      "bg-green-500/20 text-green-400 border-green-500/30",
    error:   "bg-red-500/20 text-red-400 border-red-500/30",
    warning: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
    loading: "bg-blue-500/20 text-blue-400 border-blue-500/30",
  };
  const labels = { ok: "OK", error: "Error", warning: "Aviso", loading: "Cargando" };
  const c = colors[status] || colors.loading;
  return (
    <span className={`text-xs px-2 py-0.5 rounded-full border ${c}`}>
      {labels[status] || status}
    </span>
  );
}

// ═══════════════════════════════════════════════════════════
// General Tab
// ═══════════════════════════════════════════════════════════
function GeneralTab({ config, onSave }) {
  const [form, setForm] = useState({
    app_name: "",
    system_prompt: "",
    preferred_language: "español",
    context_window_size: 8192,
    max_history_messages: 20,
    keep_recent_messages: 6,
  });
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    if (config) {
      setForm({
        app_name: config.app_name || "",
        system_prompt: config.system_prompt || "",
        preferred_language: config.preferred_language ?? "español",
        context_window_size: config.context_window_size || 8192,
        max_history_messages: config.max_history_messages || 20,
        keep_recent_messages: config.keep_recent_messages || 6,
      });
    }
  }, [config]);

  const handleSave = async () => {
    setSaving(true);
    try {
      await onSave(form);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-base font-semibold text-primary mb-4">Configuración general</h3>

        <div className="space-y-4">
          <div>
            <label className="block text-sm text-secondary mb-1">Nombre de la aplicación</label>
            <input
              type="text"
              value={form.app_name}
              onChange={(e) => setForm({ ...form, app_name: e.target.value })}
              className="w-full bg-input-bg text-primary text-sm px-3 py-2 rounded-lg border border-border focus:outline-none focus:border-accent"
              placeholder="Yotojoro IA"
            />
          </div>

          <div>
            <label className="block text-sm text-secondary mb-1">Prompt del sistema (personalidad del asistente)</label>
            <textarea
              value={form.system_prompt}
              onChange={(e) => setForm({ ...form, system_prompt: e.target.value })}
              className="w-full bg-input-bg text-primary text-sm px-3 py-2 rounded-lg border border-border focus:outline-none focus:border-accent resize-y min-h-[120px]"
              rows={5}
              placeholder="Eres un asistente experto…"
            />
          </div>

          <div>
            <label className="block text-sm text-secondary mb-1">Idioma preferido de respuesta</label>
            <div className="flex items-center gap-3">
              <input
                type="text"
                value={form.preferred_language}
                onChange={(e) => setForm({ ...form, preferred_language: e.target.value })}
                className="flex-1 bg-input-bg text-primary text-sm px-3 py-2.5 rounded-lg border border-border focus:outline-none focus:border-accent"
                placeholder="español"
              />
              <span className="text-xs text-tertiary shrink-0">Déjalo vacío para no forzar idioma</span>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm text-secondary mb-1">Ventana de contexto (tokens)</label>
              <input
                type="number"
                value={form.context_window_size}
                onChange={(e) => setForm({ ...form, context_window_size: parseInt(e.target.value) || 2048 })}
                className="w-full bg-input-bg text-primary text-sm px-3 py-2.5 rounded-lg border border-border focus:outline-none focus:border-accent"
                min={1024}
                max={131072}
                step={1024}
              />
            </div>
            <div>
              <label className="block text-sm text-secondary mb-1">Historial máximo (msgs)</label>
              <input
                type="number"
                value={form.max_history_messages}
                onChange={(e) => setForm({ ...form, max_history_messages: parseInt(e.target.value) || 10 })}
                className="w-full bg-input-bg text-primary text-sm px-3 py-2.5 rounded-lg border border-border focus:outline-none focus:border-accent"
                min={4}
                max={100}
              />
            </div>
            <div>
              <label className="block text-sm text-secondary mb-1">Msgs recientes a conservar</label>
              <input
                type="number"
                value={form.keep_recent_messages}
                onChange={(e) => setForm({ ...form, keep_recent_messages: parseInt(e.target.value) || 4 })}
                className="w-full bg-input-bg text-primary text-sm px-3 py-2.5 rounded-lg border border-border focus:outline-none focus:border-accent"
                min={2}
                max={50}
              />
            </div>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-3 pt-2">
        <button
          onClick={handleSave}
          disabled={saving}
          className="px-5 py-2.5 rounded-lg bg-accent hover:bg-accent-hover text-white text-sm font-semibold transition-colors disabled:opacity-50 active:scale-95 shadow-sm"
        >
          {saving ? "Guardando…" : "Guardar cambios"}
        </button>
        {saved && (
          <span className="flex items-center gap-1 text-sm text-green-400">
            <IconCheck className="w-4 h-4" /> Guardado
          </span>
        )}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════
// Models Tab
// ═══════════════════════════════════════════════════════════
function ModelsTab({ config, onSave, installedModels }) {
  const [installed, setInstalled] = useState(installedModels || []);
  const [pulling, setPulling] = useState(null); // model key being pulled
  const [pullProgress, setPullProgress] = useState("");
  const [pullPercent, setPullPercent] = useState(0);
  const [deleting, setDeleting] = useState(null);
  const [defaultModel, setDefaultModel] = useState(config?.default_model || "gemma2:2b");
  const [defaultVision, setDefaultVision] = useState(config?.default_vision_model || "llava:7b");
  const [routerModel, setRouterModel] = useState(config?.router_model || "qwen2.5:0.5b");
  const [routeChat, setRouteChat] = useState(config?.route_models?.chat || "gemma3:4b");
  const [routeMath, setRouteMath] = useState(config?.route_models?.math || "phi4:latest");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [customModel, setCustomModel] = useState("");

  useEffect(() => {
    if (installedModels) setInstalled(installedModels);
  }, [installedModels]);

  useEffect(() => {
    if (config) {
      setDefaultModel(config.default_model || "gemma2:2b");
      setDefaultVision(config.default_vision_model || "llava:7b");
      setRouterModel(config.router_model || "qwen2.5:0.5b");
      setRouteChat(config.route_models?.chat || "gemma3:4b");
      setRouteMath(config.route_models?.math || "phi4:latest");
    }
  }, [config]);

  const isInstalled = (key) => installed.some((m) => m === key);

  const handlePull = async (modelKey) => {
    setPulling(modelKey);
    setPullProgress("Iniciando descarga…");
    setPullPercent(0);
    try {
      await pullModel(modelKey, (data) => {
        if (data.error) {
          setPullProgress(`Error: ${data.error}`);
          return;
        }
        const status = data.status || "";
        if (data.total && data.completed) {
          const pct = Math.round((data.completed / data.total) * 100);
          setPullPercent(pct);
          setPullProgress(`${status} ${pct}%`);
        } else {
          setPullProgress(status);
        }
      });
      // Refresh installed models
      const svc = await getServicesStatus();
      setInstalled(svc.ollama?.installed_models || []);
      setPullProgress("¡Descarga completada!");
    } catch (e) {
      setPullProgress(`Error: ${e.message}`);
    } finally {
      setTimeout(() => {
        setPulling(null);
        setPullProgress("");
        setPullPercent(0);
      }, 2000);
    }
  };

  const handleDelete = async (modelKey) => {
    if (!confirm(`¿Eliminar el modelo "${modelKey}"?`)) return;
    setDeleting(modelKey);
    try {
      await deleteModel(modelKey);
      setInstalled((prev) => prev.filter((m) => m !== modelKey));
    } catch (e) {
      alert(`Error: ${e.message}`);
    } finally {
      setDeleting(null);
    }
  };

  const handleSaveDefaults = async () => {
    setSaving(true);
    try {
      await onSave({
        default_model: defaultModel,
        default_vision_model: defaultVision,
        router_model: routerModel,
        route_models: { chat: routeChat, math: routeMath },
      });
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } finally {
      setSaving(false);
    }
  };

  const handlePullCustom = () => {
    const key = customModel.trim();
    if (!key) return;
    handlePull(key);
    setCustomModel("");
  };

  const allModelKeys = Object.keys(MODEL_CATALOG);

  return (
    <div className="space-y-6">
      {/* Model defaults */}
      <div>
        <h3 className="text-base font-semibold text-primary mb-4">Modelos por defecto</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm text-secondary mb-1">Modelo de texto</label>
            <select
              value={defaultModel}
              onChange={(e) => setDefaultModel(e.target.value)}
              className="w-full bg-input-bg text-primary text-sm px-3 py-2.5 rounded-lg border border-border focus:outline-none focus:border-accent"
            >
              {allModelKeys.filter((k) => !MODEL_CATALOG[k].vision).map((k) => (
                <option key={k} value={k}>
                  {MODEL_CATALOG[k].name} ({MODEL_CATALOG[k].size}) {isInstalled(k) ? "✓" : ""}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm text-secondary mb-1">Modelo de visión</label>
            <select
              value={defaultVision}
              onChange={(e) => setDefaultVision(e.target.value)}
              className="w-full bg-input-bg text-primary text-sm px-3 py-2.5 rounded-lg border border-border focus:outline-none focus:border-accent"
            >
              {allModelKeys.filter((k) => MODEL_CATALOG[k].vision).map((k) => (
                <option key={k} value={k}>
                  {MODEL_CATALOG[k].name} ({MODEL_CATALOG[k].size}) {isInstalled(k) ? "✓" : ""}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Smart routing */}
      <div>
        <h3 className="text-sm font-semibold text-primary mb-3">Enrutamiento inteligente (Auto)</h3>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div>
            <label className="block text-xs text-secondary mb-1">Modelo clasificador</label>
            <select
              value={routerModel}
              onChange={(e) => setRouterModel(e.target.value)}
              className="w-full bg-input-bg text-primary text-xs px-2 py-2 rounded-lg border border-border focus:outline-none focus:border-accent"
            >
              {allModelKeys.map((k) => (
                <option key={k} value={k}>{MODEL_CATALOG[k].name} {isInstalled(k) ? "✓" : ""}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-xs text-secondary mb-1">Ruta: Chat →</label>
            <select
              value={routeChat}
              onChange={(e) => setRouteChat(e.target.value)}
              className="w-full bg-input-bg text-primary text-xs px-2 py-2 rounded-lg border border-border focus:outline-none focus:border-accent"
            >
              {allModelKeys.filter((k) => !MODEL_CATALOG[k].vision).map((k) => (
                <option key={k} value={k}>{MODEL_CATALOG[k].name} {isInstalled(k) ? "✓" : ""}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-xs text-secondary mb-1">Ruta: Math →</label>
            <select
              value={routeMath}
              onChange={(e) => setRouteMath(e.target.value)}
              className="w-full bg-input-bg text-primary text-xs px-2 py-2 rounded-lg border border-border focus:outline-none focus:border-accent"
            >
              {allModelKeys.filter((k) => !MODEL_CATALOG[k].vision).map((k) => (
                <option key={k} value={k}>{MODEL_CATALOG[k].name} {isInstalled(k) ? "✓" : ""}</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-3">
        <button
          onClick={handleSaveDefaults}
          disabled={saving}
          className="px-5 py-2.5 rounded-lg bg-accent hover:bg-accent-hover text-white text-sm font-semibold transition-colors disabled:opacity-50 active:scale-95 shadow-sm"
        >
          {saving ? "Guardando…" : "Guardar configuración de modelos"}
        </button>
        {saved && (
          <span className="flex items-center gap-1 text-sm text-green-400">
            <IconCheck className="w-4 h-4" /> Guardado
          </span>
        )}
      </div>

      {/* Pull custom model */}
      <div className="border-t border-border pt-4">
        <h3 className="text-sm font-semibold text-primary mb-3">Descargar modelo personalizado</h3>
        <div className="flex gap-2">
          <input
            type="text"
            value={customModel}
            onChange={(e) => setCustomModel(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handlePullCustom()}
            placeholder="ej: gemma3:4b o nombre:tag"
            className="flex-1 bg-input-bg text-primary text-sm px-3 py-2.5 rounded-lg border border-border focus:outline-none focus:border-accent placeholder:text-placeholder"
          />
          <button
            onClick={handlePullCustom}
            disabled={pulling || !customModel.trim()}
            className="px-5 py-2.5 rounded-lg bg-accent hover:bg-accent-hover text-white text-sm font-semibold transition-colors disabled:opacity-50 active:scale-95 shadow-sm"
          >
            Descargar
          </button>
        </div>
      </div>

      {/* Model catalog */}
      <div className="border-t border-border pt-4">
        <h3 className="text-sm font-semibold text-primary mb-3">Catálogo de modelos</h3>

        {/* Pull progress */}
        {pulling && (
          <div className="mb-4 p-3 rounded-lg bg-input-bg border border-border">
            <div className="flex items-center gap-2 mb-2">
              <IconLoader className="w-4 h-4 animate-spin text-accent" />
              <span className="text-sm text-primary font-medium">Descargando: {pulling}</span>
            </div>
            <div className="w-full h-2 bg-border rounded-full overflow-hidden mb-1">
              <div
                className="h-full bg-accent rounded-full transition-all duration-300"
                style={{ width: `${pullPercent}%` }}
              />
            </div>
            <p className="text-xs text-tertiary">{pullProgress}</p>
          </div>
        )}

        <div className="space-y-1">
          {allModelKeys.map((key) => {
            const info = MODEL_CATALOG[key];
            const inst = isInstalled(key);
            return (
              <div
                key={key}
                className="flex items-center justify-between px-3 py-2.5 rounded-lg hover:bg-hover transition-colors"
              >
                <div className="flex items-center gap-2 sm:gap-3 min-w-0">
                  <span
                    className={`w-2.5 h-2.5 rounded-full shrink-0 ${inst ? "bg-green-400" : "bg-border"}`}
                  />
                  <div className="min-w-0">
                    <span className="text-sm text-primary">{info.name}</span>
                    <span className="text-xs text-tertiary ml-1.5 hidden sm:inline">{key}</span>
                    {info.vision && (
                      <span className="text-xs text-accent ml-1.5">👁 visión</span>
                    )}
                  </div>
                  <span className="text-xs text-tertiary hidden sm:inline">{info.size}</span>
                </div>
                <div className="flex items-center gap-1 shrink-0">
                  {inst ? (
                    <button
                      onClick={() => handleDelete(key)}
                      disabled={deleting === key}
                      className="px-3 py-1.5 rounded-lg text-xs font-medium bg-red-500/20 text-red-400 hover:bg-red-500/30 border border-red-500/30 transition-colors disabled:opacity-50 active:scale-95"
                    >
                      {deleting === key ? "…" : "Eliminar"}
                    </button>
                  ) : (
                    <button
                      onClick={() => handlePull(key)}
                      disabled={pulling !== null}
                      className="px-3 py-1.5 rounded-lg text-xs font-medium bg-accent/20 text-accent hover:bg-accent/30 border border-accent/30 transition-colors disabled:opacity-50 active:scale-95"
                    >
                      Descargar
                    </button>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════
// Services Tab
// ═══════════════════════════════════════════════════════════
function ServicesTab() {
  const [services, setServices] = useState(null);
  const [loading, setLoading] = useState(true);
  const [testing, setTesting] = useState(null);
  const [testResult, setTestResult] = useState(null);

  const loadStatus = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getServicesStatus();
      setServices(data);
    } catch (e) {
      console.error("Error loading services:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadStatus();
  }, [loadStatus]);

  const handleTestOllama = async () => {
    setTesting("ollama");
    setTestResult(null);
    try {
      // Pick first installed model or default
      const firstModel = services?.ollama?.installed_models?.[0] || "gemma3:1b";
      const result = await testOllama(firstModel);
      setTestResult({ service: "ollama", ...result });
    } catch (e) {
      setTestResult({ service: "ollama", success: false, error: e.message });
    } finally {
      setTesting(null);
    }
  };

  const handleTestWhisper = async () => {
    setTesting("whisper");
    setTestResult(null);
    try {
      const result = await testWhisper();
      setTestResult({ service: "whisper", ...result });
    } catch (e) {
      setTestResult({ service: "whisper", success: false, error: e.message });
    } finally {
      setTesting(null);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-16">
        <IconLoader className="w-6 h-6 animate-spin text-accent" />
        <span className="ml-3 text-sm text-tertiary">Consultando servicios…</span>
      </div>
    );
  }

  const serviceList = [
    {
      key: "ollama",
      label: "Ollama (LLM)",
      description: "Motor de modelos de lenguaje",
      data: services?.ollama,
      testable: true,
      onTest: handleTestOllama,
    },
    {
      key: "whisper",
      label: "Whisper (STT)",
      description: "Reconocimiento de voz",
      data: services?.whisper,
      testable: true,
      onTest: handleTestWhisper,
    },
    {
      key: "rag",
      label: "RAG (Knowledge Base)",
      description: "Base de conocimiento con ChromaDB",
      data: services?.rag,
      testable: false,
    },
    {
      key: "tts",
      label: "TTS (Text-to-Speech)",
      description: "Conversión de texto a voz",
      data: services?.tts,
      testable: false,
    },
    {
      key: "documents",
      label: "Documentos",
      description: "Directorio de archivos",
      data: services?.documents,
      testable: false,
    },
  ];

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-base font-semibold text-primary">Estado de servicios</h3>
        <button
          onClick={loadStatus}
          className="px-3 py-1.5 rounded-lg text-xs font-medium bg-accent/20 text-accent hover:bg-accent/30 border border-accent/30 transition-colors active:scale-95"
        >
          Recargar
        </button>
      </div>

      {serviceList.map((svc) => (
        <div
          key={svc.key}
          className="p-4 rounded-lg bg-input-bg border border-border"
        >
          <div className="flex items-center justify-between mb-1">
            <div className="flex items-center gap-2">
              <h4 className="text-sm font-semibold text-primary">{svc.label}</h4>
              <StatusBadge status={svc.data?.status || "loading"} />
            </div>
            {svc.testable && (
              <button
                onClick={svc.onTest}
                disabled={testing === svc.key}
                className="px-3.5 py-1.5 rounded-lg text-xs font-medium bg-accent/20 text-accent hover:bg-accent/30 border border-accent/30 transition-colors disabled:opacity-50 active:scale-95"
              >
                {testing === svc.key ? (
                  <span className="flex items-center gap-1">
                    <IconLoader className="w-3 h-3 animate-spin" /> Probando…
                  </span>
                ) : (
                  "Probar"
                )}
              </button>
            )}
          </div>
          <p className="text-xs text-tertiary mb-1">{svc.description}</p>
          <p className="text-xs text-secondary">{svc.data?.message}</p>

          {/* Installed models list for Ollama */}
          {svc.key === "ollama" && svc.data?.installed_models?.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-1">
              {svc.data.installed_models.map((m) => (
                <span
                  key={m}
                  className="text-xs bg-hover text-secondary px-2 py-0.5 rounded"
                >
                  {m}
                </span>
              ))}
            </div>
          )}
        </div>
      ))}

      {/* Test result */}
      {testResult && (
        <div
          className={`p-4 rounded-lg border ${
            testResult.success
              ? "bg-green-500/10 border-green-500/30"
              : "bg-red-500/10 border-red-500/30"
          }`}
        >
          <h4 className="text-sm font-semibold mb-1 flex items-center gap-2">
            {testResult.success ? (
              <span className="text-green-400">✓ Test exitoso</span>
            ) : (
              <span className="text-red-400">✗ Test fallido</span>
            )}
            <span className="text-tertiary text-xs">({testResult.service})</span>
          </h4>
          {testResult.response && (
            <p className="text-xs text-secondary">Respuesta: {testResult.response}</p>
          )}
          {testResult.error && (
            <p className="text-xs text-red-400">Error: {testResult.error}</p>
          )}
          {testResult.model && (
            <p className="text-xs text-tertiary">Modelo: {testResult.model}</p>
          )}
        </div>
      )}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════
// RAG Tab
// ═══════════════════════════════════════════════════════════
function RagTab({ config, onSave }) {
  const [ragStatus, setRagStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [syncing, setSyncing] = useState(false);
  const [rebuilding, setRebuilding] = useState(false);
  const [syncResult, setSyncResult] = useState(null);
  const [ragEnabled, setRagEnabled] = useState(config?.rag_enabled ?? true);
  const [topK, setTopK] = useState(config?.rag_top_k ?? 3);
  const [minRelevance, setMinRelevance] = useState(config?.rag_min_relevance ?? 0.3);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  // Documents scan state
  const [scannedFiles, setScannedFiles] = useState(null);
  const [scanning, setScanning] = useState(false);
  const [removing, setRemoving] = useState(null);
  // Upload state
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);
  const [uploadSubfolder, setUploadSubfolder] = useState("");
  const uploadInputRef = React.useRef(null);

  useEffect(() => {
    if (config) {
      setRagEnabled(config.rag_enabled ?? true);
      setTopK(config.rag_top_k ?? 3);
      setMinRelevance(config.rag_min_relevance ?? 0.3);
    }
  }, [config]);

  const loadStatus = useCallback(async () => {
    setLoading(true);
    try {
      const [status, scan] = await Promise.all([getRagStatus(), scanDocuments()]);
      setRagStatus(status);
      setScannedFiles(scan);
    } catch (e) {
      console.error("Error loading RAG status:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadStatus();
  }, [loadStatus]);

  const handleScan = async () => {
    setScanning(true);
    try {
      const scan = await scanDocuments();
      setScannedFiles(scan);
    } catch (e) {
      console.error("Scan error:", e);
    } finally {
      setScanning(false);
    }
  };

  const handleSync = async () => {
    setSyncing(true);
    setSyncResult(null);
    try {
      const result = await syncRag();
      setSyncResult(result);
      // Refresh everything
      await loadStatus();
    } catch (e) {
      setSyncResult({ success: false, error: e.message });
    } finally {
      setSyncing(false);
    }
  };

  const handleRebuild = async () => {
    if (!confirm("¿Reconstruir toda la base de conocimiento desde cero?\nSe eliminarán todos los datos indexados y se volverán a procesar los documentos.")) return;
    setRebuilding(true);
    setSyncResult(null);
    try {
      const result = await rebuildRag();
      setSyncResult({ ...result, rebuilt: true });
      await loadStatus();
    } catch (e) {
      setSyncResult({ success: false, error: e.message });
    } finally {
      setRebuilding(false);
    }
  };

  const handleRemoveDoc = async (filePath) => {
    if (!confirm(`¿Eliminar "${filePath}" del índice?`)) return;
    setRemoving(filePath);
    try {
      await removeRagDocument(filePath);
      await loadStatus();
    } catch (e) {
      alert(`Error: ${e.message}`);
    } finally {
      setRemoving(null);
    }
  };

  const handleUpload = async (e) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    setUploading(true);
    setUploadResult(null);
    try {
      const result = await uploadDocuments(Array.from(files), uploadSubfolder.trim());
      setUploadResult(result);
      // Refresh scan after upload
      await loadStatus();
    } catch (err) {
      setUploadResult({ success: false, message: err.message });
    } finally {
      setUploading(false);
      // Reset file input so the user can re-upload the same files
      if (uploadInputRef.current) uploadInputRef.current.value = "";
    }
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      await onSave({
        rag_enabled: ragEnabled,
        rag_top_k: topK,
        rag_min_relevance: minRelevance,
      });
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } finally {
      setSaving(false);
    }
  };

  const formatSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const isBusy = syncing || rebuilding;

  return (
    <div className="space-y-6">
      {/* Status */}
      <div>
        <h3 className="text-base font-semibold text-primary mb-4">Base de conocimiento (RAG)</h3>

        {loading ? (
          <div className="flex items-center gap-2 text-sm text-tertiary">
            <IconLoader className="w-4 h-4 animate-spin" /> Cargando estado…
          </div>
        ) : ragStatus ? (
          <div className="p-4 rounded-lg bg-input-bg border border-border space-y-2">
            <div className="flex items-center gap-2">
              <StatusBadge status={ragStatus.initialized ? "ok" : ragStatus.available ? "warning" : "error"} />
              <span className="text-sm text-primary">
                {ragStatus.initialized
                  ? `${ragStatus.chunks || 0} chunks indexados de ${ragStatus.documents || 0} documentos`
                  : ragStatus.message}
              </span>
            </div>
            {ragStatus.embedding_model && (
              <p className="text-xs text-tertiary">Modelo de embeddings: {ragStatus.embedding_model}</p>
            )}
            {ragStatus.last_sync && (
              <p className="text-xs text-tertiary">Última sincronización: {new Date(ragStatus.last_sync).toLocaleString()}</p>
            )}
          </div>
        ) : null}
      </div>

      {/* Settings */}
      <div>
        <h4 className="text-sm font-semibold text-primary mb-3">Configuración RAG</h4>

        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <label className="text-sm text-secondary">RAG habilitado</label>
            <button
              onClick={() => setRagEnabled(!ragEnabled)}
              className={`relative w-12 h-6 rounded-full transition-colors ${
                ragEnabled ? "bg-accent" : "bg-border"
              }`}
            >
              <div
                className={`absolute top-0.5 w-5 h-5 rounded-full bg-white transition-transform shadow-sm ${
                  ragEnabled ? "left-[26px]" : "left-0.5"
                }`}
              />
            </button>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-secondary mb-1">Documentos a recuperar (top_k)</label>
              <input
                type="number"
                value={topK}
                onChange={(e) => setTopK(parseInt(e.target.value) || 1)}
                className="w-full bg-input-bg text-primary text-sm px-3 py-2.5 rounded-lg border border-border focus:outline-none focus:border-accent"
                min={1}
                max={20}
              />
            </div>
            <div>
              <label className="block text-sm text-secondary mb-1">Relevancia mínima (0-1)</label>
              <input
                type="number"
                value={minRelevance}
                onChange={(e) => setMinRelevance(parseFloat(e.target.value) || 0)}
                className="w-full bg-input-bg text-primary text-sm px-3 py-2.5 rounded-lg border border-border focus:outline-none focus:border-accent"
                min={0}
                max={1}
                step={0.05}
              />
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3 mt-4">
          <button
            onClick={handleSave}
            disabled={saving}
            className="px-5 py-2.5 rounded-lg bg-accent hover:bg-accent-hover text-white text-sm font-semibold transition-colors disabled:opacity-50 active:scale-95 shadow-sm"
          >
            {saving ? "Guardando…" : "Guardar configuración RAG"}
          </button>
          {saved && (
            <span className="flex items-center gap-1 text-sm text-green-400">
              <IconCheck className="w-4 h-4" /> Guardado
            </span>
          )}
        </div>
      </div>

      {/* Upload documents */}
      <div className="border-t border-border pt-4">
        <h4 className="text-sm font-semibold text-primary mb-3">Subir documentos</h4>
        <p className="text-xs text-tertiary mb-3">
          Sube archivos a la carpeta <code className="text-accent">documents/</code> para su posterior escaneo e indexación.
          {scannedFiles?.supported_extensions?.length > 0 && (
            <> Formatos aceptados: <strong>{scannedFiles.supported_extensions.join(", ")}</strong></>
          )}
        </p>

        <div className="space-y-3">
          {/* Optional subfolder */}
          <div>
            <label className="block text-xs text-secondary mb-1">Subcarpeta (opcional)</label>
            <input
              type="text"
              value={uploadSubfolder}
              onChange={(e) => setUploadSubfolder(e.target.value)}
              placeholder="Ej: recetas/postres"
              className="w-full bg-input-bg text-primary text-sm px-3 py-2 rounded-lg border border-border focus:outline-none focus:border-accent placeholder:text-placeholder"
            />
          </div>

          {/* File picker + upload button */}
          <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-2">
            <input
              ref={uploadInputRef}
              type="file"
              multiple
              accept={scannedFiles?.supported_extensions?.map(e => e).join(",") || ".txt,.md,.pdf,.docx,.xlsx,.xls"}
              onChange={handleUpload}
              className="hidden"
              disabled={uploading}
            />
            <button
              onClick={() => uploadInputRef.current?.click()}
              disabled={uploading}
              className="px-5 py-2.5 rounded-lg bg-accent hover:bg-accent-hover text-white text-sm font-semibold transition-colors disabled:opacity-50 active:scale-95 shadow-sm"
            >
              {uploading ? (
                <span className="flex items-center justify-center gap-2">
                  <IconLoader className="w-4 h-4 animate-spin" /> Subiendo…
                </span>
              ) : (
                "Seleccionar y subir archivos"
              )}
            </button>
          </div>

          {/* Upload result */}
          {uploadResult && (
            <div
              className={`p-3 rounded-lg text-sm ${
                uploadResult.success
                  ? "bg-green-500/10 border border-green-500/30 text-green-400"
                  : "bg-red-500/10 border border-red-500/30 text-red-400"
              }`}
            >
              <p>{uploadResult.message}</p>
              {uploadResult.uploaded?.length > 0 && (
                <ul className="mt-1 text-xs space-y-0.5">
                  {uploadResult.uploaded.map((f) => (
                    <li key={f.path}>✓ {f.path} ({(f.size / 1024).toFixed(1)} KB)</li>
                  ))}
                </ul>
              )}
              {uploadResult.errors?.length > 0 && (
                <ul className="mt-1 text-xs space-y-0.5">
                  {uploadResult.errors.map((e, i) => (
                    <li key={i}>✗ {e.name}: {e.error}</li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Documents scan */}
      <div className="border-t border-border pt-4">
        <div className="flex items-center justify-between mb-3">
          <div>
            <h4 className="text-sm font-semibold text-primary">Documentos en <code className="text-accent">documents/</code></h4>
            {scannedFiles && (
              <p className="text-xs text-tertiary mt-0.5">
                {scannedFiles.total} archivos encontrados · {scannedFiles.indexed_count} indexados
                {scannedFiles.supported_extensions?.length > 0 && (
                  <> · Formatos: {scannedFiles.supported_extensions.join(", ")}</>
                )}
              </p>
            )}
          </div>
          <button
            onClick={handleScan}
            disabled={scanning}
            className="px-3.5 py-1.5 rounded-lg text-xs font-medium bg-accent/20 text-accent hover:bg-accent/30 border border-accent/30 transition-colors disabled:opacity-50 active:scale-95 shrink-0"
          >
            {scanning ? (
              <span className="flex items-center gap-1">
                <IconLoader className="w-3 h-3 animate-spin" /> Escaneando…
              </span>
            ) : (
              "Escanear"
            )}
          </button>
        </div>

        {scannedFiles?.files?.length > 0 ? (
          <div className="space-y-1 max-h-64 overflow-y-auto scrollbar-thin">
            {scannedFiles.files.map((f) => (
              <div
                key={f.path}
                className="flex items-center justify-between px-3 py-2.5 rounded-lg hover:bg-hover transition-colors"
              >
                <div className="flex items-center gap-2 sm:gap-3 min-w-0 flex-1">
                  <span
                    className={`w-2.5 h-2.5 rounded-full shrink-0 ${
                      f.indexed ? "bg-green-400" : f.supported ? "bg-yellow-400" : "bg-border"
                    }`}
                    title={f.indexed ? "Indexado" : f.supported ? "Soportado, no indexado" : "Formato no soportado"}
                  />
                  <div className="min-w-0 flex-1">
                    <div className="truncate">
                      <span className="text-sm text-primary">{f.name}</span>
                    </div>
                    <div className="flex items-center gap-2 mt-0.5">
                      <span className="text-xs text-tertiary">{formatSize(f.size)}</span>
                      {f.indexed && (
                        <span className="text-xs bg-green-500/20 text-green-400 px-1.5 py-0.5 rounded">
                          {f.chunks} chunks
                        </span>
                      )}
                      {!f.supported && (
                        <span className="text-xs bg-border text-tertiary px-1.5 py-0.5 rounded">
                          no soportado
                        </span>
                      )}
                    </div>
                  </div>
                </div>
                {f.indexed && (
                  <button
                    onClick={() => handleRemoveDoc(f.path)}
                    disabled={removing === f.path}
                    className="px-3 py-1.5 rounded-lg text-xs font-medium bg-red-500/20 text-red-400 hover:bg-red-500/30 border border-red-500/30 transition-colors disabled:opacity-50 shrink-0 ml-2 active:scale-95"
                  >
                    {removing === f.path ? "…" : "Quitar"}
                  </button>
                )}
              </div>
            ))}
          </div>
        ) : scannedFiles ? (
          <p className="text-xs text-tertiary py-4 text-center">
            No hay archivos en <code className="text-accent">documents/</code>. Agrega documentos para indexar.
          </p>
        ) : null}
      </div>

      {/* Sync & Rebuild actions */}
      <div className="border-t border-border pt-4">
        <h4 className="text-sm font-semibold text-primary mb-2">Acciones</h4>
        <p className="text-xs text-tertiary mb-3">
          <strong>Sincronizar:</strong> detecta cambios (nuevos, modificados y eliminados) y actualiza el índice.<br />
          <strong>Reconstruir:</strong> elimina toda la base y la vuelve a crear desde cero.
        </p>
        <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
          <button
            onClick={handleSync}
            disabled={isBusy}
            className="px-5 py-2.5 rounded-lg bg-accent hover:bg-accent-hover text-white text-sm font-semibold transition-colors disabled:opacity-50 active:scale-95 shadow-sm"
          >
            {syncing ? (
              <span className="flex items-center justify-center gap-2">
                <IconLoader className="w-4 h-4 animate-spin" /> Sincronizando…
              </span>
            ) : (
              "Sincronizar"
            )}
          </button>
          <button
            onClick={handleRebuild}
            disabled={isBusy}
            className="px-5 py-2.5 rounded-lg bg-red-500/20 text-red-400 hover:bg-red-500/30 text-sm font-semibold transition-colors disabled:opacity-50 border border-red-500/30 active:scale-95"
          >
            {rebuilding ? (
              <span className="flex items-center justify-center gap-2">
                <IconLoader className="w-4 h-4 animate-spin" /> Reconstruyendo…
              </span>
            ) : (
              "Reconstruir base completa"
            )}
          </button>
        </div>

        {syncResult && (
          <div
            className={`mt-3 p-3 rounded-lg text-sm ${
              syncResult.success
                ? "bg-green-500/10 border border-green-500/30 text-green-400"
                : "bg-red-500/10 border border-red-500/30 text-red-400"
            }`}
          >
            {syncResult.success
              ? syncResult.rebuilt
                ? "Base reconstruida exitosamente"
                : "Sincronización completada"
              : `Error: ${syncResult.error}`}
            {syncResult.output && (
              <pre className="mt-2 text-xs text-secondary whitespace-pre-wrap max-h-40 overflow-y-auto">{syncResult.output}</pre>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════
// Main AdminPanel
// ═══════════════════════════════════════════════════════════
export default function AdminPanel({ onClose }) {
  const [activeTab, setActiveTab] = useState("general");
  const [config, setConfig] = useState(null);
  const [loading, setLoading] = useState(true);
  const [installedModels, setInstalledModels] = useState([]);

  useEffect(() => {
    Promise.all([getConfig(), getServicesStatus()])
      .then(([cfg, svc]) => {
        setConfig(cfg);
        setInstalledModels(svc.ollama?.installed_models || []);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  const handleSave = async (partial) => {
    const result = await updateConfig(partial);
    setConfig(result.config);
    return result;
  };

  return (
    <div className="fixed inset-0 z-50 flex items-end sm:items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-chat-bg border border-border rounded-t-2xl sm:rounded-2xl shadow-2xl w-full sm:max-w-3xl h-[95vh] sm:max-h-[90vh] flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-4 sm:px-6 py-3 sm:py-4 border-b border-border shrink-0">
          <h2 className="text-base sm:text-lg font-bold text-primary">Panel de administración</h2>
          <button
            onClick={onClose}
            className="p-2.5 rounded-xl bg-hover hover:bg-border text-secondary hover:text-primary transition-colors active:scale-95"
          >
            <IconClose className="w-5 h-5" />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex gap-1.5 px-4 sm:px-6 py-2.5 border-b border-border shrink-0 overflow-x-auto scrollbar-thin">
          {TABS.map((tab) => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors whitespace-nowrap active:scale-95 ${
                activeTab === tab.key
                  ? "bg-accent text-white shadow-sm"
                  : "bg-hover text-secondary hover:bg-border hover:text-primary"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4 sm:p-6 scrollbar-thin">
          {loading ? (
            <div className="flex items-center justify-center py-16">
              <IconLoader className="w-6 h-6 animate-spin text-accent" />
              <span className="ml-3 text-sm text-tertiary">Cargando configuración…</span>
            </div>
          ) : (
            <>
              {activeTab === "general" && <GeneralTab config={config} onSave={handleSave} />}
              {activeTab === "models" && (
                <ModelsTab config={config} onSave={handleSave} installedModels={installedModels} />
              )}
              {activeTab === "services" && <ServicesTab />}
              {activeTab === "rag" && <RagTab config={config} onSave={handleSave} />}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

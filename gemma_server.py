"""
FastAPI wrapper with Ollama backend + Whisper for audio transcription.

Architecture:
1. Audio arrives from frontend
2. Whisper transcribes audio to text
3. Ollama responds to the transcribed text

• POST /ask         – audio→whisper→text→ollama→response
• POST /ask_image   – image+prompt→ollama(llava)→response
• POST /ask_text    – text→ollama→response

CORS is open for http://localhost:5173 so the React front-end can call us.
"""

import base64
import os
import json
import tempfile
import ssl
import subprocess
import platform
import httpx
import numpy as np
import scipy.io.wavfile as wavfile
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict

# RAG imports
try:
    from rag_module import RAGSystem, build_rag_prompt, load_index
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("⚠️  RAG module not available. Install with: pip install chromadb")

# Fix SSL issues for model downloads
ssl._create_default_https_context = ssl._create_unverified_context

# --------------------------------------------------------------------
# Ollama configuration
# --------------------------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434"

# Available models (these should be pulled with `ollama pull <model>`)
AVAILABLE_MODELS = {
    # Gemma family
    "gemma2:2b": {
        "name": "Gemma 2 2B",
        "description": "Google Gemma 2, ligero y rápido",
        "size": "~1.6GB",
        "vision": False,
    },
    "gemma2:9b": {
        "name": "Gemma 2 9B",
        "description": "Google Gemma 2, alta calidad",
        "size": "~5.4GB",
        "vision": False,
    },
    # Llama family
    "llama3.2:3b": {
        "name": "Llama 3.2 3B",
        "description": "Meta Llama 3.2, multilingüe",
        "size": "~2GB",
        "vision": False,
    },
    "llama3.2:1b": {
        "name": "Llama 3.2 1B",
        "description": "Meta Llama 3.2, ultra ligero",
        "size": "~1.3GB",
        "vision": False,
    },
    # Vision models
    "llava:7b": {
        "name": "LLaVA 7B",
        "description": "Modelo de visión, entiende imágenes",
        "size": "~4.7GB",
        "vision": True,
    },
    "llava:13b": {
        "name": "LLaVA 13B",
        "description": "Modelo de visión, mayor calidad",
        "size": "~8GB",
        "vision": True,
    },
    # Mistral
    "mistral:7b": {
        "name": "Mistral 7B",
        "description": "Mistral AI, muy eficiente",
        "size": "~4.1GB",
        "vision": False,
    },
    # Qwen
    "qwen2.5:3b": {
        "name": "Qwen 2.5 3B",
        "description": "Alibaba Qwen 2.5, ligero",
        "size": "~1.9GB",
        "vision": False,
    },
    "qwen2.5:7b": {
        "name": "Qwen 2.5 7B",
        "description": "Alibaba Qwen 2.5, equilibrado",
        "size": "~4.7GB",
        "vision": False,
    },
    # Phi family
    "phi3:mini": {
        "name": "Phi-3 Mini",
        "description": "Microsoft Phi-3, compacto",
        "size": "~2.3GB",
        "vision": False,
    },
    "phi4:latest": {
        "name": "Phi-4 14B",
        "description": "Microsoft Phi-4, razonamiento avanzado",
        "size": "~9.1GB",
        "vision": False,
    },
    # Gemma 3 family
    "gemma3:1b": {
        "name": "Gemma 3 1B",
        "description": "Google Gemma 3, ultra ligero",
        "size": "~815MB",
        "vision": False,
    },
    "gemma3:4b": {
        "name": "Gemma 3 4B",
        "description": "Google Gemma 3, balance calidad/tamaño",
        "size": "~3.3GB",
        "vision": False,
    },
    "gemma3:12b": {
        "name": "Gemma 3 12B",
        "description": "Google Gemma 3, alta calidad",
        "size": "~8.9GB",
        "vision": False,
    },
    "gemma3:27b": {
        "name": "Gemma 3 27B",
        "description": "Google Gemma 3, máxima calidad",
        "size": "~17GB",
        "vision": False,
    },
    # DeepSeek R1 (reasoning model)
    "deepseek-r1:1.5b": {
        "name": "DeepSeek R1 1.5B",
        "description": "DeepSeek R1, razonamiento ligero",
        "size": "~1.1GB",
        "vision": False,
    },
    "deepseek-r1:7b": {
        "name": "DeepSeek R1 7B",
        "description": "DeepSeek R1, razonamiento equilibrado",
        "size": "~4.7GB",
        "vision": False,
    },
    "deepseek-r1:8b": {
        "name": "DeepSeek R1 8B",
        "description": "DeepSeek R1, razonamiento avanzado",
        "size": "~4.9GB",
        "vision": False,
    },
    "deepseek-r1:14b": {
        "name": "DeepSeek R1 14B",
        "description": "DeepSeek R1, alta capacidad",
        "size": "~9GB",
        "vision": False,
    },
    "deepseek-r1:32b": {
        "name": "DeepSeek R1 32B",
        "description": "DeepSeek R1, máximo razonamiento",
        "size": "~20GB",
        "vision": False,
    },
}

DEFAULT_MODEL = "gemma2:2b"
DEFAULT_VISION_MODEL = "llava:7b"

# --------------------------------------------------------------------
# Smart Routing Configuration
# --------------------------------------------------------------------
ROUTER_MODEL = "qwen2.5:0.5b"  # Ultra-fast model for classification

# Model routing map: category -> model
ROUTE_MODELS = {
    "math": "phi4:latest",      # Mathematical operations, calculations, proportions
    "chat": "gemma3:4b",        # Conversational, general knowledge, information
}

ROUTER_PROMPT = """Clasifica la siguiente consulta. Responde SOLO con una palabra:
- "math" si requiere cálculos, operaciones matemáticas, conversiones, proporciones, cantidades, medidas, porcentajes
- "chat" si es conversacional, información general, preguntas simples, saludos, definiciones

Consulta: {query}
Clasificación:"""

# --------------------------------------------------------------------
# Context Management Configuration
# --------------------------------------------------------------------
# Context window size for Ollama (tokens)
# Larger = longer conversations, but more memory
CONTEXT_WINDOW_SIZE = 8192  # Default: 2048, can go up to 32k for some models

# Maximum number of messages to keep in history before summarizing
MAX_HISTORY_MESSAGES = 20

# When history exceeds limit, keep these recent messages and summarize the rest
KEEP_RECENT_MESSAGES = 6

# Approximate tokens per character (conservative estimate for Spanish)
CHARS_PER_TOKEN = 3.5

# Summarization model (use fast model for efficiency)
SUMMARY_MODEL = "qwen2.5:0.5b"

SUMMARY_PROMPT = """Resume la siguiente conversación en 2-3 oraciones cortas, manteniendo los puntos clave y el contexto importante. Solo proporciona el resumen, sin introducción:

{conversation}

Resumen:"""

# --------------------------------------------------------------------
# RAG Configuration
# --------------------------------------------------------------------
# Enable/disable RAG system
RAG_ENABLED = True

# Number of documents to retrieve for context
RAG_TOP_K = 3

# Minimum relevance score (0-1, higher = more strict)
RAG_MIN_RELEVANCE = 0.3

# RAG system instance (loaded lazily)
_rag_system = None


def get_rag_system() -> Optional["RAGSystem"]:
    """Get or initialize the RAG system."""
    global _rag_system
    if not RAG_ENABLED or not RAG_AVAILABLE:
        return None
    if _rag_system is None:
        try:
            _rag_system = RAGSystem()
            print("📚 RAG system initialized")
        except Exception as e:
            print(f"⚠️  RAG initialization failed: {e}")
            return None
    return _rag_system


async def search_rag_context(query: str) -> tuple[str, List[Dict]]:
    """
    Search RAG for relevant context.
    
    Args:
        query: User's question
    
    Returns:
        Tuple of (formatted_context, raw_results)
    """
    rag = get_rag_system()
    if not rag:
        return "", []
    
    try:
        results = rag.search(query, n_results=RAG_TOP_K)
        
        # Filter by relevance
        relevant = [r for r in results if r.get("relevance", 0) >= RAG_MIN_RELEVANCE]
        
        if not relevant:
            return "", []
        
        # Format context for prompt
        context_parts = []
        for doc in relevant:
            source = doc.get("metadata", {}).get("source", "Documento")
            section = doc.get("metadata", {}).get("section", "")
            relevance = doc.get("relevance", 0) * 100
            
            header = f"[{source}]" + (f" - {section}" if section else "")
            context_parts.append(f"### {header} (relevancia: {relevance:.0f}%)\n{doc['content']}")
        
        formatted = "\n\n".join(context_parts)
        print(f"📚 RAG: Found {len(relevant)} relevant documents for: '{query[:50]}...'")
        
        return formatted, relevant
    
    except Exception as e:
        print(f"⚠️  RAG search error: {e}")
        return "", []


# --------------------------------------------------------------------
# Global Whisper model (loaded lazily)
# --------------------------------------------------------------------
whisper_model = None


def get_whisper_model():
    """Load Whisper model for speech-to-text."""
    global whisper_model
    if whisper_model is None:
        import whisper
        print("Loading Whisper model (base)...")
        whisper_model = whisper.load_model("base")
        print("Whisper model loaded.")
    return whisper_model


# --------------------------------------------------------------------
# Ollama API functions
# --------------------------------------------------------------------

async def check_ollama_running() -> bool:
    """Check if Ollama server is running."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
            return response.status_code == 200
    except:
        return False


async def get_installed_models() -> List[str]:
    """Get list of models installed in Ollama."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
    except Exception as e:
        print(f"Error getting Ollama models: {e}")
    return []


async def ollama_chat(
    model: str,
    messages: List[Dict],
    temperature: float = 0.7,
    max_tokens: int = 512,
    num_ctx: int = None,
) -> str:
    """
    Send a chat request to Ollama.
    
    Args:
        model: Model name (e.g., "gemma2:2b")
        messages: List of {"role": "user/assistant/system", "content": "..."}
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        num_ctx: Context window size (None = use global config)
    
    Returns:
        Generated text response
    """
    ctx_size = num_ctx or CONTEXT_WINDOW_SIZE
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "num_ctx": ctx_size,
                }
            },
            timeout=120.0,
        )
        
        if response.status_code != 200:
            error_text = response.text
            raise HTTPException(500, f"Ollama error: {error_text}")
        
        data = response.json()
        return data.get("message", {}).get("content", "").strip()


async def ollama_generate_with_image(
    model: str,
    prompt: str,
    image_path: str,
    temperature: float = 0.7,
) -> str:
    """
    Generate response with image using Ollama vision model.
    
    Args:
        model: Vision model name (e.g., "llava:7b")
        prompt: Text prompt about the image
        image_path: Path to image file
    
    Returns:
        Generated text response
    """
    # Read and encode image
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "options": {
                    "temperature": temperature,
                }
            },
            timeout=120.0,
        )
        
        if response.status_code != 200:
            error_text = response.text
            raise HTTPException(500, f"Ollama vision error: {error_text}")
        
        data = response.json()
        return data.get("response", "").strip()


async def route_query(query: str) -> tuple[str, str]:
    """
    Use a small model to classify the query and route to appropriate model.
    
    Args:
        query: User's query text
    
    Returns:
        Tuple of (selected_model, category)
    """
    try:
        prompt = ROUTER_PROMPT.format(query=query)
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": ROUTER_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0,  # Deterministic for classification
                        "num_predict": 10,  # Only need one word
                    }
                },
                timeout=30.0,
            )
            
            if response.status_code != 200:
                print(f"Router error, using default: {response.text}")
                return DEFAULT_MODEL, "default"
            
            data = response.json()
            category = data.get("response", "").strip().lower()
            
            # Extract just the category word
            if "math" in category:
                category = "math"
            elif "chat" in category:
                category = "chat"
            else:
                category = "chat"  # Default to chat if unclear
            
            selected_model = ROUTE_MODELS.get(category, DEFAULT_MODEL)
            print(f"🔀 Router: '{query[:50]}...' → {category} → {selected_model}")
            
            return selected_model, category
            
    except Exception as e:
        print(f"Router exception: {e}, using default model")
        return DEFAULT_MODEL, "default"


# --------------------------------------------------------------------
# FastAPI + CORS
# --------------------------------------------------------------------

app = FastAPI(title="Ollama + Whisper Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# =============================================================================
# CONTEXTO INTERNO DEL BACKEND (editar aquí para cambiar el comportamiento base)
# Este es el SYSTEM PROMPT principal que define la personalidad del asistente
# =============================================================================
INTERNAL_CONTEXT = """
Eres un asistente experto que trabaja para una empresa de tecnología.
Respondes en español de forma profesional, precisa y concisa.
Cuando no sepas algo, lo admites honestamente.
Eres amigable y servicial, ayudando al usuario con cualquier consulta.
""".strip()
# =============================================================================


# --------------------------------------------------------------------
# Context Management Functions
# --------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Estimate token count for a text (approximate)."""
    return int(len(text) / CHARS_PER_TOKEN)


def estimate_messages_tokens(messages: List[Dict]) -> int:
    """Estimate total tokens for a list of messages."""
    total = 0
    for msg in messages:
        total += estimate_tokens(msg.get("text", "") or msg.get("content", ""))
        total += 4  # Overhead per message
    return total


async def summarize_conversation(messages: List[Dict]) -> str:
    """
    Use a fast model to summarize a conversation.
    
    Args:
        messages: List of messages to summarize
    
    Returns:
        A brief summary string
    """
    if not messages:
        return ""
    
    # Format conversation for summarization
    conversation_text = ""
    for msg in messages:
        role = "Usuario" if msg.get("role") in ["user", "usuario"] else "Asistente"
        text = msg.get("text", "") or msg.get("content", "")
        conversation_text += f"{role}: {text}\n"
    
    prompt = SUMMARY_PROMPT.format(conversation=conversation_text)
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": SUMMARY_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 150,
                    }
                },
                timeout=30.0,
            )
            
            if response.status_code == 200:
                data = response.json()
                summary = data.get("response", "").strip()
                print(f"📝 Summarized {len(messages)} messages: {summary[:100]}...")
                return summary
            else:
                print(f"Summary error: {response.text}")
                return ""
    except Exception as e:
        print(f"Summary exception: {e}")
        return ""


def truncate_history_simple(
    history: List[Dict[str, str]],
    max_messages: int = MAX_HISTORY_MESSAGES
) -> List[Dict[str, str]]:
    """
    Simple truncation: keep only the most recent messages.
    
    Args:
        history: Full conversation history
        max_messages: Maximum messages to keep
    
    Returns:
        Truncated history
    """
    if len(history) <= max_messages:
        return history
    
    print(f"✂️ Truncating history from {len(history)} to {max_messages} messages")
    return history[-max_messages:]


async def manage_context(
    history: List[Dict[str, str]],
    max_messages: int = MAX_HISTORY_MESSAGES,
    keep_recent: int = KEEP_RECENT_MESSAGES,
) -> tuple[List[Dict[str, str]], str]:
    """
    Intelligently manage conversation context.
    
    If history is too long:
    1. Keep the most recent messages
    2. Summarize older messages
    3. Return truncated history + summary
    
    Args:
        history: Full conversation history
        max_messages: Threshold for triggering summarization
        keep_recent: Number of recent messages to keep
    
    Returns:
        Tuple of (truncated_history, context_summary)
    """
    if not history or len(history) <= max_messages:
        return history or [], ""
    
    # Split into old and recent
    old_messages = history[:-keep_recent]
    recent_messages = history[-keep_recent:]
    
    # Summarize old messages
    summary = await summarize_conversation(old_messages)
    
    print(f"🔄 Context managed: {len(history)} → {len(recent_messages)} msgs + summary")
    
    return recent_messages, summary


def build_system_prompt(
    user_context: str = None,
    context_summary: str = None,
    rag_context: str = None
) -> str:
    """Build system prompt combining internal context with optional user context, summary, and RAG context."""
    # INTERNAL_CONTEXT es el prompt base principal
    system_prompt = INTERNAL_CONTEXT if INTERNAL_CONTEXT else "Eres un asistente amigable y útil. Responde en español de forma concisa."
    
    # Agregar contexto RAG (documentos relevantes de la base de conocimiento)
    # PRIORIDAD: La base de conocimiento tiene precedencia sobre el conocimiento general
    if rag_context:
        system_prompt = f"""{system_prompt}

## Base de Conocimiento (PRIORIDAD ALTA)
IMPORTANTE: Usa PRIORITARIAMENTE la siguiente información de nuestra base de conocimiento para responder.
Si la información relevante está en estos documentos, basa tu respuesta en ellos.
Solo complementa con tu conocimiento general si los documentos no cubren completamente la pregunta.
Si nada de los documentos es relevante a la pregunta, responde con tu conocimiento general.
Cita la fuente del documento cuando uses información de la base de conocimiento.

{rag_context}
"""
    
    # Agregar resumen de conversación anterior si hay
    if context_summary:
        system_prompt = f"{system_prompt}\n\nResumen de la conversación anterior: {context_summary}"
    
    # Agregar contexto del usuario (desde el frontend)
    if user_context:
        system_prompt = f"{system_prompt}\n\nContexto del usuario: {user_context}"
    
    return system_prompt


def build_ollama_messages(
    system_prompt: str,
    user_text: str,
    history: List[Dict[str, str]] = None
) -> List[Dict]:
    """
    Build messages array for Ollama chat API.
    
    Args:
        system_prompt: System instructions
        user_text: Current user message
        history: Previous messages [{"role": "user" or "model", "text": "..."}]
    
    Returns:
        messages array for Ollama
    """
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add history
    if history:
        for msg in history:
            role = "assistant" if msg["role"] == "model" else "user"
            messages.append({"role": role, "content": msg["text"]})
    
    # Add current message
    messages.append({"role": "user", "content": user_text})
    
    return messages


# --------------------------------------------------------------------
# /models  — list available models
# --------------------------------------------------------------------

@app.get("/models")
async def list_models():
    """Return list of available Ollama models."""
    installed = await get_installed_models()
    
    # Add "auto" option first for smart routing
    models = [{
        "key": "auto",
        "name": "🔀 Auto (Smart Routing)",
        "description": f"Routing automático: math→{ROUTE_MODELS['math']}, chat→{ROUTE_MODELS['chat']}",
        "size": "~0",
        "family": "router",
        "vision": False,
        "loaded": ROUTER_MODEL.split(":")[0] in " ".join(installed),
        "current": False,
    }]
    
    for key, info in AVAILABLE_MODELS.items():
        # Check if model name matches any installed model
        is_loaded = any(key in m or m.startswith(key.split(":")[0]) for m in installed)
        
        models.append({
            "key": key,
            "name": info["name"],
            "description": info["description"],
            "size": info["size"],
            "family": "ollama",
            "vision": info.get("vision", False),
            "loaded": is_loaded,
            "current": False,
        })
    
    return {
        "models": models,
        "default": "auto",  # Default to auto routing
        "current": None,
        "installed": installed,
        "router_model": ROUTER_MODEL,
        "route_models": ROUTE_MODELS,
    }


@app.delete("/models/cache")
async def clear_model_cache():
    """
    Delete all installed Ollama models and report freed space.
    """
    installed = await get_installed_models()
    
    if not installed:
        return {
            "deleted": [],
            "errors": [],
            "freed_gb": 0,
            "message": "No hay modelos instalados para eliminar"
        }
    
    deleted = []
    errors = []
    freed_bytes = 0
    
    # Get model sizes before deleting
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10.0)
            if response.status_code == 200:
                models_data = response.json().get("models", [])
                model_sizes = {m["name"]: m.get("size", 0) for m in models_data}
            else:
                model_sizes = {}
    except:
        model_sizes = {}
    
    # Delete each model
    async with httpx.AsyncClient() as client:
        for model_name in installed:
            try:
                response = await client.delete(
                    f"{OLLAMA_BASE_URL}/api/delete",
                    json={"name": model_name},
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    size = model_sizes.get(model_name, 0)
                    freed_bytes += size
                    deleted.append(model_name)
                    print(f"Deleted {model_name} ({size / 1e9:.2f} GB)")
                else:
                    errors.append({"model": model_name, "error": response.text})
                    
            except Exception as e:
                errors.append({"model": model_name, "error": str(e)})
    
    freed_gb = round(freed_bytes / 1e9, 2)
    
    return {
        "deleted": deleted,
        "errors": errors,
        "freed_gb": freed_gb,
        "message": f"Eliminados {len(deleted)} modelos, liberados {freed_gb} GB"
    }


# --------------------------------------------------------------------
# /ask  — audio blob (base-64)  →  whisper → ollama → text
# --------------------------------------------------------------------

class AudioPayload(BaseModel):
    data: str  # base-64 WAV data
    model: Optional[str] = None
    context: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None


@app.post("/ask")
async def ask_audio(payload: AudioPayload):
    """Process audio: Whisper transcribes it, then Ollama responds."""
    wav_path = None
    try:
        print("\n" + "=" * 60)
        print("Received audio request")
        
        # Decode base64 audio
        wav_bytes = base64.b64decode(payload.data)
        print(f"Decoded {len(wav_bytes)} bytes")
        
        # Save to temp file for Whisper
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            wav_path = tmp.name
            tmp.write(wav_bytes)
        
        # Get audio info
        sample_rate, audio_data = wavfile.read(wav_path)
        duration = len(audio_data) / sample_rate
        print(f"Audio: {sample_rate}Hz, {duration:.2f}s")
        
        # Normalize audio volume
        if len(audio_data) > 0:
            audio_float = audio_data.astype(np.float32)
            max_val = np.max(np.abs(audio_float))
            if max_val > 0:
                audio_normalized = (audio_float / max_val) * 0.9 * 32767
                audio_normalized = audio_normalized.astype(np.int16)
                wavfile.write(wav_path, sample_rate, audio_normalized)
        
        # Step 1: Transcribe with Whisper
        print("\n--- STEP 1: Whisper Transcription ---")
        whisper_m = get_whisper_model()
        
        initial_prompt = "Esta es una conversación en español. El usuario hace preguntas o da instrucciones."
        whisper_result = whisper_m.transcribe(
            wav_path,
            language="es",
            task="transcribe",
            initial_prompt=initial_prompt,
            temperature=0,
            condition_on_previous_text=False,
            fp16=False
        )
        user_text = whisper_result["text"].strip()
        print(f"Transcribed: '{user_text}'")
        
        if not user_text:
            return {"text": "No pude entender lo que dijiste. ¿Puedes repetirlo?"}
        
        # Step 2: Generate response with Ollama
        model = payload.model or "auto"
        routed_category = None
        
        # Smart routing: auto-select model based on query type
        if model == "auto":
            model, routed_category = await route_query(user_text)
            print(f"\n--- STEP 2: Auto-routed to {model} ({routed_category}) ---")
        else:
            print(f"\n--- STEP 2: Ollama Response (model: {model}) ---")
        
        if payload.context:
            ctx_preview = payload.context[:100] + '...' if len(payload.context) > 100 else payload.context
            print(f"With context: '{ctx_preview}'")
        
        # Manage conversation context (truncate + summarize if needed)
        managed_history, context_summary = await manage_context(payload.history or [])
        
        if payload.history:
            original_count = len(payload.history)
            managed_count = len(managed_history)
            if original_count != managed_count:
                print(f"\n--- Context Management: {original_count} → {managed_count} messages ---")
                if context_summary:
                    print(f"Summary: {context_summary[:100]}...")
            else:
                print(f"\n--- Chat History ({managed_count} messages) ---")
            
            for i, msg in enumerate(managed_history):
                preview = msg['text'][:80] + '...' if len(msg['text']) > 80 else msg['text']
                print(f"  [{i+1}] {msg['role']}: {preview}")
        
        # Search RAG for relevant context
        rag_context, rag_results = await search_rag_context(user_text)
        
        system_prompt = build_system_prompt(payload.context, context_summary, rag_context)
        messages = build_ollama_messages(system_prompt, user_text, managed_history)
        
        response = await ollama_chat(model, messages)
        print(f"Ollama response: '{response}'")
        print("=" * 60 + "\n")
        
        result = {"text": response, "transcription": user_text, "model": model}
        if routed_category:
            result["routed_category"] = routed_category
        if rag_results:
            result["rag_sources"] = list({r.get("metadata", {}).get("source", "") for r in rag_results})
            result["rag_chunks"] = [
                {
                    "content": r.get("content", "")[:500],
                    "source": r.get("metadata", {}).get("source", "Documento"),
                    "section": r.get("metadata", {}).get("section", ""),
                    "relevance": round(r.get("relevance", 0), 4),
                }
                for r in rag_results
            ]
        return result
    
    except Exception as exc:
        import traceback
        error_msg = f"Error: {str(exc)}\n{traceback.format_exc()}"
        print(error_msg)
        raise HTTPException(500, error_msg) from exc
    finally:
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)


# --------------------------------------------------------------------
# /ask_text  — text prompt → ollama → text
# --------------------------------------------------------------------

class TextPayload(BaseModel):
    text: str
    model: Optional[str] = None
    context: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None


@app.post("/ask_text")
async def ask_text(payload: TextPayload):
    """Process text directly: sends to Ollama."""
    try:
        print("\n" + "=" * 60)
        print("Received text request")
        
        user_text = payload.text.strip()
        print(f"User text: '{user_text}'")
        
        if not user_text:
            return {"text": "No recibí ningún texto. ¿Puedes intentarlo de nuevo?"}
        
        model = payload.model or "auto"
        routed_category = None
        
        # Smart routing: auto-select model based on query type
        if model == "auto":
            model, routed_category = await route_query(user_text)
            print(f"\n--- Auto-routed to {model} ({routed_category}) ---")
        else:
            print(f"\n--- Ollama Response (model: {model}) ---")
        
        if payload.context:
            ctx_preview = payload.context[:100] + '...' if len(payload.context) > 100 else payload.context
            print(f"With context: '{ctx_preview}'")
        
        # Manage conversation context (truncate + summarize if needed)
        managed_history, context_summary = await manage_context(payload.history or [])
        
        if payload.history:
            original_count = len(payload.history)
            managed_count = len(managed_history)
            if original_count != managed_count:
                print(f"\n--- Context Management: {original_count} → {managed_count} messages ---")
                if context_summary:
                    print(f"Summary: {context_summary[:100]}...")
            else:
                print(f"\n--- Chat History ({managed_count} messages) ---")
            
            for i, msg in enumerate(managed_history):
                preview = msg['text'][:80] + '...' if len(msg['text']) > 80 else msg['text']
                print(f"  [{i+1}] {msg['role']}: {preview}")
        
        # Search RAG for relevant context
        rag_context, rag_results = await search_rag_context(user_text)
        
        system_prompt = build_system_prompt(payload.context, context_summary, rag_context)
        messages = build_ollama_messages(system_prompt, user_text, managed_history)
        
        response = await ollama_chat(model, messages)
        print(f"Ollama response: '{response}'")
        print("=" * 60 + "\n")
        
        result = {"text": response, "model": model}
        if routed_category:
            result["routed_category"] = routed_category
        if rag_results:
            result["rag_sources"] = list({r.get("metadata", {}).get("source", "") for r in rag_results})
            result["rag_chunks"] = [
                {
                    "content": r.get("content", "")[:500],
                    "source": r.get("metadata", {}).get("source", "Documento"),
                    "section": r.get("metadata", {}).get("section", ""),
                    "relevance": round(r.get("relevance", 0), 4),
                }
                for r in rag_results
            ]
        return result
    
    except Exception as exc:
        import traceback
        error_msg = f"Error: {str(exc)}\n{traceback.format_exc()}"
        print(error_msg)
        raise HTTPException(500, error_msg) from exc


# --------------------------------------------------------------------
# /ask_image  — multipart(form-data) → llava → text
# --------------------------------------------------------------------

@app.post("/ask_image")
async def ask_image(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    context: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
):
    """Process image with text prompt using Ollama vision model (LLaVA)."""
    img_path = None
    try:
        print(f"\nReceived image request with prompt: {prompt}")
        
        # Select vision model
        vision_model = model or DEFAULT_VISION_MODEL
        
        # Check if model supports vision
        if vision_model in AVAILABLE_MODELS and not AVAILABLE_MODELS[vision_model].get("vision"):
            vision_model = DEFAULT_VISION_MODEL
            print(f"Model doesn't support vision, using {vision_model}")
        
        suffix = os.path.splitext(image.filename)[1] or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            img_path = tmp.name
            tmp.write(await image.read())
        print(f"Saved image to {img_path}")
        
        # Build prompt with context
        full_prompt = prompt
        if context:
            full_prompt = f"{context}\n\n{prompt}"
        
        response = await ollama_generate_with_image(vision_model, full_prompt, img_path)
        print(f"Response: {response}")
        
        return {"text": response, "model": vision_model}
    
    except Exception as exc:
        import traceback
        error_msg = f"Error: {str(exc)}\n{traceback.format_exc()}"
        print(error_msg)
        raise HTTPException(500, error_msg) from exc
    finally:
        if img_path and os.path.exists(img_path):
            os.remove(img_path)


# --------------------------------------------------------------------
# Text-to-Speech (native macOS or cross-platform)
# --------------------------------------------------------------------

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    rate: Optional[int] = 180

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """Convert text to speech using native TTS."""
    try:
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="No text provided")
        
        system = platform.system()
        
        if system == "Darwin":  # macOS
            with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as aiff_file:
                aiff_path = aiff_file.name
            
            try:
                voice = request.voice or "Paulina"
                rate = request.rate or 180
                
                cmd = ["say", "-v", voice, "-r", str(rate), "-o", aiff_path, text]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode != 0:
                    raise HTTPException(status_code=500, detail="TTS generation failed")
                
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
                    wav_path = wav_file.name
                
                convert_cmd = ["afconvert", "-f", "WAVE", "-d", "LEI16@22050", aiff_path, wav_path]
                subprocess.run(convert_cmd, capture_output=True, timeout=10)
                
                with open(wav_path, "rb") as f:
                    audio_data = f.read()
                
                audio_b64 = base64.b64encode(audio_data).decode("utf-8")
                
                os.remove(aiff_path)
                os.remove(wav_path)
                
                return {
                    "audio": audio_b64,
                    "format": "wav",
                    "voice": voice,
                    "engine": "macos_native"
                }
                
            except subprocess.TimeoutExpired:
                raise HTTPException(status_code=500, detail="TTS timeout")
            finally:
                if os.path.exists(aiff_path):
                    os.remove(aiff_path)
        
        elif system == "Windows":
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
                wav_path = wav_file.name
            
            try:
                ps_script = f'''
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Rate = 1
$voices = $synth.GetInstalledVoices()
foreach ($voice in $voices) {{
    if ($voice.VoiceInfo.Culture.Name -like "es-*") {{
        $synth.SelectVoice($voice.VoiceInfo.Name)
        break
    }}
}}
$synth.SetOutputToWaveFile("{wav_path}")
$synth.Speak("{text.replace('"', '`"').replace("'", "''")}")
$synth.Dispose()
'''
                
                cmd = ["powershell", "-Command", ps_script]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode != 0:
                    raise HTTPException(status_code=500, detail="TTS generation failed")
                
                with open(wav_path, "rb") as f:
                    audio_data = f.read()
                
                audio_b64 = base64.b64encode(audio_data).decode("utf-8")
                
                return {
                    "audio": audio_b64,
                    "format": "wav",
                    "voice": "Windows Spanish",
                    "engine": "windows_sapi"
                }
                
            except subprocess.TimeoutExpired:
                raise HTTPException(status_code=500, detail="TTS timeout")
            finally:
                if os.path.exists(wav_path):
                    os.remove(wav_path)
        
        else:
            return {
                "error": "native_tts_unavailable",
                "message": "Use browser speech synthesis",
                "engine": None
            }
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"TTS error: {e}")
        return {
            "error": "tts_failed",
            "message": str(e),
            "engine": None
        }


@app.get("/tts/voices")
async def get_available_voices():
    """Get list of available Spanish TTS voices."""
    system = platform.system()
    
    if system == "Darwin":
        try:
            result = subprocess.run(
                ["say", "-v", "?"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            spanish_voices = []
            for line in result.stdout.split("\n"):
                if "es_" in line or "es-" in line:
                    parts = line.split()
                    if parts:
                        spanish_voices.append({
                            "name": parts[0],
                            "language": "es",
                            "engine": "macos_native"
                        })
            
            return {
                "voices": spanish_voices,
                "default": "Paulina",
                "engine": "macos_native"
            }
        except Exception as e:
            print(f"Error getting voices: {e}")
            return {"voices": [], "engine": None}
    
    return {"voices": [], "engine": None}


# --------------------------------------------------------------------
# RAG Admin Endpoints
# --------------------------------------------------------------------

@app.get("/rag/status")
async def rag_status():
    """Get RAG system status and statistics."""
    if not RAG_AVAILABLE:
        return {
            "enabled": False,
            "available": False,
            "message": "RAG module not installed. Run: pip install chromadb"
        }
    
    if not RAG_ENABLED:
        return {
            "enabled": False,
            "available": True,
            "message": "RAG is disabled in configuration"
        }
    
    rag = get_rag_system()
    if not rag:
        return {
            "enabled": True,
            "available": True,
            "initialized": False,
            "message": "RAG system failed to initialize"
        }
    
    stats = rag.get_stats()
    index = load_index()
    
    return {
        "enabled": True,
        "available": True,
        "initialized": True,
        "documents": len(index.get("files", {})),
        "chunks": stats.get("total_documents", 0),
        "embedding_model": stats.get("embedding_model", "unknown"),
        "last_sync": index.get("last_sync"),
        "config": {
            "top_k": RAG_TOP_K,
            "min_relevance": RAG_MIN_RELEVANCE
        }
    }


@app.get("/rag/documents")
async def rag_documents():
    """List indexed documents."""
    if not RAG_AVAILABLE or not RAG_ENABLED:
        return {"documents": [], "error": "RAG not available"}
    
    index = load_index()
    documents = []
    
    for rel_path, info in index.get("files", {}).items():
        documents.append({
            "path": rel_path,
            "chunks": info.get("chunks", 0),
            "indexed_at": info.get("indexed_at", ""),
        })
    
    return {"documents": documents, "total": len(documents)}


@app.post("/rag/search")
async def rag_search_endpoint(query: str, n_results: int = 3):
    """Test RAG search."""
    if not RAG_AVAILABLE or not RAG_ENABLED:
        return {"results": [], "error": "RAG not available"}
    
    rag = get_rag_system()
    if not rag:
        return {"results": [], "error": "RAG not initialized"}
    
    results = rag.search(query, n_results=n_results)
    
    return {
        "query": query,
        "results": [
            {
                "content": r["content"][:500] + "..." if len(r["content"]) > 500 else r["content"],
                "source": r.get("metadata", {}).get("source", ""),
                "relevance": round(r.get("relevance", 0) * 100, 1)
            }
            for r in results
        ]
    }


@app.post("/rag/sync")
async def rag_sync_endpoint():
    """Trigger RAG synchronization (runs rag_admin.py sync)."""
    import subprocess
    
    try:
        result = subprocess.run(
            ["python", "rag_admin.py", "sync"],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        # Reload RAG system after sync
        global _rag_system
        _rag_system = None
        
        return {
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr if result.returncode != 0 else None
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# --------------------------------------------------------------------
# Document serving (for document viewer in frontend)
# --------------------------------------------------------------------

DOCUMENTS_DIR = Path("documents")


@app.get("/documents/{file_path:path}")
async def serve_document(file_path: str):
    """Serve a document file from the documents directory."""
    full_path = (DOCUMENTS_DIR / file_path).resolve()
    
    # Security: ensure the path is inside the documents directory
    if not str(full_path).startswith(str(DOCUMENTS_DIR.resolve())):
        raise HTTPException(status_code=403, detail="Access denied")
    
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Map extensions to MIME types
    mime_map = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".doc": "application/msword",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xls": "application/vnd.ms-excel",
        ".txt": "text/plain; charset=utf-8",
        ".md": "text/markdown; charset=utf-8",
        ".csv": "text/csv; charset=utf-8",
        ".json": "application/json",
        ".html": "text/html; charset=utf-8",
        ".htm": "text/html; charset=utf-8",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }
    
    ext = full_path.suffix.lower()
    media_type = mime_map.get(ext, "application/octet-stream")
    
    return FileResponse(
        path=str(full_path),
        media_type=media_type,
        filename=full_path.name,
    )


@app.get("/documents/{file_path:path}/text")
async def get_document_text(file_path: str):
    """Get text content of a document (extracted via document_processors)."""
    full_path = (DOCUMENTS_DIR / file_path).resolve()
    
    if not str(full_path).startswith(str(DOCUMENTS_DIR.resolve())):
        raise HTTPException(status_code=403, detail="Access denied")
    
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Document not found")
    
    ext = full_path.suffix.lower()
    
    try:
        if ext == ".txt" or ext == ".md" or ext == ".csv":
            content = full_path.read_text(encoding="utf-8")
        elif ext == ".pdf":
            try:
                from document_processors import extract_text_from_pdf
                content = extract_text_from_pdf(str(full_path))
            except ImportError:
                import subprocess
                result = subprocess.run(
                    ["python", "-c", f"from document_processors import extract_text_from_pdf; print(extract_text_from_pdf('{full_path}'))"],
                    capture_output=True, text=True
                )
                content = result.stdout if result.returncode == 0 else f"Error extracting text: {result.stderr}"
        elif ext == ".docx":
            try:
                from document_processors import extract_text_from_docx
                content = extract_text_from_docx(str(full_path))
            except ImportError:
                content = "[Cannot extract text: document_processors not available]"
        elif ext == ".xlsx" or ext == ".xls":
            try:
                from document_processors import extract_text_from_excel
                content = extract_text_from_excel(str(full_path))
            except ImportError:
                content = "[Cannot extract text: document_processors not available]"
        else:
            content = "[Unsupported format for text extraction]"
        
        return {"content": content, "filename": full_path.name, "extension": ext}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading document: {str(e)}")


# --------------------------------------------------------------------
# Admin Configuration (persistent JSON file)
# --------------------------------------------------------------------

CONFIG_FILE = Path("app_config.json")

DEFAULT_CONFIG = {
    "app_name": "Yotojoro IA",
    "system_prompt": INTERNAL_CONTEXT,
    "default_model": DEFAULT_MODEL,
    "default_vision_model": DEFAULT_VISION_MODEL,
    "router_model": ROUTER_MODEL,
    "route_models": ROUTE_MODELS,
    "rag_enabled": RAG_ENABLED,
    "rag_top_k": RAG_TOP_K,
    "rag_min_relevance": RAG_MIN_RELEVANCE,
    "context_window_size": CONTEXT_WINDOW_SIZE,
    "max_history_messages": MAX_HISTORY_MESSAGES,
    "keep_recent_messages": KEEP_RECENT_MESSAGES,
}


def load_config() -> dict:
    """Load app configuration from JSON file, or return defaults."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            # Merge with defaults so new keys are present
            merged = {**DEFAULT_CONFIG, **saved}
            return merged
        except Exception as e:
            print(f"⚠️  Error loading config: {e}")
    return dict(DEFAULT_CONFIG)


def save_config(config: dict):
    """Persist configuration to JSON file."""
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def apply_config(config: dict):
    """Apply configuration values to runtime globals."""
    global INTERNAL_CONTEXT, DEFAULT_MODEL, DEFAULT_VISION_MODEL
    global ROUTER_MODEL, ROUTE_MODELS
    global RAG_ENABLED, RAG_TOP_K, RAG_MIN_RELEVANCE
    global CONTEXT_WINDOW_SIZE, MAX_HISTORY_MESSAGES, KEEP_RECENT_MESSAGES

    INTERNAL_CONTEXT = config.get("system_prompt", INTERNAL_CONTEXT)
    DEFAULT_MODEL = config.get("default_model", DEFAULT_MODEL)
    DEFAULT_VISION_MODEL = config.get("default_vision_model", DEFAULT_VISION_MODEL)
    ROUTER_MODEL = config.get("router_model", ROUTER_MODEL)
    ROUTE_MODELS = config.get("route_models", ROUTE_MODELS)
    RAG_ENABLED = config.get("rag_enabled", RAG_ENABLED)
    RAG_TOP_K = config.get("rag_top_k", RAG_TOP_K)
    RAG_MIN_RELEVANCE = config.get("rag_min_relevance", RAG_MIN_RELEVANCE)
    CONTEXT_WINDOW_SIZE = config.get("context_window_size", CONTEXT_WINDOW_SIZE)
    MAX_HISTORY_MESSAGES = config.get("max_history_messages", MAX_HISTORY_MESSAGES)
    KEEP_RECENT_MESSAGES = config.get("keep_recent_messages", KEEP_RECENT_MESSAGES)


# Apply saved config on startup
_startup_config = load_config()
apply_config(_startup_config)


# --------------------------------------------------------------------
# Admin API Endpoints
# --------------------------------------------------------------------

@app.get("/admin/config")
async def get_admin_config():
    """Return current application configuration."""
    config = load_config()
    return config


class ConfigUpdate(BaseModel):
    app_name: Optional[str] = None
    system_prompt: Optional[str] = None
    default_model: Optional[str] = None
    default_vision_model: Optional[str] = None
    router_model: Optional[str] = None
    route_models: Optional[Dict[str, str]] = None
    rag_enabled: Optional[bool] = None
    rag_top_k: Optional[int] = None
    rag_min_relevance: Optional[float] = None
    context_window_size: Optional[int] = None
    max_history_messages: Optional[int] = None
    keep_recent_messages: Optional[int] = None


@app.put("/admin/config")
async def update_admin_config(update: ConfigUpdate):
    """Update application configuration (partial update)."""
    config = load_config()
    update_dict = update.model_dump(exclude_none=True)
    config.update(update_dict)
    save_config(config)
    apply_config(config)
    return {"success": True, "config": config}


@app.get("/admin/services/status")
async def admin_services_status():
    """Test all backend services and return their status."""
    results = {}

    # 1. Ollama
    try:
        ollama_ok = await check_ollama_running()
        installed = await get_installed_models() if ollama_ok else []
        results["ollama"] = {
            "status": "ok" if ollama_ok else "error",
            "message": f"{len(installed)} modelos instalados" if ollama_ok else "Ollama no está ejecutándose. Ejecuta: ollama serve",
            "installed_models": installed,
        }
    except Exception as e:
        results["ollama"] = {"status": "error", "message": str(e)}

    # 2. Whisper
    try:
        import whisper
        results["whisper"] = {
            "status": "ok",
            "message": "Módulo whisper disponible",
            "model_loaded": whisper_model is not None,
        }
    except ImportError:
        results["whisper"] = {
            "status": "error",
            "message": "Whisper no instalado. Ejecuta: pip install openai-whisper",
        }

    # 3. RAG
    try:
        if not RAG_AVAILABLE:
            results["rag"] = {
                "status": "error",
                "message": "Módulo RAG no disponible. Ejecuta: pip install chromadb",
            }
        elif not RAG_ENABLED:
            results["rag"] = {
                "status": "warning",
                "message": "RAG está deshabilitado en la configuración",
            }
        else:
            rag = get_rag_system()
            if rag:
                stats = rag.get_stats()
                results["rag"] = {
                    "status": "ok",
                    "message": f"{stats.get('total_documents', 0)} chunks indexados",
                    "chunks": stats.get("total_documents", 0),
                }
            else:
                results["rag"] = {
                    "status": "error",
                    "message": "RAG no pudo inicializarse",
                }
    except Exception as e:
        results["rag"] = {"status": "error", "message": str(e)}

    # 4. TTS
    try:
        system = platform.system()
        if system == "Darwin":
            result = subprocess.run(["which", "say"], capture_output=True, timeout=5)
            results["tts"] = {
                "status": "ok" if result.returncode == 0 else "error",
                "message": "TTS nativo de macOS disponible" if result.returncode == 0 else "Comando 'say' no encontrado",
                "engine": "macos_native",
            }
        elif system == "Windows":
            results["tts"] = {
                "status": "ok",
                "message": "TTS SAPI de Windows disponible",
                "engine": "windows_sapi",
            }
        else:
            results["tts"] = {
                "status": "warning",
                "message": "TTS nativo no disponible, usando TTS del navegador",
                "engine": "browser",
            }
    except Exception as e:
        results["tts"] = {"status": "error", "message": str(e)}

    # 5. Documents directory
    try:
        docs_path = Path("documents")
        if docs_path.exists():
            file_count = sum(1 for _ in docs_path.rglob("*") if _.is_file())
            results["documents"] = {
                "status": "ok",
                "message": f"{file_count} archivos en directorio de documentos",
                "path": str(docs_path.resolve()),
            }
        else:
            results["documents"] = {
                "status": "warning",
                "message": "Directorio 'documents/' no existe",
            }
    except Exception as e:
        results["documents"] = {"status": "error", "message": str(e)}

    return results


@app.post("/admin/models/pull")
async def admin_pull_model(model: str):
    """Pull (download) an Ollama model. Returns a streaming response with progress."""
    try:
        async def stream_pull():
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_BASE_URL}/api/pull",
                    json={"name": model},
                    timeout=None,
                ) as response:
                    if response.status_code != 200:
                        yield json.dumps({"error": f"Ollama error: {response.status_code}"}) + "\n"
                        return
                    async for line in response.aiter_lines():
                        if line.strip():
                            yield line + "\n"

        return StreamingResponse(stream_pull(), media_type="application/x-ndjson")
    except Exception as e:
        raise HTTPException(500, f"Error pulling model: {str(e)}")


@app.delete("/admin/models/{model_name:path}")
async def admin_delete_model(model_name: str):
    """Delete a specific Ollama model."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.request(
                "DELETE",
                f"{OLLAMA_BASE_URL}/api/delete",
                json={"name": model_name},
                timeout=30.0,
            )
            if response.status_code == 200:
                return {"success": True, "message": f"Modelo '{model_name}' eliminado"}
            else:
                raise HTTPException(response.status_code, response.text)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/admin/test/ollama")
async def admin_test_ollama(model: str = "gemma3:1b"):
    """Test Ollama with a simple prompt."""
    try:
        response = await ollama_chat(
            model,
            [{"role": "user", "content": "Responde solo con: Hola, funciono correctamente."}],
            temperature=0,
            max_tokens=30,
        )
        return {"success": True, "model": model, "response": response}
    except Exception as e:
        return {"success": False, "model": model, "error": str(e)}


@app.post("/admin/test/whisper")
async def admin_test_whisper():
    """Test Whisper model loading."""
    try:
        model = get_whisper_model()
        return {
            "success": True,
            "message": "Whisper cargado correctamente",
            "model_type": "base",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# --------------------------------------------------------------------
# Health check
# --------------------------------------------------------------------

@app.get("/health")
async def health():
    ollama_ok = await check_ollama_running()
    return {
        "status": "ok" if ollama_ok else "degraded",
        "ollama": "running" if ollama_ok else "not running",
        "message": "Run 'ollama serve' to start Ollama" if not ollama_ok else None
    }


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("Starting Ollama + Whisper Server")
    print("=" * 60)
    print(f"Ollama URL: {OLLAMA_BASE_URL}")
    print(f"Default model: {DEFAULT_MODEL}")
    print(f"Vision model: {DEFAULT_VISION_MODEL}")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)

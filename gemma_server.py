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
import tempfile
import ssl
import subprocess
import platform
import httpx
import numpy as np
import scipy.io.wavfile as wavfile
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict

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
    # Phi
    "phi3:mini": {
        "name": "Phi-3 Mini",
        "description": "Microsoft Phi-3, compacto",
        "size": "~2.3GB",
        "vision": False,
    },
}

DEFAULT_MODEL = "gemma2:2b"
DEFAULT_VISION_MODEL = "llava:7b"

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
    max_tokens: int = 256,
) -> str:
    """
    Send a chat request to Ollama.
    
    Args:
        model: Model name (e.g., "gemma2:2b")
        messages: List of {"role": "user/assistant/system", "content": "..."}
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
    
    Returns:
        Generated text response
    """
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


# --------------------------------------------------------------------
# FastAPI + CORS
# --------------------------------------------------------------------

app = FastAPI(title="Ollama + Whisper Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# =============================================================================
# CONTEXTO INTERNO DEL BACKEND (editar aquí para cambiar el comportamiento base)
# =============================================================================
INTERNAL_CONTEXT = """
Eres un asistente experto que trabaja para una empresa de tecnología.
Siempre respondes de forma profesional, precisa y concisa.
Cuando no sepas algo, lo admites honestamente.
""".strip()
# =============================================================================


def build_system_prompt(user_context: str = None) -> str:
    """Build system prompt combining internal context with optional user context."""
    base_prompt = "Eres un asistente amigable y útil. Responde en español de forma concisa."
    
    combined_context = INTERNAL_CONTEXT
    if user_context:
        combined_context = f"{INTERNAL_CONTEXT}\n\n{user_context}"
    
    if combined_context:
        return f"{base_prompt}\n\nContexto adicional: {combined_context}"
    return base_prompt


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
    
    models = []
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
        "default": DEFAULT_MODEL,
        "current": None,
        "installed": installed,
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
        model = payload.model or DEFAULT_MODEL
        print(f"\n--- STEP 2: Ollama Response (model: {model}) ---")
        
        if payload.context:
            ctx_preview = payload.context[:100] + '...' if len(payload.context) > 100 else payload.context
            print(f"With context: '{ctx_preview}'")
        
        if payload.history:
            print(f"\n--- Chat History ({len(payload.history)} messages) ---")
            for i, msg in enumerate(payload.history):
                preview = msg['text'][:80] + '...' if len(msg['text']) > 80 else msg['text']
                print(f"  [{i+1}] {msg['role']}: {preview}")
        
        system_prompt = build_system_prompt(payload.context)
        messages = build_ollama_messages(system_prompt, user_text, payload.history)
        
        response = await ollama_chat(model, messages)
        print(f"Ollama response: '{response}'")
        print("=" * 60 + "\n")
        
        return {"text": response, "transcription": user_text, "model": model}
    
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
        
        model = payload.model or DEFAULT_MODEL
        print(f"\n--- Ollama Response (model: {model}) ---")
        
        if payload.context:
            ctx_preview = payload.context[:100] + '...' if len(payload.context) > 100 else payload.context
            print(f"With context: '{ctx_preview}'")
        
        if payload.history:
            print(f"\n--- Chat History ({len(payload.history)} messages) ---")
            for i, msg in enumerate(payload.history):
                preview = msg['text'][:80] + '...' if len(msg['text']) > 80 else msg['text']
                print(f"  [{i+1}] {msg['role']}: {preview}")
        
        system_prompt = build_system_prompt(payload.context)
        messages = build_ollama_messages(system_prompt, user_text, payload.history)
        
        response = await ollama_chat(model, messages)
        print(f"Ollama response: '{response}'")
        print("=" * 60 + "\n")
        
        return {"text": response, "model": model}
    
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

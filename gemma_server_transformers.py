"""
FastAPI wrapper for Gemma-3n model with Whisper for audio transcription.

Architecture:
1. Audio arrives from frontend
2. Whisper transcribes audio to text
3. Gemma3n responds to the transcribed text

• POST /ask         – audio→whisper→text→gemma→response
• POST /ask_image   – image+prompt→gemma→response

CORS is open for http://localhost:5173 so the React front-end can call us.
"""

import base64
import os
import shutil
import tempfile
import torch
import ssl
import subprocess
import platform
import numpy as np
import scipy.io.wavfile as wavfile
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

# Fix SSL issues for model downloads
ssl._create_default_https_context = ssl._create_unverified_context

# --------------------------------------------------------------------
# Available Gemma models
# --------------------------------------------------------------------

AVAILABLE_MODELS = {
    # Gemma 3n family (multimodal: text, image, audio)
    "gemma-3n-e2b-it": {
        "id": "google/gemma-3n-e2b-it",
        "name": "Gemma 3n E2B",
        "description": "2B efectivos, multimodal, ligero",
        "size": "~4GB",
        "family": "gemma3n"
    },
    "gemma-3n-e4b-it": {
        "id": "google/gemma-3n-e4b-it",
        "name": "Gemma 3n E4B",
        "description": "4B efectivos, multimodal, potente",
        "size": "~8GB",
        "family": "gemma3n"
    },
    # Gemma 3 family (text + image)
    "gemma-3-1b-it": {
        "id": "google/gemma-3-1b-it",
        "name": "Gemma 3 1B",
        "description": "Ultra ligero, respuestas básicas",
        "size": "~2GB",
        "family": "gemma3"
    },
    "gemma-3-4b-it": {
        "id": "google/gemma-3-4b-it",
        "name": "Gemma 3 4B",
        "description": "Buen balance calidad/tamaño",
        "size": "~8GB",
        "family": "gemma3"
    },
    "gemma-3-12b-it": {
        "id": "google/gemma-3-12b-it",
        "name": "Gemma 3 12B",
        "description": "Alta calidad, requiere más RAM",
        "size": "~24GB",
        "family": "gemma3"
    },
    "gemma-3-27b-it": {
        "id": "google/gemma-3-27b-it",
        "name": "Gemma 3 27B",
        "description": "Máxima calidad, requiere GPU",
        "size": "~54GB",
        "family": "gemma3"
    },
    # Gemma 2 family (text only)
    "gemma-2-2b-it": {
        "id": "google/gemma-2-2b-it",
        "name": "Gemma 2 2B",
        "description": "Solo texto, muy eficiente",
        "size": "~4GB",
        "family": "gemma2"
    },
    "gemma-2-9b-it": {
        "id": "google/gemma-2-9b-it",
        "name": "Gemma 2 9B",
        "description": "Solo texto, buena calidad",
        "size": "~18GB",
        "family": "gemma2"
    },
    "gemma-2-27b-it": {
        "id": "google/gemma-2-27b-it",
        "name": "Gemma 2 27B",
        "description": "Solo texto, alta calidad",
        "size": "~54GB",
        "family": "gemma2"
    },
}

DEFAULT_MODEL = "gemma-3-1b-it"

# --------------------------------------------------------------------
# Global models (loaded lazily)
# --------------------------------------------------------------------

whisper_model = None
gemma_models = {}  # Cache for loaded Gemma models
current_model_key = None


def get_whisper_model():
    """Load Whisper model for speech-to-text."""
    global whisper_model
    if whisper_model is None:
        import whisper
        print("Loading Whisper model (base)...")
        whisper_model = whisper.load_model("base")
        print("Whisper model loaded.")
    return whisper_model


def get_gemma_model_and_processor(model_key: str = None):
    """Load Gemma model for text generation. Supports Gemma 2, 3, and 3n families."""
    global gemma_models, current_model_key
    
    if model_key is None:
        model_key = DEFAULT_MODEL
    
    if model_key not in AVAILABLE_MODELS:
        raise ValueError(f"Model '{model_key}' not found. Available: {list(AVAILABLE_MODELS.keys())}")
    
    # Check if model is already loaded
    if model_key in gemma_models:
        current_model_key = model_key
        return gemma_models[model_key]
    
    model_info = AVAILABLE_MODELS[model_key]
    model_id = model_info["id"]
    family = model_info["family"]
    
    print(f"Loading {family} model: {model_id}...")
    
    # Load appropriate model class based on family
    if family == "gemma3n":
        from transformers import AutoProcessor, Gemma3nForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_id)
        model = Gemma3nForConditionalGeneration.from_pretrained(
            model_id,
            device_map="cpu",
            torch_dtype=torch.float32,
        )
    elif family == "gemma3":
        # Gemma 3 text-only models use AutoTokenizer, not AutoProcessor
        from transformers import AutoTokenizer, AutoModelForCausalLM
        processor = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cpu",
            torch_dtype=torch.float32,
        )
    else:  # gemma2
        from transformers import AutoTokenizer, AutoModelForCausalLM
        processor = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cpu",
            torch_dtype=torch.float32,
        )
    
    model.eval()
    print(f"Model loaded: {model_key}")
    
    # Cache the model
    gemma_models[model_key] = (model, processor)
    current_model_key = model_key
    
    return model, processor


# --------------------------------------------------------------------
# FastAPI + CORS
# --------------------------------------------------------------------

app = FastAPI(title="Gemma3n + Whisper Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------
# /models  — list available models
# --------------------------------------------------------------------

@app.get("/models")
async def list_models():
    """Return list of available Gemma models."""
    models = []
    for key, info in AVAILABLE_MODELS.items():
        models.append({
            "key": key,
            "name": info["name"],
            "description": info["description"],
            "size": info["size"],
            "family": info["family"],
            "loaded": key in gemma_models,
            "current": key == current_model_key,
        })
    return {"models": models, "default": DEFAULT_MODEL, "current": current_model_key}


@app.delete("/models/cache")
async def clear_model_cache():
    """Clear downloaded models from HuggingFace cache."""
    global gemma_models, current_model_key
    
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    deleted = []
    errors = []
    freed_bytes = 0
    
    for model_key, info in AVAILABLE_MODELS.items():
        # Convert model id to cache folder name: google/gemma-3-1b-it -> models--google--gemma-3-1b-it
        model_id = info["id"]
        cache_name = f"models--{model_id.replace('/', '--')}"
        cache_path = cache_dir / cache_name
        
        if cache_path.exists():
            try:
                # Calculate size before deleting
                size = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file())
                freed_bytes += size
                shutil.rmtree(cache_path)
                deleted.append(model_key)
                print(f"Deleted cache for {model_key} ({size / 1e9:.2f} GB)")
            except Exception as e:
                errors.append({"model": model_key, "error": str(e)})
    
    # Clear loaded models from memory
    gemma_models.clear()
    current_model_key = None
    
    return {
        "deleted": deleted,
        "errors": errors,
        "freed_gb": round(freed_bytes / 1e9, 2),
        "message": f"Deleted {len(deleted)} models, freed {freed_bytes / 1e9:.2f} GB"
    }


# --------------------------------------------------------------------
# /ask  — audio blob (base-64)  →  whisper → gemma → text
# --------------------------------------------------------------------

class AudioPayload(BaseModel):
    data: str  # base-64 WAV data
    model: Optional[str] = None  # optional model key
    context: Optional[str] = None  # optional context for the chat
    history: Optional[List[Dict[str, str]]] = None  # conversation history [{"role": "user/model", "text": "..."}]


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
    
    # Combine internal context with user context from frontend
    combined_context = INTERNAL_CONTEXT
    if user_context:
        combined_context = f"{INTERNAL_CONTEXT}\n\n{user_context}"
    
    if combined_context:
        return f"{base_prompt}\n\nContexto adicional: {combined_context}"
    return base_prompt


def build_messages_for_gemma(model_family: str, system_prompt: str, user_text: str, history: List[Dict[str, str]] = None):
    """
    Build messages array for Gemma model with conversation history.
    
    Args:
        model_family: 'gemma2', 'gemma3', or 'gemma3n'
        system_prompt: System instructions
        user_text: Current user message
        history: Previous messages [{"role": "user" or "model", "text": "..."}]
    
    Returns:
        messages array formatted for the model family
    """
    if model_family == "gemma2":
        # Gemma 2 uses simple chat template
        # Build conversation string with history
        conversation_parts = [system_prompt]
        
        if history:
            for msg in history:
                role_label = "Usuario" if msg["role"] == "user" else "Asistente"
                conversation_parts.append(f"{role_label}: {msg['text']}")
        
        conversation_parts.append(f"Usuario: {user_text}")
        full_content = "\n\n".join(conversation_parts)
        
        return [{"role": "user", "content": full_content}]
    else:
        # Gemma 3 and 3n use processor with structured content
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
        ]
        
        # Add history messages
        if history:
            for msg in history:
                role = msg["role"] if msg["role"] in ["user", "assistant"] else ("user" if msg["role"] == "user" else "assistant")
                # Gemma uses 'assistant' not 'model'
                if msg["role"] == "model":
                    role = "assistant"
                messages.append({
                    "role": role,
                    "content": [{"type": "text", "text": msg["text"]}],
                })
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": user_text}],
        })
        
        return messages


@app.post("/ask")
async def ask_audio(payload: AudioPayload):
    """
    Process audio: Whisper transcribes it, then Gemma responds.
    """
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
        print(f"Saved to {wav_path}")
        
        # Get audio info for logging
        sample_rate, audio_data = wavfile.read(wav_path)
        duration = len(audio_data) / sample_rate
        print(f"Audio: {sample_rate}Hz, {duration:.2f}s, shape={audio_data.shape}")
        
        # Normalize audio volume for better Whisper transcription
        if len(audio_data) > 0:
            audio_float = audio_data.astype(np.float32)
            max_val = np.max(np.abs(audio_float))
            if max_val > 0:
                # Normalize to 90% of max range to avoid clipping
                audio_normalized = (audio_float / max_val) * 0.9 * 32767
                audio_normalized = audio_normalized.astype(np.int16)
                wavfile.write(wav_path, sample_rate, audio_normalized)
                print(f"Audio normalized (peak: {max_val:.0f} -> {np.max(np.abs(audio_normalized)):.0f})")
        
        # Step 1: Transcribe with Whisper (optimized for Spanish)
        print("\n--- STEP 1: Whisper Transcription ---")
        whisper_m = get_whisper_model()
        
        # Whisper optimization for Spanish:
        # - language="es": Force Spanish detection (skips language detection overhead)
        # - task="transcribe": Explicit transcription mode
        # - initial_prompt: Primes the model with Spanish context and common phrases
        # - temperature=0: More deterministic, better for clear speech
        # - condition_on_previous_text=False: Prevents hallucination loops
        initial_prompt = "Esta es una conversación en español. El usuario hace preguntas o da instrucciones."
        
        whisper_result = whisper_m.transcribe(
            wav_path,
            language="es",
            task="transcribe",
            initial_prompt=initial_prompt,
            temperature=0,
            condition_on_previous_text=False,
            fp16=False  # Better accuracy on CPU
        )
        user_text = whisper_result["text"].strip()
        print(f"Transcribed: '{user_text}'")
        
        if not user_text:
            return {"text": "No pude entender lo que dijiste. ¿Puedes repetirlo?"}
        
        # Step 2: Generate response with Gemma
        model_key = payload.model or DEFAULT_MODEL
        print(f"\n--- STEP 2: Gemma Response (model: {model_key}) ---")
        model, processor = get_gemma_model_and_processor(model_key)
        model_family = AVAILABLE_MODELS[model_key]["family"]
        
        print(f"Sending to Gemma: '{user_text}'")
        if payload.context:
            print(f"With context: '{payload.context[:100]}...'" if len(payload.context) > 100 else f"With context: '{payload.context}'")
        if payload.history:
            print(f"\n--- Chat History ({len(payload.history)} messages) ---")
            for i, msg in enumerate(payload.history):
                role = msg.get('role', 'unknown')
                text = msg.get('text', '')
                preview = text[:80] + '...' if len(text) > 80 else text
                print(f"  [{i+1}] {role}: {preview}")
            print("--- End History ---\n")
        
        system_prompt = build_system_prompt(payload.context)
        messages = build_messages_for_gemma(model_family, system_prompt, user_text, payload.history)
        
        # Process messages for model
        if model_family == "gemma2":
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(model.device)
            input_ids = inputs
        else:
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
            input_ids = inputs["input_ids"]
        
        print("Generating response...")
        with torch.inference_mode():
            if model_family == "gemma2":
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
        
        # Decode only the generated part
        input_len = input_ids.shape[-1]
        response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
        response = response.strip()
        
        print(f"Gemma response: '{response}'")
        print("=" * 60 + "\n")
        
        return {"text": response, "transcription": user_text, "model": current_model_key}
    
    except Exception as exc:
        import traceback
        error_msg = f"Error: {str(exc)}\n{traceback.format_exc()}"
        print(error_msg)
        raise HTTPException(500, error_msg) from exc
    finally:
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)


# --------------------------------------------------------------------
# /ask_text  — text prompt → gemma → text (bypasses Whisper)
# --------------------------------------------------------------------

class TextPayload(BaseModel):
    text: str  # user's text prompt
    model: Optional[str] = None  # optional model key
    context: Optional[str] = None  # optional context for the chat
    history: Optional[List[Dict[str, str]]] = None  # conversation history [{"role": "user/model", "text": "..."}]


@app.post("/ask_text")
async def ask_text(payload: TextPayload):
    """
    Process text directly: sends to Gemma without Whisper.
    """
    try:
        print("\n" + "=" * 60)
        print("Received text request")
        
        user_text = payload.text.strip()
        print(f"User text: '{user_text}'")
        
        if not user_text:
            return {"text": "No recibí ningún texto. ¿Puedes intentarlo de nuevo?"}
        
        # Generate response with Gemma
        model_key = payload.model or DEFAULT_MODEL
        print(f"\n--- Gemma Response (model: {model_key}) ---")
        model, processor = get_gemma_model_and_processor(model_key)
        model_family = AVAILABLE_MODELS[model_key]["family"]
        
        print(f"Sending to Gemma: '{user_text}'")
        if payload.context:
            print(f"With context: '{payload.context[:100]}...'" if len(payload.context) > 100 else f"With context: '{payload.context}'")
        if payload.history:
            print(f"\n--- Chat History ({len(payload.history)} messages) ---")
            for i, msg in enumerate(payload.history):
                role = msg.get('role', 'unknown')
                text = msg.get('text', '')
                preview = text[:80] + '...' if len(text) > 80 else text
                print(f"  [{i+1}] {role}: {preview}")
            print("--- End History ---\n")
        
        system_prompt = build_system_prompt(payload.context)
        
        # Build messages with history support
        messages = build_messages_for_gemma(model_family, system_prompt, user_text, payload.history)
        
        # Different handling for Gemma 2 (tokenizer) vs Gemma 3/3n (processor)
        if model_family == "gemma2":
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(model.device)
            input_ids = inputs
        else:
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
            input_ids = inputs["input_ids"]
        
        print("Generating response...")
        with torch.inference_mode():
            if model_family == "gemma2":
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
        
        # Decode only the generated part
        input_len = input_ids.shape[-1]
        response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
        response = response.strip()
        
        print(f"Gemma response: '{response}'")
        print("=" * 60 + "\n")
        
        return {"text": response, "model": current_model_key}
    
    except Exception as exc:
        import traceback
        error_msg = f"Error: {str(exc)}\n{traceback.format_exc()}"
        print(error_msg)
        raise HTTPException(500, error_msg) from exc


# --------------------------------------------------------------------
# /ask_image  — multipart(form-data)  →  text
# --------------------------------------------------------------------

@app.post("/ask_image")
async def ask_image(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    context: Optional[str] = Form(None),
):
    """Process image with text prompt using Gemma3n vision."""
    img_path = None
    try:
        print(f"\nReceived image request with prompt: {prompt}")
        if context:
            print(f"With context: '{context[:100]}...'" if len(context) > 100 else f"With context: '{context}'")
        
        suffix = os.path.splitext(image.filename)[1] or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            img_path = tmp.name
            tmp.write(await image.read())
        print(f"Saved image to {img_path}")
        
        model, processor = get_gemma_model_and_processor()
        
        system_prompt = build_system_prompt(context)
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        
        print("Generating response...")
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
            )
        
        input_len = inputs["input_ids"].shape[-1]
        response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
        response = response.strip()
        
        print(f"Response: {response}")
        return {"text": response}
    
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
    voice: Optional[str] = None  # Voice name (e.g., "Paulina", "Monica")
    rate: Optional[int] = 180    # Words per minute

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """
    Convert text to speech using native TTS.
    Returns audio as base64-encoded WAV.
    """
    try:
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="No text provided")
        
        system = platform.system()
        
        if system == "Darwin":  # macOS
            # Use macOS 'say' command with AIFF output, then convert to WAV
            with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as aiff_file:
                aiff_path = aiff_file.name
            
            try:
                # Spanish voices on macOS: Paulina (Mexico), Monica (Spain), Jorge (Spain)
                voice = request.voice or "Paulina"
                rate = request.rate or 180
                
                # Generate speech with 'say' command
                cmd = ["say", "-v", voice, "-r", str(rate), "-o", aiff_path, text]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode != 0:
                    print(f"say command error: {result.stderr}")
                    raise HTTPException(status_code=500, detail="TTS generation failed")
                
                # Convert AIFF to WAV using afconvert
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
                    wav_path = wav_file.name
                
                convert_cmd = ["afconvert", "-f", "WAVE", "-d", "LEI16@22050", aiff_path, wav_path]
                subprocess.run(convert_cmd, capture_output=True, timeout=10)
                
                # Read and encode the WAV file
                with open(wav_path, "rb") as f:
                    audio_data = f.read()
                
                audio_b64 = base64.b64encode(audio_data).decode("utf-8")
                
                # Clean up temp files
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
                # Ensure cleanup
                if os.path.exists(aiff_path):
                    os.remove(aiff_path)
        
        elif system == "Windows":
            # Use PowerShell with SAPI.SpVoice for Windows TTS
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
                wav_path = wav_file.name
            
            try:
                # PowerShell script to generate speech and save as WAV
                # Uses .NET SpeechSynthesizer which outputs WAV directly
                ps_script = f'''
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Rate = 1
# Try to find a Spanish voice
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
                    print(f"PowerShell TTS error: {result.stderr}")
                    raise HTTPException(status_code=500, detail="TTS generation failed")
                
                # Read and encode the WAV file
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
            # Linux or other - return error to use browser TTS
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
    """
    Get list of available Spanish TTS voices.
    """
    system = platform.system()
    
    if system == "Darwin":
        try:
            # Get voices from macOS
            result = subprocess.run(
                ["say", "-v", "?"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            # Parse Spanish voices
            spanish_voices = []
            for line in result.stdout.split("\n"):
                if "es_" in line or "es-" in line:
                    # Format: "VoiceName    language    # comment"
                    parts = line.split()
                    if parts:
                        voice_name = parts[0]
                        spanish_voices.append({
                            "name": voice_name,
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
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    print("Starting server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)

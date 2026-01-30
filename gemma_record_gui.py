import threading
import tempfile
import os
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, font as tkfont

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

import torch
from transformers import AutoProcessor, Gemma3nForConditionalGeneration

# ------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------
MODEL_ID = "google/gemma-3n-e4b-it"
INPUT_DEVICE_INDEX = 7          # PortAudio index mapping to hw:1,6
CHANNELS            = 2
SAMPLE_RATE         = 48_000
DURATION_SEC        = 4         # seconds to record per click
SYSTEM_PROMPT       = (
    "You are a friendly assistant. Respond in a natural, conversational tone. "
    "Avoid numbered or bulleted lists; instead write short sentences or paragraphs."
)

# ------------------------------------------------------
# AUDIO UTILS
# ------------------------------------------------------

def record_wav(path: str):
    """Capture audio from the chosen device and save as 16‑bit WAV."""
    sd.default.device = (INPUT_DEVICE_INDEX, None)
    buf = sd.rec(int(DURATION_SEC * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                 channels=CHANNELS, dtype="int16")
    sd.wait()
    write(path, SAMPLE_RATE, buf)

# ------------------------------------------------------
# MODEL SINGLETON (load once, reuse)
# ------------------------------------------------------

_model = None
_processor = None
_model_lock = threading.Lock()


def _materialize_meta_tensor_attrs(module, device="cpu"):
    """
    Find and materialize meta tensors stored as plain attributes (not parameters/buffers).
    
    Some modules create tensors in __init__ with torch.tensor() that don't get registered
    as parameters or buffers. When model loading happens in a meta device context,
    these become meta tensors that won't be moved by .to(device).
    """
    # Handle specific known cases where we know the config value
    if hasattr(module, 'post_layer_scale'):
        attr = module.post_layer_scale
        if torch.is_tensor(attr) and attr.device.type == "meta":
            # Get value from config
            if hasattr(module, 'config') and hasattr(module.config, 'conf_residual_weight'):
                value = module.config.conf_residual_weight
                new_tensor = torch.tensor(value, dtype=attr.dtype, device=device)
                module.post_layer_scale = new_tensor
                print(f"  Materialized post_layer_scale = {value} on {device}")
    
    # Handle any other meta tensors as scalar 0 (fallback)
    for name in list(dir(module)):
        if name.startswith('_') or name == 'post_layer_scale':
            continue
        try:
            attr = getattr(module, name)
            if torch.is_tensor(attr) and attr.device.type == "meta":
                # For scalar tensors, assume 0 as default
                if attr.numel() == 1:
                    new_tensor = torch.zeros(attr.shape, dtype=attr.dtype, device=device)
                    setattr(module, name, new_tensor)
                    print(f"  Materialized meta attr: {name} -> {device} (default 0)")
        except Exception:
            pass
    
    # Recurse into child modules
    for child_name, child in module._modules.items():
        if child is not None:
            _materialize_meta_tensor_attrs(child, device)


def get_model_and_processor():
    global _model, _processor
    with _model_lock:
        if _model is None or _processor is None:
            print("Initializing Processor...")
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

            print("Loading model on CPU with float32 (required for audio processing)...")
            print("Note: This is slower but necessary for correct audio understanding")

            _processor = AutoProcessor.from_pretrained(MODEL_ID)

            # Load model on CPU with float32 - the ONLY configuration that works for audio
            with torch.no_grad():
                _model = Gemma3nForConditionalGeneration.from_pretrained(
                    MODEL_ID,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=False,
                    device_map='cpu',
                )

            # Materialize meta tensor attributes in audio components
            if hasattr(_model, "model") and hasattr(_model.model, "audio_tower"):
                print("Materializing meta tensor attributes in audio tower...")
                _materialize_meta_tensor_attrs(_model.model.audio_tower, device="cpu")
                
            if hasattr(_model, "model") and hasattr(_model.model, "embed_audio"):
                print("Materializing meta tensor attributes in embed_audio...")
                _materialize_meta_tensor_attrs(_model.model.embed_audio, device="cpu")
            
            _model.eval()

            print(f"Model successfully loaded on {_model.device}")
        return _model, _processor

# ------------------------------------------------------
# SIMPLE SANITISER TO AVOID UNICODE GLYPHS MISSING IN SOME FONTS
# ------------------------------------------------------

_REPLACE_MAP = {
    "•": "-",
    "▪": "-",
    "●": "-",
    "◦": "-",
    "—": "-",
    "–": "-",
}

import re

def sanitize(text: str) -> str:
    """Clean up model output, removing garbage and repetitive patterns."""
    # Replace unicode bullets
    for bad, good in _REPLACE_MAP.items():
        text = text.replace(bad, good)
    
    # Remove garbage patterns (base64-like, repeated special chars, etc.)
    # Stop at first occurrence of garbage patterns
    garbage_patterns = [
        r'm-code\}',  # Common garbage start
        r'm-hmm\.',   # Repetitive filler (keep first occurrence)
        r'[💚🌸✨].*$',  # Emoji followed by garbage
        r'::[0-9:]+$',  # Repeated ::0::0
        r'(oh\s+){5,}',  # Repeated "oh oh oh..."
        r'([a-zA-Z]{1,3}\s*){20,}',  # Repeated short words
        r'\{💚.*$',  # Emoji in braces garbage
    ]
    
    for pattern in garbage_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Keep text before the garbage
            text = text[:match.start()].strip()
    
    # Remove trailing incomplete sentences (ending with unfinished words)
    # Keep complete sentences only
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if sentences:
        # Filter out very short or garbage sentences
        clean_sentences = []
        for s in sentences:
            s = s.strip()
            # Skip if too short, mostly symbols, or repetitive
            if len(s) < 3:
                continue
            if len(re.sub(r'[^a-zA-ZáéíóúüñÁÉÍÓÚÜÑ\s]', '', s)) < len(s) * 0.3:
                continue  # Too many special chars
            clean_sentences.append(s)
        text = ' '.join(clean_sentences)
    
    return text.strip()

# ------------------------------------------------------
# GUI
# ------------------------------------------------------


class GemmaGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Gemma‑3n Conversational Audio Demo")

        # 4‑K scaling: triple default font sizes
        for f_name in (
            "TkDefaultFont", "TkTextFont", "TkHeadingFont", "TkFixedFont",
            "TkMenuFont", "TkCaptionFont", "TkSmallCaptionFont",
            "TkIconFont", "TkTooltipFont",
        ):
            try:
                fnt = tkfont.nametofont(f_name)
                fnt.configure(size=int(fnt.cget("size") * 3))
            except tk.TclError:
                pass

        # Widgets
        self.record_btn = ttk.Button(self, text="🎙️  Record", command=self.start_record)
        self.record_btn.grid(row=0, column=0, padx=20, pady=20, sticky="ew")

        self.output = scrolledtext.ScrolledText(
            self, width=60, height=20, wrap="word", font=("Helvetica", 12)
        )
        self.output.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # Disable recording until model is loaded
        self.record_btn.config(state="disabled")
        self._append_output("Loading model … please wait.\n")
        threading.Thread(target=self._load_model_thread, daemon=True).start()

    # --------------------------------------------------
    # THREAD‑SAFE OUTPUT
    # --------------------------------------------------

    def _append_output(self, txt: str):
        """Safely append text from any thread."""
        def inner():
            self.output.insert(tk.END, txt)
            self.output.see(tk.END)
        self.output.after(0, inner)

    # --------------------------------------------------
    # MODEL LOADER THREAD
    # --------------------------------------------------

    def _load_model_thread(self):
        try:
            get_model_and_processor()
            self._append_output("Model loaded. You can click Record.\n")
        except Exception as e:
            self._append_output(f"Error loading model: {e}\n")
            messagebox.showerror("Model Load Error", str(e))
        finally:
            # Enable the record button whether load succeeded or failed
            self.record_btn.after(0, lambda: self.record_btn.config(state="normal"))

    # --------------------------------------------------
    # RECORD → RUN MODEL in background thread
    # --------------------------------------------------

    def start_record(self):
        self.record_btn.config(state="disabled")
        self.output.delete("1.0", tk.END)
        threading.Thread(target=self._record_and_generate, daemon=True).start()

    def _record_and_generate(self):
        try:
            self._append_output(f"Recording… speak now ({DURATION_SEC} s)\n")
            with tempfile.TemporaryDirectory() as td:
                wav_path = os.path.join(td, "input.wav")
                record_wav(wav_path)

                self._append_output("Processing with Gemma … this may take a moment.\n")
                model, processor = get_model_and_processor()

                messages = [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": SYSTEM_PROMPT},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Here is my audio message:"},
                            {"type": "audio", "audio": wav_path},
                        ],
                    },
                ]

                inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                """ print("Model device:", next(model.parameters()).device)
                for k, v in inputs.items():
                    print(k, v.device, v.dtype) """
                
                with torch.inference_mode():
                    generation = model.generate(
                        **inputs, max_new_tokens=256, disable_compile=True
                    )

                input_len = inputs["input_ids"].shape[-1]
                response = processor.decode(
                    generation[0][input_len:], skip_special_tokens=True
                )
                response = sanitize(response)

                self._append_output("\n===== Gemma response =====\n" + response + "\n")

        except Exception as e:
            self._append_output(f"Error: {e}\n")
            messagebox.showerror("Error", str(e))
        finally:
            self.record_btn.after(0, lambda: self.record_btn.config(state="normal"))

# ------------------------------------------------------
# MAIN
# ------------------------------------------------------


if __name__ == "__main__":
    app = GemmaGUI()
    # Extra scaling for high‑DPI—each logical pixel gets 3× physical pixels
    app.tk.call("tk", "scaling", 3.0)
    app.mainloop()

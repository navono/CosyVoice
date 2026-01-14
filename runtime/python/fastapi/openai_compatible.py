# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys

# IMPORTANT: Set CUDA_VISIBLE_DEVICES BEFORE importing torch
# PyTorch initializes CUDA on import, so this must be done first
device_arg = None
for i, arg in enumerate(sys.argv):
    if arg == '--device' and i + 1 < len(sys.argv):
        device_arg = sys.argv[i + 1]
        break
if device_arg is None:
    device_arg = os.getenv('DEVICE', '0')
os.environ['CUDA_VISIBLE_DEVICES'] = device_arg

import argparse
import logging
import io
import base64
from typing import Optional, Literal
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import numpy as np
import torch
import torchaudio
import requests

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2, CosyVoice3, AutoModel

logging.getLogger('matplotlib').setLevel(logging.WARNING)

# OpenAI API compatible models
app = FastAPI(
    title="CosyVoice OpenAI Compatible API",
    description="OpenAI-compatible Audio API for CosyVoice TTS",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global model instance
cosyvoice: Optional[CosyVoice | CosyVoice2 | CosyVoice3] = None


# Voice mapping - OpenAI voices to CosyVoice speakers
VOICE_MAPPING = {
    'alloy': 'female_source',  # Default female voice
    'echo': 'male_source',     # Default male voice
    'fable': 'zh_female_emo_1',
    'onyx': 'zh_male_emo_1',
    'nova': 'zh_female_source',
    'shimmer': 'zh_male_source',
}


def load_prompt_audio(prompt_wav: str) -> np.ndarray:
    """Load prompt audio from base64 string, URL, or file path"""
    if prompt_wav is None:
        raise ValueError("prompt_wav is required")

    # Check if it's a URL
    if urlparse(prompt_wav).scheme in ('http', 'https'):
        response = requests.get(prompt_wav)
        response.raise_for_status()
        audio_data = response.content
        waveform, sample_rate = torchaudio.load(io.BytesIO(audio_data))

    # Check if it's base64 encoded
    elif prompt_wav.startswith('data:audio'):
        # Remove data URL prefix (e.g., "data:audio/wav;base64,")
        if ',' in prompt_wav:
            prompt_wav = prompt_wav.split(',', 1)[1]
        audio_data = base64.b64decode(prompt_wav)
        waveform, sample_rate = torchaudio.load(io.BytesIO(audio_data))

    # Check if it's raw base64
    elif not prompt_wav.startswith('/') and not os.path.exists(prompt_wav):
        try:
            audio_data = base64.b64decode(prompt_wav)
            waveform, sample_rate = torchaudio.load(io.BytesIO(audio_data))
        except Exception:
            raise ValueError(f"Invalid prompt_wav format: {prompt_wav[:50]}...")

    # Treat as file path
    else:
        waveform, sample_rate = torchaudio.load(prompt_wav)

    # Convert to numpy and resample to 16kHz if needed
    audio = waveform.numpy()[0]  # Get first channel
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        audio = resampler(waveform).numpy()[0]

    return audio


def get_prompt_wav_path(prompt_wav: str) -> str:
    """Get file path for prompt audio, handling base64/URL by saving to temp file"""
    import tempfile
    import uuid

    # If it's already a file path, return it
    if os.path.exists(prompt_wav):
        return prompt_wav

    # For base64 or URL, save to temp file
    if prompt_wav.startswith('data:audio') or urlparse(prompt_wav).scheme in ('http', 'https'):
        # Load audio data
        if urlparse(prompt_wav).scheme in ('http', 'https'):
            response = requests.get(prompt_wav)
            response.raise_for_status()
            audio_data = response.content
        else:
            # Base64 encoded
            if ',' in prompt_wav:
                prompt_wav = prompt_wav.split(',', 1)[1]
            audio_data = base64.b64decode(prompt_wav)

        # Save to temp file
        temp_dir = tempfile.gettempdir()
        temp_filename = os.path.join(temp_dir, f"prompt_audio_{uuid.uuid4().hex[:8]}.wav")
        with open(temp_filename, 'wb') as f:
            f.write(audio_data)
        return temp_filename

    # If it doesn't exist and isn't a URL/base64, return as-is (let model handle error)
    return prompt_wav


def generate_audio_data(model_output, output_format: str = 'wav'):
    """Generate audio data in specified format"""
    for i in model_output:
        audio = i['tts_speech'].numpy()
        sample_rate = 22050

        # Convert to int16 PCM
        audio_int16 = (audio * (2 ** 15)).astype(np.int16)

        if output_format == 'pcm':
            # Raw PCM data
            yield audio_int16.tobytes()
        elif output_format == 'wav':
            # WAV format with header
            import wave
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            yield buf.getvalue()
        else:
            # For mp3, opus, etc., use torchaudio if available
            try:
                audio_tensor = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
                buf = io.BytesIO()
                torchaudio.save(buf, audio_tensor, sample_rate, format=output_format.upper())
                yield buf.getvalue()
            except Exception:
                # Fallback to wav if format not supported
                import wave
                buf = io.BytesIO()
                with wave.open(buf, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_int16.tobytes())
                yield buf.getvalue()


# OpenAI API compatible request models
class CreateSpeechRequest(BaseModel):
    model: str = Field(..., description="The TTS model to use (e.g., tts-1, tts-1-hd)")
    input: str = Field(..., min_length=1, description="The text to generate audio for")
    voice: str = Field(..., description="The voice to use (alloy, echo, fable, onyx, nova, shimmer)")
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="mp3",
        description="The format to audio in"
    )
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="The speed of the generated audio")
    # CosyVoice specific parameters
    prompt_text: Optional[str] = Field(None, description="Prompt text for zero-shot voice cloning")
    prompt_wav: Optional[str] = Field(None, description="Base64 encoded audio or URL for voice reference (used with prompt_text)")
    instruct_text: Optional[str] = Field(None, description="Instruction text for instruct mode")


class VoicesResponse(BaseModel):
    voices: list


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI compatible)"""
    return {
        "object": "list",
        "data": [
            {
                "id": "tts-1",
                "object": "model",
                "created": 1677610602,
                "owned_by": "cosyvoice"
            },
            {
                "id": "tts-1-hd",
                "object": "model",
                "created": 1677610602,
                "owned_by": "cosyvoice"
            }
        ]
    }


@app.get("/v1/voices")
async def list_voices():
    """List available voices (OpenAI compatible)"""
    available_spks = cosyvoice.list_available_spks() if cosyvoice else list(VOICE_MAPPING.keys())
    return {
        "voices": [
            {
                "voice": voice,
                "name": voice.capitalize(),
                "description": f"{voice.capitalize()} voice"
            }
            for voice in VOICE_MAPPING.keys()
        ]
    }


@app.post("/v1/audio/speech")
async def create_speech(request: CreateSpeechRequest, http_request: Request):
    """Create audio from text (OpenAI compatible endpoint)

    Supports multiple inference modes based on provided parameters:
    - SFT mode: Default mode, uses pre-defined speaker voices
    - Zero-shot mode: When prompt_text and prompt_wav are provided
    - Cross-lingual mode: When only prompt_wav is provided
    - Instruct mode: When instruct_text is provided (CosyVoice1 only)
    - Instruct2 mode: When instruct_text and prompt_wav are provided (CosyVoice2/3)
    """
    global cosyvoice

    if cosyvoice is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate speed parameter
    speed = max(0.25, min(4.0, request.speed))

    # Determine inference mode based on provided parameters
    mode = "sft"
    if request.prompt_text and request.prompt_wav:
        mode = "zero_shot"
    elif request.prompt_wav and not request.prompt_text:
        mode = "cross_lingual"
    elif request.instruct_text and request.prompt_wav:
        mode = "instruct2"
    elif request.instruct_text:
        mode = "instruct"

    logging.info(f"Using {mode} mode for inference")

    try:
        # Get content type based on response format
        content_type_map = {
            'mp3': 'audio/mpeg',
            'opus': 'audio/opus',
            'aac': 'audio/aac',
            'flac': 'audio/flac',
            'wav': 'audio/wav',
            'pcm': 'audio/pcm',
        }
        content_type = content_type_map.get(request.response_format, 'audio/wav')

        # Generate audio based on mode
        if mode == "zero_shot":
            # Zero-shot mode: clone voice from prompt
            prompt_wav_path = get_prompt_wav_path(request.prompt_wav)
            model_output = cosyvoice.inference_zero_shot(
                tts_text=request.input,
                prompt_text=request.prompt_text,
                prompt_wav=prompt_wav_path,
                stream=False,
                speed=speed
            )
        elif mode == "cross_lingual":
            # Cross-lingual mode: clone voice without text prompt
            prompt_wav_path = get_prompt_wav_path(request.prompt_wav)
            model_output = cosyvoice.inference_cross_lingual(
                tts_text=request.input,
                prompt_wav=prompt_wav_path,
                stream=False,
                speed=speed
            )
        elif mode == "instruct":
            # Instruct mode: use instruction with speaker
            available_spks = cosyvoice.list_available_spks()
            if not available_spks:
                raise HTTPException(
                    status_code=400,
                    detail="Instruct mode not supported by this model. Please provide both instruct_text and prompt_wav for instruct2 mode."
                )
            spk_id = VOICE_MAPPING.get(request.voice, available_spks[0])
            model_output = cosyvoice.inference_instruct(
                tts_text=request.input,
                spk_id=spk_id,
                instruct_text=request.instruct_text,
                stream=False,
                speed=speed
            )
        elif mode == "instruct2":
            # Instruct2 mode: use instruction with prompt audio
            prompt_wav_path = get_prompt_wav_path(request.prompt_wav)
            model_output = cosyvoice.inference_instruct2(
                tts_text=request.input,
                instruct_text=request.instruct_text,
                prompt_wav=prompt_wav_path,
                stream=False,
                speed=speed
            )
        else:
            # Default SFT mode: use pre-defined speaker
            available_spks = cosyvoice.list_available_spks()
            if not available_spks:
                raise HTTPException(
                    status_code=400,
                    detail="SFT mode not supported by this model. Please provide prompt_wav for zero-shot/cross-lingual mode, or instruct_text for instruct mode."
                )
            spk_id = VOICE_MAPPING.get(request.voice, available_spks[0])
            model_output = cosyvoice.inference_sft(
                tts_text=request.input,
                spk_id=spk_id,
                stream=False,
                speed=speed
            )

        # Collect all audio data
        audio_chunks = []
        for chunk in generate_audio_data(model_output, request.response_format):
            audio_chunks.append(chunk)

        audio_data = b''.join(audio_chunks)

        return StreamingResponse(
            iter([audio_data]),
            media_type=content_type,
            headers={
                "Content-Disposition": f'attachment; filename="speech.{request.response_format}"'
            }
        )

    except Exception as e:
        logging.error(f"Error generating speech: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")


# Legacy CosyVoice endpoints (for backward compatibility)
@app.get("/inference_sft")
@app.post("/inference_sft")
async def inference_sft(tts_text: str, spk_id: str):
    """Legacy SFT inference endpoint"""
    if cosyvoice is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    model_output = cosyvoice.inference_sft(tts_text, spk_id)

    def generate_data(model_output):
        for i in model_output:
            tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
            yield tts_audio

    return StreamingResponse(generate_data(model_output))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": cosyvoice is not None,
        "available_speakers": cosyvoice.list_available_spks() if cosyvoice else []
    }


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "CosyVoice OpenAI Compatible API",
        "version": "1.0.0",
        "endpoints": {
            "audio_speech": "/v1/audio/speech",
            "models": "/v1/models",
            "voices": "/v1/voices",
            "health": "/health"
        },
        "documentation": "https://platform.openai.com/docs/api-reference/audio/createSpeech"
    }


def load_model(model_dir: str, load_jit: bool = False, load_trt: bool = False,
               load_vllm: bool = False, fp16: bool = False, trt_concurrent: int = 1,
               device: str = '0'):
    """Load CosyVoice model"""
    global cosyvoice

    # Log CUDA device info (CUDA_VISIBLE_DEVICES was set before torch import)
    if torch.cuda.is_available():
        logging.info(f"Using GPU device: cuda (mapped to physical GPU via CUDA_VISIBLE_DEVICES={device})")
    else:
        logging.warning("CUDA is not available, using CPU")

    logging.info(f"Loading model from: {model_dir}")
    logging.info(f"Model directory exists: {os.path.exists(model_dir)}")
    if os.path.exists(model_dir):
        logging.info(f"Model directory contents: {os.listdir(model_dir)}")
    else:
        logging.error(f"Model directory does not exist: {model_dir}")
        logging.info(f"Current working directory: {os.getcwd()}")
        logging.info(f"Environment MODEL_DIR: {os.getenv('MODEL_DIR', 'not set')}")

    try:
        # Try AutoModel first to automatically detect model type
        cosyvoice = AutoModel(
            model_dir=model_dir,
            load_jit=load_jit,
            load_trt=load_trt,
            load_vllm=load_vllm,
            fp16=fp16,
            trt_concurrent=trt_concurrent
        )
        logging.info(f"Model loaded successfully using AutoModel")
    except Exception as e:
        logging.warning(f"AutoModel failed: {e}, trying specific model types...")
        # Fallback to trying specific model types
        for ModelClass, name in [(CosyVoice, 'CosyVoice'), (CosyVoice2, 'CosyVoice2'), (CosyVoice3, 'CosyVoice3')]:
            try:
                if name == 'CosyVoice':
                    cosyvoice = ModelClass(model_dir, load_jit=load_jit, load_trt=load_trt, fp16=fp16, trt_concurrent=trt_concurrent)
                elif name in ['CosyVoice2', 'CosyVoice3']:
                    cosyvoice = ModelClass(model_dir, load_trt=load_trt, load_vllm=load_vllm, fp16=fp16, trt_concurrent=trt_concurrent)
                logging.info(f"Model loaded successfully as {name}")
                break
            except Exception:
                continue
        else:
            raise TypeError('No valid model type found!')

    available_spks = cosyvoice.list_available_spks()
    logging.info(f"Available speakers: {available_spks}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CosyVoice OpenAI Compatible API Server")
    parser.add_argument('--port',
                        type=int,
                        default=8000,
                        help='Port to run the server on (default: 8000)')
    parser.add_argument('--host',
                        type=str,
                        default='0.0.0.0',
                        help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/Fun-CosyVoice3-0.5B',
                        help='Local path or ModelScope repo ID')
    parser.add_argument('--load_jit',
                        action='store_true',
                        help='Load JIT compiled model')
    parser.add_argument('--load_trt',
                        action='store_true',
                        help='Load TensorRT model')
    parser.add_argument('--load_vllm',
                        action='store_true',
                        help='Load VLLM model (CosyVoice2/3 only)')
    parser.add_argument('--fp16',
                        action='store_true',
                        help='Use FP16 precision')
    parser.add_argument('--trt_concurrent',
                        type=int,
                        default=1,
                        help='TensorRT concurrent instances')
    parser.add_argument('--device',
                        type=str,
                        default='0',
                        help='CUDA device to use (e.g., "0", "1", "0,1", "-1" for CPU)')
    parser.add_argument('--workers',
                        type=int,
                        default=1,
                        help='Number of worker processes')

    args = parser.parse_args()

    # Load model
    load_model(
        model_dir=args.model_dir,
        load_jit=args.load_jit,
        load_trt=args.load_trt,
        load_vllm=args.load_vllm,
        fp16=args.fp16,
        trt_concurrent=args.trt_concurrent,
        device=args.device
    )

    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers
    )

#!/usr/bin/env python3
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Simple HTTP client example for CosyVoice OpenAI Compatible API
#
# Usage:
#   python tests/example_client.py --mode sft
#   python tests/example_client.py --mode zero_shot --prompt-wav reference.wav
#   python tests/example_client.py --mode instruct --instruct "用开心的语气"

import argparse
import base64
import os
import sys
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def encode_audio(file_path: str) -> str:
    """Encode audio file to base64"""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def sft_mode(text: str, voice: str, output: str, speed: float, format: str):
    """SFT Mode: Use pre-defined speaker voices"""
    print(f"🎤 SFT Mode: voice={voice}")

    payload = {
        "model": "tts-1",
        "input": text,
        "voice": voice,
        "response_format": format,
        "speed": speed,
    }

    response = requests.post(f"{API_BASE_URL}/v1/audio/speech", json=payload, timeout=300)
    response.raise_for_status()

    with open(output, "wb") as f:
        f.write(response.content)

    print(f"✅ Audio saved to: {output}")


def zero_shot_mode(text: str, prompt_text: str, prompt_wav: str, output: str, speed: float, format: str):
    """Zero-shot Mode: Clone voice from prompt audio"""
    print(f"🎤 Zero-shot Mode: prompt_text={prompt_text[:30]}...")

    # Read and encode audio if it's a file
    if os.path.exists(prompt_wav):
        with open(prompt_wav, "rb") as f:
            audio_data = base64.b64encode(f.read()).decode("utf-8")
        prompt_wav_data = f"data:audio/wav;base64,{audio_data}"
    else:
        # Assume it's a URL or already base64
        prompt_wav_data = prompt_wav

    payload = {
        "model": "tts-1",
        "input": text,
        "voice": "alloy",  # Not used in zero-shot mode
        "response_format": format,
        "speed": speed,
        "prompt_text": prompt_text,
        "prompt_wav": prompt_wav_data,
    }

    response = requests.post(f"{API_BASE_URL}/v1/audio/speech", json=payload, timeout=300)
    response.raise_for_status()

    with open(output, "wb") as f:
        f.write(response.content)

    print(f"✅ Audio saved to: {output}")


def cross_lingual_mode(text: str, prompt_wav: str, output: str, speed: float, format: str):
    """Cross-lingual Mode: Clone voice without text prompt"""
    print(f"🎤 Cross-lingual Mode")

    # Read and encode audio if it's a file
    if os.path.exists(prompt_wav):
        with open(prompt_wav, "rb") as f:
            audio_data = base64.b64encode(f.read()).decode("utf-8")
        prompt_wav_data = f"data:audio/wav;base64,{audio_data}"
    else:
        prompt_wav_data = prompt_wav

    payload = {
        "model": "tts-1",
        "input": text,
        "voice": "alloy",  # Not used in cross-lingual mode
        "response_format": format,
        "speed": speed,
        "prompt_wav": prompt_wav_data,
    }

    response = requests.post(f"{API_BASE_URL}/v1/audio/speech", json=payload, timeout=300)
    response.raise_for_status()

    with open(output, "wb") as f:
        f.write(response.content)

    print(f"✅ Audio saved to: {output}")


def instruct_mode(text: str, instruct_text: str, voice: str, output: str, speed: float, format: str):
    """Instruct Mode: Use instruction with speaker"""
    print(f"🎤 Instruct Mode: instruct={instruct_text}")

    payload = {
        "model": "tts-1",
        "input": text,
        "voice": voice,
        "response_format": format,
        "speed": speed,
        "instruct_text": instruct_text,
    }

    response = requests.post(f"{API_BASE_URL}/v1/audio/speech", json=payload, timeout=300)
    response.raise_for_status()

    with open(output, "wb") as f:
        f.write(response.content)

    print(f"✅ Audio saved to: {output}")


def instruct2_mode(text: str, instruct_text: str, prompt_wav: str, output: str, speed: float, format: str):
    """Instruct2 Mode: Use instruction with prompt audio"""
    print(f"🎤 Instruct2 Mode: instruct={instruct_text}")

    # Read and encode audio if it's a file
    if os.path.exists(prompt_wav):
        with open(prompt_wav, "rb") as f:
            audio_data = base64.b64encode(f.read()).decode("utf-8")
        prompt_wav_data = f"data:audio/wav;base64,{audio_data}"
    else:
        prompt_wav_data = prompt_wav

    payload = {
        "model": "tts-1",
        "input": text,
        "voice": "alloy",
        "response_format": format,
        "speed": speed,
        "instruct_text": instruct_text,
        "prompt_wav": prompt_wav_data,
    }

    response = requests.post(f"{API_BASE_URL}/v1/audio/speech", json=payload, timeout=300)
    response.raise_for_status()

    with open(output, "wb") as f:
        f.write(response.content)

    print(f"✅ Audio saved to: {output}")


def main():
    parser = argparse.ArgumentParser(description="CosyVoice OpenAI API Client Example")
    parser.add_argument("--api-url", default=API_BASE_URL, help="API base URL")
    parser.add_argument("--mode", required=True,
                       choices=["sft", "zero_shot", "cross_lingual", "instruct", "instruct2"],
                       help="Inference mode")
    parser.add_argument("--text", default="你好，这是一个测试。", help="Text to synthesize")
    parser.add_argument("--voice", default="alloy", help="Voice for SFT/Instruct mode")
    parser.add_argument("--output", default="output.wav", help="Output audio file")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed (0.25-4.0)")
    parser.add_argument("--format", default="wav", choices=["wav", "mp3", "pcm"],
                       help="Audio format")
    parser.add_argument("--prompt-text", help="Prompt text for zero-shot mode")
    parser.add_argument("--prompt-wav", help="Prompt audio file for zero-shot/cross-lingual/instruct2 mode")
    parser.add_argument("--instruct", help="Instruction text for instruct/instruct2 mode")

    args = parser.parse_args()

    # Update API URL from env
    global API_BASE_URL
    API_BASE_URL = args.api_url

    # Check server health
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
        print(f"✅ Server is healthy: {response.json()['status']}")
    except Exception as e:
        print(f"❌ Server health check failed: {e}")
        sys.exit(1)

    # Route to appropriate mode
    try:
        if args.mode == "sft":
            sft_mode(args.text, args.voice, args.output, args.speed, args.format)
        elif args.mode == "zero_shot":
            if not args.prompt_text or not args.prompt_wav:
                print("❌ zero_shot mode requires --prompt-text and --prompt-wav")
                sys.exit(1)
            zero_shot_mode(args.text, args.prompt_text, args.prompt_wav, args.output, args.speed, args.format)
        elif args.mode == "cross_lingual":
            if not args.prompt_wav:
                print("❌ cross_lingual mode requires --prompt-wav")
                sys.exit(1)
            cross_lingual_mode(args.text, args.prompt_wav, args.output, args.speed, args.format)
        elif args.mode == "instruct":
            if not args.instruct:
                print("❌ instruct mode requires --instruct")
                sys.exit(1)
            instruct_mode(args.text, args.instruct, args.voice, args.output, args.speed, args.format)
        elif args.mode == "instruct2":
            if not args.instruct or not args.prompt_wav:
                print("❌ instruct2 mode requires --instruct and --prompt-wav")
                sys.exit(1)
            instruct2_mode(args.text, args.instruct, args.prompt_wav, args.output, args.speed, args.format)
    except requests.exceptions.HTTPError as e:
        print(f"❌ API request failed: {e}")
        if e.response is not None:
            print(f"   Response: {e.response.text}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

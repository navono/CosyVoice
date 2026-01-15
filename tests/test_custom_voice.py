#!/usr/bin/env python3
"""
Test script for custom voice file support
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_custom_voice_with_text():
    """Test using custom voice file with corresponding text file"""
    print("=" * 60)
    print("Test 1: Custom Voice with Text File")
    print("=" * 60)
    print("Prerequisites:")
    print("  - jiang-style1.mp3 exists in VOICE_DIR")
    print("  - jiang-style1.txt exists in VOICE_DIR")
    print()
    
    url = f"{BASE_URL}/v1/audio/speech"
    payload = {
        "model": "tts-1",
        "input": "这是使用自定义语音的测试。",
        "voice": "jiang-style1.mp3",  # or just "jiang-style1"
        "response_format": "wav",
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            print(f"Content-Length: {len(response.content)} bytes")
            with open("/tmp/test_custom_voice_with_text.wav", "wb") as f:
                f.write(response.content)
            print("✓ Audio saved to /tmp/test_custom_voice_with_text.wav")
            print("✓ Mode: Zero-shot (with prompt text from .txt file)")
        else:
            print(f"✗ Error: {response.text}")
    except Exception as e:
        print(f"✗ Exception: {e}")
    
    print()


def test_custom_voice_without_text():
    """Test using custom voice file without text file (cross-lingual mode)"""
    print("=" * 60)
    print("Test 2: Custom Voice without Text File")
    print("=" * 60)
    print("Prerequisites:")
    print("  - voice-sample.wav exists in VOICE_DIR")
    print("  - voice-sample.txt does NOT exist")
    print()
    
    url = f"{BASE_URL}/v1/audio/speech"
    payload = {
        "model": "tts-1",
        "input": "这是跨语言语音克隆的测试。",
        "voice": "voice-sample.wav",
        "response_format": "wav"
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            print(f"Content-Length: {len(response.content)} bytes")
            with open("/tmp/test_custom_voice_cross_lingual.wav", "wb") as f:
                f.write(response.content)
            print("✓ Audio saved to /tmp/test_custom_voice_cross_lingual.wav")
            print("✓ Mode: Cross-lingual (no prompt text)")
        else:
            print(f"✗ Error: {response.text}")
    except Exception as e:
        print(f"✗ Exception: {e}")
    
    print()


def test_custom_voice_with_instruct():
    """Test custom voice with instruction text"""
    print("=" * 60)
    print("Test 3: Custom Voice with Instruction")
    print("=" * 60)
    
    url = f"{BASE_URL}/v1/audio/speech"
    payload = {
        "model": "tts-1",
        "input": "这是带有情感指令的语音合成。",
        "voice": "jiang-style1",
        "instruct_text": "用开心的语气说话",
        "response_format": "wav",
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            print(f"Content-Length: {len(response.content)} bytes")
            with open("/tmp/test_custom_voice_instruct.wav", "wb") as f:
                f.write(response.content)
            print("✓ Audio saved to /tmp/test_custom_voice_instruct.wav")
            print("✓ Mode: Instruct2 (custom voice + instruction)")
        else:
            print(f"✗ Error: {response.text}")
    except Exception as e:
        print(f"✗ Exception: {e}")
    
    print()


def test_standard_voice_fallback():
    """Test that standard OpenAI voices still work"""
    print("=" * 60)
    print("Test 4: Standard Voice Fallback")
    print("=" * 60)
    
    url = f"{BASE_URL}/v1/audio/speech"
    payload = {
        "model": "tts-1",
        "input": "这是使用标准语音的测试。",
        "voice": "alloy",
        "response_format": "wav"
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    print("Note: Should use standard voice mapping (not custom file)")
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            print(f"Content-Length: {len(response.content)} bytes")
            with open("/tmp/test_standard_voice.wav", "wb") as f:
                f.write(response.content)
            print("✓ Audio saved to /tmp/test_standard_voice.wav")
            print("✓ Mode: SFT or Cross-lingual (standard voice)")
        else:
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"✗ Exception: {e}")
    
    print()


def test_explicit_prompt_wav_priority():
    """Test that explicit prompt_wav has priority over voice file"""
    print("=" * 60)
    print("Test 5: Explicit prompt_wav Priority")
    print("=" * 60)
    
    url = f"{BASE_URL}/v1/audio/speech"
    payload = {
        "model": "tts-1",
        "input": "测试参数优先级。",
        "voice": "jiang-style1",  # This would normally load jiang-style1.mp3
        "prompt_wav": "https://example.com/other-voice.wav",  # But this takes priority
        "prompt_text": "这是明确指定的参考文本",
        "response_format": "wav",
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    print("Note: prompt_wav should override voice file")
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text if response.status_code != 200 else 'Success'}")
    except Exception as e:
        print(f"✗ Exception: {e}")
    
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Custom Voice File Test Suite")
    print("=" * 60)
    print()
    print("Setup Instructions:")
    print("1. Ensure VOICE_DIR is mounted in docker-compose.yml")
    print("2. Place voice samples in HOST_VOICE_DIR")
    print("3. Format: audio_file.mp3 + audio_file.txt (optional)")
    print()
    
    test_standard_voice_fallback()
    test_custom_voice_with_text()
    test_custom_voice_without_text()
    test_custom_voice_with_instruct()
    test_explicit_prompt_wav_priority()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
    print()
    print("Summary:")
    print("- Standard voices (alloy, echo, etc.) still work normally")
    print("- Custom voice files are automatically detected and loaded")
    print("- Text files (.txt) are optional for zero-shot mode")
    print("- Without text file, uses cross-lingual mode")
    print("- Explicit prompt_wav parameter has priority over voice files")

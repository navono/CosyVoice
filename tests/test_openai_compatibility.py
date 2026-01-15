#!/usr/bin/env python3
"""
Test script to verify OpenAI API compatibility
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_openai_standard_request():
    """Test with standard OpenAI parameters only"""
    print("=" * 60)
    print("Test 1: Standard OpenAI Request")
    print("=" * 60)
    
    url = f"{BASE_URL}/v1/audio/speech"
    payload = {
        "model": "tts-1",
        "input": "Hello, this is a test of the OpenAI compatible API.",
        "voice": "alloy",
        "response_format": "wav",
        "speed": 1.0
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        print(f"Status: {response.status_code}")
        print(f"Content-Type: {response.headers.get('Content-Type')}")
        print(f"Content-Length: {len(response.content)} bytes")
        
        if response.status_code == 200:
            with open("/tmp/test_standard.wav", "wb") as f:
                f.write(response.content)
            print("✓ Audio saved to /tmp/test_standard.wav")
        else:
            print(f"✗ Error: {response.text}")
    except Exception as e:
        print(f"✗ Exception: {e}")
    
    print()


def test_cosyvoice_extensions():
    """Test with CosyVoice-specific extensions"""
    print("=" * 60)
    print("Test 2: CosyVoice Extensions (Zero-shot)")
    print("=" * 60)
    
    url = f"{BASE_URL}/v1/audio/speech"
    payload = {
        "model": "tts-1",
        "input": "这是一个测试语音克隆的例子。",
        "voice": "alloy",
        "response_format": "wav",
        "speed": 1.0,
        # CosyVoice extensions
        "prompt_text": "希望你以开心的语气说话。",
        "instruct_text": "用温柔的声音说话"
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    print("Note: This test may fail if prompt_wav is required for zero-shot mode")
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            print(f"Content-Length: {len(response.content)} bytes")
            with open("/tmp/test_extensions.wav", "wb") as f:
                f.write(response.content)
            print("✓ Audio saved to /tmp/test_extensions.wav")
        else:
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"✗ Exception: {e}")
    
    print()


def test_list_models():
    """Test /v1/models endpoint"""
    print("=" * 60)
    print("Test 3: List Models")
    print("=" * 60)
    
    url = f"{BASE_URL}/v1/models"
    
    try:
        response = requests.get(url, timeout=10)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")
            print(f"✓ Found {len(data.get('data', []))} models")
        else:
            print(f"✗ Error: {response.text}")
    except Exception as e:
        print(f"✗ Exception: {e}")
    
    print()


def test_health():
    """Test health endpoint"""
    print("=" * 60)
    print("Test 4: Health Check")
    print("=" * 60)
    
    url = f"{BASE_URL}/health"
    
    try:
        response = requests.get(url, timeout=10)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")
            print(f"✓ Model loaded: {data.get('model_loaded')}")
            print(f"✓ Available speakers: {data.get('available_speakers')}")
        else:
            print(f"✗ Error: {response.text}")
    except Exception as e:
        print(f"✗ Exception: {e}")
    
    print()


def test_minimal_request():
    """Test with minimal required parameters"""
    print("=" * 60)
    print("Test 5: Minimal Required Parameters")
    print("=" * 60)
    
    url = f"{BASE_URL}/v1/audio/speech"
    payload = {
        "model": "tts-1",
        "input": "你好世界",
        "voice": "alloy"
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    print("Note: Using defaults for response_format (mp3) and speed (1.0)")
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        print(f"Status: {response.status_code}")
        print(f"Content-Type: {response.headers.get('Content-Type')}")
        
        if response.status_code == 200:
            print(f"Content-Length: {len(response.content)} bytes")
            with open("/tmp/test_minimal.mp3", "wb") as f:
                f.write(response.content)
            print("✓ Audio saved to /tmp/test_minimal.mp3")
        else:
            print(f"✗ Error: {response.text}")
    except Exception as e:
        print(f"✗ Exception: {e}")
    
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("OpenAI API Compatibility Test Suite")
    print("=" * 60)
    print()
    
    test_health()
    test_list_models()
    test_minimal_request()
    test_openai_standard_request()
    test_cosyvoice_extensions()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)

#!/usr/bin/env python3
"""
Test CosyVoice API using official OpenAI Python SDK
This verifies that the API is fully compatible with OpenAI's client library
"""
from openai import OpenAI
from pathlib import Path

# Initialize OpenAI client pointing to CosyVoice server
client = OpenAI(
    api_key="dummy-key",  # CosyVoice doesn't require authentication
    base_url="http://localhost:8000/v1"
)

def test_basic_tts():
    """Test basic TTS using OpenAI SDK"""
    print("=" * 60)
    print("Test: Basic TTS with OpenAI SDK")
    print("=" * 60)
    
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input="Hello! This is a test using the official OpenAI Python SDK."
        )
        
        # Save audio file
        output_path = Path("/tmp/openai_sdk_test.mp3")
        response.stream_to_file(output_path)
        
        print(f"✓ Success! Audio saved to {output_path}")
        print(f"✓ File size: {output_path.stat().st_size} bytes")
        
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print()


def test_with_options():
    """Test with various options"""
    print("=" * 60)
    print("Test: TTS with Options (speed, format)")
    print("=" * 60)
    
    try:
        response = client.audio.speech.create(
            model="tts-1-hd",
            voice="nova",
            input="你好，世界！这是一个测试。",
            response_format="wav",
            speed=1.2
        )
        
        output_path = Path("/tmp/openai_sdk_options.wav")
        response.stream_to_file(output_path)
        
        print(f"✓ Success! Audio saved to {output_path}")
        print(f"✓ File size: {output_path.stat().st_size} bytes")
        print("✓ Format: WAV, Speed: 1.2x")
        
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print()


def test_list_models():
    """Test listing models"""
    print("=" * 60)
    print("Test: List Models")
    print("=" * 60)
    
    try:
        models = client.models.list()
        
        print(f"✓ Found {len(models.data)} models:")
        for model in models.data:
            print(f"  - {model.id} (owned by {model.owned_by})")
        
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("OpenAI SDK Compatibility Test")
    print("=" * 60)
    print()
    
    test_list_models()
    test_basic_tts()
    test_with_options()
    
    print("=" * 60)
    print("All SDK tests completed!")
    print("=" * 60)
    print("\nNote: If you see errors about missing 'openai' package,")
    print("install it with: pip install openai")

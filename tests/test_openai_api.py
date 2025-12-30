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
"""
Test cases for CosyVoice OpenAI Compatible API

Run tests:
    pytest tests/test_openai_api.py -v
    pytest tests/test_openai_api.py::test_sft_mode -v
    pytest tests/test_openai_api.py -k "zero_shot" -v

Requirements:
    pip install pytest requests httpx
"""
import os
import base64
import pytest
import requests
import shutil
from typing import Optional


# API configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
MODEL_DIR = os.getenv("MODEL_DIR", "./pretrained_models/Fun-CosyVoice3-0.5B")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs")

# clear output dir
def clear_output_dir():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

class CosyVoiceAPIClient:
    """Client for CosyVoice OpenAI Compatible API"""

    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.timeout = 300  # 5 minutes timeout for TTS generation

    def health_check(self) -> dict:
        """Check API health status"""
        response = requests.get(f"{self.base_url}/health", timeout=10)
        response.raise_for_status()
        return response.json()

    def list_models(self) -> dict:
        """List available models"""
        response = requests.get(f"{self.base_url}/v1/models", timeout=10)
        response.raise_for_status()
        return response.json()

    def list_voices(self) -> dict:
        """List available voices"""
        response = requests.get(f"{self.base_url}/v1/voices", timeout=10)
        response.raise_for_status()
        return response.json()

    def create_speech(
        self,
        text: str,
        voice: str = "alloy",
        response_format: str = "wav",
        speed: float = 1.0,
        prompt_text: Optional[str] = None,
        prompt_wav: Optional[str] = None,
        instruct_text: Optional[str] = None,
    ) -> bytes:
        """Create speech from text

        Args:
            text: Text to convert to speech
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
            response_format: Audio format (mp3, opus, aac, flac, wav, pcm)
            speed: Speed of the generated audio (0.25 - 4.0)
            prompt_text: Prompt text for zero-shot mode
            prompt_wav: Prompt audio path, URL, or base64 for zero-shot/cross-lingual mode
            instruct_text: Instruction text for instruct mode

        Returns:
            Audio data as bytes
        """
        payload = {
            "model": "tts-1",
            "input": text,
            "voice": voice,
            "response_format": response_format,
            "speed": speed,
        }

        # Add optional parameters
        if prompt_text:
            payload["prompt_text"] = prompt_text
        if prompt_wav:
            payload["prompt_wav"] = prompt_wav
        if instruct_text:
            payload["instruct_text"] = instruct_text

        response = requests.post(
            f"{self.base_url}/v1/audio/speech",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.content


def encode_audio_to_base64(audio_path: str) -> str:
    """Encode audio file to base64 string"""
    with open(audio_path, "rb") as f:
        audio_data = f.read()
    return base64.b64encode(audio_data).decode("utf-8")


# Test fixtures
@pytest.fixture(scope="module")
def api_client():
    """Create API client fixture"""
    client = CosyVoiceAPIClient(API_BASE_URL)

    # Wait for server to be ready
    import time
    max_retries = 30
    for i in range(max_retries):
        try:
            health = client.health_check()
            if health.get("model_loaded"):
                break
        except requests.exceptions.RequestException:
            if i < max_retries - 1:
                time.sleep(2)
            else:
                raise

    return client


@pytest.fixture(scope="module")
def supports_sft(api_client: CosyVoiceAPIClient):
    """Check if model supports SFT mode (has available speakers)"""
    health = api_client.health_check()
    return len(health.get("available_speakers", [])) > 0


@pytest.fixture(scope="module", autouse=True)
def output_dir():
    """Create and cleanup output directory for test audio files"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    yield OUTPUT_DIR
    # Cleanup is optional, comment out if you want to keep the files


def save_audio_output(audio_data: bytes, test_name: str, output_path: str = OUTPUT_DIR):
    """Save audio data to file for inspection"""
    filename = os.path.join(output_path, f"{test_name}.wav")
    with open(filename, 'wb') as f:
        f.write(audio_data)
    print(f"  Audio saved to: {filename}")


@pytest.fixture(scope="module")
def sample_audio_path():
    """Path to sample audio file for testing"""
    # Try to find a sample audio file
    possible_paths = [
        "/home/pingqixing/assets/voices/Female_03.wav",
        "./tests/fixtures/reference.wav",
        "./pretrained_models/Fun-CosyVoice3-0.5B/reference.wav",
        "./reference.wav",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    # Return None if no sample audio found
    pytest.skip("No sample audio file found for testing")
    return None


@pytest.fixture(scope="module")
def sample_audio_text():
    """Get reference text for the sample audio"""
    possible_paths = [
        "/home/pingqixing/assets/voices/Female_03.txt",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return f.read().strip()

    # Default fallback text
    return "我们更决心加强于强化规则，无论是国际间或是在各国之内"


# ============================================================================
# Health and Info Tests
# ============================================================================

def test_health_check(api_client: CosyVoiceAPIClient):
    """Test health check endpoint"""
    health = api_client.health_check()

    assert health["status"] == "healthy"
    assert health["model_loaded"] is True
    assert isinstance(health.get("available_speakers"), list)


def test_list_models(api_client: CosyVoiceAPIClient):
    """Test list models endpoint"""
    models = api_client.list_models()

    assert models["object"] == "list"
    assert len(models["data"]) > 0
    assert models["data"][0]["id"] in ["tts-1", "tts-1-hd"]


def test_list_voices(api_client: CosyVoiceAPIClient):
    """Test list voices endpoint"""
    voices = api_client.list_voices()

    assert "voices" in voices
    assert len(voices["voices"]) > 0
    voice_ids = [v["voice"] for v in voices["voices"]]
    expected_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    for voice in expected_voices:
        assert voice in voice_ids


# ============================================================================
# SFT Mode Tests
# ============================================================================

@pytest.mark.parametrize("voice", ["alloy", "echo", "nova"])
@pytest.mark.parametrize("text", ["你好，世界！", "Hello, world!", "这是一个测试。"])
@pytest.mark.skipif("os.getenv('SKIP_SFT_TESTS', 'false') == 'true'", reason="SFT mode not supported by model")
def test_sft_mode(api_client: CosyVoiceAPIClient, voice: str, text: str):
    """Test SFT mode with different voices and texts"""
    audio_data = api_client.create_speech(
        text=text,
        voice=voice,
        response_format="wav",
    )

    assert len(audio_data) > 0
    # Verify WAV header
    assert audio_data[:4] == b"RIFF"
    assert audio_data[8:12] == b"WAVE"


@pytest.mark.skipif("os.getenv('SKIP_SFT_TESTS', 'false') == 'true'", reason="SFT mode not supported by model")
def test_sft_mode_different_formats(api_client: CosyVoiceAPIClient):
    """Test SFT mode with different audio formats"""
    text = "测试音频格式"
    formats = ["wav", "mp3", "pcm"]

    for fmt in formats:
        audio_data = api_client.create_speech(
            text=text,
            voice="alloy",
            response_format=fmt,
        )
        assert len(audio_data) > 0


@pytest.mark.skipif("os.getenv('SKIP_SFT_TESTS', 'false') == 'true'", reason="SFT mode not supported by model")
def test_sft_mode_speed(api_client: CosyVoiceAPIClient):
    """Test SFT mode with different speeds"""
    text = "测试语速"
    speeds = [0.5, 1.0, 1.5, 2.0]

    for speed in speeds:
        audio_data = api_client.create_speech(
            text=text,
            voice="alloy",
            response_format="wav",
            speed=speed,
        )
        assert len(audio_data) > 0


# ============================================================================
# Zero-shot Mode Tests (CosyVoice3 supported)
# ============================================================================

def test_zero_shot_mode_with_file(api_client: CosyVoiceAPIClient, sample_audio_path: str, sample_audio_text: str):
    """Test zero-shot mode with file path"""
    # CosyVoice3 requires system prompt in prompt_text
    prompt_text_with_system = f"You are a helpful assistant.<|endofprompt|>{sample_audio_text}"
    audio_data = api_client.create_speech(
        text="你好，世界！",
        voice="alloy",
        prompt_text=prompt_text_with_system,
        prompt_wav=sample_audio_path,
        response_format="wav",
    )

    assert len(audio_data) > 0
    assert audio_data[:4] == b"RIFF"
    save_audio_output(audio_data, "zero_shot_chinese")


def test_zero_shot_mode_with_base64(api_client: CosyVoiceAPIClient, sample_audio_path: str, sample_audio_text: str):
    """Test zero-shot mode with base64 encoded audio"""
    base64_audio = encode_audio_to_base64(sample_audio_path)
    prompt_text_with_system = f"You are a helpful assistant.<|endofprompt|>{sample_audio_text}"

    audio_data = api_client.create_speech(
        text="这是用base64编码音频合成的语音",
        voice="alloy",
        prompt_text=prompt_text_with_system,
        prompt_wav=f"data:audio/wav;base64,{base64_audio}",
        response_format="wav",
    )

    assert len(audio_data) > 0
    assert audio_data[:4] == b"RIFF"


@pytest.mark.skip(reason="URL test requires network access")
def test_zero_shot_mode_with_url(api_client: CosyVoiceAPIClient):
    """Test zero-shot mode with URL (using a public sample audio URL)"""
    audio_url = "https://github.com/PaddlePaddle/PaddleSpeech/raw/develop/paddlespeech/t2s/exps/wavernn_test_audio.wav"

    audio_data = api_client.create_speech(
        text="这是用URL音频合成的语音",
        voice="alloy",
        prompt_text="Reference text",
        prompt_wav=audio_url,
        response_format="wav",
    )

    assert len(audio_data) > 0


# ============================================================================
# Cross-lingual Mode Tests (CosyVoice3 supported)
# ============================================================================

def test_cross_lingual_mode(api_client: CosyVoiceAPIClient, sample_audio_path: str):
    """Test cross-lingual mode with only prompt_wav"""
    audio_data = api_client.create_speech(
        text="跨语言模式测试",
        voice="alloy",
        prompt_wav=sample_audio_path,
        response_format="wav",
    )

    assert len(audio_data) > 0
    assert audio_data[:4] == b"RIFF"
    save_audio_output(audio_data, "cross_lingual_chinese")


# ============================================================================
# Instruct Mode Tests
# ============================================================================

@pytest.mark.skipif("os.getenv('SKIP_SFT_TESTS', 'false') == 'true'", reason="Instruct mode not supported by model")
def test_instruct_mode(api_client: CosyVoiceAPIClient):
    """Test instruct mode with instruct_text"""
    audio_data = api_client.create_speech(
        text="这是指令模式合成的语音",
        voice="alloy",
        instruct_text="用温柔的语气说话",
        response_format="wav",
    )

    assert len(audio_data) > 0
    assert audio_data[:4] == b"RIFF"


@pytest.mark.parametrize("instruct", [
    "用开心的语气",
    "用悲伤的语气",
    "用激动的语气",
    "用平静的语气",
    "用温柔的语气",
])
@pytest.mark.skipif("os.getenv('SKIP_SFT_TESTS', 'false') == 'true'", reason="Instruct mode not supported by model")
def test_instruct_mode_different_instructions(api_client: CosyVoiceAPIClient, instruct: str):
    """Test instruct mode with different instructions"""
    audio_data = api_client.create_speech(
        text="测试不同的指令",
        voice="alloy",
        instruct_text=instruct,
        response_format="wav",
    )

    assert len(audio_data) > 0


# ============================================================================
# Instruct2 Mode Tests (CosyVoice3 supported)
# ============================================================================

def test_instruct2_mode(api_client: CosyVoiceAPIClient, sample_audio_path: str):
    """Test instruct2 mode with instruct_text and prompt_wav"""
    # CosyVoice3 requires system prompt in instruct_text
    instruct_text_with_system = "You are a helpful assistant. 请用温柔的语气说话。<|endofprompt|>"
    audio_data = api_client.create_speech(
        text="这是指令模式2合成的语音",
        voice="alloy",
        instruct_text=instruct_text_with_system,
        prompt_wav=sample_audio_path,
        response_format="wav",
    )

    assert len(audio_data) > 0
    assert audio_data[:4] == b"RIFF"
    save_audio_output(audio_data, "instruct2_gentle")


@pytest.mark.parametrize("instruct_suffix", [
    "请用开心的语气说话。<|endofprompt|>",
    "请用激动的语气说话。<|endofprompt|>",
    "请用平静的语气说话。<|endofprompt|>",
])
def test_instruct2_mode_different_instructions(api_client: CosyVoiceAPIClient, sample_audio_path: str, instruct_suffix: str):
    """Test instruct2 mode with different instructions"""
    instruct_text_with_system = f"You are a helpful assistant. {instruct_suffix}"
    audio_data = api_client.create_speech(
        text="测试不同的指令",
        voice="alloy",
        instruct_text=instruct_text_with_system,
        prompt_wav=sample_audio_path,
        response_format="wav",
    )

    assert len(audio_data) > 0


# ============================================================================
# Additional CosyVoice3 Tests
# ============================================================================

def test_instruct2_cantonese(api_client: CosyVoiceAPIClient, sample_audio_path: str):
    """Test instruct2 mode with Cantonese dialect"""
    instruct_text = "You are a helpful assistant. 请用广东话表达。<|endofprompt|>"
    audio_data = api_client.create_speech(
        text="好少咯，一般系放嗰啲国庆啊，中秋嗰啲可能会咯。",
        voice="alloy",
        instruct_text=instruct_text,
        prompt_wav=sample_audio_path,
        response_format="wav",
    )

    assert len(audio_data) > 0
    save_audio_output(audio_data, "instruct2_cantonese")


def test_instruct2_fast_speed(api_client: CosyVoiceAPIClient, sample_audio_path: str):
    """Test instruct2 mode with fast speed instruction"""
    instruct_text = "You are a helpful assistant. 请用尽可能快地语速说这句话。<|endofprompt|>"
    audio_data = api_client.create_speech(
        text="收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐。",
        voice="alloy",
        instruct_text=instruct_text,
        prompt_wav=sample_audio_path,
        response_format="wav",
    )

    assert len(audio_data) > 0
    save_audio_output(audio_data, "instruct2_fast_speed")


@pytest.mark.parametrize("instruct_text,text", [
    ("You are a helpful assistant. 请用悲伤的语气表达。<|endofprompt|>", "今天天气不好，我感到很失落。"),
    ("You are a helpful assistant. 请用愤怒的语气表达。<|endofprompt|>", "这真是太令人愤怒了！"),
    ("You are a helpful assistant. 请用讲述故事的语气表达。<|endofprompt|>", "很久很久以前，在一个遥远的地方，住着一位勇敢的骑士。"),
    ("You are a helpful assistant. 请用正式的语气表达。<|endofprompt|>", "尊敬的各位来宾，欢迎光临今天的会议。"),
])
def test_instruct2_various_styles(api_client: CosyVoiceAPIClient, sample_audio_path: str, instruct_text: str, text: str):
    """Test instruct2 mode with various styles and emotions"""
    audio_data = api_client.create_speech(
        text=text,
        voice="alloy",
        instruct_text=instruct_text,
        prompt_wav=sample_audio_path,
        response_format="wav",
    )

    assert len(audio_data) > 0


# ============================================================================
# Additional CosyVoice3 Tests
# ============================================================================

def test_zero_shot_long_text(api_client: CosyVoiceAPIClient, sample_audio_path: str, sample_audio_text: str):
    """Test zero-shot mode with long text"""
    long_text = "这是一个长文本测试。" * 10
    prompt_text_with_system = f"You are a helpful assistant.<|endofprompt|>{sample_audio_text}"

    audio_data = api_client.create_speech(
        text=long_text,
        voice="alloy",
        prompt_text=prompt_text_with_system,
        prompt_wav=sample_audio_path,
        response_format="wav",
    )

    assert len(audio_data) > 0
    save_audio_output(audio_data, "zero_shot_long_text")


def test_cross_lingual_english(api_client: CosyVoiceAPIClient, sample_audio_path: str):
    """Test cross-lingual mode with English text"""
    audio_data = api_client.create_speech(
        text="Hello, this is a cross-lingual test.",
        voice="alloy",
        prompt_wav=sample_audio_path,
        response_format="wav",
    )

    assert len(audio_data) > 0
    save_audio_output(audio_data, "cross_lingual_english")


# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.skipif("os.getenv('SKIP_SFT_TESTS', 'false') == 'true'", reason="SFT mode not supported by model")
def test_invalid_voice(api_client: CosyVoiceAPIClient):
    """Test with invalid voice (should fall back to first available)"""
    audio_data = api_client.create_speech(
        text="测试无效音色",
        voice="invalid_voice",
        response_format="wav",
    )

    # Should still return audio using fallback voice
    assert len(audio_data) > 0


@pytest.mark.skipif("os.getenv('SKIP_SFT_TESTS', 'false') == 'true'", reason="SFT mode not supported by model")
def test_invalid_speed_low(api_client: CosyVoiceAPIClient):
    """Test with speed below minimum (should be clamped to 0.25)"""
    audio_data = api_client.create_speech(
        text="测试低语速",
        voice="alloy",
        speed=0.1,  # Will be clamped to 0.25
        response_format="wav",
    )

    assert len(audio_data) > 0


@pytest.mark.skipif("os.getenv('SKIP_SFT_TESTS', 'false') == 'true'", reason="SFT mode not supported by model")
def test_invalid_speed_high(api_client: CosyVoiceAPIClient):
    """Test with speed above maximum (should be clamped to 4.0)"""
    audio_data = api_client.create_speech(
        text="测试高语速",
        voice="alloy",
        speed=10.0,  # Will be clamped to 4.0
        response_format="wav",
    )

    assert len(audio_data) > 0


def test_empty_text(api_client: CosyVoiceAPIClient):
    """Test with empty text (should fail)"""
    with pytest.raises(requests.exceptions.HTTPError):
        api_client.create_speech(
            text="",
            voice="alloy",
            response_format="wav",
        )


# ============================================================================
# Long Text Tests
# ============================================================================

@pytest.mark.skipif("os.getenv('SKIP_SFT_TESTS', 'false') == 'true'", reason="SFT mode not supported by model")
def test_long_text(api_client: CosyVoiceAPIClient):
    """Test with long text"""
    long_text = "这是一个长文本测试。" * 20

    audio_data = api_client.create_speech(
        text=long_text,
        voice="alloy",
        response_format="wav",
    )

    assert len(audio_data) > 0


# ============================================================================
# Concurrent Request Tests
# ============================================================================

@pytest.mark.parametrize("count", [3, 5])
@pytest.mark.skipif("os.getenv('SKIP_SFT_TESTS', 'false') == 'true'", reason="SFT mode not supported by model")
def test_concurrent_requests(api_client: CosyVoiceAPIClient, count: int):
    """Test multiple concurrent requests"""
    import concurrent.futures

    def generate_speech(voice: str):
        return api_client.create_speech(
            text=f"并发测试 {voice}",
            voice=voice,
            response_format="wav",
        )

    voices = ["alloy", "echo", "nova", "fable", "onyx"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as executor:
        futures = [executor.submit(generate_speech, voices[i % len(voices)]) for i in range(count)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(results) == count
    for audio_data in results:
        assert len(audio_data) > 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])

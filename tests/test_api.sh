#!/bin/bash
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Test script for CosyVoice OpenAI Compatible API using curl
#
# Usage:
#   bash tests/test_api.sh

API_URL="${API_URL:-http://localhost:8000}"

echo "========================================"
echo "CosyVoice OpenAI API Test Script"
echo "========================================"
echo "API URL: $API_URL"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Function to run a test
run_test() {
    local test_name="$1"
    local command="$2"

    echo -n "Testing: $test_name ... "

    if eval "$command" > /dev/null 2>&1; then
        echo -e "${GREEN}PASSED${NC}"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}FAILED${NC}"
        ((TESTS_FAILED++))
        return 1
    fi
}

# Wait for server to be ready
echo "Waiting for server to be ready..."
for i in {1..30}; do
    if curl -s "$API_URL/health" | grep -q "healthy"; then
        echo -e "${GREEN}Server is ready!${NC}"
        echo ""
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}Server is not responding. Please start the server first.${NC}"
        exit 1
    fi
    sleep 2
done

# ============================================================================
# Health Check Tests
# ============================================================================

echo "=== Health Check Tests ==="

run_test "Health check" \
    "curl -s -f '$API_URL/health'"

run_test "List models" \
    "curl -s -f '$API_URL/v1/models'"

run_test "List voices" \
    "curl -s -f '$API_URL/v1/voices'"

echo ""

# ============================================================================
# SFT Mode Tests
# ============================================================================

echo "=== SFT Mode Tests ==="

run_test "SFT mode - Chinese text" \
    "curl -s -f -X POST '$API_URL/v1/audio/speech' \
    -H 'Content-Type: application/json' \
    -d '{\"model\":\"tts-1\",\"input\":\"你好，世界！\",\"voice\":\"alloy\",\"response_format\":\"wav\"}' \
    -o /tmp/test_sft_chinese.wav"

run_test "SFT mode - English text" \
    "curl -s -f -X POST '$API_URL/v1/audio/speech' \
    -H 'Content-Type: application/json' \
    -d '{\"model\":\"tts-1\",\"input\":\"Hello, world!\",\"voice\":\"echo\",\"response_format\":\"wav\"}' \
    -o /tmp/test_sft_english.wav"

run_test "SFT mode - Different voices" \
    "curl -s -f -X POST '$API_URL/v1/audio/speech' \
    -H 'Content-Type: application/json' \
    -d '{\"model\":\"tts-1\",\"input\":\"测试不同音色\",\"voice\":\"nova\",\"response_format\":\"wav\"}' \
    -o /tmp/test_sft_voice.wav"

run_test "SFT mode - Speed 0.5" \
    "curl -s -f -X POST '$API_URL/v1/audio/speech' \
    -H 'Content-Type: application/json' \
    -d '{\"model\":\"tts-1\",\"input\":\"测试语速\",\"voice\":\"alloy\",\"response_format\":\"wav\",\"speed\":0.5}' \
    -o /tmp/test_sft_speed.wav"

run_test "SFT mode - Speed 2.0" \
    "curl -s -f -X POST '$API_URL/v1/audio/speech' \
    -H 'Content-Type: application/json' \
    -d '{\"model\":\"tts-1\",\"input\":\"测试语速\",\"voice\":\"alloy\",\"response_format\":\"wav\",\"speed\":2.0}' \
    -o /tmp/test_sft_speed2.wav"

run_test "SFT mode - MP3 format" \
    "curl -s -f -X POST '$API_URL/v1/audio/speech' \
    -H 'Content-Type: application/json' \
    -d '{\"model\":\"tts-1\",\"input\":\"测试音频格式\",\"voice\":\"alloy\",\"response_format\":\"mp3\"}' \
    -o /tmp/test_sft_mp3.mp3"

echo ""

# ============================================================================
# Zero-shot Mode Tests
# ============================================================================

echo "=== Zero-shot Mode Tests ==="

# Check if reference audio exists
REFERENCE_AUDIO="./tests/fixtures/reference.wav"
if [ -f "$REFERENCE_AUDIO" ]; then
    # Encode audio to base64
    BASE64_AUDIO=$(base64 -i "$REFERENCE_AUDIO" | tr -d '\n')

    run_test "Zero-shot mode with base64" \
        "curl -s -f -X POST '$API_URL/v1/audio/speech' \
        -H 'Content-Type: application/json' \
        -d '{\"model\":\"tts-1\",\"input\":\"这是零样本模式测试\",\"prompt_text\":\"参考文本\",\"prompt_wav\":\"data:audio/wav;base64,$BASE64_AUDIO\",\"response_format\":\"wav\"}' \
        -o /tmp/test_zero_shot.wav"
else
    echo -e "${YELLOW}Skipping zero-shot tests (no reference audio found)${NC}"
fi

echo ""

# ============================================================================
# Instruct Mode Tests
# ============================================================================

echo "=== Instruct Mode Tests ==="

run_test "Instruct mode - Happy tone" \
    "curl -s -f -X POST '$API_URL/v1/audio/speech' \
    -H 'Content-Type: application/json' \
    -d '{\"model\":\"tts-1\",\"input\":\"测试指令模式\",\"voice\":\"alloy\",\"instruct_text\":\"用开心的语气说话\",\"response_format\":\"wav\"}' \
    -o /tmp/test_instruct.wav"

run_test "Instruct mode - Gentle tone" \
    "curl -s -f -X POST '$API_URL/v1/audio/speech' \
    -H 'Content-Type: application/json' \
    -d '{\"model\":\"tts-1\",\"input\":\"测试指令模式\",\"voice\":\"alloy\",\"instruct_text\":\"用温柔的语气说话\",\"response_format\":\"wav\"}' \
    -o /tmp/test_instruct2.wav"

echo ""

# ============================================================================
# Cross-lingual Mode Tests
# ============================================================================

echo "=== Cross-lingual Mode Tests ==="

if [ -f "$REFERENCE_AUDIO" ]; then
    run_test "Cross-lingual mode" \
        "curl -s -f -X POST '$API_URL/v1/audio/speech' \
        -H 'Content-Type: application/json' \
        -d '{\"model\":\"tts-1\",\"input\":\"跨语言模式测试\",\"prompt_wav\":\"data:audio/wav;base64,$BASE64_AUDIO\",\"response_format\":\"wav\"}' \
        -o /tmp/test_cross_lingual.wav"
else
    echo -e "${YELLOW}Skipping cross-lingual tests (no reference audio found)${NC}"
fi

echo ""

# ============================================================================
# Instruct2 Mode Tests
# ============================================================================

echo "=== Instruct2 Mode Tests ==="

if [ -f "$REFERENCE_AUDIO" ]; then
    run_test "Instruct2 mode" \
        "curl -s -f -X POST '$API_URL/v1/audio/speech' \
        -H 'Content-Type: application/json' \
        -d '{\"model\":\"tts-1\",\"input\":\"测试指令模式2\",\"instruct_text\":\"用激动的语气说话\",\"prompt_wav\":\"data:audio/wav;base64,$BASE64_AUDIO\",\"response_format\":\"wav\"}' \
        -o /tmp/test_instruct2_mode.wav"
else
    echo -e "${YELLOW}Skipping instruct2 tests (no reference audio found)${NC}"
fi

echo ""

# ============================================================================
# Error Handling Tests
# ============================================================================

echo "=== Error Handling Tests ==="

run_test "Invalid speed (should clamp)" \
    "curl -s -f -X POST '$API_URL/v1/audio/speech' \
    -H 'Content-Type: application/json' \
    -d '{\"model\":\"tts-1\",\"input\":\"测试\",\"voice\":\"alloy\",\"response_format\":\"wav\",\"speed\":10.0}' \
    -o /tmp/test_invalid_speed.wav"

echo ""

# ============================================================================
# Summary
# ============================================================================

echo "========================================"
echo "Test Summary"
echo "========================================"
echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
echo -e "${RED}Failed: $TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    echo ""
    echo "Generated audio files in /tmp/:"
    ls -lh /tmp/test_*.wav /tmp/test_*.mp3 2>/dev/null || echo "No audio files found"
    exit 0
else
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
fi

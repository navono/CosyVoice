#!/bin/bash
# Test script for /v1/voices API endpoint

echo "=========================================="
echo "Testing /v1/voices API endpoint"
echo "=========================================="
echo ""

# Test 1: Get available voices
echo "Test 1: GET /v1/voices"
echo "----------------------------------------"
curl -s http://localhost:8000/v1/voices | python3 -m json.tool
echo ""
echo ""

# Test 2: Health check
echo "Test 2: GET /health"
echo "----------------------------------------"
curl -s http://localhost:8000/health | python3 -m json.tool
echo ""
echo ""

# Test 3: Check VOICE_DIR in container
echo "Test 3: Check VOICE_DIR in container"
echo "----------------------------------------"
docker exec cosyvoice-openai bash -c 'echo "VOICE_DIR=$VOICE_DIR" && ls -la $VOICE_DIR 2>&1 || echo "Directory does not exist"'
echo ""
echo ""

echo "=========================================="
echo "Setup Instructions:"
echo "=========================================="
echo "1. Create voice directory on host:"
echo "   mkdir -p /mnt/e/OneDrive/data_sync/audio_samples"
echo ""
echo "2. Add voice files:"
echo "   cp your-voice.mp3 /mnt/e/OneDrive/data_sync/audio_samples/"
echo "   echo 'Reference text' > /mnt/e/OneDrive/data_sync/audio_samples/your-voice.txt"
echo ""
echo "3. Restart container:"
echo "   docker compose restart"
echo ""
echo "4. Test API:"
echo "   curl http://localhost:8000/v1/voices"
echo ""

#!/bin/bash

# Quick test script for individual audio files
# Usage: ./test_individual_files.sh

API_URL="http://localhost:8000/api/voice-detection"
API_KEY="demo_key_123"

echo "=================================================="
echo "Testing AI Voice Detection API"
echo "=================================================="
echo ""

# Test 1: English AI_GENERATED
echo "📝 Test 1/5: English_voice_AI_GENERATED.mp3"
echo "Expected: AI_GENERATED"
echo "Response:"
curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -H "x-api-key: $API_KEY" \
  -d "{
    \"language\": \"English\",
    \"audioFormat\": \"mp3\",
    \"audioBase64\": \"$(base64 -i English_voice_AI_GENERATED.mp3 | tr -d '\n')\"
  }" | python3 -m json.tool
echo ""
echo ""

# Test 2: Hindi HUMAN
echo "📝 Test 2/5: Hindi_Voice_HUMAN.mp3"
echo "Expected: HUMAN"
echo "Response:"
curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -H "x-api-key: $API_KEY" \
  -d "{
    \"language\": \"Hindi\",
    \"audioFormat\": \"mp3\",
    \"audioBase64\": \"$(base64 -i Hindi_Voice_HUMAN.mp3 | tr -d '\n')\"
  }" | python3 -m json.tool
echo ""
echo ""

# Test 3: Malayalam AI_GENERATED
echo "📝 Test 3/5: Malayalam_AI_GENERATED.mp3"
echo "Expected: AI_GENERATED"
echo "Response:"
curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -H "x-api-key: $API_KEY" \
  -d "{
    \"language\": \"Malayalam\",
    \"audioFormat\": \"mp3\",
    \"audioBase64\": \"$(base64 -i Malayalam_AI_GENERATED.mp3 | tr -d '\n')\"
  }" | python3 -m json.tool
echo ""
echo ""

# Test 4: Tamil HUMAN
echo "📝 Test 4/5: TAMIL_VOICE__HUMAN.mp3"
echo "Expected: HUMAN"
echo "Response:"
curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -H "x-api-key: $API_KEY" \
  -d "{
    \"language\": \"Tamil\",
    \"audioFormat\": \"mp3\",
    \"audioBase64\": \"$(base64 -i TAMIL_VOICE__HUMAN.mp3 | tr -d '\n')\"
  }" | python3 -m json.tool
echo ""
echo ""

# Test 5: Telugu AI_GENERATED
echo "📝 Test 5/5: Telugu_Voice_AI_GENERATED.mp3"
echo "Expected: AI_GENERATED"
echo "Response:"
curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -H "x-api-key: $API_KEY" \
  -d "{
    \"language\": \"Telugu\",
    \"audioFormat\": \"mp3\",
    \"audioBase64\": \"$(base64 -i Telugu_Voice_AI_GENERATED.mp3 | tr -d '\n')\"
  }" | python3 -m json.tool
echo ""
echo ""

echo "=================================================="
echo "✅ All tests completed!"
echo "=================================================="

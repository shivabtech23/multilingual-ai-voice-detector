#!/usr/bin/env python3
"""
API Evaluation Script - Based on Hackathon Evaluation Guide
Tests the API locally with the provided test audio files.
"""

import requests
import base64
import json
import os
from pathlib import Path

def evaluate_voice_detection_api(endpoint_url, api_key, test_files):
    """
    Evaluate voice detection API using the exact logic from the hackathon evaluator.
    """
    if not endpoint_url:
        print("❌ Error: Endpoint URL is required")
        return False

    if not test_files or len(test_files) == 0:
        print("❌ Error: No test files provided")
        return False

    total_files = len(test_files)
    score_per_file = 100 / total_files
    total_score = 0
    file_results = []

    print(f"\n{'='*60}")
    print(f"🚀 Starting Evaluation")
    print(f"{'='*60}")
    print(f"Endpoint: {endpoint_url}")
    print(f"Total Test Files: {total_files}")
    print(f"Score per File: {score_per_file:.2f}")
    print(f"{'='*60}\n")

    for idx, file_data in enumerate(test_files):
        language = file_data.get('language', 'English')
        file_path = file_data.get('file_path', '')
        expected_classification = file_data.get('expected_classification', '')

        print(f"📝 Test {idx + 1}/{total_files}: {Path(file_path).name}")

        if not file_path or not expected_classification:
            result = {
                'fileIndex': idx,
                'language': language,
                'expectedClassification': expected_classification,
                'status': 'skipped',
                'message': 'Missing file path or expected classification',
                'score': 0
            }
            file_results.append(result)
            print(f"   ⚠️  Skipped: Missing file path or expected classification\n")
            continue

        # Read and encode audio file
        try:
            with open(file_path, 'rb') as audio_file:
                audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')
        except Exception as e:
            result = {
                'fileIndex': idx,
                'language': language,
                'expectedClassification': expected_classification,
                'status': 'failed',
                'message': f'Failed to read audio file: {str(e)}',
                'score': 0
            }
            file_results.append(result)
            print(f"   ❌ Failed to read file: {str(e)}\n")
            continue

        # Prepare request
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': api_key
        }

        request_body = {
            'language': language,
            'audioFormat': 'mp3',
            'audioBase64': audio_base64
        }

        try:
            import time
            start_time = time.time()

            # Send request
            response = requests.post(endpoint_url, headers=headers, json=request_body, timeout=30)

            latency = time.time() - start_time

            # Check HTTP status code
            if response.status_code != 200:
                result = {
                    'fileIndex': idx,
                    'language': language,
                    'expectedClassification': expected_classification,
                    'status': 'failed',
                    'message': f'API returned status {response.status_code}',
                    'score': 0
                }
                file_results.append(result)
                print(f"   ❌ HTTP Status: {response.status_code}")
                print(f"   Response: {response.text}\n")
                continue

            response_data = response.json()

            # Validate response format
            response_status = response_data.get('status', '')
            response_classification = response_data.get('classification', '')
            confidence_score = response_data.get('confidenceScore', None)

            # Validate required fields
            if not response_status or not response_classification or confidence_score is None:
                result = {
                    'fileIndex': idx,
                    'language': language,
                    'expectedClassification': expected_classification,
                    'status': 'failed',
                    'message': 'Response missing required fields (status, classification, confidenceScore)',
                    'responseData': response_data,
                    'score': 0
                }
                file_results.append(result)
                print(f"   ❌ Missing required fields")
                print(f"   Response: {json.dumps(response_data, indent=2)}\n")
                continue

            # Validate status is success
            if response_status != 'success':
                result = {
                    'fileIndex': idx,
                    'language': language,
                    'expectedClassification': expected_classification,
                    'status': 'failed',
                    'message': f'API returned status: {response_status}',
                    'responseData': response_data,
                    'score': 0
                }
                file_results.append(result)
                print(f"   ❌ Status not 'success': {response_status}\n")
                continue

            # Validate confidenceScore range
            if not isinstance(confidence_score, (int, float)) or confidence_score < 0 or confidence_score > 1:
                result = {
                    'fileIndex': idx,
                    'language': language,
                    'expectedClassification': expected_classification,
                    'status': 'failed',
                    'message': f'Invalid confidenceScore: {confidence_score}. Must be between 0 and 1',
                    'responseData': response_data,
                    'score': 0
                }
                file_results.append(result)
                print(f"   ❌ Invalid confidence score: {confidence_score} (must be 0-1)\n")
                continue

            # Validate classification value
            valid_classifications = ['HUMAN', 'AI_GENERATED']
            if response_classification not in valid_classifications:
                result = {
                    'fileIndex': idx,
                    'language': language,
                    'expectedClassification': expected_classification,
                    'status': 'failed',
                    'message': f'Invalid classification: {response_classification}. Must be HUMAN or AI_GENERATED',
                    'responseData': response_data,
                    'score': 0
                }
                file_results.append(result)
                print(f"   ❌ Invalid classification: {response_classification}\n")
                continue

            # Score calculation
            file_score = 0
            if response_classification == expected_classification:
                # Scale score by confidence
                if confidence_score >= 0.8:
                    file_score = score_per_file
                    confidence_tier = "100%"
                elif confidence_score >= 0.6:
                    file_score = score_per_file * 0.75
                    confidence_tier = "75%"
                elif confidence_score >= 0.4:
                    file_score = score_per_file * 0.5
                    confidence_tier = "50%"
                else:
                    file_score = score_per_file * 0.25
                    confidence_tier = "25%"

                total_score += file_score

                result = {
                    'fileIndex': idx,
                    'language': language,
                    'expectedClassification': expected_classification,
                    'actualClassification': response_classification,
                    'confidenceScore': confidence_score,
                    'latency': round(latency, 2),
                    'status': 'success',
                    'matched': True,
                    'score': round(file_score, 2),
                    'responseData': response_data
                }
                file_results.append(result)
                print(f"   ✅ Classification: {response_classification} (Correct!)")
                print(f"   📊 Confidence: {confidence_score:.2f} → {confidence_tier} of points")
                print(f"   ⏱️  Latency: {latency:.2f}s")
                print(f"   🎯 Score: {file_score:.2f}/{score_per_file:.2f}\n")
            else:
                result = {
                    'fileIndex': idx,
                    'language': language,
                    'expectedClassification': expected_classification,
                    'actualClassification': response_classification,
                    'confidenceScore': confidence_score,
                    'latency': round(latency, 2),
                    'status': 'success',
                    'matched': False,
                    'score': 0,
                    'responseData': response_data
                }
                file_results.append(result)
                print(f"   ❌ Classification: {response_classification} (Expected: {expected_classification})")
                print(f"   📊 Confidence: {confidence_score:.2f}")
                print(f"   ⏱️  Latency: {latency:.2f}s")
                print(f"   🎯 Score: 0/{score_per_file:.2f}\n")

        except requests.exceptions.Timeout:
            result = {
                'fileIndex': idx,
                'language': language,
                'expectedClassification': expected_classification,
                'status': 'failed',
                'message': 'Request timed out (>30 seconds)',
                'score': 0
            }
            file_results.append(result)
            print(f"   ⏱️  Timeout: Request took longer than 30 seconds\n")
        except requests.exceptions.ConnectionError:
            result = {
                'fileIndex': idx,
                'language': language,
                'expectedClassification': expected_classification,
                'status': 'failed',
                'message': 'Connection error - unable to reach endpoint',
                'score': 0
            }
            file_results.append(result)
            print(f"   🔌 Connection Error: Unable to reach endpoint\n")
        except Exception as e:
            result = {
                'fileIndex': idx,
                'language': language,
                'expectedClassification': expected_classification,
                'status': 'failed',
                'message': str(e),
                'score': 0
            }
            file_results.append(result)
            print(f"   ❌ Error: {str(e)}\n")

    # Calculate final score
    final_score = round(total_score)

    # Print summary
    print(f"{'='*60}")
    print(f"📊 EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total Files Tested: {total_files}")
    print(f"Final Score: {final_score}/100")
    print(f"{'='*60}\n")

    # Print detailed results
    successful = sum(1 for r in file_results if r.get('matched', False))
    failed = sum(1 for r in file_results if r['status'] == 'failed')
    wrong = sum(1 for r in file_results if r['status'] == 'success' and not r.get('matched', False))

    print(f"✅ Correct Classifications: {successful}/{total_files}")
    print(f"❌ Wrong Classifications: {wrong}/{total_files}")
    print(f"⚠️  Failed/Errors: {failed}/{total_files}\n")

    # Save detailed results to JSON
    results_file = 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'finalScore': final_score,
            'totalFiles': total_files,
            'scorePerFile': round(score_per_file, 2),
            'successfulClassifications': successful,
            'wrongClassifications': wrong,
            'failedTests': failed,
            'fileResults': file_results
        }, f, indent=2)

    print(f"💾 Detailed results saved to: {results_file}\n")

    return final_score == 100


if __name__ == '__main__':
    # Local endpoint
    ENDPOINT_URL = 'http://localhost:8000/api/voice-detection'
    API_KEY = 'demo_key_123'

    # Test files based on provided MP3s
    TEST_FILES = [
        {
            'language': 'English',
            'file_path': 'English_voice_AI_GENERATED.mp3',
            'expected_classification': 'AI_GENERATED'
        },
        {
            'language': 'Hindi',
            'file_path': 'Hindi_Voice_HUMAN.mp3',
            'expected_classification': 'HUMAN'
        },
        {
            'language': 'Malayalam',
            'file_path': 'Malayalam_AI_GENERATED.mp3',
            'expected_classification': 'AI_GENERATED'
        },
        {
            'language': 'Tamil',
            'file_path': 'TAMIL_VOICE__HUMAN.mp3',
            'expected_classification': 'HUMAN'
        },
        {
            'language': 'Telugu',
            'file_path': 'Telugu_Voice_AI_GENERATED.mp3',
            'expected_classification': 'AI_GENERATED'
        }
    ]

    # Check if files exist
    missing_files = [f['file_path'] for f in TEST_FILES if not os.path.exists(f['file_path'])]
    if missing_files:
        print(f"❌ Error: Missing test files: {missing_files}")
        print("Please ensure all MP3 files are in the current directory.")
        exit(1)

    # Run evaluation
    success = evaluate_voice_detection_api(ENDPOINT_URL, API_KEY, TEST_FILES)

    if success:
        print("🎉 SUCCESS! Your API achieved 100/100 points!")
    else:
        print("⚠️  API did not achieve perfect score. Review results above.")

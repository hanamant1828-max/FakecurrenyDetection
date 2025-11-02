"""
Test script to verify predictions are independent for each image
This will test with alternating genuine and fake images
"""
import requests
import os
import time

test_images = [
    ('demo_images/genuine_sample_1.jpg', 'genuine'),
    ('demo_images/fake_sample_1.jpg', 'fake'),
    ('demo_images/genuine_sample_2.jpg', 'genuine'),
    ('demo_images/fake_sample_2.jpg', 'fake'),
    ('demo_images/genuine_sample_3.jpg', 'genuine'),
    ('demo_images/fake_sample_3.jpg', 'fake'),
]

API_URL = 'http://127.0.0.1:5000/predict'

print("="*80)
print("TESTING INDEPENDENT PREDICTIONS FOR INDIAN ₹500 NOTES")
print("="*80)
print("\nThis test verifies that each image is processed independently")
print("and predictions are NOT reused from previous images.\n")

results = []
all_passed = True

for image_path, expected_class in test_images:
    if not os.path.exists(image_path):
        print(f"⚠️  SKIPPED: {image_path} (file not found)")
        continue
    
    print(f"\n{'='*80}")
    print(f"Testing: {image_path}")
    print(f"Expected: {expected_class.upper()}")
    print(f"{'='*80}")
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(API_URL, files=files, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            prediction = data['prediction'].lower()
            confidence = data['confidence']
            fake_prob = data['probabilities']['fake']
            genuine_prob = data['probabilities']['genuine']
            
            print(f"✓ Response received")
            print(f"  Prediction: {prediction.upper()}")
            print(f"  Confidence: {confidence:.2f}%")
            print(f"  Probabilities - Fake: {fake_prob:.2f}%, Genuine: {genuine_prob:.2f}%")
            
            is_correct = prediction == expected_class
            if is_correct:
                print(f"✓ CORRECT PREDICTION")
            else:
                print(f"✗ INCORRECT PREDICTION (Expected: {expected_class}, Got: {prediction})")
                all_passed = False
            
            results.append({
                'image': os.path.basename(image_path),
                'expected': expected_class,
                'predicted': prediction,
                'correct': is_correct,
                'confidence': confidence
            })
        else:
            print(f"✗ ERROR: HTTP {response.status_code}")
            print(f"  {response.text}")
            all_passed = False
    
    except Exception as e:
        print(f"✗ EXCEPTION: {str(e)}")
        all_passed = False
    
    time.sleep(0.5)

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}\n")

if results:
    correct = sum(1 for r in results if r['correct'])
    total = len(results)
    accuracy = (correct / total) * 100
    
    print(f"Results: {correct}/{total} correct ({accuracy:.1f}% accuracy)")
    print("\nDetailed Results:")
    for r in results:
        status = "✓" if r['correct'] else "✗"
        print(f"  {status} {r['image']:25s} - Expected: {r['expected']:7s}, Got: {r['predicted']:7s}, Confidence: {r['confidence']:.1f}%")
    
    if all_passed and accuracy == 100:
        print("\n✓✓✓ ALL TESTS PASSED! ✓✓✓")
        print("Each image was processed independently with correct predictions!")
    else:
        print("\n⚠️ SOME TESTS FAILED")
        print("Check if predictions are independent and model is properly trained.")
else:
    print("⚠️ NO TESTS COMPLETED")

print(f"\n{'='*80}")

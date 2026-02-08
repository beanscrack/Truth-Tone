import requests
import json
import time
import os

def test_analyze_fake_file():
    url = "http://localhost:8000/analyze"
    fake_file_path = "frontend/public/test-audio-fake.mp3"
    
    if not os.path.exists(fake_file_path):
        print(f"File not found: {fake_file_path}")
        return

    print(f"Uploading {fake_file_path} to verify detection logic...")
    
    with open(fake_file_path, 'rb') as f:
        files = {'file': (os.path.basename(fake_file_path), f, 'audio/mpeg')}
        try:
            start_time = time.time()
            response = requests.post(url, files=files)
            response.raise_for_status()
            
            data = response.json()
            duration = time.time() - start_time
            
            print(f"\n✅ Analysis completed in {duration:.2f}s")
            
            print(f"Verdict: {data.get('verdict')}")
            print(f"Confidence Score: {data.get('confidence_score')}")
            print(f"Explanation: {data.get('explanation')}")
            
            if data.get('verdict') == 'FAKE' or data.get('verdict') == 'LIKELY FAKE':
                print("\nSUCCESS: Model correctly identified the audio as fake/manipulated.")
            else:
                print("\nFAILURE: Model still thinks this is real.")
                
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")

if __name__ == "__main__":
    test_analyze_fake_file()

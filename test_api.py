import requests

def test_analyze():
    url = "http://localhost:8000/analyze"
    # Create a dummy file
    files = {'file': ('test.wav', b'dummy content', 'audio/wav')}
    
    try:
        response = requests.post(url, files=files)
        response.raise_for_status()
        data = response.json()
        
        print("Analyze Response Status:", response.status_code)
        print("Keys in response:", data.keys())
        print("Verdict:", data.get('verdict'))
        print("Confidence:", data.get('confidence_score'))
        
        # Validate critical fields
        required_keys = ['verdict', 'confidence_score', 'explanation', 'analysis', 'timeline_data', 'audio_fingerprint']
        for key in required_keys:
            if key not in data:
                print(f"FAILED: Missing key {key}")
                return False
        
        print("Analyze Test PASSED")
        return True
    except Exception as e:
        print(f"Analyze Test FAILED: {e}")
        return False

if __name__ == "__main__":
    test_analyze()

import requests
import json

def test_api():
    base_url = "http://localhost:8000"
    
    print("Testing health endpoint...")
    health_response = requests.get(f"{base_url}/health")
    print(f"Health status: {health_response.status_code}")
    print(f"Health response: {health_response.json()}")
    
    print("\nTesting chat completions endpoint...")
    chat_data = {
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    chat_response = requests.post(
        f"{base_url}/v1/chat/completions",
        json=chat_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Chat status: {chat_response.status_code}")
    if chat_response.status_code == 200:
        response_data = chat_response.json()
        print(f"Response: {response_data['choices'][0]['message']['content']}")
        print(f"Usage: {response_data['usage']}")
    else:
        print(f"Error: {chat_response.text}")

if __name__ == "__main__":
    test_api()
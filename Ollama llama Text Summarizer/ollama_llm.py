import requests
ollama_url = "http://127.0.0.1:11434" # Replace your ollama url here 

def llm_response(prompt, model="llama3.1"):
  data = {
      "prompt": prompt,
      "model": model,
      "stream": False
  }

  response = requests.post(f"{ollama_url}/api/generate", json=data)

  if response.status_code == 200:
    return response.json().get("response", "No response Found")
  else:
    return f"Error: {response.status_code}, {response.text}"

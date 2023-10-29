import requests, json

prompt = '说：你好'
history = [

]

messages = []
for chat in history :
    messages.append({"role": "user", "content": chat[0] })
    messages.append({"role": "assistant", "content": chat[1] })
messages.append({"role": "user", "content": prompt })

request_data = {
    "model": "qwen",
    "messages": messages,
    "temperature": 0.7,
    "max_tokens": 1024,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0
}

headers = {"Content-Type": "application/json"}
api_url = "http://192.168.72.126:8000/v1/chat/completions"

# 发送请求并获取响应
response = requests.post(api_url, headers=headers, json=request_data)

result = json.loads(response.content)

response = result['choices'][0]['message']['content']

history.append((prompt, response))

print(response, history)


from openai import OpenAI
client = OpenAI(
    base_url="https://26c4-35-240-169-47.ngrok-free.app/v1",
    api_key="token-abc123",
)

completion = client.chat.completions.create(
  model="mistralai/Mistral-7B-Instruct-v0.2",
  messages=[
    {"role": "user", "content": "안녕!"}
  ]
)

print(completion.choices[0].message)
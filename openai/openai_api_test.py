import openai
import config

# openai.api_key = "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"
openai.api_key = API_KEY

def getMessage(prompt):
  completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", 
    messages=[{"role": "user", "content": prompt}]
  )
  #print(completion)
  for c in completion["choices"]:
    message = c["message"]["content"]
    #print(c["message"]["content"])
  return message

def main():
  while True:
    print("You: ", end="")
    prompt = input()
    if prompt == 'q' or prompt == 'Q':
      break
    print("ChatGPT: ", getMessage(prompt), end="\n\n")

if __name__ == "__main__":
  main()
  
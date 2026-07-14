import gradio as gr
import os
from huggingface_hub import InferenceClient
hf_token = os.environ.get("HF_TOKEN")

client = InferenceClient("Qwen/Qwen2.5-7B-Instruct", token = "hf_token")

def respond(message, history):
    
    messages = [{"role": "system", "content": "You are a friendly chatbot."}]
    
    if history:
        messages.extend(history)
        
    messages.append({"role": "user", "content": message})
    
    response = client.chat_completion(
        messages,
        max_tokens=20,
        temperature = .2
    )
    
    return response.choices[0].message.content.strip()

chatbot = gr.ChatInterface(respond)

chatbot.launch()
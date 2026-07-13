import gradio as gr
from huggingface_hub import InferenceClient

client = InferenceClient("Qwen/Qwen2.5-7B-Instruct")

def respond(message, history):
    
    messages = [{"role": "system", "content": "You are a friendly chatbot."}]
    
    if history:
        messages.extend(history)
        
    messages.append({"role": "user", "content": message})
    
    response = client.chat_completion(
        messages,
        max_tokens=100
    )
    
    return response.choices[0].message.content.strip()

chatbot = gr.ChatInterface(respond)

chatbot.launch()
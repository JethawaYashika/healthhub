import gradio as gr
import os
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

hf_token = os.environ.get("HF_TOKEN")
client = InferenceClient("Qwen/Qwen2.5-7B-Instruct", token=hf_token)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

KB_FILES = {
    "Gaining Weight": "knowledgeGain.txt",
    "Maintaining Weight": "knowledgeMaintain.txt",
    "Losing Weight": "knowledgeLose.txt",
}

def load_and_chunk(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    return chunks

knowledge_bases = {}
for goal_name, filepath in KB_FILES.items():
    chunks = load_and_chunk(filepath)
    embeddings = embedding_model.encode(chunks, convert_to_tensor=True)
    knowledge_bases[goal_name] = {
        "chunks": chunks,
        "embeddings": embeddings,
    }

def retrieve_context(query, goal_name, top_k=3):
    kb = knowledge_bases[goal_name]
    chunks = kb["chunks"]
    chunk_embeddings = kb["embeddings"]

    query_embedding = embedding_model.encode(query, convert_to_tensor=True)

    similarities = torch.nn.functional.cosine_similarity(
        query_embedding.unsqueeze(0), chunk_embeddings
    )

    top_results = torch.topk(similarities, k=min(top_k, len(chunks)))
    top_indices = top_results.indices.tolist()

    relevant_chunks = [chunks[i] for i in top_indices]
    return "\n\n".join(relevant_chunks)

def respond(message, history, goal_name):
    context = retrieve_context(message, goal_name)

    system_prompt = (
        "You are a friendly, knowledgeable nutrition and fitness chatbot. "
        "Use the following reference information to help answer the user's "
        "question. If the reference information doesn't contain a relevant "
        "answer, say so honestly rather than making something up.\n\n"
        f"Reference information:\n{context}"
    )

    messages = [{"role": "system", "content": system_prompt}]

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": message})

    response = client.chat_completion(
        messages,
        max_tokens=300,
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()

chatbot = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Dropdown(
            choices=["Gaining Weight", "Maintaining Weight", "Losing Weight"],
            value="Maintaining Weight",
            label="What's your goal?",
        )
    ],
    title="Health Hub",
    description="Select your goal below, then ask a question. The chatbot will pull relevant information from a knowledge base tailored to your goal.",
)

if __name__ == "__main__":
    chatbot.launch()
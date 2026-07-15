import gradio as gr
import os
import json
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import torch

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


PROFILE_FILE = "user_profiles.json"

def load_profiles():
    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_profiles(profiles):
    with open(PROFILE_FILE, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)

def history_to_messages(history):
    """Convert [[user_msg, bot_msg], ...] tuple history into the
    role-based dict format the chat completion API expects."""
    messages = []
    for user_msg, bot_msg in history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if bot_msg:
            messages.append({"role": "assistant", "content": bot_msg})
    return messages

def respond(message, history, profile):
    goal_name = profile.get("goal", "Maintaining Weight")
    context = retrieve_context(message, goal_name)

    system_prompt = (
        "You are a friendly, knowledgeable nutrition and fitness chatbot. "
        "Use the reference information below to help answer the user's "
        "question. If it doesn't contain a relevant answer, say so "
        "honestly rather than making something up.\n\n"
        f"User profile — name: {profile.get('name')}, age: {profile.get('age')}, "
        f"height: {profile.get('height')}, weight: {profile.get('weight')}, "
        f"gender: {profile.get('gender')}, goal: {goal_name}\n\n"
        f"Reference information:\n{context}"
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history_to_messages(history))
    messages.append({"role": "user", "content": message})

    response = client.chat_completion(
        messages,
        max_tokens=300,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

def user_send(message, history, profile):
    if not message.strip():
        return history, ""
    reply = respond(message, history, profile)
    history = history + [[message, reply]]
    return history, ""


def handle_login(profile: gr.OAuthProfile | None):
    if profile is None:
        # Not logged in yet
        return (
            gr.update(visible=False),  # intake_group
            gr.update(visible=False),  # chat_group
            "Please sign in above to get started.",
            {},                        # profile_state
            "",                        # username_state
            [],                        # chatbot_ui
        )

    username = profile.username
    profiles = load_profiles()

    if username in profiles:
        saved_profile = profiles[username]
        greeting = (
            f"Welcome back, {saved_profile['name']}! I remember your goal "
            f"is **{saved_profile['goal']}**. Ask me anything!"
        )
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            "",
            saved_profile,
            username,
            [[None, greeting]],
        )
    else:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            f"Hi {username}! Please tell us a bit about yourself to get started.",
            {},
            username,
            [],
        )

def save_new_profile(username_state, name, goal, height, weight, age, gender):
    profile = {
        "name": name.strip() if name else username_state,
        "goal": goal,
        "height": height,
        "weight": weight,
        "age": age,
        "gender": gender,
    }

    profiles = load_profiles()
    profiles[username_state] = profile
    save_profiles(profiles)

    greeting = (
        f"Thanks, {profile['name']}! I've saved your info and I'll remember "
        f"it next time you sign in. Your goal is **{goal}** — ask me anything!"
    )
    history = [[None, greeting]]

    return (
        gr.update(visible=False),  # intake_group
        gr.update(visible=True),   # chat_group
        profile,                   # profile_state
        history,                   # chatbot_ui
    )


with gr.Blocks(title="Nutrition & Fitness Chatbot") as demo:
    profile_state = gr.State({})
    username_state = gr.State("")

    gr.Markdown("# 🥗 Nutrition & Fitness Chatbot")
    gr.LoginButton()
    login_status = gr.Markdown("")

   
    with gr.Group(visible=False) as intake_group:
        gr.Markdown("### Tell us a bit about yourself")
        name_input = gr.Textbox(label="Name")
        goal_input = gr.Dropdown(
            choices=["Gaining Weight", "Maintaining Weight", "Losing Weight"],
            label="Health Goal",
        )
        height_input = gr.Textbox(label="Height", placeholder="e.g. 5'8\"")
        weight_input = gr.Textbox(label="Weight", placeholder="e.g. 160 lbs")
        age_input = gr.Number(label="Age")
        gender_input = gr.Dropdown(
            choices=["Female", "Male", "Non-binary", "Prefer not to say"],
            label="Gender",
        )
        intake_submit = gr.Button("Save & Start Chatting", variant="primary")


    with gr.Group(visible=False) as chat_group:
        chatbot_ui = gr.Chatbot(label="Chat")
        msg_input = gr.Textbox(label="Your message", placeholder="Ask me something...")
        send_btn = gr.Button("Send")
        
    demo.load(
        handle_login,
        inputs=None,
        outputs=[intake_group, chat_group, login_status, profile_state, username_state, chatbot_ui],
    )

    intake_submit.click(
        save_new_profile,
        inputs=[username_state, name_input, goal_input, height_input, weight_input, age_input, gender_input],
        outputs=[intake_group, chat_group, profile_state, chatbot_ui],
    )

    send_btn.click(
        user_send,
        inputs=[msg_input, chatbot_ui, profile_state],
        outputs=[chatbot_ui, msg_input],
    )
    msg_input.submit(
        user_send,
        inputs=[msg_input, chatbot_ui, profile_state],
        outputs=[chatbot_ui, msg_input],
    )

if __name__ == "__main__":
    demo.launch()
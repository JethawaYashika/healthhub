import gradio as gr
import os
import json
import hashlib
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import torch
#hello
hf_token = os.environ.get("HF_TOKEN")
client = InferenceClient("Qwen/Qwen2.5-7B-Instruct", token=hf_token)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

KB_FILES = {
    "Gaining Weight": "knowledgeGain.txt",
    "Maintaining Weight": "knowledgeMaintain.txt",
    "Losing Weight": "knowledgeLose.txt",
}

DIETARY_PREFERENCES = [
    "Vegan",
    "Vegetarian",
    "Pescatarian",
    "Keto",
    "Paleo",
    "Gluten-Free",
    "Dairy-Free",
    "Low-FODMAP",
    "Mediterranean",
    "Kosher",
]

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


USERS_FILE = "users.json"

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)

def hash_password(password):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def check_login(username, password):
    username = (username or "").strip()
    users = load_users()

    if not username or not password:
        return (
            gr.update(visible=True),   
            gr.update(visible=False),  
            gr.update(visible=False),  
            {},                        
            "Please enter both a username and password.",
            [],   
            gr.update(value="")
        )

    if username not in users:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            {},
            "No account found for that username. Click Sign Up to create one.",
            [],
            gr.update(value="")
        )

    if users[username]["password_hash"] != hash_password(password):
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            {},
            "Incorrect password. Please try again.",
            [],
            gr.update(value="")
        )

    profile = users[username]["profile"]
    display_name = profile.get("name",username)
    header_html = f""" <div style="display: flex; align-items: center; justify-content: flex-end; gap: 12px; padding: 10px; margin-bottom: 10px;"> <span style="font-size: 1.1rem; font-weight: bold; color: #1B5E5F; font-family: sans-serif;"> Hi, {display_name}! </span> <img src="https://api.dicebear.com/7.x/adventurer/svg?seed={display_name}" alt="Profile" style="width: 45px; height: 45px; border-radius: 50%; border: 2px solid #2A8A8C; background-color: white;" /> </div> """
    greeting = (
        f"Welcome back, {profile['name']}! I remember your goal is "
        f"**{profile['goal']}**. Ask me anything!"
    )
    history = [{"role": "assistant", "content": greeting}]
    return (
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        profile,
        "",
        history,
        gr.update(value=header_html)
    )

def start_signup():
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        "",
    )

def complete_signup(username, password, password_confirm, name, goal, height, weight, age, gender, dietary_prefs, other_prefs):
    users = load_users()
    
    username = (username or "").strip()
    
    if not username or not password or not password_confirm:
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            "Please fill in all fields including password.",
            {},
            [],
            gr.update(value="")
        )
    
    if password != password_confirm:
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            "Passwords do not match. Please try again.",
            {},
            [],
            gr.update(value="")
        )
    
    if username in users:
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            "That username is already taken. Please try a different one.",
            {},
            [],
            gr.update(value="")
        )
    
    combined_prefs = list(dietary_prefs) if dietary_prefs else []
    if other_prefs and other_prefs.strip():
        combined_prefs.append(f"Other: {other_prefs.strip()}")

    profile = {
        "name": name.strip() if name else username,
        "goal": goal,
        "height": height,
        "weight": weight,
        "age": age,
        "gender": gender,
        "dietary_preferences": combined_prefs,
    }

    users[username] = {
        "password_hash": hash_password(password),
        "profile": profile,
    }
    save_users(users)
    header_html = f""" <div style="display: flex; align-items: center; justify-content: flex-end; gap: 12px; padding: 10px; margin-bottom: 10px;"> <span style="font-size: 1.1rem; font-weight: bold; color: #1B5E5F; font-family: sans-serif;"> Hi, {display_name}! </span> <img src="https://api.dicebear.com/7.x/adventurer/svg?seed={display_name}" alt="Profile" style="width: 45px; height: 45px; border-radius: 50%; border: 2px solid #2A8A8C; background-color: white;" /> </div> """

    prefs_str = ", ".join(combined_prefs) if combined_prefs else "No specific preferences"
    greeting = (
        f"Thanks, {profile['name']}! Your account is set up and I'll remember "
        f"your info next time you log in. Your goal is **{goal}** and your dietary preferences "
        f"are: **{prefs_str}** — ask me anything!"
    )
    history = [{"role": "assistant", "content": greeting}]

    return (
        gr.update(visible=False),  
        gr.update(visible=True),   
        "",                        
        profile,                   
        history,  
        gr.update(value=header_html)
    )

def respond(message, history, profile):
    goal_name = profile.get("goal", "Maintaining Weight")
    dietary_prefs = profile.get("dietary_preferences", [])
    prefs_str = ", ".join(dietary_prefs) if dietary_prefs else "No specific dietary preferences"
    
    context = retrieve_context(message, goal_name)

    system_prompt = (
        "You are a friendly, knowledgeable nutrition and fitness chatbot. "
        "Use the reference information below to help answer the user's "
        "question. If it doesn't contain a relevant answer, say so "
        "honestly rather than making something up.\n\n"
        f"User profile — name: {profile.get('name')}, age: {profile.get('age')}, "
        f"height: {profile.get('height')}, weight: {profile.get('weight')}, "
        f"gender: {profile.get('gender')}, goal: {goal_name}, "
        f"dietary preferences: {prefs_str}\n\n"
        f"Reference information:\n{context}"
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    response = client.chat_completion(
        messages,
        max_tokens=1500,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

def user_send(message, history, profile):
    if not message.strip():
        return history, ""
    history = history + [{"role": "user", "content": message}]
    reply = respond(message, history[:-1], profile)
    history = history + [{"role": "assistant", "content": reply}]
    return history, ""
    
custom_css = """
@import url('https://googleapis.com');
* { 
    font-family: 'DM Serif Display', serif !important; 
}
body {
    background: #3F9E91;
    background: linear-gradient(135deg, rgba(63, 158, 145, 1) 23%, rgba(4, 194, 159, 1) 100%);
    margin: 0 !important;
    padding: 0 !important;
}
.gradio-container {
    background: transparent !important;
    max-width: 950px;
    margin: 0 auto !important;
    padding: 30px 20px !important;
}
/* Header title */
.gradio-container > h1 {
    text-align: center;
    color: white !important;
    font-size: 42px !important;
    font-weight: 600 !important;
    margin: 0 0 30px 0 !important;
    letter-spacing: 0.5px;
}
/* All groups get cream background */
.gradio-group {
    background-color: #D8DDD8 !important;
    border-color: #D8DDD8 !important;
}
.gradio-group .block, .dark .gradio-group, .dark .gradio-group .block {
    background-color: #D8DDD8 !important;
    border-color: #D8DDD8 !important;
}
/* Markdown headings */
.gradio-markdown h3 {
    color: #1B5E5F !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    margin: 0 0 20px 0 !important;
}
.gradio-markdown p {
    color: #2D2D2D !important;
    font-size: 15px !important;
    margin: 10px 0 !important;
}
/* Text inputs */
input[type=text], input[type=password], textarea {
    background: #1A1A1A !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
    font-size: 15px !important;
}
input::placeholder, textarea::placeholder, 
input::-webkit-input-placeholder, textarea::-webkit-input-placeholder {
    color: #A0A0A0 !important;
    opacity: 1 !important;
}
input:focus, textarea:focus {
    outline: none !important;
    border: 2px solid #2A8A8C !important;
    box-shadow: 0 0 8px rgba(42, 138, 140, 0.3) !important;
}
/* Number inputs */
input[type=number] {
    background: #1A1A1A !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
    font-size: 15px !important;
}
/* Dropdowns */
select {
    background: #1A1A1A !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
    font-size: 15px !important;
}
select option {
    background: #1A1A1A !important;
    color: white !important;
}
/* Checkbox styling */
.gradio-checkboxgroup {
    display: flex;
    flex-direction: column;
    gap: 10px !important;
}
.gradio-checkboxgroup label {
    background: white !important;
    color: #2D2D2D !important;
    border: 2px solid #C0C0BA !important;
    border-radius: 8px !important;
    padding: 12px 14px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    font-weight: 500 !important;
}
.gradio-checkboxgroup label:hover {
    border-color: #2A8A8C !important;
    background: #F8F8F8 !important;
}
.gradio-checkboxgroup input[type=checkbox]:checked + label {
    background: #2A8A8C !important;
    color: white !important;
    border-color: #2A8A8C !important;
}
/* Primary buttons */
.gradio-button {
    background: #2A8A8C !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 12px rgba(42, 138, 140, 0.3) !important;
}
.gradio-button:hover {
    background: #1B5E5F !important;
    box-shadow: 0 6px 16px rgba(27, 94, 95, 0.4) !important;
    transform: translateY(-1px) !important;
}
/* Secondary buttons */
button[variant=secondary] {
    background: white !important;
    color: #2A8A8C !important;
    border: 2px solid #2A8A8C !important;
}
button[variant=secondary]:hover {
    background: #2A8A8C !important;
    color: white !important;
}
/* Chat styling */
.gradio-chatbot {
    background: #F5F5F3 !important;
    border-radius: 14px !important;
    border: none !important;
    padding: 20px !important;
    min-height: 400px !important;
}
.gradio-chatbot > div {
    background: #F5F5F3 !important;
}
.gradio-chatbot .message-wrap {
    background: transparent !important;
}
.gradio-chatbot .message {
    border-radius: 10px !important;
    padding: 14px 16px !important;
    font-size: 15px !important;
    margin: 10px 0 !important;
    max-width: 85%;
    line-height: 1.5 !important;
}
.gradio-chatbot .message.user {
    background: #7FB3D5 !important;
    color: white !important;
    margin-left: auto !important;
    margin-right: 0 !important;
    text-align: left !important;
}
.gradio-chatbot .message.assistant {
    background: #B8DFE8 !important;
    color: #2D2D2D !important;
    margin-right: auto !important;
    margin-left: 0 !important;
    text-align: left !important;
}
/* Labels */
label {
    color: #2D2D2D !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    margin-bottom: 8px !important;
}
/* Hide the Chat label for chatbot */
.gradio-chatbot ~ label {
    display: none !important;
}
/* Row container */
.gradio-row {
    gap: 16px !important;
}
/* Status message */
.gradio-textbox, .gradio-number, .gradio-dropdown {
    margin-bottom: 16px !important;
}
@media (max-width: 600px) {
    .gradio-container {
        padding: 15px 10px !important;
    }
    .gradio-group {
        padding: 20px !important;
        border-radius: 16px !important;
    }
    .gradio-container > h1 {
        font-size: 32px !important;
    }
}
"""

with gr.Blocks(title="HealthHub", css=custom_css) as demo:
    profile_state = gr.State({})

    gr.Markdown("# 🥗 HealthHub")

    with gr.Group(visible=True) as auth_group:
        gr.Markdown("### Log in or create an account")
        username_input = gr.Textbox(label="Username")
        password_input = gr.Textbox(label="Password", type="password")
        with gr.Row():
            login_btn = gr.Button("Login", variant="primary")
            signup_btn = gr.Button("Sign Up")
        auth_status = gr.Markdown("")

    with gr.Group(visible=False) as signup_group:
        gr.Markdown("### Create Your Account")
        
        signup_username_input = gr.Textbox(label="Username")
        signup_password_input = gr.Textbox(label="Password", type="password")
        signup_password_confirm = gr.Textbox(label="Confirm Password", type="password")
        
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
        
        gr.Markdown("### Dietary Preferences (select all that apply)")
        dietary_input = gr.CheckboxGroup(
            choices=DIETARY_PREFERENCES,
            label="Which dietary preferences apply to you?",
        )
        other_dietary_input = gr.Textbox(
            label="Other (please specify if not listed above)",
            placeholder="e.g. Nut allergy, Halal, Etc.",
            interactive=True,
        )
        
        signup_status = gr.Markdown("")
        signup_submit = gr.Button("Create Account & Start Chatting", variant="primary")

    with gr.Group(visible=False) as chat_group:
        profile_header = gr.HTML(visible=True)
        chatbot_ui = gr.Chatbot(label="Chat")
        msg_input = gr.Textbox(label="Your message", placeholder="Ask me something...")
        send_btn = gr.Button("Send")

    login_btn.click(
        check_login,
        inputs=[username_input, password_input],
        outputs=[auth_group, signup_group, chat_group, profile_state, auth_status, chatbot_ui, profile_header]
    )

    signup_btn.click(
        start_signup,
        inputs=[],
        outputs=[auth_group, signup_group],
    ).then(
        lambda: ("", ""),
        outputs=[username_input, password_input]
    )

    signup_submit.click(
        complete_signup,
        inputs=[signup_username_input, signup_password_input, signup_password_confirm, name_input, goal_input, height_input, weight_input, age_input, gender_input, dietary_input, other_dietary_input],
        outputs=[signup_group, chat_group, signup_status, profile_state, chatbot_ui, profile_header]
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
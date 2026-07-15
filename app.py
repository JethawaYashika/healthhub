import gradio as gr
import os
import json
import hashlib
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import torch

hf_token = os.environ.get("HF_TOKEN")
client = InferenceClient("Qwen/Qwen2.5-7B-Instruct", token=hf_token)

# -------------------------------------------------------------------
# 1. Load the embedding model (used to turn text into vectors)
# -------------------------------------------------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------------------------------------------
# 2. Load and chunk each knowledge base file
# -------------------------------------------------------------------
KB_FILES = {
    "Gaining Weight": "knowledgeGain.txt",
    "Maintaining Weight": "knowledgeMaintain.txt",
    "Losing Weight": "knowledgeLose.txt",
}

# Dietary preferences
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

# -------------------------------------------------------------------
# 3. Retrieval function
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# 4. Self-contained user accounts (no Hugging Face login required)
#    Passwords are hashed before storage — a reasonable baseline for
#    a class project, though not the same as production-grade auth.
#
#    NOTE: On the free tier of Hugging Face Spaces, local files are
#    wiped when the Space restarts/sleeps unless you enable
#    "Persistent Storage" in your Space settings.
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# 5. Login / signup logic
# -------------------------------------------------------------------
def check_login(username, password):
    username = (username or "").strip()
    users = load_users()

    if not username or not password:
        return (
            gr.update(visible=True),   # auth_group
            gr.update(visible=False),  # signup_group
            gr.update(visible=False),  # chat_group
            {},                        # profile_state
            "Please enter both a username and password.",
            [],                        # chatbot_ui
        )

    if username not in users:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            {},
            "No account found for that username. Click Sign Up to create one.",
            [],
        )

    if users[username]["password_hash"] != hash_password(password):
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            {},
            "Incorrect password. Please try again.",
            [],
        )

    profile = users[username]["profile"]
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
    )

def start_signup(username, password):
    username = (username or "").strip()
    users = load_users()

    if not username or not password:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            "",
            "",
            "Please enter both a username and password.",
        )

    if username in users:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            "",
            "",
            "That username is already taken. Please log in instead.",
        )

    return (
        gr.update(visible=False),
        gr.update(visible=True),
        username,
        password,
        "",
    )

def complete_signup(username_state, password_state, name, goal, height, weight, age, gender, dietary_prefs, other_prefs):
    users = load_users()
    
    # Combine selected preferences with custom "other" preferences
    combined_prefs = list(dietary_prefs) if dietary_prefs else []
    if other_prefs and other_prefs.strip():
        combined_prefs.append(f"Other: {other_prefs.strip()}")

    profile = {
        "name": name.strip() if name else username_state,
        "goal": goal,
        "height": height,
        "weight": weight,
        "age": age,
        "gender": gender,
        "dietary_preferences": combined_prefs,
    }

    users[username_state] = {
        "password_hash": hash_password(password_state),
        "profile": profile,
    }
    save_users(users)

    prefs_str = ", ".join(combined_prefs) if combined_prefs else "No specific preferences"
    greeting = (
        f"Thanks, {profile['name']}! Your account is set up and I'll remember "
        f"your info next time you log in. Your goal is **{goal}** and your dietary preferences "
        f"are: **{prefs_str}** — ask me anything!"
    )
    history = [{"role": "assistant", "content": greeting}]

    return (
        gr.update(visible=False),  # signup_group
        gr.update(visible=True),   # chat_group
        profile,                   # profile_state
        history,                   # chatbot_ui
    )

# -------------------------------------------------------------------
# 6. Chat response function
# -------------------------------------------------------------------
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
        max_tokens=300,
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

# -------------------------------------------------------------------
# 7. Gradio interface
# -------------------------------------------------------------------
with gr.Blocks(title="Nutrition & Fitness Chatbot") as demo:
    profile_state = gr.State({})
    username_state = gr.State("")
    password_state = gr.State("")

    gr.Markdown("# 🥗 Nutrition & Fitness Chatbot")

    # --- Screen 1: login / signup ---
    with gr.Group(visible=True) as auth_group:
        gr.Markdown("### Log in or create an account")
        username_input = gr.Textbox(label="Username")
        password_input = gr.Textbox(label="Password", type="password")
        with gr.Row():
            login_btn = gr.Button("Login", variant="primary")
            signup_btn = gr.Button("Sign Up")
        auth_status = gr.Markdown("")

    # --- Screen 2: new account profile form ---
    with gr.Group(visible=False) as signup_group:
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
        
        # New dietary preferences section
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
        
        signup_submit = gr.Button("Create Account & Start Chatting", variant="primary")

    # --- Screen 3: chat ---
    with gr.Group(visible=False) as chat_group:
        chatbot_ui = gr.Chatbot(label="Chat")
        msg_input = gr.Textbox(label="Your message", placeholder="Ask me something...")
        send_btn = gr.Button("Send")

    # --- Wiring ---
    login_btn.click(
        check_login,
        inputs=[username_input, password_input],
        outputs=[auth_group, signup_group, chat_group, profile_state, auth_status, chatbot_ui],
    )

    signup_btn.click(
        start_signup,
        inputs=[username_input, password_input],
        outputs=[auth_group, signup_group, username_state, password_state, auth_status],
    )

    signup_submit.click(
        complete_signup,
        inputs=[username_state, password_state, name_input, goal_input, height_input, weight_input, age_input, gender_input, dietary_input, other_dietary_input],
        outputs=[signup_group, chat_group, profile_state, chatbot_ui],
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
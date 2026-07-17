import os
import streamlit as st
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer, util
import numpy as np
import time
import re
import transformers
import json
import uuid

st.set_page_config(page_title="Walmart Virtual Assistant", page_icon="🛒", layout="wide")

# Custom CSS for Professional Walmart Branding & ChatGPT style sidebar
st.markdown("""
<style>
    /* Main Background & Fonts */
    .stApp {
        font-family: 'Bogle', 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* Header styling */
    .walmart-header {
        background-color: #0071CE;
        padding: 1rem 2rem;
        color: white;
        border-radius: 8px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        display: flex;
        align-items: center;
        gap: 15px;
    }
    .walmart-header h1 {
        color: white;
        margin: 0;
        font-size: 1.8rem;
        font-weight: 600;
    }
    
    /* Chat Bubbles */
    .stChatMessage {
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
    }

    /* Make sidebar buttons look like ChatGPT history */
    [data-testid="stSidebar"] .stButton > button {
        border: none;
        background-color: transparent;
        text-align: left;
        justify-content: flex-start;
        padding-left: 10px;
        font-weight: normal;
        border-radius: 6px;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: rgba(150, 150, 150, 0.1);
    }
    /* New Chat button specifically */
    [data-testid="stSidebar"] .stButton:first-of-type > button {
        border: 1px solid rgba(150, 150, 150, 0.2);
        font-weight: bold;
    }
</style>

<div class="walmart-header">
    <div style="font-size: 2rem;">🛒</div>
    <div>
        <h1>Walmart Virtual Support Assistant</h1>
        <div style="font-size: 0.9rem; color: #FFC220; font-weight: bold;">Enterprise RAG System</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Constants
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GPT2_MODEL = "distilgpt2"
TOP_K = 3
CLARITY_THRESHOLD = 0.40
ACCEPT_SIM_THRESHOLD = 0.65
DEVICE = "cpu"
HISTORY_FILE = "chat_history.json"

transformers.logging.set_verbosity_error()

@st.cache_resource
def load_models():
    embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)
    tokenizer = GPT2Tokenizer.from_pretrained(GPT2_MODEL)
    model = GPT2LMHeadModel.from_pretrained(GPT2_MODEL).to(DEVICE)
    model.config.pad_token_id = model.config.eos_token_id
    return embed_model, tokenizer, model

@st.cache_data
def load_data():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, 'CustomerServiceDataSet.csv')
        data = pd.read_csv(csv_path, encoding='latin-1')
        return data
    except Exception as e:
        st.error(f"Error loading dataset `CustomerServiceDataSet.csv`. Please make sure it is in the same folder as this script. Error: {e}")
        return pd.DataFrame(columns=["category", "problem", "solution"])

embed_model, tokenizer, model = load_models()
data = load_data()

@st.cache_resource
def compute_problem_embeddings(_embed_model, problems):
    if not problems:
        return None
    return _embed_model.encode(problems, convert_to_tensor=True, show_progress_bar=False)

problem_texts = data["problem"].astype(str).tolist() if not data.empty else []
problem_embeddings = compute_problem_embeddings(embed_model, problem_texts)

# --- History Management ---
def load_history():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, HISTORY_FILE)
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_history(history):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, HISTORY_FILE)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = load_history()
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome to Walmart Support. How may I assist you today?"}]
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = []

def mem_add(role, text):
    emb = embed_model.encode(text, convert_to_tensor=True)
    st.session_state.conversation_memory.append({"role": role, "text": text, "embedding": emb, "time": time.time()})

def mem_retrieve_similar(query, top_k=3):
    if not st.session_state.conversation_memory:
        return []
    q_emb = embed_model.encode(query, convert_to_tensor=True)
    mem_embs = torch.stack([m["embedding"] for m in st.session_state.conversation_memory]).squeeze(1)
    sims = util.cos_sim(q_emb, mem_embs)[0]
    topk = torch.topk(sims, k=min(top_k, len(sims)))
    results = []
    for score, idx in zip(topk.values.tolist(), topk.indices.tolist()):
        results.append({"score": float(score), "entry": st.session_state.conversation_memory[idx]})
    return results

def retrieve_candidates(user_input, k=TOP_K):
    if data.empty or problem_embeddings is None:
        return []
    user_words = set(user_input.lower().split())
    exact_hits = []

    for idx, row in data.iterrows():
        problem_words = set(str(row["problem"]).lower().split())
        overlap = len(user_words & problem_words)
        if overlap >= 2:
            exact_hits.append({
                "score": 1.0,
                "idx": idx,
                "problem": str(row["problem"]),
                "solution": str(row["solution"])
            })

    if exact_hits:
        return exact_hits[:k]

    q_emb = embed_model.encode(user_input, convert_to_tensor=True)
    cos_scores = util.cos_sim(q_emb, problem_embeddings)[0]
    topk = torch.topk(cos_scores, k=min(k, len(cos_scores)))
    candidates = []
    for score, idx in zip(topk.values.tolist(), topk.indices.tolist()):
        sol = str(data["solution"].iloc[idx])
        prob = str(data["problem"].iloc[idx])
        candidates.append({
            "score": float(score),
            "idx": int(idx),
            "problem": prob,
            "solution": sol
        })
    return candidates

def gpt2_generate_text(prompt, max_new_tokens=60, stop_tokens=None):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.2,
        top_k=0,
        top_p=1.0,
        pad_token_id=model.config.eos_token_id,
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded[len(prompt):].strip()

def gpt2_accept_reject_prompt(user_input, candidate):
    prompt = (
        "You are a terse verifier. Given a customer's issue and a proposed solution, "
        "respond in EXACT format:\n\n"
        "DECISION: <ACCEPT or REJECT>\n"
        "REASON: <one short sentence>\n\n"
        f"Customer issue: {user_input}\n\nProposed solution: {candidate}\n\n"
        "DECISION:"
    )
    raw = gpt2_generate_text(prompt, max_new_tokens=40)
    out = raw.strip().splitlines()
    if not out:
        return {"decision": "REJECT", "reason": "No response from verifier."}
    first_line = out[0].upper()
    decision = "ACCEPT" if "ACCEPT" in first_line else ("REJECT" if "REJECT" in first_line else None)
    reason = ""
    if decision is None:
        if "YES" in first_line or "OK" in first_line:
            decision = "ACCEPT"
        else:
            decision = "REJECT"
    if len(out) >= 2:
        reason = out[1].strip()
    else:
        if "REASON:" in raw:
            reason = raw.split("REASON:")[-1].strip().split("\n")[0]
    return {"decision": decision, "reason": reason}

def agent_r_retrieve_and_rephrase(user_input, k=TOP_K, rephrase_with_gpt2=True):
    candidates = retrieve_candidates(user_input, k=k)
    mem_hits = mem_retrieve_similar(user_input, top_k=2)
    mem_context = " ".join([h["entry"]["text"] for h in mem_hits]) if mem_hits else ""
    enriched_candidates = []
    for c in candidates:
        solution = c["solution"]
        if rephrase_with_gpt2 and c["score"] >= CLARITY_THRESHOLD:
            prompt = (
                "Rephrase this solution clearly and professionally, keep meaning exactly, "
                "in one or two short sentences:\n\n"
                f"{solution}\n\nRephrased:"
            )
            try:
                rephrased = gpt2_generate_text(prompt, max_new_tokens=80)
                if len(rephrased.split()) >= 3:
                    lines = re.split(r'[.!?]', rephrased)
                    cleaned = ". ".join([ln.strip() for ln in lines if ln.strip()][:2]).strip() + "."
                    solution_text = cleaned
                else:
                    solution_text = solution
            except Exception:
                solution_text = solution
        else:
            solution_text = solution

        enriched_candidates.append({
            "idx": c["idx"],
            "problem": c["problem"],
            "raw_solution": c["solution"],
            "solution": solution_text,
            "retrieval_score": c["score"],
            "mem_context": mem_context
        })
    return enriched_candidates

def agent_v_verify_and_plan(user_input, candidate):
    q_emb = embed_model.encode(user_input, convert_to_tensor=True)
    cand_emb = embed_model.encode(candidate["solution"], convert_to_tensor=True)
    sim_q_cand = float(util.cos_sim(q_emb, cand_emb)[0])

    ds_emb = embed_model.encode(candidate["raw_solution"], convert_to_tensor=True)
    sim_cand_ds = float(util.cos_sim(cand_emb, ds_emb)[0])

    heuristic_score = (candidate["retrieval_score"] * 0.6) + (sim_q_cand * 0.3) + (sim_cand_ds * 0.1)

    try:
        vr = gpt2_accept_reject_prompt(user_input, candidate["solution"])
        decision = vr["decision"]
        reason = vr["reason"]
    except Exception as e:
        decision = "REJECT"
        reason = f"Verifier error: {e}"

    accepted_by_gpt2 = True if decision == "ACCEPT" else False
    combined_confidence = float(np.clip(heuristic_score * 1.0 + (0.15 if accepted_by_gpt2 else -0.15), 0.0, 1.0))

    plan = None
    clean_sol = candidate["solution"]
    clean_sol = "\n".join(
      ln for ln in clean_sol.split("\n")
      if "rephrased" not in ln.lower()
    )
    candidate["clean_solution"] = clean_sol.strip()

    acceptance = False
    if combined_confidence >= ACCEPT_SIM_THRESHOLD:
        acceptance = True
    elif accepted_by_gpt2 and combined_confidence >= 0.45:
        acceptance = True
    else:
        acceptance = False

    return {
        "acceptance": acceptance,
        "score": combined_confidence,
        "reason": reason,
        "sim_q_cand": sim_q_cand,
        "sim_cand_ds": sim_cand_ds,
        "plan": plan
    }

def add_sympathy(solution_text, user_input):
    prompt = (
        "You are a polite customer service agent. "
        "Add a short empathetic remark to this response without changing the solution meaning. "
        "Keep it 1 sentence max.\n\n"
        f"Customer said: {user_input}\n"
        f"Original solution: {solution_text}\n\n"
        "Empathetic reply:"
    )
    try:
        empathetic_text = gpt2_generate_text(prompt, max_new_tokens=50)
        empathetic_text = empathetic_text.split("\n")[0].strip()
        if empathetic_text:
            return empathetic_text
        else:
            return solution_text
    except Exception:
        return solution_text

def dialogue_manager(user_input):
    mem_add("user", user_input)
    candidates = agent_r_retrieve_and_rephrase(user_input, k=TOP_K, rephrase_with_gpt2=True)
    if not candidates:
        return {"reply": "I couldn't find any information in the dataset. Please ensure the dataset is loaded."}

    evaluated = []
    for cand in candidates:
        v = agent_v_verify_and_plan(user_input, cand)
        evaluated.append({"candidate": cand, "verif": v})

    evaluated = sorted(evaluated, key=lambda x: (x["verif"]["acceptance"], x["verif"]["score"]), reverse=True)
    best = evaluated[0]

    if not best["verif"]["acceptance"] or best["verif"]["score"] < 0.40:
        clarifying_question = "I couldn't find a confident match. Could you give one short extra detail (order #, date, or exact error message)?"
        mem_add("bot", clarifying_question)
        return {"reply": clarifying_question, "awaits_clarification": True, "candidates": evaluated}

    cand = best["candidate"]
    ver = best["verif"]
    reply_parts = []
    solution_text = cand.get("clean_solution", cand["solution"])
    
    negative_keywords = ["issue", "problem", "trouble", "error", "delay", "failed", "cannot", "unable",]
    if any(word in user_input.lower() for word in negative_keywords):
      empathetic_text = add_sympathy(solution_text, user_input)
      solution_text = f"{empathetic_text} {solution_text}"

    reply_parts.append(solution_text)

    if ver["plan"]:
        reply_parts.append("\nSuggested steps:\n" + "\n".join([f"{i+1}. {s}" for i, s in enumerate(ver["plan"])]))

    reply_parts.append(f"\n*(Confidence: {ver['score']:.2f})*")
    final_reply = "\n\n".join(reply_parts)
    mem_add("bot", final_reply)
    return {"reply": final_reply, "awaits_clarification": False, "candidates": evaluated}

# UI Rendering and Sidebar Layout
with st.sidebar:
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.current_session_id = str(uuid.uuid4())
        st.session_state.messages = [{"role": "assistant", "content": "Welcome to Walmart Support. How may I assist you today?"}]
        st.session_state.conversation_memory = []
        st.rerun()
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.85rem; font-weight:bold; color:#888;'>Chat History</p>", unsafe_allow_html=True)
    
    # Render saved history
    if not st.session_state.history:
        st.caption("No past conversations.")
    else:
        # Sort by timestamp (newest first) or just reverse iteration
        for session_id in reversed(list(st.session_state.history.keys())):
            session_data = st.session_state.history[session_id]
            title = session_data.get("title", "New Conversation")
            short_title = title[:28] + "..." if len(title) > 28 else title
            
            # Highlight current session visually
            is_current = (session_id == st.session_state.current_session_id)
            label = f"{'💬' if not is_current else '🔹'} {short_title}"
            
            if st.button(label, key=session_id, use_container_width=True):
                st.session_state.current_session_id = session_id
                st.session_state.messages = session_data["messages"]
                # Clear short term memory on history switch to save encoding overhead
                st.session_state.conversation_memory = [] 
                st.rerun()
            
    # Bottom spacer and user profile mock
    st.markdown("<br>" * 10 + "<hr>", unsafe_allow_html=True)
    st.markdown("👤 **Current User**<br><span style='font-size:0.8rem; color:gray;'>Enterprise Account</span>", unsafe_allow_html=True)

chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        avatar = "🛒" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

# chat_input must be outside of columns
if prompt := st.chat_input("Type your question here..."):
    # Assign a title to the session if this is the very first user message
    user_msg_count = sum(1 for m in st.session_state.messages if m["role"] == "user")
    if user_msg_count == 0:
        st.session_state.history[st.session_state.current_session_id] = {
            "title": prompt,
            "messages": []
        }
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with chat_container:
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
            
        with st.chat_message("assistant", avatar="🛒"):
            with st.spinner("Processing request..."):
                response = dialogue_manager(prompt)
                reply = response.get("reply", "We are currently experiencing technical difficulties. Please try again.")
                st.markdown(reply)
                
                # Show debug info about AI Reasoning
                if response.get("candidates"):
                    with st.expander("🔍 System Diagnostics & Retrieval Logs"):
                        for i, cand_data in enumerate(response["candidates"]):
                            cand = cand_data["candidate"]
                            ver = cand_data["verif"]
                            st.markdown(f"**Retrieval Candidate {i+1}:** `{cand['problem']}`")
                            status_icon = "✅" if ver['acceptance'] else "❌"
                            st.markdown(f"> **Status:** {status_icon} {ver['acceptance']} | **Confidence:** `{ver['score']:.2f}`\n> \n> **Decision Context:** {ver['reason']}")
                            if i < len(response["candidates"]) - 1:
                                st.divider()
                                
    st.session_state.messages.append({"role": "assistant", "content": reply})
    
    # Save to history
    st.session_state.history[st.session_state.current_session_id] = {
        "title": st.session_state.history[st.session_state.current_session_id]["title"] if st.session_state.current_session_id in st.session_state.history else prompt,
        "messages": st.session_state.messages
    }
    save_history(st.session_state.history)

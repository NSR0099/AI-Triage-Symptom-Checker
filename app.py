import json
import os
import pickle
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

# --------------------------
# Gemini API setup (OpenAI-compatible client)
API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if not API_KEY:
    API_KEY = "AIzaSyBCeqUH0aOHQqh2pDqdHlasQJjoWJQ1ZxI"  # fallback Gemini key
client = OpenAI(
    api_key=API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# --------------------------
# Load and process data
@st.cache_resource
def load_db():
    with open("ai_triage_10000_extended.json", "r", encoding="utf-8") as f:
        return json.load(f)["diseases"]

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def build_or_load_symptom_map(_db):
    """Load cached symptom map if available, else build and save it."""
    if os.path.exists("symptom_map.pkl"):
        st.info("✅ Loaded cached symptom map from disk.")
        with open("symptom_map.pkl", "rb") as f:
            return pickle.load(f)

    st.warning("⚙️ Building symptom map... this may take a few minutes.")
    _model = load_model()
    symptom_map = []
    total = len(_db)
    pb = st.progress(0)
    status = st.empty()

    for i, disease in enumerate(_db):
        status.text(f"🔄 Processing {i+1}/{total}: {disease['Name']}")
        for sym in disease["Symptoms"]:
            emb = _model.encode(sym["Name"], convert_to_tensor=True)
            symptom_map.append(
                (disease["Disease_ID"], disease["Name"], sym["Name"], sym["Severity"], emb)
            )
        pb.progress((i + 1) / total)

    status.empty()
    pb.empty()

    # Save for future runs
    with open("symptom_map.pkl", "wb") as f:
        pickle.dump(symptom_map, f)

    st.success("💾 Symptom map built and saved as symptom_map.pkl")
    return symptom_map

# --------------------------
# Initialize
db = load_db()
model = load_model()
symptom_map = build_or_load_symptom_map(db)

# --------------------------
# Symptom checker
def symptom_checker(user_symptoms, age=None, sex=None, top_n=3):
    scores = {}
    embs = [model.encode(s, convert_to_tensor=True) for s in user_symptoms]
    for did, dname, sname, sev, emb in symptom_map:
        for ue in embs:
            sim = util.cos_sim(ue, emb).item()
            if sim > 0.6:
                entry = scores.setdefault(did, {"disease": dname, "score": 0, "matches": []})
                entry["score"] += sim
                entry["matches"].append(sname)
    ranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    for r in ranked:
        r["confidence"] = min(100, int(r["score"] * 20))
    return ranked[:top_n]

# --------------------------
# Follow-up with Gemini
def generate_followup(user_symptoms, candidate_conditions, demographics, asked_questions):
    prompt = f"""
    The patient reported symptoms: {', '.join(user_symptoms)}.
    Candidate conditions: {', '.join([c['disease'] for c in candidate_conditions])}.
    Current demographics: {json.dumps(demographics)}.
    Already asked: {asked_questions}.

    Suggest ONE new follow-up question that has NOT been asked before.
    It can be about:
    - Additional symptoms
    - Demographics (age, sex, pregnancy)
    - Lifestyle (smoking, alcohol, obesity)

    Keep it short and specific.
    """
    resp = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": "You are a helpful medical triage assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return resp.choices[0].message.content.strip()

# --------------------------
# Summary with Gemini
def generate_summary(results, demographics, symptoms):
    prompt = f"""
    Patient demographics: {json.dumps(demographics)}.
    Reported symptoms: {', '.join(symptoms)}.
    Candidate conditions with confidence: {json.dumps(results)}.

    Provide a short triage summary:
    - The most likely condition
    - Why it matches the symptoms
    - Urgency (routine vs urgent)
    - Disclaimer: This is not medical advice
    """
    resp = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": "You are a concise medical triage assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return resp.choices[0].message.content.strip()

# --------------------------
# Chatbot UI
st.title("🩺 AI Triage Chatbot with Gemini")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "symptoms" not in st.session_state:
    st.session_state.symptoms = []
if "demographics" not in st.session_state:
    st.session_state.demographics = {
        "age": None,
        "sex": None,
        "pregnant": None,
        "smoking": None,
        "alcohol": None,
        "obesity": None,
    }
if "awaiting_followup" not in st.session_state:
    st.session_state.awaiting_followup = False
if "last_conditions" not in st.session_state:
    st.session_state.last_conditions = []
if "asked_questions" not in st.session_state:
    st.session_state.asked_questions = []

CONFIDENCE_THRESHOLD = 70

# Greeting + first question
if not st.session_state.messages:
    greeting = "👋 Hello! I'm your AI health assistant."
    first_q = "👉 What symptoms or health concerns are you experiencing right now?"
    st.session_state.messages.append({"role": "assistant", "content": f"{greeting}\n\n{first_q}"})

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Handle user input
if user_input := st.chat_input("Type your response..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    bot_reply = ""

    if st.session_state.awaiting_followup:
        ans = user_input.lower().strip()

        # Capture demographics
        if ans.isdigit() and st.session_state.demographics["age"] is None:
            st.session_state.demographics["age"] = int(ans)
        elif ans in ["male", "female", "other"] and st.session_state.demographics["sex"] is None:
            st.session_state.demographics["sex"] = ans.capitalize()
        elif "preg" in ans and st.session_state.demographics["pregnant"] is None:
            st.session_state.demographics["pregnant"] = "yes" in ans
        elif "smok" in ans and st.session_state.demographics["smoking"] is None:
            st.session_state.demographics["smoking"] = "yes" in ans
        elif "alcohol" in ans and st.session_state.demographics["alcohol"] is None:
            st.session_state.demographics["alcohol"] = "yes" in ans
        elif "obese" in ans or "overweight" in ans:
            st.session_state.demographics["obesity"] = True
        else:
            st.session_state.symptoms.append(ans)

        results = symptom_checker(st.session_state.symptoms)
        bot_reply = "Thanks! I updated my analysis.\n\n"

        if not results:
            bot_reply += "❌ No matching conditions found."
        else:
            top_conf = results[0]["confidence"]
            bot_reply += "Here are updated possible conditions:\n"
            for r in results:
                bot_reply += f"- 🦠 **{r['disease']}** ({r['confidence']}% confidence, matches: {', '.join(r['matches'])})\n"

            if top_conf >= CONFIDENCE_THRESHOLD:
                summary = generate_summary(results, st.session_state.demographics, st.session_state.symptoms)
                bot_reply += f"\n✅ Confidence reached ({top_conf}%).\n\n{summary}"
                st.session_state.awaiting_followup = False
            else:
                q = generate_followup(st.session_state.symptoms, results, st.session_state.demographics, st.session_state.asked_questions)
                if q not in st.session_state.asked_questions:
                    st.session_state.asked_questions.append(q)
                bot_reply += f"\n👉 {q}"
                st.session_state.awaiting_followup = True
                st.session_state.last_conditions = results

    else:
        st.session_state.symptoms.extend([s.strip() for s in user_input.split(",") if s.strip()])
        results = symptom_checker(st.session_state.symptoms)

        if not results:
            bot_reply = "❌ No matching conditions found. Try adding more details."
        else:
            top_conf = results[0]["confidence"]
            bot_reply = "Based on your input, here are possible conditions:\n"
            for r in results:
                bot_reply += f"- 🦠 **{r['disease']}** ({r['confidence']}% confidence, matches: {', '.join(r['matches'])})\n"

            if top_conf >= CONFIDENCE_THRESHOLD:
                summary = generate_summary(results, st.session_state.demographics, st.session_state.symptoms)
                bot_reply += f"\n✅ Confidence reached ({top_conf}%).\n\n{summary}"
                st.session_state.awaiting_followup = False
            else:
                q = generate_followup(st.session_state.symptoms, results, st.session_state.demographics, st.session_state.asked_questions)
                if q not in st.session_state.asked_questions:
                    st.session_state.asked_questions.append(q)
                bot_reply += f"\n👉 {q}"
                st.session_state.awaiting_followup = True
                st.session_state.last_conditions = results

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.write(bot_reply)

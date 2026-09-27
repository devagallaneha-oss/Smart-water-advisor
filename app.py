from fastapi import FastAPI, UploadFile, File
import pandas as pd
import uuid
import ollama

app = FastAPI(title="Smart Water Usage Advisor API")

# In-memory session storage
SESSIONS = {}

# ---------------------------
# Utility Functions
# ---------------------------

def analyze_data(df):
    daily = df.groupby("date")["liters"].sum().to_dict()
    activity = df.groupby("activity")["liters"].sum().to_dict()
    return daily, activity

def detect_spikes(daily):
    avg = sum(daily.values()) / len(daily)
    return {d: v for d, v in daily.items() if v > avg * 1.3}

def rule_based_advice(activity):
    tips = []

    if activity.get("Shower", 0) > 120:
        tips.append("Reduce shower time by 2 minutes.")
    if activity.get("Laundry", 0) > 120:
        tips.append("Use laundry machines only on full load.")
    if activity.get("Kitchen", 0) > 70:
        tips.append("Avoid running tap continuously while washing dishes.")

    if not tips:
        tips.append("Your water usage is already efficient.")

    return tips

def ollama_advice(activity):
    prompt = f"""
    Household water usage data: {activity}
    Give 3 short, practical water-saving suggestions.
    """
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

# ---------------------------
# API ENDPOINTS
# ---------------------------

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload CSV → returns session_id
    """
    df = pd.read_csv(file.file)
    session_id = str(uuid.uuid4())

    daily, activity = analyze_data(df)

    SESSIONS[session_id] = {
        "daily": daily,
        "activity": activity
    }

    return {
        "message": "Data uploaded successfully",
        "session_id": session_id
    }

@app.get("/dashboard/{session_id}")
def dashboard(session_id: str):
    """
    Returns analytics summary
    """
    data = SESSIONS.get(session_id)
    if not data:
        return {"error": "Invalid session ID"}

    spikes = detect_spikes(data["daily"])

    return {
        "daily_usage": data["daily"],
        "activity_usage": data["activity"],
        "high_usage_days": spikes
    }

@app.get("/advisor/rules/{session_id}")
def rule_advisor(session_id: str):
    """
    Rule-based water saving advice
    """
    data = SESSIONS.get(session_id)
    if not data:
        return {"error": "Invalid session ID"}

    tips = rule_based_advice(data["activity"])
    return {"tips": tips}

@app.get("/advisor/ai/{session_id}")
def ai_advisor(session_id: str):
    """
    AI-powered advice using Ollama
    """
    data = SESSIONS.get(session_id)
    if not data:
        return {"error": "Invalid session ID"}

    advice = ollama_advice(data["activity"])
    return {"ai_advice": advice}

@app.get("/weekly-plan")
def weekly_plan():
    """
    Static weekly water-saving plan
    """
    return {
        "plan": [
            "Limit showers to 5 minutes",
            "Do laundry twice a week only",
            "Reuse RO water for plants",
            "Turn off tap while brushing"
        ],
        "estimated_savings": "25–30%"
    }
import os
import json
import sqlite3
from fastapi import FastAPI, Request, Response, Query
import httpx
from groq import Groq
from twilio.twiml.voice_response import VoiceResponse, Gather

app = FastAPI()

# --- ১. ডেটাবেস সেটআপ (SQLite) ---
def init_db():
    conn = sqlite3.connect("appointments.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS bookings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT,
            phone TEXT,
            doctor TEXT,
            date TEXT,
            time_slot TEXT,
            channel TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ডাক্তার ও সময়সূচি
DOCTORS = {
    "ডাঃ সেন (জেনারেল ফিজিশিয়ান)": ["সকাল ১০:০০", "সকাল ১১:০০", "বিকাল ৫:০০"],
    "ডাঃ রায় (দন্ত চিকিৎসক)": ["দুপুর ১২:০০", "বিকাল ৪:০০", "সন্ধ্যা ৬:০০"],
    "ডাঃ দাস (হৃদরোগ বিশেষজ্ঞ)": ["সকাল ১১:৩০", "দুপুর ৩:০০"]
}

# --- ২. বুকিং ও স্লট চেক ফাংশন ---
def check_available_slots(doctor_name: str, date: str) -> str:
    matched_doc = next((d for d in DOCTORS if doctor_name.lower() in d.lower()), None)
    if not matched_doc:
        return f"এই ডাক্তারকে পাওয়া যায়নি। আমাদের ডাক্তারগণ: {', '.join(DOCTORS.keys())}"

    conn = sqlite3.connect("appointments.db")
    c = conn.cursor()
    c.execute("SELECT time_slot FROM bookings WHERE doctor = ? AND date = ?", (matched_doc, date))
    booked = {row[0] for row in c.fetchall()}
    conn.close()

    free_slots = [s for s in DOCTORS[matched_doc] if s not in booked]
    if not free_slots:
        return f"{date} তারিখে {matched_doc}-এর কোনো স্লট খালি নেই।"
    return f"{date} তারিখে {matched_doc}-এর ফাঁকা সময়: {', '.join(free_slots)}"


def book_appointment(patient_name: str, phone: str, doctor_name: str, date: str, time_slot: str, channel: str = "WhatsApp") -> str:
    matched_doc = next((d for d in DOCTORS if doctor_name.lower() in d.lower()), None)
    if not matched_doc:
        return "বুকিং হয়নি: সঠিক ডাক্তারের নাম বলুন।"

    conn = sqlite3.connect("appointments.db")
    c = conn.cursor()
    c.execute(
        "SELECT id FROM bookings WHERE doctor = ? AND date = ? AND time_slot = ?",
        (matched_doc, date, time_slot)
    )
    if c.fetchone():
        conn.close()
        return f"দুঃখিত, {time_slot} স্লটটি আগেই বুক হয়ে গেছে।"

    c.execute(
        "INSERT INTO bookings (patient_name, phone, doctor, date, time_slot, channel) VALUES (?, ?, ?, ?, ?, ?)",
        (patient_name, phone, matched_doc, date, time_slot, channel)
    )
    booking_id = c.lastrowid
    conn.commit()
    conn.close()
    return f"বুকিং কনফার্ম হয়েছে! আইডি: #{booking_id}, রোগী: {patient_name}, ডাক্তার: {matched_doc}, তারিখ: {date}, সময়: {time_slot}।"


# --- ৩. Groq AI ও টুলস সেটআপ ---
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_available_slots",
            "description": "কোনো নির্দিষ্ট তারিখে ডাক্তারের ফাঁকা সময় চেক করে।",
            "parameters": {
                "type": "object",
                "properties": {
                    "doctor_name": {"type": "string", "description": "ডাক্তারের নাম"},
                    "date": {"type": "string", "description": "তারিখ (যেমন: 20-11-2026)"}
                },
                "required": ["doctor_name", "date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": "রোগীর ডাক্তার অ্যাপয়েন্টমেন্ট কনফার্ম ও ডেটাবেসে সেভ করে।",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_name": {"type": "string", "description": "রোগীর নাম"},
                    "phone": {"type": "string", "description": "রোগীর ফোন নম্বর"},
                    "doctor_name": {"type": "string", "description": "ডাক্তারের নাম"},
                    "date": {"type": "string", "description": "তারিখ"},
                    "time_slot": {"type": "string", "description": "সময় (যেমন: সকাল ১০:০০)"}
                },
                "required": ["patient_name", "doctor_name", "date", "time_slot"]
            }
        }
    }
]

SYSTEM_PROMPT = f"""
তুমি একজন ভদ্র ক্লিনিক রিসেপশনিস্ট AI। রোগীদের ডাক্তার বুক করতে সাহায্য করো।
আমাদের ডাক্তারগণ: {', '.join(DOCTORS.keys())}।
নিয়ম:
১. রোগী যে ভাষায় বলবে (বাংলা, বাংলিশ বা ইংরেজি), সেই ভাষায় সহজ ও সংক্ষিপ্ত উত্তর দাও।
২. বুকিং করার আগে অবশ্যই জেনে নাও: রোগীর নাম, ডাক্তারের নাম, তারিখ এবং সময়।
৩. প্রথমে check_available_slots দিয়ে ফাঁকা সময় চেক করে তারপর book_appointment দিয়ে বুক করো।
৪. ফোন কলের ক্ষেত্রে উত্তর ১-২ লাইনের মধ্যে রাখবে এবং কোনো স্টার (*) চিহ্ন ব্যবহার করবে না।
"""

user_sessions = {}
MODEL_NAME = "llama-3.3-70b-versatile"


def ask_ai(session_id: str, user_message: str, channel: str, phone: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "সার্ভারে GROQ_API_KEY সেট করা নেই। Render-এর Environment Variables চেক করুন।"

    client = Groq(api_key=api_key)

    if session_id not in user_sessions:
        user_sessions[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

    messages = user_sessions[session_id]
    messages.append({
        "role": "user",
        "content": f"[Channel: {channel}, User Phone: {phone}] রোগী বলল: {user_message}"
    })

    # মেমোরি ঠিক রাখতে শেষ ২০টি মেসেজ রাখা
    if len(messages) > 21:
        messages = [messages[0]] + messages[-20:]
        user_sessions[session_id] = messages

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.3,
            max_tokens=300
        )

        response_msg = response.choices[0].message

        # AI যদি কোনো টুল কল করতে চায়
        if response_msg.tool_calls:
            messages.append({
                "role": "assistant",
                "content": response_msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in response_msg.tool_calls
                ]
            })

            for tc in response_msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                if tc.function.name == "check_available_slots":
                    result = check_available_slots(
                        doctor_name=args.get("doctor_name", ""),
                        date=args.get("date", "")
                    )
                elif tc.function.name == "book_appointment":
                    result = book_appointment(
                        patient_name=args.get("patient_name", "Unknown"),
                        phone=args.get("phone") or phone,
                        doctor_name=args.get("doctor_name", ""),
                        date=args.get("date", ""),
                        time_slot=args.get("time_slot", ""),
                        channel=channel
                    )
                else:
                    result = "অজানা ফাংশন।"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": result
                })

            final_response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.3,
                max_tokens=300
            )
            final_text = final_response.choices[0].message.content or "ধন্যবাদ!"
        else:
            final_text = response_msg.content or "দুঃখিত, বুঝতে পারিনি। দয়া করে আবার বলুন।"

        messages.append({"role": "assistant", "content": final_text})
        return final_text

    except Exception as e:
        print("Groq Error:", e)
        return "দুঃখিত, এই মুহূর্তে একটু টেকনিক্যাল সমস্যা হচ্ছে। কিছুক্ষণ পর আবার চেষ্টা করুন।"


# --- ৪. Render Health Check ---
@app.get("/")
async def home():
    return {"status": "Doctor Booking AI (Groq + WhatsApp + Call) লাইভ আছে!"}


# --- ৫. WhatsApp এন্ডপয়েন্ট ---
@app.get("/whatsapp")
async def verify_whatsapp(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
    hub_challenge: str = Query(None, alias="hub.challenge")
):
    if hub_mode == "subscribe" and hub_verify_token == os.getenv("VERIFY_TOKEN"):
        return Response(content=hub_challenge, media_type="text/plain")
    return Response(content="Forbidden", status_code=403)


@app.post("/whatsapp")
async def whatsapp_webhook(request: Request):
    data = await request.json()
    try:
        entry = data["entry"][0]["changes"][0]["value"]
        if "messages" in entry:
            msg = entry["messages"][0]
            sender_phone = msg["from"]
            text_body = msg.get("text", {}).get("body", "")

            if text_body:
                reply_text = ask_ai(f"wa_{sender_phone}", text_body, "WhatsApp", sender_phone)
                await send_whatsapp_message(sender_phone, reply_text)
    except (KeyError, IndexError):
        pass
    return {"status": "ok"}


async def send_whatsapp_message(to_phone: str, text: str):
    url = f"https://graph.facebook.com/v21.0/{os.getenv('WHATSAPP_PHONE_ID')}/messages"
    headers = {
        "Authorization": f"Bearer {os.getenv('WHATSAPP_TOKEN')}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_phone,
        "type": "text",
        "text": {"body": text}
    }
    async with httpx.AsyncClient() as http_client:
        await http_client.post(url, headers=headers, json=payload)


# --- ৬. Twilio ফোন কল এন্ডপয়েন্ট ---
@app.post("/voice")
async def incoming_call():
    vr = VoiceResponse()
    gather = Gather(
        input="speech",
        action="/voice/process",
        language="bn-IN",
        speech_timeout="auto"
    )
    gather.say("নমস্কার! আমাদের ক্লিনিকে স্বাগতম। আপনি কোন ডাক্তারের অ্যাপয়েন্টমেন্ট বুক করতে চান?", language="bn-IN")
    vr.append(gather)
    return Response(content=str(vr), media_type="application/xml")


@app.post("/voice/process")
async def process_call_speech(request: Request):
    form = await request.form()
    caller_phone = form.get("From", "Unknown")
    call_sid = form.get("CallSid", caller_phone)
    user_speech = form.get("SpeechResult", "")

    vr = VoiceResponse()
    if not user_speech:
        vr.say("আমি আপনার কথা শুনতে পাইনি। দয়া করে আবার বলুন।", language="bn-IN")
        vr.redirect("/voice")
        return Response(content=str(vr), media_type="application/xml")

    ai_reply = ask_ai(f"call_{call_sid}", user_speech, "Phone Call", caller_phone)

    gather = Gather(
        input="speech",
        action="/voice/process",
        language="bn-IN",
        speech_timeout="auto"
    )
    gather.say(ai_reply, language="bn-IN")
    vr.append(gather)
    return Response(content=str(vr), media_type="application/xml")

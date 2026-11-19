import os
import json
import sqlite3
from fastapi import FastAPI, Request, Response, BackgroundTasks, Query
from groq import Groq
from twilio.rest import Client as TwilioClient
from twilio.twiml.voice_response import VoiceResponse, Gather
from twilio.twiml.messaging_response import MessagingResponse

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
    "Dr. Sen (General Physician)": ["10:00 AM", "11:00 AM", "05:00 PM"],
    "Dr. Roy (Dentist)": ["12:00 PM", "04:00 PM", "06:00 PM"],
    "Dr. Das (Cardiologist)": ["11:30 AM", "03:00 PM"]
}

# বাংলা ভয়েস সেটিংস (Twilio Google TTS)
VOICE_NAME = os.getenv("TWILIO_VOICE", "Google.bn-IN-Standard-A")
VOICE_LANG = os.getenv("TWILIO_LANG", "bn-IN")

# --- ২. বুকিং ও স্লট চেক ফাংশন ---
def check_available_slots(doctor_name: str, date: str) -> str:
    matched_doc = next((d for d in DOCTORS if doctor_name.lower() in d.lower()), None)
    if not matched_doc:
        return f"Doctor not found. Available doctors: {', '.join(DOCTORS.keys())}"

    conn = sqlite3.connect("appointments.db")
    c = conn.cursor()
    c.execute("SELECT time_slot FROM bookings WHERE doctor = ? AND date = ?", (matched_doc, date))
    booked = {row[0] for row in c.fetchall()}
    conn.close()

    free_slots = [s for s in DOCTORS[matched_doc] if s not in booked]
    if not free_slots:
        return f"No free slots for {matched_doc} on {date}."
    return f"Available slots for {matched_doc} on {date}: {', '.join(free_slots)}"


def book_appointment(patient_name: str, phone: str, doctor_name: str, date: str, time_slot: str, channel: str = "Phone Call") -> str:
    matched_doc = next((d for d in DOCTORS if doctor_name.lower() in d.lower()), None)
    if not matched_doc:
        return "Booking failed: Please choose a valid doctor."

    conn = sqlite3.connect("appointments.db")
    c = conn.cursor()
    c.execute(
        "SELECT id FROM bookings WHERE doctor = ? AND date = ? AND time_slot = ?",
        (matched_doc, date, time_slot)
    )
    if c.fetchone():
        conn.close()
        return f"Sorry, {time_slot} is already booked."

    c.execute(
        "INSERT INTO bookings (patient_name, phone, doctor, date, time_slot, channel) VALUES (?, ?, ?, ?, ?, ?)",
        (patient_name, phone, matched_doc, date, time_slot, channel)
    )
    booking_id = c.lastrowid
    conn.commit()
    conn.close()
    return f"Booking Confirmed! ID: #{booking_id}, Patient: {patient_name}, Doctor: {matched_doc}, Date: {date}, Time: {time_slot}."


# --- ৩. Groq অটো-মডেল ও টুলস সেটআপ ---
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_available_slots",
            "description": "Check available appointment time slots for a doctor on a given date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doctor_name": {"type": "string", "description": "Name of the doctor, e.g., Dr. Sen"},
                    "date": {"type": "string", "description": "Appointment date, e.g., 20-11-2026"}
                },
                "required": ["doctor_name", "date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": "Confirm and save a patient's doctor appointment in the database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_name": {"type": "string", "description": "Patient full name"},
                    "phone": {"type": "string", "description": "Patient phone number"},
                    "doctor_name": {"type": "string", "description": "Doctor name"},
                    "date": {"type": "string", "description": "Appointment date"},
                    "time_slot": {"type": "string", "description": "Selected time slot, e.g., 10:00 AM"}
                },
                "required": ["patient_name", "doctor_name", "date", "time_slot"]
            }
        }
    }
]

SYSTEM_PROMPT = f"""
You are a polite clinic receptionist AI helping patients book doctor appointments.
Available Doctors: {', '.join(DOCTORS.keys())}.
Rules:
1. Reply in Bengali (বাংলা) or the same language the patient uses.
2. Before booking, collect: patient name, doctor name, date, and preferred time slot.
3. First call check_available_slots to verify free slots, then call book_appointment to confirm.
4. For Phone Calls, reply in 1-2 short natural spoken sentences in Bengali script. Never use symbols like *, #, or markdown.
"""

user_sessions = {}
ACTIVE_MODEL = None


def get_active_model(client: Groq) -> str:
    global ACTIVE_MODEL
    env_model = os.getenv("GROQ_MODEL")
    if env_model:
        return env_model
    if ACTIVE_MODEL:
        return ACTIVE_MODEL

    models_resp = client.models.list()
    all_ids = [m.id for m in models_resp.data]

    blocked = ("whisper", "tts", "guard", "playai", "embed", "rerank", "orpheus", "safeguard", "distil", "compound")
    chat_models = [m for m in all_ids if not any(b in m.lower() for b in blocked)]

    preferred = [
        "llama-4-maverick",
        "llama-4-scout",
        "llama-3.3-70b",
        "llama-3.1-8b-instant",
        "qwen",
        "gpt-oss",
        "llama",
    ]
    for pref in preferred:
        for m in chat_models:
            if pref in m.lower():
                ACTIVE_MODEL = m
                return ACTIVE_MODEL

    if chat_models:
        ACTIVE_MODEL = chat_models[0]
        return ACTIVE_MODEL

    raise RuntimeError(f"No usable chat model found. Available: {all_ids}")


def ask_ai(session_id: str, user_message: str, channel: str, phone: str) -> str:
    global ACTIVE_MODEL
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "সার্ভারে গ্রক এপিআই কী সেট করা নেই।"

    client = Groq(api_key=api_key)

    if session_id not in user_sessions:
        user_sessions[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

    messages = user_sessions[session_id]
    messages.append({
        "role": "user",
        "content": f"[Channel: {channel}, Phone: {phone}] {user_message}"
    })

    if len(messages) > 21:
        messages = [messages[0]] + messages[-20:]
        user_sessions[session_id] = messages

    try:
        model_name = get_active_model(client)

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.3,
                max_tokens=250
            )
        except Exception as tool_err:
            print("Tool call fallback:", tool_err)
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.3,
                max_tokens=250
            )

        response_msg = response.choices[0].message

        if getattr(response_msg, "tool_calls", None):
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
                    result = "Unknown tool."

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": result
                })

            final_response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.3,
                max_tokens=250
            )
            final_text = final_response.choices[0].message.content or "ধন্যবাদ!"
        else:
            final_text = response_msg.content or "দুঃখিত, বুঝতে পারিনি। দয়া করে আবার বলুন।"

        final_text = final_text.replace("*", "").replace("#", "").strip()
        messages.append({"role": "assistant", "content": final_text})
        return final_text

    except Exception as e:
        ACTIVE_MODEL = None
        print("Groq Error:", e)
        return "দুঃখিত, একটু টেকনিক্যাল সমস্যা হচ্ছে। কিছুক্ষণ পর আবার চেষ্টা করুন।"


def send_whatsapp_via_api(from_wa: str, to_wa: str, clean_phone: str, user_msg: str):
    try:
        reply_text = ask_ai(f"wa_{clean_phone}", user_msg, "WhatsApp", clean_phone)
        sid = os.getenv("TWILIO_ACCOUNT_SID")
        token = os.getenv("TWILIO_AUTH_TOKEN")
        twilio_client = TwilioClient(sid, token)
        sent = twilio_client.messages.create(from_=from_wa, to=to_wa, body=reply_text)
        print(f"Twilio API Sent OK -> SID: {sent.sid}")
    except Exception as e:
        print(f"Twilio API Send Error -> {e}")


# --- ৪. হোম ও বুকিং লিস্ট ---
@app.get("/")
async def home():
    return {"status": "Doctor Booking AI (Voice Fixed + WhatsApp) লাইভ আছে!"}


@app.get("/bookings")
async def view_bookings():
    conn = sqlite3.connect("appointments.db")
    c = conn.cursor()
    c.execute("SELECT id, patient_name, phone, doctor, date, time_slot, channel FROM bookings ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return {
        "bookings": [
            {
                "id": r[0], "patient_name": r[1], "phone": r[2],
                "doctor": r[3], "date": r[4], "time_slot": r[5], "channel": r[6]
            }
            for r in rows
        ]
    }


# --- ৫. AI থেকে আপনার ফোনে ফ্রি কল আনার লিংক ---
@app.get("/call-me")
async def trigger_outbound_call(request: Request, to: str = Query(default=None)):
    sid = os.getenv("TWILIO_ACCOUNT_SID")
    token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_PHONE_NUMBER")
    to_number = to or os.getenv("MY_PHONE_NUMBER")

    if not all([sid, token, from_number, to_number]):
        return {
            "error": "Render-এর Environment-এ TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER এবং MY_PHONE_NUMBER সেট করুন।"
        }

    voice_url = "https://doctor-ai-dosj.onrender.com/voice"
    try:
        twilio_client = TwilioClient(sid, token)
        call = twilio_client.calls.create(
            to=to_number,
            from_=from_number,
            url=voice_url,
            method="POST"
        )
        return {"status": "কল যাচ্ছে! ফোন রিসিভ করুন।", "call_sid": call.sid}
    except Exception as e:
        return {"error": str(e)}


# --- ৬. Twilio Phone Call এন্ডপয়েন্ট (GET ও POST দুটোই সাপোর্ট করে) ---
@app.api_route("/voice", methods=["GET", "POST"])
async def incoming_call():
    vr = VoiceResponse()
    gather = Gather(
        input="speech",
        action="/voice/process",
        method="POST",
        language=VOICE_LANG,
        speech_timeout="auto"
    )
    gather.say(
        "নমস্কার! আমাদের ক্লিনিকে স্বাগতম। আপনি কোন ডাক্তারের অ্যাপয়েন্টমেন্ট বুক করতে চান?",
        voice=VOICE_NAME,
        language=VOICE_LANG
    )
    vr.append(gather)
    vr.say("কোনো উত্তর পাওয়া যায়নি। আবার চেষ্টা করুন।", voice=VOICE_NAME, language=VOICE_LANG)
    return Response(content=str(vr), media_type="text/xml")


@app.api_route("/voice/process", methods=["GET", "POST"])
async def process_call_speech(request: Request):
    data = await request.form() if request.method == "POST" else request.query_params
    caller_phone = str(data.get("From", "Unknown"))
    call_sid = str(data.get("CallSid", caller_phone))
    user_speech = str(data.get("SpeechResult", "")).strip()
    print(f"Call Speech -> {user_speech}")

    vr = VoiceResponse()
    if not user_speech:
        vr.say("আমি আপনার কথা শুনতে পাইনি। দয়া করে আবার বলুন।", voice=VOICE_NAME, language=VOICE_LANG)
        vr.redirect("/voice", method="POST")
        return Response(content=str(vr), media_type="text/xml")

    ai_reply = ask_ai(f"call_{call_sid}", user_speech, "Phone Call", caller_phone)
    print(f"Call Reply -> {ai_reply}")

    gather = Gather(
        input="speech",
        action="/voice/process",
        method="POST",
        language=VOICE_LANG,
        speech_timeout="auto"
    )
    gather.say(ai_reply, voice=VOICE_NAME, language=VOICE_LANG)
    vr.append(gather)
    return Response(content=str(vr), media_type="text/xml")


# --- ৭. Twilio WhatsApp এন্ডপয়েন্ট ---
@app.post("/whatsapp")
async def twilio_whatsapp(request: Request, background_tasks: BackgroundTasks):
    form = await request.form()
    from_raw = str(form.get("From", "")).strip()
    to_raw = str(form.get("To", "")).strip()
    user_msg = str(form.get("Body", "")).strip()

    if not user_msg:
        return Response(content=str(MessagingResponse()), media_type="text/xml")

    clean_phone = from_raw.replace("whatsapp:", "")

    if os.getenv("TWILIO_ACCOUNT_SID") and os.getenv("TWILIO_AUTH_TOKEN"):
        from_wa = to_raw if to_raw.startswith("whatsapp:") else f"whatsapp:{to_raw}"
        to_wa = from_raw if from_raw.startswith("whatsapp:") else f"whatsapp:{from_raw}"
        background_tasks.add_task(send_whatsapp_via_api, from_wa, to_wa, clean_phone, user_msg)
        return Response(content=str(MessagingResponse()), media_type="text/xml")

    reply_text = ask_ai(f"wa_{clean_phone}", user_msg, "WhatsApp", clean_phone)
    resp = MessagingResponse()
    resp.message(reply_text)
    return Response(content=str(resp), media_type="text/xml")

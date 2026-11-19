import os
import json
import sqlite3
from fastapi import FastAPI, Request, Response
from groq import Groq
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

# ডাক্তার ও সময়সূচি (ইচ্ছেমতো বদলে নিতে পারেন)
DOCTORS = {
    "Dr. Sen (General Physician)": ["10:00 AM", "11:00 AM", "05:00 PM"],
    "Dr. Roy (Dentist)": ["12:00 PM", "04:00 PM", "06:00 PM"],
    "Dr. Das (Cardiologist)": ["11:30 AM", "03:00 PM"]
}

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


def book_appointment(patient_name: str, phone: str, doctor_name: str, date: str, time_slot: str, channel: str = "WhatsApp") -> str:
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


# --- ৩. Groq AI ও টুলস সেটআপ ---
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
1. Reply in the same language the patient uses (Bengali, Banglish, or English). Keep replies short and clear.
2. Before booking, collect: patient name, doctor name, date, and preferred time slot.
3. First call check_available_slots to verify free slots, then call book_appointment to confirm.
4. For Phone Calls, keep answers within 1-2 short sentences and do not use markdown or asterisks (*).
"""

user_sessions = {}
MODEL_NAME = "llama-3.3-70b-versatile"


def ask_ai(session_id: str, user_message: str, channel: str, phone: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "সার্ভারে GROQ_API_KEY সেট করা নেই। Render-এর Environment চেক করুন।"

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
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.3,
                max_tokens=300
            )
        except Exception as tool_err:
            print("Tool call fallback:", tool_err)
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.3,
                max_tokens=300
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
        return "দুঃখিত, একটু টেকনিক্যাল সমস্যা হচ্ছে। কিছুক্ষণ পর আবার চেষ্টা করুন।"


# --- ৪. হোম ও বুকিং লিস্ট দেখার লিংক ---
@app.get("/")
async def home():
    return {"status": "Doctor Booking AI (Groq + Twilio WhatsApp & Call) লাইভ আছে!"}


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
                "id": r[0],
                "patient_name": r[1],
                "phone": r[2],
                "doctor": r[3],
                "date": r[4],
                "time_slot": r[5],
                "channel": r[6]
            }
            for r in rows
        ]
    }


# --- ৫. Twilio WhatsApp এন্ডপয়েন্ট ---
@app.post("/whatsapp")
async def twilio_whatsapp(request: Request):
    form = await request.form()
    sender_raw = form.get("From", "Unknown")
    sender_phone = str(sender_raw).replace("whatsapp:", "")
    user_msg = str(form.get("Body", "")).strip()

    resp = MessagingResponse()
    if user_msg:
        reply_text = ask_ai(f"wa_{sender_phone}", user_msg, "WhatsApp", sender_phone)
        resp.message(reply_text)
    return Response(content=str(resp), media_type="application/xml; charset=utf-8")


# --- ৬. Twilio Phone Call এন্ডপয়েন্ট ---
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
    return Response(content=str(vr), media_type="application/xml; charset=utf-8")


@app.post("/voice/process")
async def process_call_speech(request: Request):
    form = await request.form()
    caller_phone = str(form.get("From", "Unknown"))
    call_sid = str(form.get("CallSid", caller_phone))
    user_speech = str(form.get("SpeechResult", "")).strip()

    vr = VoiceResponse()
    if not user_speech:
        vr.say("আমি আপনার কথা শুনতে পাইনি। দয়া করে আবার বলুন।", language="bn-IN")
        vr.redirect("/voice")
        return Response(content=str(vr), media_type="application/xml; charset=utf-8")

    ai_reply = ask_ai(f"call_{call_sid}", user_speech, "Phone Call", caller_phone)

    gather = Gather(
        input="speech",
        action="/voice/process",
        language="bn-IN",
        speech_timeout="auto"
    )
    gather.say(ai_reply, language="bn-IN")
    vr.append(gather)
    return Response(content=str(vr), media_type="application/xml; charset=utf-8")

# GenAI Meeting Summarizer & Action Tracker — Base Code

This canvas contains a minimal full‑stack starter for a meeting summarizer + action tracker using Whisper (OpenAI transcription), GPT‑4 for summarization/action extraction, and a simple CrewAI‑like action management. It includes:

- Backend: FastAPI (Python) with endpoints to upload audio, transcribe, summarize, and manage actions (SQLite).
- Worker: simple transcription + summarization flow using OpenAI API (Whisper + GPT-4)
- Frontend: React app to upload audio, view transcript & summary, and CRUD actions.

> Replace `OPENAI_API_KEY` with your key in environment or secret manager. This is base code — expand error handling, auth, rate limits, and security for production.

---

## File tree

```
genai-meeting/
├─ backend/
│  ├─ app/
│  │  ├─ main.py
│  │  ├─ db.py
│  │  ├─ models.py
│  │  └─ worker.py
│  └─ requirements.txt
├─ frontend/
│  ├─ package.json
│  └─ src/
│     ├─ App.jsx
│     └─ index.jsx
└─ README.md
```

---

## backend/requirements.txt

```
fastapi==0.95.2
uvicorn[standard]==0.22.0
requests==2.31.0
openai==1.0.0
python-multipart==0.0.6
databases==0.6.0
aiosqlite==0.18.0
pydantic==1.10.7
sqlalchemy==1.4.55
```

---

## backend/app/db.py

```python
# db.py — simple SQLite + SQLAlchemy engine and helper
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./meetings.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

metadata = MetaData()
```

---

## backend/app/models.py

```python
# models.py — SQLAlchemy models for meetings, actions
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.sql import func
from .db import Base

class Meeting(Base):
    __tablename__ = "meetings"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=True)
    transcript = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class ActionItem(Base):
    __tablename__ = "actions"
    id = Column(Integer, primary_key=True, index=True)
    meeting_id = Column(Integer, ForeignKey("meetings.id"), nullable=False)
    description = Column(Text, nullable=False)
    assignee = Column(String(255), nullable=True)
    done = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
```

---

## backend/app/worker.py

```python
# worker.py — transcription and summarization helpers
import os
from openai import OpenAI

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Transcribe using Whisper (OpenAI speech-to-text)
def transcribe_audio(file_path: str) -> str:
    # Uses OpenAI's speech API (whisper) — adapt if using local whisper or different client
    with open(file_path, "rb") as f:
        resp = client.audio.transcriptions.create(model="whisper-1", file=f)
    return resp.text

# Summarize & extract action items using GPT-4
def summarize_and_extract_actions(transcript: str) -> dict:
    prompt = f"""
You are a meeting assistant. Given the meeting transcript below, produce:
1) A concise bullet-point summary (3-6 bullets).
2) A list of action items (JSON array) with: description, suggested assignee (if mentioned), and due suggestion.

Transcript:
"""
    prompt += transcript + "\n\nRespond with JSON like:\n{\n  \"summary\": [\"...\"],\n  \"actions\": [{\"description\":\"...\", \"assignee\":\"...\", \"due\":\"...\"}]\n}\n"

    completion = client.chat.completions.create(
        model="gpt-4o-mini", # change to gpt-4 or gpt-4o depending on availability
        messages=[{"role":"user","content":prompt}],
        temperature=0.1,
        max_tokens=800,
    )

    text = completion.choices[0].message.content
    # Try to parse JSON from response
    import json, re
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        # fallback: put whole text into summary
        return {"summary": [text[:1000]], "actions": []}
    payload = json.loads(m.group(0))
    return payload
```

---

## backend/app/main.py

```python
# main.py — FastAPI app
import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from .db import SessionLocal, engine
from .models import Base, Meeting, ActionItem
from .worker import transcribe_audio, summarize_and_extract_actions

Base.metadata.create_all(bind=engine)
app = FastAPI(title="GenAI Meeting Summarizer")

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/api/upload-audio")
async def upload_audio(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # save file
    file_path = f"{UPLOAD_DIR}/{file.filename}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # create meeting record
    meeting = Meeting(title=file.filename)
    db.add(meeting)
    db.commit()
    db.refresh(meeting)

    # transcribe & summarize (synchronous for base code)
    try:
        transcript = transcribe_audio(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    result = summarize_and_extract_actions(transcript)

    meeting.transcript = transcript
    meeting.summary = "\n".join(result.get("summary", [])) if result.get("summary") else None
    db.add(meeting)
    db.commit()

    # save actions
    actions = result.get("actions", [])
    saved = []
    for a in actions:
        ai = ActionItem(meeting_id=meeting.id, description=a.get("description",""), assignee=a.get("assignee"))
        db.add(ai)
        db.commit()
        db.refresh(ai)
        saved.append({"id": ai.id, "description": ai.description, "assignee": ai.assignee})

    return JSONResponse({"meeting": {"id": meeting.id, "title": meeting.title, "summary": meeting.summary}, "actions": saved})

@app.get("/api/meetings/{meeting_id}")
def get_meeting(meeting_id: int, db: Session = Depends(get_db)):
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if not meeting:
        raise HTTPException(404, "Meeting not found")
    actions = db.query(ActionItem).filter(ActionItem.meeting_id == meeting_id).all()
    return {"id": meeting.id, "title": meeting.title, "transcript": meeting.transcript, "summary": meeting.summary, "actions": [{"id": a.id, "description": a.description, "assignee": a.assignee, "done": a.done} for a in actions]}

@app.post("/api/meetings/{meeting_id}/actions")
def create_action(meeting_id: int, payload: dict, db: Session = Depends(get_db)):
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if not meeting:
        raise HTTPException(404, "Meeting not found")
    ai = ActionItem(meeting_id=meeting_id, description=payload.get("description"), assignee=payload.get("assignee"))
    db.add(ai)
    db.commit()
    db.refresh(ai)
    return {"id": ai.id, "description": ai.description, "assignee": ai.assignee}

@app.patch("/api/actions/{action_id}")
def update_action(action_id: int, payload: dict, db: Session = Depends(get_db)):
    ai = db.query(ActionItem).filter(ActionItem.id == action_id).first()
    if not ai:
        raise HTTPException(404, "Action not found")
    if "done" in payload:
        ai.done = payload["done"]
    if "assignee" in payload:
        ai.assignee = payload["assignee"]
    if "description" in payload:
        ai.description = payload["description"]
    db.add(ai)
    db.commit()
    db.refresh(ai)
    return {"id": ai.id, "description": ai.description, "assignee": ai.assignee, "done": ai.done}
```

---

## frontend/package.json

```json
{
  "name": "genai-meetings-frontend",
  "version": "0.1.0",
  "private": true,
  "dependencies": {
    "react": "18.2.0",
    "react-dom": "18.2.0",
    "vite": "^4.0.0"
  },
  "scripts": {
    "dev": "vite"
  }
}
```

---

## frontend/src/App.jsx

```jsx
import React, {useState} from 'react'

export default function App(){
  const [file, setFile] = useState(null)
  const [meeting, setMeeting] = useState(null)
  const [actions, setActions] = useState([])

  const upload = async () =>{
    if(!file) return alert('Choose file')
    const fd = new FormData()
    fd.append('file', file)
    const res = await fetch('/api/upload-audio', {method:'POST', body: fd})
    const data = await res.json()
    setMeeting(data.meeting)
    setActions(data.actions)
  }

  return (
    <div style={{padding:20}}>
      <h1>GenAI Meeting Summarizer</h1>
      <input type="file" accept="audio/*" onChange={(e)=>setFile(e.target.files[0])} />
      <button onClick={upload}>Upload & Process</button>

      {meeting && (
        <div style={{marginTop:20}}>
          <h2>{meeting.title}</h2>
          <h3>Summary</h3>
          <pre>{meeting.summary}</pre>

          <h3>Action Items</h3>
          <ul>
            {actions.map(a=> <li key={a.id}>{a.description} — {a.assignee}</li>)}
          </ul>
        </div>
      )}
    </div>
  )
}
```

---

## frontend/src/index.jsx

```jsx
import React from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'

createRoot(document.getElementById('root')).render(<App />)
```

---

## .env.example

```
OPENAI_API_KEY=sk-...
DATABASE_URL=sqlite:///./meetings.db
UPLOAD_DIR=./uploads
```

---

## README.md (snippet)

```
# GenAI Meeting Summarizer & Action Tracker

1. Backend
   - cd backend
   - python -m venv venv
   - pip install -r requirements.txt
   - export OPENAI_API_KEY=...
   - uvicorn app.main:app --reload --port 8000

2. Frontend (simple static dev server or proxy)
   - cd frontend
   - npm install
   - npm run dev

API proxy: in dev, configure the frontend dev server to proxy /api to http://localhost:8000
```

---

### Next steps & improvements (suggested)

- Add authentication (JWT) and per-user meetings
- Use background worker (Celery/RQ) to transcribe & summarize asynchronously
- Improve prompt engineering for better action extraction and due-date inference
- Integrate CrewAI or task-management API for assigning tasks automatically
- Add real-time collaboration, meeting speaker diarization, and timestamped highlights


---

If you want, I can now:
- generate the full files as downloadable files,
- convert the frontend to TypeScript + Tailwind UI,
- add a Celery background worker and Dockerfiles,
- wire up a simple UI for marking actions done.

Tell me which next step to generate and I'll add it directly into this project canvas.


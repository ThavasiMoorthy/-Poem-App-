import os
import logging
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from backend.poem_engine import PoemEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Tamil Poem Generator")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
MODEL_PATH = os.path.join(BASE_DIR, "model_data", "model5_epoch3.pt")
TOKENIZER_PATH = os.path.join(BASE_DIR, "model_data", "tamil_sp_model5.model")

templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Initialize Poem Engine
try:
    poem_engine = PoemEngine(MODEL_PATH, TOKENIZER_PATH)
    logger.info("Poem Engine initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Poem Engine: {e}")
    poem_engine = None

class ChatRequest(BaseModel):
    message: str # We'll use this as the subject/prompt

import json
import asyncio
from fastapi.responses import StreamingResponse

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

async def poem_streamer(prompt: str):
    """
    Helper to stream processed poem tokens.
    """
    last_text = ""
    stop_tags = ["<குறிச்சொற்கள்>", "<முடிவு>", "<துவக்கம்>", "<வழிமுறை>", "</வழிமுறை>", "<பொருள்>", "<கருப்பொருள்>", "<பாணி>"]
    
    for full_text in poem_engine.generate_stream(prompt):
        # Determine if any stop tag has appeared in the full generated text
        should_stop = False
        display_text = full_text
        
        for tag in ["<குறிச்சொற்கள்>", "<முடிவு>"]:
            if tag in display_text:
                display_text = display_text.split(tag)[0]
                should_stop = True
        
        # Calculate delta from the cleaned version
        if display_text.startswith(last_text):
            delta = display_text[len(last_text):]
            if delta:
                # Remove any other intermediate tags (unlikely but safe)
                for tag in stop_tags:
                    delta = delta.replace(tag, "")
                
                if delta:
                    yield f"data: {json.dumps({'text': delta})}\n\n"
            last_text = display_text
            
        if should_stop:
            break
        
        await asyncio.sleep(0.01)
    
    yield "data: [DONE]\n\n"

@app.post("/api/chat")
async def chat_endpoint(data: ChatRequest):
    if not poem_engine:
        return {"response": "Server Error: Poem Engine not loaded.", "source": "error"}
    
    try:
        subject = data.message.strip()
        theme = "மகிழ்ச்சி"
        prompt = f"<துவக்கம்>\n<வழிமுறை>\nகீழ்க்கண்ட தகவல்களை அடிப்படையாகக் கொண்டு\nஒரு புதுக்கவிதையை எழுதவும்.\n</வழிமுறை>\n\n<பொருள்> {subject}\n<கருப்பொருள்> {theme}\n<பாணி> புதுக்கவிதை\n"
        
        logger.info(f"Starting stream for subject: {subject}")
        return StreamingResponse(poem_streamer(prompt), media_type="text/event-stream")
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        return {"response": f"Error: {str(e)}", "source": "error"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

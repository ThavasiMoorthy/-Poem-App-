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

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/chat")
async def chat_endpoint(data: ChatRequest):
    if not poem_engine:
        return {"response": "Server Error: Poem Engine not loaded.", "source": "error"}
    
    try:
        subject = data.message.strip()
        # Default theme set to 'மகிழ்ச்சி' (Happiness) as requested
        theme = "மகிழ்ச்சி"
        
        # Construct the specialized prompt
        prompt = f"<துவக்கம்>\n<வழிமுறை>\nகீழ்க்கண்ட தகவல்களை அடிப்படையாகக் கொண்டு\nஒரு புதுக்கவிதையை எழுதவும்.\n</வழிமுறை>\n\n<பொருள்> {subject}\n<கருப்பொருள்> {theme}\n<பாணி> புதுக்கவிதை\n"
        
        logger.info(f"Generating poem for subject: {subject} with theme: {theme}")
        full_output = poem_engine.generate(prompt)
        
        # --- Filtering Logic ---
        # We only want the poem text that comes after "<பாணி> புதுக்கவிதை"
        # and ends before "<குறிச்சொற்கள்>" or "<முடிவு>"
        
        response_text = full_output
        
        if "<பாணி> புதுக்கவிதை" in response_text:
            response_text = response_text.split("<பாணி> புதுக்கவிதை")[-1].strip()
        
        if "<குறிச்சொற்கள்>" in response_text:
            response_text = response_text.split("<குறிச்சொற்கள்>")[0].strip()
            
        if "<முடிவு>" in response_text:
            response_text = response_text.split("<முடிவு>")[0].strip()
            
        # Clean up any remaining leading/trailing whitespace or extra tags
        response_text = response_text.strip()
            
        logger.info(f"Generated filtered response: {response_text[:100]}...")
        return {"response": response_text, "source": "ai"}
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return {"response": f"Error: {str(e)}", "source": "error"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

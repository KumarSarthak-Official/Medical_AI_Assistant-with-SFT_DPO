import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import TextIteratorStreamer
from threading import Thread
from unsloth import FastLanguageModel
import uvicorn

app = FastAPI(title="Medical AI API")

# Load the Model from Hugging Face
MODEL_NAME = "kumarsarthak98/medical-llama-3.2-3b-sft-dpo"
print("Loading model into VRAM...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Receives a question, formats it, and streams the LLM response."""
    messages = [
        {"role": "system", "content": "You are a knowledgeable medical AI assistant."},
        {"role": "user", "content": request.question}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(**inputs, streamer=streamer, max_new_tokens=512, temperature=0.3)

    # Start generation in a background thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Generator for Server-Sent Events (SSE)
    def token_generator():
        for token in streamer:
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
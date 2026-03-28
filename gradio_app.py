import gradio as gr
import httpx

# Point this to your FastAPI server
API_URL = "http://localhost:8000/chat"

def stream_from_api(question, history):
    """Sends the question to the FastAPI backend and yields the streaming response."""
    history = history or []
    history.append([question, ""])
    
    try:
        with httpx.Client(timeout=60.0) as client:
            with client.stream("POST", API_URL, json={"question": question}) as response:
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        token = line[6:] # Strip the "data: " prefix
                        if token == "[DONE]":
                            break
                        history[-1][1] += token
                        yield history, ""
    except Exception as e:
        yield history + [["", f"Error connecting to backend: {str(e)}"]], ""

# Build the UI
demo = gr.ChatInterface(
    stream_from_api,
    title="🏥 Medical AI Assistant (Client)",
    description="This lightweight UI streams responses from a decoupled FastAPI backend.",
    examples=[
        "What are the early warning signs of Type 2 Diabetes?",
        "How do SSRIs work in the brain?",
    ]
)

if __name__ == "__main__":
    demo.launch()
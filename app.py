import os
import threading
from groq import Groq
import gradio as gr
import streamlit as st
import streamlit.components.v1 as components

# 1. Page Configuration
st.set_page_config(page_title="Professional AI Assistant", page_icon="🤖", layout="wide")

# 2. Get Groq API Key from Streamlit Secrets or Environment Variable
groq_api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

if not groq_api_key:
    st.error("⚠️ Please add your `GROQ_API_KEY` in Streamlit Cloud Secrets settings.")
    st.stop()

# Initialize Groq Client
client = Groq(api_key=groq_api_key)

SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a highly capable, professional, and friendly AI Assistant. "
        "Provide clear, concise, and accurate responses. Use clean Markdown formatting."
    )
}

# 3. Gradio Chat Prediction Function
def predict(message, history):
    messages = [SYSTEM_PROMPT]
    
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
        
    messages.append({"role": "user", "content": message})
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            stream=True
        )
        
        partial_message = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                partial_message += chunk.choices[0].delta.content
                yield partial_message

    except Exception as e:
        yield f"⚠️ Error: {str(e)}"

# 4. Function to run Gradio in a background thread
@st.cache_resource
def launch_gradio():
    demo = gr.ChatInterface(
        fn=predict,
        title="🤖 Professional AI Assistant",
        description="Powered by Groq & Llama 3.3 70B",
        textbox=gr.Textbox(placeholder="Type your message here...", scale=7),
        examples=[
            "Explain quantum computing simply.",
            "Write a Python script to sort a list.",
            "Give me 3 tips for effective time management."
        ]
    )
    
    # Launch Gradio on port 7860
    thread = threading.Thread(
        target=demo.launch,
        kwargs={"server_name": "127.0.0.1", "server_port": 7860, "inline": False, "prevent_thread_lock": True},
        daemon=True
    )
    thread.start()

# Launch Gradio once
launch_gradio()

# 5. Render Gradio UI inside Streamlit Web App
st.markdown("## 🤖 Professional AI Assistant")
st.caption("Gradio UI running inside Streamlit Cloud")

# Embed Gradio Interface using iframe
components.iframe("http://127.0.0.1:7860", height=800, scrolling=True)
  

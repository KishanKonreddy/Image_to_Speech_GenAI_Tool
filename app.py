# 🛠 Patch to avoid torch.classes crash with Streamlit
import sys
import types
import torch

torch.classes = types.SimpleNamespace()
sys.modules['torch.classes'] = torch.classes

# 🌐 Standard imports
import os
import time
from typing import Any
import requests
import streamlit as st
from dotenv import find_dotenv, load_dotenv
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from transformers import pipeline
from gtts import gTTS  # ✅ gTTS for audio

from utils.custom import css_code

# 🔐 Load API keys
load_dotenv(find_dotenv())
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def progress_bar(amount_of_time: int) -> Any:
    progress_text = "Please wait, Generative models hard at work"
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(amount_of_time):
        time.sleep(0.04)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()


def generate_text_from_image(image_path: str) -> str:
    image_to_text: Any = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    generated_text: str = image_to_text(image_path)[0]["generated_text"]
    print(f"IMAGE INPUT: {image_path}")
    print(f"GENERATED TEXT OUTPUT: {generated_text}")
    return generated_text


def generate_story_from_text(scenario: str) -> str:
    prompt_template: str = f"""
    You are a talented story teller who can create a story from a simple narrative.
    Create a story using the following scenario; the story should be maximum 50 words long;

    CONTEXT: {scenario}
    STORY:
    """
    prompt: PromptTemplate = PromptTemplate(template=prompt_template, input_variables=["scenario"])
    llm: Any = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.9)
    story_llm: Any = LLMChain(llm=llm, prompt=prompt, verbose=True)
    generated_story: str = story_llm.predict(scenario=scenario)
    print(f"TEXT INPUT: {scenario}")
    print(f"GENERATED STORY OUTPUT: {generated_story}")
    return generated_story


def generate_speech_from_text(message: str, audio_path: str) -> None:
    """
    Uses Google Text-to-Speech (gTTS) to convert text into audio and save it as MP3.
    """
    try:
        tts = gTTS(text=message, lang='en')
        tts.save(audio_path)
        print(f"✅ Audio saved to {audio_path}")
    except Exception as e:
        print("❌ gTTS audio generation failed:", str(e))
        st.error("Failed to generate audio. Please check your internet connection.")


def main() -> None:
    st.set_page_config(page_title="IMAGE TO STORY CONVERTER", page_icon="🖼️")
    st.markdown(css_code, unsafe_allow_html=True)

    # ✅ Ensure folders exist
    os.makedirs("audio-img", exist_ok=True)
    os.makedirs("img-audio", exist_ok=True)

    with st.sidebar:
        st.image("img/images.png")
        st.write("---")
        st.write("AI App created by")
        st.write("Kishan Kumar Reddy Konreddy")
        st.write("Dinesh Buruboyina")
        st.write("Shriya Reddy")
        st.write("Paavani")

    st.header("Image-to-Story Converter")
    uploaded_file: Any = st.file_uploader("Please choose a file to upload", type="jpg")

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data: Any = uploaded_file.getvalue()

        # ✅ Save image with its original name in audio-img/
        image_filename = uploaded_file.name
        image_path = os.path.join("audio-img", image_filename)
        with open(image_path, "wb") as file:
            file.write(bytes_data)

        st.image(image_path, caption="Uploaded Image", use_column_width=True)
        progress_bar(100)

        # ✅ Extract base name (without extension) for audio filename
        image_base_name = os.path.splitext(image_filename)[0]
        audio_filename = f"{image_base_name}.mp3"
        audio_path = os.path.join("img-audio", audio_filename)

        scenario: str = generate_text_from_image(image_path)
        story: str = generate_story_from_text(scenario)
        generate_speech_from_text(story, audio_path)

        with st.expander("Generated Image scenario"):
            st.write(scenario)
        with st.expander("Generated short story"):
            st.write(story)

        st.audio(audio_path)


if __name__ == "__main__":
    main()
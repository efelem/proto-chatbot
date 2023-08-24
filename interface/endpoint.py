import os

import gradio as gr
from dotenv import load_dotenv
from model import load_model
from theme import DeepSquare

from utils import seed_everything

load_dotenv()


def inference(prompt):
    for resp_text in model(prompt, max_tokens=1024):
        yield resp_text


MODEL_NAME = os.getenv("MODEL_NAME", "oasst-pythia-13b")
CACHE_DIR = os.getenv("CACHE_DIR", "/opt/models")
RANDOM_SEED = os.getenv("RANDOM_SEED", "true").lower() == "true"

seed_everything(None if RANDOM_SEED else 42)

model, new_conv = load_model(MODEL_NAME, CACHE_DIR)

with gr.Blocks(theme=DeepSquare()) as happ:
    with gr.Row():
        with gr.Column():
            question = gr.Textbox(label="Prompt")
            send_btn = gr.Button(value="Get answer")
        with gr.Column():
            answer = gr.Textbox(label="Answer")

    send_btn.click(inference, inputs=question, outputs=answer, api_name="predict")
    examples = gr.Examples(
        examples=["I went to the supermarket yesterday.", "Helen is a good swimmer."],
        inputs=[question],
    )


happ.queue(concurrency_count=2, max_size=100).launch(
    share=False, show_api=True, debug=True
)

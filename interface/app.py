import json
import sys
import os
import uuid
import threading

import gradio as gr
from conversation import get_default_conv_template
from dotenv import load_dotenv
from gradio_client import Client
from theme import DeepSquare

load_dotenv()

MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"
N_CONV_MAX = int(os.getenv("N_CONV_MAX", "100"))
STREAM = os.getenv("STREAM", "true").lower() == "true"
ENDPOINT_URL = os.getenv("ENDPOINT_URL", None)

CACHE_DIR = os.getenv("CACHE_DIR", "/opt/models")
RANDOM_SEED = os.getenv("RANDOM_SEED", "true").lower() == "true"
MODEL_NAME = os.getenv("MODEL_NAME", "oasst-pythia-13b")


if ENDPOINT_URL:
    client = Client(ENDPOINT_URL)
    print("Using api for the model")

    def model(prompt, max_tokens=1024):
        return client.submit(
            prompt,  # str representing string value in 'Prompt' Textbox component
            api_name="/predict",
        )

    def conv_generator():
        return get_default_conv_template(MODEL_NAME).copy()

else:
    print("Launching model internally")
    from model import load_model

    model, conv_generator = load_model(MODEL_NAME, CACHE_DIR)


class State:
    def __init__(self, conv_generator):
        """
        Initialize the State object.

        :param conv_generator: A callable that returns a new conversation object.
        :param to_gradio_chatbo4t: A callable that converts conversation objects to Gradio chatbot format.
        :param from_gradio_chatbot: A callable that converts Gradio chatbot format to conversation objects.
        """
        self.lock = threading.Lock()  # Initialize the lock 
        self.conv_generator = conv_generator
        self.current = self.conv_generator()
        self.current_key = str(uuid.uuid4())
        self.clear()

    def clear(self):
        """
        Clear the current state, initializing an empty conversation list and resetting attributes.
        """
        self.filename = "conversations.json"
        self.data = []
        self.data_id_link = {}
        self.data_number = 0
        for n in range(N_CONV_MAX):
            self.data.append({"key": "", "history": None})
        self.write_history()

    def get_state(self):
        """
        Get the current state data.

        :return: A list of conversation dictionaries.
        """
        return self.data

    def load_conversation_by_key(self, key):
        """
        Retrieve one conversation history for a given key.

        :param key: The unique identifier of the conversation.
        :return: The conversation history as a list of tuples (instruction, answer).
        """
        it = self.data_id_link[key]
        self.current = self.data[it]["history"]
        self.current_key = key

        return self.current.to_gradio_chatbot()

    def set_current_history(self):
        print(f"Set current history for conversation id {self.current_key}")
        self.set_history(self.current_key,self.current)

    def set_history(self, key, conversation):
        """
        Set one conversation history for a given key.

        :param key: The unique identifier of the conversation.
        :param history: The conversation history as a list of tuples (instruction, answer).
        """
        if key not in self.data_id_link:
            self.data_id_link[key] = self.data_number
            self.data_number += 1
        it = self.data_id_link[key]
        self.data[it]["key"] = key
        self.data[it]["history"] = conversation
        self.write_history()

    def start_new_conversation(self):
        if not self.current_key or self.current.is_empty() or self.current_key not in self.data_id_link:
            print("Error no conversation present")        
        else:
            self.data_id_link[key] = self.data_number
            self.data_number += 1
            it = self.data_id_link[self.current_key]
            self.data[it]["history"] = self.current
        self.current_key = str(uuid.uuid4())
        conversations.current = conversations.conv_generator()

    def load_full_history_from_file(self, filename=None):
        """
        Load the full conversation history from a file.

        :param filename: The path to the file containing the conversation history. Defaults to the value of self.filename.
        """
        if not filename:
            filename = self.filename
        try:
            self.data_number = 0
            with open(filename, "r") as file:
                data = file.read()
                data_list = json.loads(data)
                for conv in data_list:
                    key = conv["key"]
                    history = conv["history"] 
                    conversation = self.conv_generator()
                    conversation.from_gradio_chatbot(history)
                    self.set_history(key, conversation)
        except FileNotFoundError:
            print("History file not found")
            pass

    def write_history(self):
        """
        Write the modified conversation history to the file.
        """
        data = json.dumps(self.get_data())
        if len(data) == 0:
            return
        with open(self.filename, "w") as file:
            file.write(data)

    def get_data(self):
        """
        Get a list of conversation dictionaries with non-empty keys.

        :return: A list of conversation dictionaries with non-empty keys.
        """
        data_list = []
        for item in self.data:
            if item["key"] and item["key"] != "":
                data_list.append({"key": item["key"], "history": item["history"].to_gradio_chatbot()})
        return data_list



global conversations
conversations = State(conv_generator)


with gr.Blocks(theme=DeepSquare(), css="./style.css") as demo:
    key = gr.State(value=[])

    def load_conversations_list(force_visible=False, write_history=True):
        out = []
        print("Loading conversations list")
        for id in conversations.data:
            if id["history"]:
                out.append(gr.Textbox.update(id["key"], visible=True))
            else:
                out.append(gr.Textbox.update(visible=force_visible))
        if write_history:
            conversations.write_history()
        return out[::-1]

    def add_user_prompt(key, history, instruction):
        print("Add user prompt")
        if len(key) == 0:
            key.append(conversations.current_key)
        history = history + [[instruction, None]]
        conversations.current.append_message(
            conversations.current.roles[0], instruction
        )
        conversations.current.append_message(conversations.current.roles[1], None)
        return history, "", key[0]

    def load_one_conversation(new_key, key, history):
        print("Load one conversations")
        if len(key) == 0:
            key.append(new_key)
        else:
            key[0] = new_key
        history = conversations.load_conversation_by_key(key[0])

        return key[0], history

    def bot(key, history):
        print("Bot speaking")
        prompt = conversations.current.get_prompt()
        history[-1][1] = ""
        for resp_text in model(prompt, max_tokens=1024):
            history[-1][1] += resp_text
            yield history, conversations.filename

        conversations.current.messages[-1][1] = history[-1][1]
        print("Set current history")
        conversations.set_current_history()
        yield history, conversations.filename

    def clear(history):
        conversations.start_new_conversation()
        return "", [], []

    def upload_full_history(file):
        conversations.clear()
        conversations.load_full_history_from_file(file.name)
        return load_conversations_list(force_visible=False, write_history=False)

    def load_first_conversation():
        key = conversations.data[0]["key"]
        return key

    logo = "file/assets/logo.svg"
    with gr.Row():
        with gr.Column(scale=0.2):
            with gr.Row():
                gr.Markdown(
                    f"""
                <h1>
                    <center>
                        <a href="https://www.deepsquare.io" style="display: inline-block; vertical-align: middle;">
                            <img src="{logo}" width="30" height="30"  style="display:block; margin:auto;">
                        </a>
                            on demand chatbot
                    </center>
                </h1>
                """,
                    elem_id="title",
                )
                start_new_btn = gr.Button("Start a new chat")
                load_one_btn = gr.Textbox(
                    placeholder="No conversation loaded",
                    label="Conversation id",
                )
                buttons = []
                for id in conversations.data:
                    b = gr.Button(
                        f"Conversation {id['key']}",
                        visible=False,
                        elem_classes="conversations",
                    )
                    buttons.append(b)
                download = gr.File(
                    conversations.filename,
                    type="file",
                    interactive=True,
                    label="Load / Download conversations",
                )

        with gr.Column(scale=0.8):
            chatbot = gr.Chatbot([], elem_id="chatbot").style(height=380)

            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter",
            ).style(container=False)

    txt.submit(
        add_user_prompt, [key, chatbot, txt], [chatbot, txt, load_one_btn], api_name="predict"
    ).then(bot, [key, chatbot], [chatbot, download], api_name="predict2")

    dep = demo.load(load_conversations_list, None, buttons)

    download.upload(upload_full_history, download, buttons).then(
        clear, chatbot, [load_one_btn, key, chatbot]
    ).then(load_conversations_list, None, buttons)

    load_one_btn.submit(
        load_one_conversation, [load_one_btn, key, chatbot], [load_one_btn, chatbot]
    )
    start_new_btn.click(clear, chatbot, [load_one_btn, key, chatbot]).then(
        load_conversations_list, None, buttons
    )
    for button in buttons:
        button.click(
            load_one_conversation, [button, key, chatbot], [load_one_btn, chatbot]
        ).then(load_conversations_list, None, buttons)

demo.queue(concurrency_count=2, max_size=100).launch(
    share=MOCK_MODE == "true", show_api=False, debug=True
)

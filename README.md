# Proto-Chatbot 

- OpenSource Chatbot Gradio Interface 
- Can run locally or on DeepSquare distributed super computer 

## Setup

```
conda create -n chatbot python=3.10
conda activate chatbot
conda install -y cudatoolkit
pip install -r requirements.txt
```

```
/opt/models/
├── models--deepsquare--alpaca-lora-7b
│   ├── tokenizer
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   └── tokenizer.model
│   ├── 4bit-128g.safetensors
│   ├── config.json
│   ├── generation_config.json
│   ├── pytorch_model.bin.index.json
│   ├── pytorch_model-00001-of-00007.bin
│   ├── ...
│   └── pytorch_model-00007-of-00007.bin
├── models--deepsquare--alpaca-lora-13b
│   └── ...
└── ...
```

### Containerized environment

```
docker build -t squarefactory/chatbot .
```

## Command line

```
python model.py -s="how are you" -s="are you sure?"
```


from docker image:

```
docker run \
  --gpus all
  --ipc=host
  --ulimit memlock=-1 --ulimit stack=67108864  \
  -v ${PWD}:/app \
  -v /opt/models:/opt/models \
  squarefactory/chatbot \
  python model.py -s="how to end capitalism?"
```

## Gradio app

> Tips: To set model directory, use `MODELS_DIR` env variable

```
gradio view.py
```


### Decouple model and view

export gradio engine.py
export ENGINE_URL=https://url_to_engine_api gradio view.py

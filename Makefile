# Load environment variables from .env file
ifneq (,$(wildcard .env))
    include .env
    export
endif

# Default values if not set in .env
MODEL_DIR ?= pretrained_models/Fun-CosyVoice3-0.5B
PORT ?= 8000
HOST ?= 0.0.0.0
DEVICE ?= 0
IMAGE_NAME ?= cosyvoice:openai-latest
CONDA_ENV ?= cosyvoice
VOICE_DIR ?= /workspace/voices
HOST_VOICE_DIR ?= /mnt/e/OneDrive/data_sync/audio_samples

CONDA_ACTIVATE := . $$(conda info --base)/etc/profile.d/conda.sh && conda activate $(CONDA_ENV) &&

gradio:
	python webui.py --port 15000 --model_dir $(MODEL_DIR) --device $(DEVICE)

openai:
	python runtime/python/fastapi/openai_compatible.py --model_dir $(MODEL_DIR) --port $(PORT) --host $(HOST) --device $(DEVICE) --voice_dir $(VOICE_DIR)

openai-jit:
	$(CONDA_ACTIVATE) python runtime/python/fastapi/openai_compatible.py --model_dir $(MODEL_DIR) --port $(PORT) --host $(HOST) --device $(DEVICE) --load_jit

openai-trt:
	$(CONDA_ACTIVATE) python runtime/python/fastapi/openai_compatible.py --model_dir $(MODEL_DIR) --port $(PORT) --host $(HOST) --device $(DEVICE) --load_trt

openai-vllm:
	$(CONDA_ACTIVATE) python runtime/python/fastapi/openai_compatible.py --model_dir $(MODEL_DIR) --port $(PORT) --host $(HOST) --device $(DEVICE) --load_vllm

openai-fp16:
	$(CONDA_ACTIVATE) python runtime/python/fastapi/openai_compatible.py --model_dir $(MODEL_DIR) --port $(PORT) --host $(HOST) --device $(DEVICE) --fp16

up:
	MODEL_DIR=$(MODEL_DIR) PORT=$(PORT) HOST=$(HOST) DEVICE=$(DEVICE) VOICE_DIR=$(VOICE_DIR) docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

build:
	cp .env.example .env &&
	git submodule init && git submodule update &&
	docker build --network=host -t $(IMAGE_NAME) -f docker/Dockerfile .

test-api:
	bash tests/test_api.sh

test-py:
	$(CONDA_ACTIVATE) SKIP_SFT_TESTS=true python tests/test_openai_api.py

.PHONY: gradio openai openai-jit openai-trt openai-vllm openai-fp16 up down logs build test test-api test-py
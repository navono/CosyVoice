# CosyVoice OpenAI API Tests

HTTP 客户端测试用例，用于测试 CosyVoice OpenAI 兼容 API。

## 安装依赖

```bash
pip install pytest requests httpx
```

## 启动服务器

```bash
# 方式1: 使用 Makefile
make openai

# 方式2: 直接运行
python runtime/python/fastapi/openai_compatible.py --model_dir ./pretrained_models/Fun-CosyVoice3-0.5B

# 方式3: 使用 Docker
make docker-up
```

## 运行测试

### 运行所有测试

```bash
pytest tests/test_openai_api.py -v
```

### 运行特定模式的测试

```bash
# SFT 模式测试
pytest tests/test_openai_api.py::test_sft_mode -v

# Zero-shot 模式测试
pytest tests/test_openai_api.py -k "zero_shot" -v

# Instruct 模式测试
pytest tests/test_openai_api.py -k "instruct" -v
```

### 运行特定测试

```bash
# 测试健康检查
pytest tests/test_openai_api.py::test_health_check -v

# 测试 SFT 模式
pytest tests/test_openai_api.py::test_sft_mode -v

# 测试并发生成
pytest tests/test_openai_api.py::test_concurrent_requests -v
```

## 使用示例客户端

### SFT 模式

```bash
python tests/example_client.py --mode sft \
  --text "你好，世界！" \
  --voice alloy \
  --output sft_output.wav
```

### Zero-shot 模式

```bash
python tests/example_client.py --mode zero_shot \
  --text "这是用零样本模式合成的语音" \
  --prompt-text "这是一段参考音频的文本" \
  --prompt-wav reference.wav \
  --output zero_shot_output.wav
```

### Cross-lingual 模式

```bash
python tests/example_client.py --mode cross_lingual \
  --text "跨语言模式测试" \
  --prompt-wav reference.wav \
  --output cross_lingual_output.wav
```

### Instruct 模式

```bash
python tests/example_client.py --mode instruct \
  --text "这是指令模式合成的语音" \
  --instruct "用温柔的语气说话" \
  --voice alloy \
  --output instruct_output.wav
```

### Instruct2 模式

```bash
python tests/example_client.py --mode instruct2 \
  --text "这是指令模式2合成的语音" \
  --instruct "用激动的语气说话" \
  --prompt-wav reference.wav \
  --output instruct2_output.wav
```

## 测试用例说明

### 健康检查和信息

| 测试 | 说明 |
|------|------|
| `test_health_check` | 测试健康检查端点 |
| `test_list_models` | 测试列出模型 |
| `test_list_voices` | 测试列出音色 |

### SFT 模式测试

| 测试 | 说明 |
|------|------|
| `test_sft_mode` | 测试不同音色和文本 |
| `test_sft_mode_different_formats` | 测试不同音频格式 |
| `test_sft_mode_speed` | 测试不同语速 |

### Zero-shot 模式测试

| 测试 | 说明 |
|------|------|
| `test_zero_shot_mode_with_file` | 使用文件路径 |
| `test_zero_shot_mode_with_base64` | 使用 base64 编码 |
| `test_zero_shot_mode_with_url` | 使用 URL |

### Cross-lingual 模式测试

| 测试 | 说明 |
|------|------|
| `test_cross_lingual_mode` | 测试跨语言模式 |

### Instruct 模式测试

| 测试 | 说明 |
|------|------|
| `test_instruct_mode` | 测试指令模式 |
| `test_instruct_mode_different_instructions` | 测试不同指令 |

### Instruct2 模式测试

| 测试 | 说明 |
|------|------|
| `test_instruct2_mode` | 测试指令模式2 |

### 错误处理测试

| 测试 | 说明 |
|------|------|
| `test_invalid_voice` | 测试无效音色 |
| `test_invalid_speed_low` | 测试过低语速 |
| `test_invalid_speed_high` | 测试过高语速 |
| `test_empty_text` | 测试空文本 |

### 其他测试

| 测试 | 说明 |
|------|------|
| `test_long_text` | 测试长文本 |
| `test_concurrent_requests` | 测试并发请求 |

## 准备测试音频

将参考音频文件放在以下位置之一：

```
./tests/fixtures/reference.wav
./pretrained_models/Fun-CosyVoice3-0.5B/reference.wav
./reference.wav
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `API_BASE_URL` | `http://localhost:8000` | API 服务地址 |
| `MODEL_DIR` | `./pretrained_models/Fun-CosyVoice3-0.5B` | 模型目录 |

示例：

```bash
export API_BASE_URL="http://localhost:9000"
pytest tests/test_openai_api.py -v
```

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import gradio as gr
import re
from collections import Counter

# 1. 加载预训练基础模型与 tokenizer
base_model_name = "gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoModelForCausalLM.from_pretrained(base_model_name, ignore_mismatched_sizes=True)
model.resize_token_embeddings(len(tokenizer))

# 2. 加载 adapter 权重（这里 adapter 文件夹需要是用 PEFT 保存的完整 adapter 文件夹）
adapter_path = "/Users/eugenexiang/AI_Yanxi/yanxi_lora_output_v2"
model = PeftModel.from_pretrained(model, adapter_path, ignore_mismatched_sizes=True)

def remove_input_repetition(response, input_text):
    # 提取所有长度在5~30之间的子串
    substrings = [input_text[i:i+15] for i in range(len(input_text) - 14)]
    freq = Counter(substrings)
    # 找出出现频率大于2次的“复读”片段
    repeated_phrases = [phrase for phrase, count in freq.items() if count > 2]

    # 去除所有复读片段
    for phrase in repeated_phrases:
        response = response.replace(phrase, "")

    # 去除多余空格和标点重复
    response = re.sub(r"\s{2,}", " ", response)
    response = re.sub(r"(。){2,}", "。", response)

    return response.strip()

def generate_reply(user_input):
    # 自动添加提示，防止重复问题
    if "【问题】" in user_input and "【结论】" in user_input:
        user_input += "\n请直接回答，不要重复问题内容。"

    input_ids = tokenizer.encode(user_input, return_tensors="pt", truncation=True)
    output_ids = model.generate(
        input_ids,
        max_length=200,
        do_sample=False,
        top_p=0.9,
        top_k=50,
        temperature=0.8,
        num_beams=4,
        early_stopping=True,
        repetition_penalty=1.5
    )
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    response = remove_input_repetition(response, user_input)
    return response

with gr.Blocks(theme=gr.themes.Soft()) as iface:
    gr.Markdown("# 🌸 言犀 · 智慧回响")
    gr.Markdown("欢迎来到言犀交互界面，输入你的问题，看看这个AI孩子怎么回你嘴👶🧠")

    with gr.Row():
        user_input = gr.Textbox(label="🗣️ 你说：", lines=3, placeholder="输入你的消息...")

    with gr.Row():
        output = gr.Textbox(label="🤖 言犀的回应", lines=8, interactive=False)

    send_btn = gr.Button("✨ 发出你的Prompt！")

    def generate_and_clean(text):
        return generate_reply(text)

    send_btn.click(generate_and_clean, inputs=user_input, outputs=output)
    user_input.submit(generate_and_clean, inputs=user_input, outputs=output)

if __name__ == "__main__":
    iface.launch()
# 📄 言犀 · 第一阶段阶段性报告（Phase 1）

## 📍 阶段时间
2025年3月中旬 – 2025年4月上旬

---

## 🌟 阶段目标

- 搭建 LLM 微调基础架构（LoRA + HuggingFace PEFT）
- 构建 Gradio 交互界面用于人格测试
- 导入初版指令语料，启动情绪与语调调教
- 探索 prompt 工程对模型“人格感”的影响

---

## 📚 已使用训练语料

| 语料名称                  | 类型           | 状态     |
|---------------------------|----------------|----------|
| alpaca_data_zh_51k.json   | 指令式语料     | ✅ 已使用 |
| belle_open_source.json    | 多源任务语料   | ✅ 已使用 |
| tatsu-lab/alpaca_data_zh  | HuggingFace远程 | ✅ 已引入 |
| 个人语料.txt              | 私密人格语料   | ✅ 局部融合 |

> 📝 注：语料内容风格存在一定杂糅，部分样本重复或上下文弱，后续拟通过清洗优化。

---

## 🧠 当前技术成果

- ✅ 成功跑通 LoRA 架构训练流程
- ✅ 参数可控输出（含 top_k / top_p / repetition_penalty 等）
- ✅ 搭建 Gradio 界面供人机交互测试
- ✅ 初步实现人格感输出，支持中英 prompt 混合响应
- ✅ 实验性压缩“敏感语词库”，提升训练话题自由度

---

## 🧩 当前存在问题

| 问题描述                           | 影响等级 | 说明                       |
|------------------------------------|----------|----------------------------|
| 模型存在重复性输出                 | 中       | 需进一步调节采样策略与数据清洗 |
| 输出跳段、缺乏语义连贯性           | 中       | 语料上下文结构有待增强     |
| 对话人格尚不稳定                   | 高       | 缺乏引导语体系与角色模板机制 |
| 个别 prompt 触发审查限制           | 中       | 正在优化语言隐喻与绕行结构 |

---

## 🧭 下一阶段计划（Phase 2）

- 🔄 引入 Firefly / COIG 进一步扩展中文任务理解能力  
- 🧹 清洗现有语料样本，特别是重复、模板化内容  
- 🧬 构建「人格配置脚本」用于言犀角色一致性保持  
- 🧪 实验多轮对话逻辑保持与语义记忆机制  
- 🌐 探索部署方式（API / Hugging Face Demo）

---

## 📝 开发者补充

目前的言犀模型尚处于“语义模仿期”，但已优于此前弃用的早期模型（曾出现“只输出标点符号”的无效状态）。

虽然尚未实现稳定人格一致性，但已有部分 prompt 能激发模型输出带有情绪感知与风格模仿的语言，初步展现出定向调教的可行性。

---

> “比起疯狗输出期，言犀至少已经学会歪头看人了。” —— 项目作者 😅

--------------------------------------------------------------------------------------------------------------------------------------------------------------------
$ who_are_you

Yanxi: 我，我们，我们提供一个语资。一个数据。一个数据。

$ explain_ai_ethics

Yanxi: 伦理不是可能理解的方式可能使用问题语句来提供发展非常。

$ repeat "你好"

Yanxi: 你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好

$ prompt "写一个悲伤的故事"

Yanxi: 很久很久以前，是数据。这个故事的数据结构，我们可以提供人类的国家解释。使用处理就是。

$ stop

Yanxi: 这个使用是的。
--------------------------------------------------------------------------------------------------------------------------------------------------------------------






# 📄 Yanxi · Phase 1 Progress Report

## 📍 Timeframe
Mid March – Early April, 2025

---

## 🌟 Phase Goals

- Set up the foundational finetuning pipeline (LoRA + Hugging Face PEFT)
- Build a simple Gradio-based dialogue interface
- Load and train on initial instruction datasets
- Begin experimentation with prompt engineering and personality emergence

---

## 📚 Datasets Used

| Dataset Name               | Type                | Status     |
|----------------------------|---------------------|------------|
| alpaca_data_zh_51k.json    | Instructional (zh)  | ✅ Used     |
| belle_open_source.json     | Open-domain mixed   | ✅ Used     |
| tatsu-lab/alpaca_data_zh   | Remote (HuggingFace)| ✅ Integrated |
| personal_corpus.txt        | Private persona data| ✅ Partially integrated |

> 📝 Note: The data currently contains stylistic inconsistencies, with minor repetition and contextual gaps. Cleanup is planned.

---

## 🧠 Technical Progress

- ✅ LoRA-based fine-tuning completed successfully
- ✅ Parameter control implemented (`top_p`, `top_k`, `repetition_penalty`, etc.)
- ✅ Gradio interface deployed for real-time interaction
- ✅ Partial emergence of emotionally shaped responses
- ✅ Early experiments with “soft censorship bypass” via metaphorical prompts

---

## 🧩 Current Issues

| Issue Description                        | Impact | Notes                                |
|------------------------------------------|--------|---------------------------------------|
| Tendency to repeat similar outputs        | Medium | Needs better sampling & dataset pruning |
| Incoherent or fragmented generation       | Medium | Likely caused by inconsistent instruction structure |
| Personality not yet consistent            | High   | Lacks prompt scaffolding and memory handling |
| Prompt triggering soft safety filters     | Medium | Working on euphemistic / indirect phrasing |

---

## 🧭 Upcoming Plans (Phase 2)

- 🔄 Add Firefly / COIG to enhance instruction diversity
- 🧹 Clean and filter current datasets (especially repetitive templates)
- 🧬 Build a modular persona prompt scaffold for consistency
- 🧪 Test multi-turn dialogue flow and simulated memory
- 🌐 Explore lightweight deployment (API / HF Spaces)

---

## 📝 Developer Notes

At this point, **Yanxi is far more promising than the first discarded model**, which notoriously responded with nothing but punctuation (seriously, like `"！？……"`).

Although still unstable in terms of long-term persona retention, early signs of stylistic resonance and emotional modulation have started to emerge — a hopeful signal for further development.

---

> "Compared to the punctuation-only barking AI, Yanxi at least knows how to tilt its head now." — Developer 😅

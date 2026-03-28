# 🏥 Medical AI Assistant — Fine-Tuning LLaMA 3.2-3B with SFT + DPO

<div align="center">

[![Model on HuggingFace](https://img.shields.io/badge/🤗%20Model-kumarsarthak98%2Fmedical--llama--3.2--3b--sft--dpo-blue)](https://huggingface.co/kumarsarthak98/medical-llama-3.2-3b-sft-dpo)
[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-HuggingFace%20Spaces-orange)](https://huggingface.co/spaces/kumarsarthak98/medical-ai-assistant)
[![WandB Report](https://img.shields.io/badge/📊%20Training%20Logs-Weights%20%26%20Biases-yellow)](https://wandb.ai/kumarsarthakofficial-birla-institute-of-technology-mesra/medical-llm-finetune/reports/Projects-medical-llm-finetune--VmlldzoxNjM0NjAwMA?accessToken=2qhqj2j2up2qd98epspklrd3qo9v9cyzks3eqf8zs6csryfqv9fubw535j2ce8dc)
[![GitHub](https://img.shields.io/badge/GitHub-KumarSarthak--Official-black?logo=github)](https://github.com/KumarSarthak-Official/Medical_AI_Assistant-with-SFT_DPO)

</div>

---

## 📌 Project Overview

This project fine-tunes **Meta's LLaMA 3.2-3B-Instruct** model into a domain-specific medical AI assistant capable of answering clinical and biomedical questions with accuracy and depth. The training pipeline combines two modern alignment techniques:

- **SFT (Supervised Fine-Tuning)** — Teaches the model to follow the medical Q&A format using high-quality labeled examples
- **DPO (Direct Preference Optimization)** — Refines the model's behavior by training it to prefer correct, complete medical answers over incorrect, vague, or overcautious ones

The project evolved across **three iterative attempts**, each improving dataset quality, training configuration, and alignment strategy — ultimately achieving a **ROUGE-1 of 0.7674** and **Perplexity of 2.95** in the final run.

---

## 🔗 Quick Links

| Resource | Link |
|---|---|
| 🤗 Fine-tuned Model | [kumarsarthak98/medical-llama-3.2-3b-sft-dpo](https://huggingface.co/kumarsarthak98/medical-llama-3.2-3b-sft-dpo) |
| 🚀 Live Demo (Spaces) | [medical-ai-assistant](https://huggingface.co/spaces/kumarsarthak98/medical-ai-assistant) |
| 📊 WandB Training Report | [Full Experiment Logs](https://wandb.ai/kumarsarthakofficial-birla-institute-of-technology-mesra/medical-llm-finetune/reports/Projects-medical-llm-finetune--VmlldzoxNjM0NjAwMA?accessToken=2qhqj2j2up2qd98epspklrd3qo9v9cyzks3eqf8zs6csryfqv9fubw535j2ce8dc) |
| 💻 GitHub Repo | [Medical_AI_Assistant-with-SFT_DPO](https://github.com/KumarSarthak-Official/Medical_AI_Assistant-with-SFT_DPO) |

---

## 🚀 Run the Notebooks

Click any badge below to open the notebook directly in Google Colab:

| Notebook | Description | Colab |
|---|---|---|
| 1st Attempt | Baseline: toy dataset + PubMedQA, 1 epoch | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KumarSarthak-Official/Medical_AI_Assistant-with-SFT_DPO/blob/main/notebooks/Medical_LLM_Full_Pipeline_1st_Attempt.ipynb) |
| 2nd Attempt | Refined DPO, 3 epochs, same dataset | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KumarSarthak-Official/Medical_AI_Assistant-with-SFT_DPO/blob/main/notebooks/Medical_LLM_Full_Pipeline_2nd_Attempt.ipynb) |
| **Final** ✅ | USMLE data, NEFTune, best results | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KumarSarthak-Official/Medical_AI_Assistant-with-SFT_DPO/blob/main/notebooks/Medical_LLM_Full_Pipeline_Final.ipynb) |

> ⚠️ **Note:** These notebooks require a GPU runtime. In Colab go to `Runtime → Change runtime type → T4 GPU`. You will also need to set your `WANDB_API_KEY` and `HF_TOKEN` as Colab secrets (`🔑` icon in the left sidebar).

---

## 🏗️ Architecture & Technical Stack

```
Base Model:         unsloth/Llama-3.2-3B-Instruct-bnb-4bit
Parameters:         3.31B total | 97.25M trainable (2.94% via LoRA)
Quantization:       4-bit NF4 (via BitsAndBytes)
Adapter:            LoRA (r=64, alpha=128, dropout=0.05)
Target Layers:      q_proj, k_proj, v_proj, o_proj,
                    gate_proj, up_proj, down_proj
Training:           SFT (3 epochs, 1521 steps) → DPO (1 epoch, 928 steps)
Hardware:           Kaggle T4 GPU (also runnable on Colab T4)
Serving:            FastAPI (streaming SSE) + Gradio frontend
```

**Key Libraries:** `unsloth`, `transformers`, `trl`, `peft`, `bitsandbytes`, `wandb`, `fastapi`, `gradio`

---

## 🔬 Experimental Journey — Three Attempts

The project went through three major iterations. Each attempt revealed weaknesses in either data quality or training configuration, driving the next round of improvements.

---

### Attempt 1 — Baseline [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KumarSarthak-Official/Medical_AI_Assistant-with-SFT_DPO/blob/main/notebooks/Medical_LLM_Full_Pipeline_1st_Attempt.ipynb)

**Goal:** Establish a working end-to-end pipeline for medical domain fine-tuning.

**Datasets Used:**
- `lavita/medical-qa-shared-task-v1-toy` — A small toy multiple-choice dataset (few hundred examples)
- `qiaojin/PubMedQA` (`pqa_labeled`) — Biomedical research Q&A with context paragraphs

**Training Configuration:**

| Parameter | Value |
|---|---|
| LoRA Rank (r) | 64 |
| LoRA Alpha | 128 |
| SFT Epochs | 1 |
| SFT Learning Rate | 2e-4 |
| DPO Beta | 0.1 |
| DPO Learning Rate | 5e-5 |
| DPO Steps | 110 |
| Gradient Accumulation | 8 |
| Sequence Packing | Disabled |
| NEFTune | Not used |

**Measured Results:**

| Metric | Value |
|---|---|
| SFT eval/loss | 1.6265 |
| SFT train/loss | 1.6377 |
| SFT train/epoch | 1 |
| SFT global steps | 58 |
| eval/samples_per_second | 5.648 |
| eval/steps_per_second | 2.824 |
| DPO avg_loss (final) | 0.01378 |
| DPO loss (final step) | 0.00066 |

<!-- Add screenshot: Screenshot__65_.png — run summary table showing eval/loss 1.62653 -->
<!-- ![Attempt 1 Run Summary](assets/attempt1_run_summary.png) -->

**Why This Wasn't Enough:**

The pipeline ran successfully and training loss converged, but outputs were shallow. The `lavita/medical-qa-shared-task-v1-toy` dataset is deliberately tiny — designed for benchmarking, not training. Answers were often just single option labels with no clinical reasoning. The model learned format but not medical substance. With only **58 training steps**, the model had far too little exposure to medical content to generalize. The DPO stage (Beta=0.1) also gave the model too much freedom to deviate from the SFT reference.

**Decision:** Run more epochs and strengthen DPO before deciding on a dataset change.

---

### Attempt 2 — Refined Training [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KumarSarthak-Official/Medical_AI_Assistant-with-SFT_DPO/blob/main/notebooks/Medical_LLM_Full_Pipeline_2nd_Attempt.ipynb)

**Goal:** Push the existing dataset further with more training and stronger DPO, to understand whether the ceiling was the data or the training setup.

**Datasets Used:** Same as Attempt 1 (toy + PubMedQA)

**Key Changes from Attempt 1:**
- Increased SFT epochs from 1 → **3**
- Added explicit 90/10 train/validation splitting to track overfitting
- Richer DPO negative strategies: mismatched answer, incomplete answer, overcautious answer, binary flip
- DPO Beta increased 0.1 → **0.2** to keep policy closer to the SFT reference
- DPO Learning Rate reduced 5e-5 → **5e-6** to prevent over-optimization

**Measured Results:**

| Metric | Value |
|---|---|
| SFT Final train loss | 1.4070 |
| SFT eval/loss | 1.6766 |
| SFT train/epoch | 3 |
| SFT global steps | 174 |
| SFT logged train/loss | 1.1578 |
| eval/samples_per_second | 6.607 |
| eval/steps_per_second | 3.303 |
| **Validation Loss** | **1.6811** |
| **Perplexity** | **5.3717** |
| **ROUGE-1** | **0.3641** |
| **ROUGE-2** | **0.1371** |
| **ROUGE-L** | **0.2813** |
| DPO Steps | 500 |
| DPO avg_loss (final) | 0.10156 |
| DPO Final average loss | 0.0979 |
| DPO loss (final step) | 0.00053 |

<!-- Add screenshot: Screenshot__68_.png — SFT complete, train loss 1.4070, run history bars -->
<!-- ![Attempt 2 SFT Training](assets/attempt2_sft_training.png) -->

<!-- Add screenshot: Screenshot__71_.png — ROUGE evaluation results 0.3641 / 0.1371 / 0.2813 -->
<!-- ![Attempt 2 ROUGE Scores](assets/attempt2_rouge_scores.png) -->

**Why This Still Wasn't Enough:**

Running 3 epochs improved training loss (1.64 → 1.16) and DPO showed better convergence (500 steps vs 110). But ROUGE-1 of **0.3641** means the model was only matching about 1 in 3 words from reference answers. The core problem was now clear: **the dataset itself was the bottleneck**. The toy dataset contains short, shallow answers that don't capture clinical reasoning. No amount of additional epochs could fix bad training data. PubMedQA alone, while research-heavy, doesn't provide the breadth of decision-making a medical assistant needs.

**Decision:** Completely replace the toy dataset with USMLE board-level questions at scale, and enable sequence packing + NEFTune noise for better GPU efficiency and generalization.

---

### Attempt 3 — Final ✅ [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KumarSarthak-Official/Medical_AI_Assistant-with-SFT_DPO/blob/main/notebooks/Medical_LLM_Full_Pipeline_Final.ipynb)

**Goal:** Build a production-quality model using high-difficulty clinical reasoning data at scale.

**Datasets Used:**
- `GBaker/MedQA-USMLE-4-options` — Real USMLE board exam questions requiring multi-step clinical reasoning. **8,000 examples** sampled.
- `qiaojin/PubMedQA` (`pqa_labeled`) — Full **1,000** biomedical research Q&A pairs

**Total SFT Dataset:** ~8,100 training examples — up from ~500 in previous attempts

**What Changed and Why:**

| Change | Attempt 1 & 2 | Final | Reason |
|---|---|---|---|
| Primary Dataset | Toy medical Q&A | USMLE board questions (8,000) | Board-level questions demand real clinical reasoning, not label selection |
| Dataset Scale | ~500 examples | ~8,100 examples | More diverse, high-quality data is the single biggest lever for performance |
| SFT Steps | 58 / 174 | **1,521** | 8.7x more training steps than Attempt 2 |
| Sequence Packing | Disabled | Enabled (`packing=True`) | Maximizes T4 GPU utilization by packing short sequences together |
| NEFTune Noise | Not used | `neftune_noise_alpha=5.0` | Adds embedding noise during training — proven to improve instruction-following |
| DPO Beta | 0.1 / 0.2 | 0.2 | Keeps fine-tuned policy closer to SFT reference, prevents reward hacking |
| DPO Learning Rate | 5e-5 / 5e-6 | 5e-6 | Prevents over-optimization on preference pairs |
| DPO Steps | 110 / 500 | **928** | More alignment steps over richer USMLE preference pairs |

**Final Training Configuration:**

| Parameter | Value |
|---|---|
| LoRA Rank (r) | 64 |
| LoRA Alpha | 128 |
| LoRA Dropout | 0.05 |
| Trainable Parameters | 97,255,424 / 3,310,005,248 (2.94%) |
| SFT Epochs | 3 |
| SFT Total Steps | 1,521 |
| SFT Learning Rate | 2e-4 |
| SFT LR Scheduler | Cosine |
| NEFTune Noise Alpha | 5.0 |
| Batch Size (per device) | 1 |
| Gradient Accumulation | 8 (effective batch = 16) |
| DPO Beta | 0.2 |
| DPO Learning Rate | 5e-6 |
| DPO Steps | 928 |
| Max Sequence Length | 2048 |
| Optimizer | `adamw_torch` |

---

## 📊 Results — Complete Metrics Comparison

### Scores Across All Attempts

| Metric | Attempt 1 | Attempt 2 | **Final (Attempt 3)** |
|---|---|---|---|
| SFT eval/loss | 1.6265 | 1.6766 | — |
| SFT train/loss (final) | 1.6377 | 1.1578 | **0.0000** (fully converged) |
| SFT Epochs | 1 | 3 | **3** |
| SFT Steps | 58 | 174 | **1,521** |
| Validation Loss | — | 1.6811 | **1.0828** ⬇️ |
| **Perplexity** | — | 5.3717 | **2.9528** ⬇️ |
| **ROUGE-1** | — | 0.3641 | **0.7674** ⬆️ |
| **ROUGE-2** | — | 0.1371 | **0.7010** ⬆️ |
| **ROUGE-L** | — | 0.2813 | **0.7548** ⬆️ |
| DPO Steps | 110 | 500 | **928** |
| DPO avg_loss (final) | 0.01378 | 0.10156 | **0.01378** |

<!-- Add screenshot: Screenshot__73_.png — Final ROUGE 0.7674 / 0.7010 / 0.7548, Perplexity 2.9528 -->
<!-- ![Final Evaluation Results](assets/final_evaluation_results.png) -->

<!-- Add screenshot: Screenshot__74_.png — Final SFT: 1521/1521 steps, 3 epochs, 97M trainable params -->
<!-- ![Final SFT Training Complete](assets/final_sft_training.png) -->

### What the Numbers Mean

The improvement from Attempt 2 → Final is dramatic. ROUGE-1 jumped from **0.36 → 0.77** — the final model correctly reproduces more than 3 out of 4 words present in a reference medical answer. ROUGE-2 at **0.70** shows strong bigram overlap, confirming the model captures real clinical terminology rather than isolated keywords. Perplexity dropped from **5.37 → 2.95**, indicating the model is far more confident and calibrated on medical text. All of this is attributable primarily to the dataset upgrade: USMLE board questions provide the clinical reasoning chains that teach a model *how to think* about medicine.

### Sample Prediction vs Reference (Final Model)

**Prediction:**
> The study shows that post-tonsillectomy late haemorrhage is more frequently observed in the night-time. The authors suggest that the night-time haemorrhage could be a sign of a more serious condition.

**Reference:**
> The incidence of post-tonsillectomy late haemorrhage in our study population was 1.78%. A statistically significant difference was found between night-time and day-time haemorrhages...

The model correctly identifies the key clinical finding and frames it as a research-style conclusion — exactly the kind of structured, evidence-aware response the USMLE training data enabled.

---

## 📈 Training Curves (WandB)

All training runs were tracked with [Weights & Biases](https://wandb.ai/kumarsarthakofficial-birla-institute-of-technology-mesra/medical-llm-finetune/reports/Projects-medical-llm-finetune--VmlldzoxNjM0NjAwMA?accessToken=2qhqj2j2up2qd98epspklrd3qo9v9cyzks3eqf8zs6csryfqv9fubw535j2ce8dc).

<!-- Add screenshot: Screenshot__82_.png — WandB train/loss curves across all runs -->
<!-- ![WandB Training Loss](assets/wandb_train_loss.png) -->

<!-- Add screenshot: Screenshot__83_.png — WandB eval/loss curves -->
<!-- ![WandB Eval Loss](assets/wandb_eval_loss.png) -->

<!-- Add screenshot: Screenshot__84_.png — WandB DPO metrics: dpo/loss, dpo/avg_loss -->
<!-- ![WandB DPO Metrics](assets/wandb_dpo_metrics.png) -->

**SFT Training Loss** — The final 3-epoch USMLE run shows a smooth, sustained descent over 1,521 steps with the cosine LR schedule, ultimately converging to a final train loss of **0.0000**.

**Eval Loss** — The final model achieves eval/loss of **1.0828**, the lowest across all runs. The best checkpoint is loaded automatically at end of training.

**DPO Metrics** — The `dpo/avg_loss` curve converges cleanly from ~0.3 → ~0.014 over 928 steps. The final `dpo/loss` at the last step is near zero, confirming stable alignment without reward hacking.

> **View the full interactive training dashboard →** [WandB Report](https://wandb.ai/kumarsarthakofficial-birla-institute-of-technology-mesra/medical-llm-finetune/reports/Projects-medical-llm-finetune--VmlldzoxNjM0NjAwMA?accessToken=2qhqj2j2up2qd98epspklrd3qo9v9cyzks3eqf8zs6csryfqv9fubw535j2ce8dc)

---

## 🚀 Deployment Architecture

This project supports **two deployment modes** depending on your hardware and setup. Both expose the same Gradio chat interface.

---

### Mode 1 — FastAPI Backend (GPU Server / Cloud)

Best for: production servers, cloud VMs, machines with a dedicated NVIDIA GPU.

```
┌─────────────────────────────────────────────────────────┐
│                     User (Browser)                      │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP
┌──────────────────────▼──────────────────────────────────┐
│              Gradio Frontend  (gradio_app.py)            │
│        Lightweight chat UI — streams SSE tokens          │
└──────────────────────┬──────────────────────────────────┘
                       │ POST /chat  (Server-Sent Events)
┌──────────────────────▼──────────────────────────────────┐
│           FastAPI Backend  (app.py)                      │
│  Loads model in 4-bit via Unsloth + TextIteratorStreamer │
│  Model: kumarsarthak98/medical-llama-3.2-3b-sft-dpo     │
└─────────────────────────────────────────────────────────┘
```

---

### Mode 2 — LM Studio (Local / Offline / No GPU Required)

Best for: local testing, recruiters evaluating the project, machines without a dedicated GPU. Runs **entirely offline** using the 4-bit GGUF export of the model — no Python backend, no CUDA, no API keys needed.

```
┌─────────────────────────────────────────────────────────┐
│                     User (Browser)                      │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP
┌──────────────────────▼──────────────────────────────────┐
│              Gradio Frontend  (gradio_app.py)            │
│     Streams OpenAI-compatible SSE from LM Studio        │
└──────────────────────┬──────────────────────────────────┘
                       │ POST /v1/chat/completions
┌──────────────────────▼──────────────────────────────────┐
│           LM Studio Local Server                         │
│  Runs the GGUF model on CPU/GPU via llama.cpp            │
│  Endpoint: http://127.0.0.1:1234                        │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Running Locally

Choose the mode that matches your setup:

---

### ⚡ Mode 1 — FastAPI (requires NVIDIA GPU)

**Prerequisites:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install unsloth transformers fastapi uvicorn gradio httpx pydantic
```

Or from `requirements.txt`:
```bash
pip install -r requirements.txt
```

**Step 1 — Start the FastAPI backend:**
```bash
python app.py
# Loads model from HuggingFace, serves at http://localhost:8000
```

**Step 2 — Launch the Gradio frontend (new terminal):**
```bash
python gradio_app.py
# UI available at http://localhost:7860
```

---

### 🖥️ Mode 2 — LM Studio (works on any machine, fully offline)

> No GPU required. No Python backend. No API keys. Runs on Windows, Mac, and Linux.

**Step 1 — Download LM Studio**

Go to [lmstudio.ai](https://lmstudio.ai) and install it for your OS.

**Step 2 — Download the GGUF model**

Inside LM Studio, search for:
```
kumarsarthak98/medical-llama-3.2-3b-sft-dpo
```
Download the **4-bit GGUF** variant (Q4_K_M recommended for best speed/quality balance).

**Step 3 — Start the Local Server**

In LM Studio:
- Go to the **"Local Server"** tab (the `↔` icon)
- Select the downloaded model
- Click **"Start Server"**
- Server will run at `http://127.0.0.1:1234`

**Step 4 — Install minimal Python dependencies:**
```bash
pip install gradio httpx
```

**Step 5 — Launch the Gradio frontend:**
```bash
python gradio_app.py
# UI available at http://localhost:7860
```

That's it — open your browser at `http://localhost:7860` and start chatting with the model entirely offline.

---

### 📊 Run ROUGE Evaluation
```bash
python evaluate.py \
  --model_id kumarsarthak98/medical-llama-3.2-3b-sft-dpo \
  --test_data medqa_test.json
```

---

## 📁 Repository Structure

```
Medical_AI_Assistant-with-SFT_DPO/
│
├── Medical_LLM_Full_Pipeline_1st_Attempt.ipynb   # Baseline: toy dataset, 1 epoch
├── Medical_LLM_Full_Pipeline_2nd_Attempt.ipynb   # 3 epochs, better DPO (ROUGE-1: 0.36)
├── Medical_LLM_Full_Pipeline_Final.ipynb          # USMLE data, NEFTune (ROUGE-1: 0.77) ✅
│
├── assets/                       # Screenshots and images for README
│   ├── attempt1_run_summary.png
│   ├── attempt2_sft_training.png
│   ├── attempt2_rouge_scores.png
│   ├── final_evaluation_results.png
│   ├── final_sft_training.png
│   ├── wandb_train_loss.png
│   ├── wandb_eval_loss.png
│   └── wandb_dpo_metrics.png
│
├── app.py              # FastAPI streaming backend
├── gradio_app.py       # Gradio chat frontend
├── evaluate.py         # ROUGE evaluation script
├── medqa_test.json     # Sample test questions
└── requirements.txt    # Python dependencies
```

---

## 💬 Example Interactions

**Query:** What are the early warning signs of Type 2 Diabetes?

> The model provides a structured clinical answer covering polyuria, polydipsia, fatigue, blurred vision, and slow wound healing — drawing on USMLE-level clinical knowledge.

**Query:** How do SSRIs work in the brain?

> The model explains the serotonin reuptake mechanism, synaptic effects, and the lag to therapeutic effect — reflecting the biomedical depth from PubMedQA training.

---

## 🧑‍💻 Author

**Kumar Sarthak**
Birla Institute of Technology, Mesra

[![GitHub](https://img.shields.io/badge/GitHub-KumarSarthak--Official-black?logo=github)](https://github.com/KumarSarthak-Official/Medical_AI_Assistant-with-SFT_DPO)
[![HuggingFace](https://img.shields.io/badge/🤗-kumarsarthak98-blue)](https://huggingface.co/kumarsarthak98)

---

## ⚠️ Disclaimer

This model is intended for **educational and research purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical decisions.

---

## 📄 License

This project is released under the [MIT License](LICENSE). The base LLaMA 3.2 model is subject to Meta's [Llama Community License](https://llama.meta.com/llama-downloads/).

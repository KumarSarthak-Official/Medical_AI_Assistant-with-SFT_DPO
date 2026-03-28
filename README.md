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

The project evolved across **three iterative attempts**, each improving dataset quality, training configuration, and alignment strategy — ultimately producing a production-ready model deployed as a streaming API and Gradio application.

---

## 🔗 Quick Links

| Resource | Link |
|---|---|
| 🤗 Fine-tuned Model | [kumarsarthak98/medical-llama-3.2-3b-sft-dpo](https://huggingface.co/kumarsarthak98/medical-llama-3.2-3b-sft-dpo) |
| 🚀 Live Demo (Spaces) | [medical-ai-assistant](https://huggingface.co/spaces/kumarsarthak98/medical-ai-assistant) |
| 📊 WandB Training Report | [Full Experiment Logs](https://wandb.ai/kumarsarthakofficial-birla-institute-of-technology-mesra/medical-llm-finetune/reports/Projects-medical-llm-finetune--VmlldzoxNjM0NjAwMA?accessToken=2qhqj2j2up2qd98epspklrd3qo9v9cyzks3eqf8zs6csryfqv9fubw535j2ce8dc) |
| 💻 GitHub Repo | [Medical_AI_Assistant-with-SFT_DPO](https://github.com/KumarSarthak-Official/Medical_AI_Assistant-with-SFT_DPO) |

---

## 🏗️ Architecture & Technical Stack

```
Base Model:   unsloth/Llama-3.2-3B-Instruct-bnb-4bit
Quantization: 4-bit NF4 (via BitsAndBytes)
Adapter:      LoRA (r=64, alpha=128, dropout=0.05)
Target Layers: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
Training:     SFT → DPO (two-stage alignment)
Hardware:     Kaggle T4 GPU
Serving:      FastAPI (streaming SSE) + Gradio frontend
```

**Key Libraries:** `unsloth`, `transformers`, `trl`, `peft`, `bitsandbytes`, `wandb`, `fastapi`, `gradio`

---

## 🔬 Experimental Journey — Three Attempts

The project went through three major iterations. Each attempt exposed weaknesses in either the data or the training setup, driving the next round of improvements.

---

### Attempt 1 — `Medical_LLM_Full_Pipeline_1st_Attempt.ipynb`

**Goal:** Establish a baseline pipeline for medical domain fine-tuning.

**Datasets Used:**
- `lavita/medical-qa-shared-task-v1-toy` — A small toy multiple-choice dataset (~few hundred examples)
- `qiaojin/PubMedQA` (`pqa_labeled`) — Biomedical research Q&A with context paragraphs

**Training Config:**
| Parameter | Value |
|---|---|
| LoRA Rank (r) | 64 |
| LoRA Alpha | 128 |
| SFT Epochs | 1 |
| DPO Beta | 0.1 |
| DPO Learning Rate | 5e-5 |
| Gradient Accumulation | 8 |
| Optimizer | `adamw_torch` |

**Result & Why It Wasn't Enough:**

The pipeline ran successfully and the training loss converged, but the model's outputs were shallow. The `medical-qa-shared-task-v1-toy` dataset is deliberately small and designed for benchmarking, not for training a generalist medical assistant. Answers were often just single option labels (e.g., "Option 3") with no clinical reasoning. The model learned to mimic the format but not the substance of medical knowledge. The DPO stage at this step was also relatively mild (Beta=0.1), giving the model too much freedom to deviate from the reference.

**Decision:** Switch to a larger, higher-quality dataset that contains real clinical reasoning.

---

### Attempt 2 — `Medical_LLM_Full_Pipeline_2nd_Attempt.ipynb`

**Goal:** Improve the SFT data format and training stability while keeping the same dataset combination as a sanity check before committing to a full dataset swap.

**Datasets Used:** Same as Attempt 1 (toy + PubMedQA)

**Key Changes from Attempt 1:**
- Refined the `format_medquad` function to produce cleaner question-answer formatting
- Added explicit train/validation splitting (90/10) to monitor overfitting
- Stabilized DPO rejected-answer generation with more diverse negative strategies:
  - **Mismatched answer** — pulling a correct answer from a different question
  - **Incomplete answer** — truncating the answer and adding a cop-out disclaimer
  - **Overcautious answer** — replacing the answer with a vague "consult a clinician" response
  - **Binary flip** — inverting Yes/No responses

**Result & Why It Still Wasn't Enough:**

The model improved in format consistency but still lacked depth. The core problem became clear: the toy dataset's answers are too short and too shallow. PubMedQA alone, while research-heavy, doesn't capture the clinical decision-making a medical assistant needs. The eval loss curves were reasonable, but ROUGE scores on open-ended medical questions remained low because the model had never seen the kind of structured clinical reasoning required for real patient-facing scenarios.

**Decision:** Completely replace the toy dataset with proper USMLE board-level questions, dramatically scale up training data, and restructure the SFT training.

---

### Attempt 3 (Final) — `Medical_LLM_Full_Pipeline_Final.ipynb` ✅

**Goal:** Build a production-quality model using high-difficulty clinical reasoning data at scale.

**Datasets Used:**
- `GBaker/MedQA-USMLE-4-options` — Real USMLE (United States Medical Licensing Examination) board questions requiring multi-step clinical reasoning. **8,000 examples** sampled.
- `qiaojin/PubMedQA` (`pqa_labeled`) — Full 1,000 biomedical research Q&A pairs

**Total SFT Dataset:** ~9,000 examples (90% train / 10% validation)

**What Changed and Why:**

| Change | Attempt 1 & 2 | Final | Reason |
|---|---|---|---|
| Primary Dataset | Toy medical Q&A | USMLE board questions | Board-level questions demand genuine multi-step clinical reasoning, not just option selection |
| Dataset Size | ~500 examples | ~9,000 examples | More data reduces memorization and improves generalization |
| SFT Epochs | 1 | 3 | The larger dataset needs more passes to converge well |
| Sequence Packing | Disabled | Enabled (`packing=True`) | Packs short sequences together to maximize GPU utilization and throughput on T4 |
| NEFTune Noise | Not used | `neftune_noise_alpha=5.0` | Adds random noise to embeddings during training, improving generalization and instruction-following |
| DPO Beta | 0.1 | 0.2 | Higher beta keeps the fine-tuned policy closer to the SFT reference, preventing reward hacking |
| DPO Learning Rate | 5e-5 | 5e-6 | 10x smaller LR for DPO prevents over-optimization on the preference pairs |

**Training Config (Final):**
| Parameter | Value |
|---|---|
| LoRA Rank (r) | 64 |
| LoRA Alpha | 128 |
| LoRA Dropout | 0.05 |
| SFT Epochs | 3 |
| SFT Learning Rate | 2e-4 |
| SFT LR Scheduler | Cosine |
| NEFTune Noise Alpha | 5.0 |
| DPO Beta | 0.2 |
| DPO Learning Rate | 5e-6 |
| Gradient Accumulation | 8 |
| Max Sequence Length | 2048 |
| Optimizer | `adamw_torch` |

---

## 📊 Training Results (WandB)

All training runs were tracked with [Weights & Biases](https://wandb.ai/kumarsarthakofficial-birla-institute-of-technology-mesra/medical-llm-finetune/reports/Projects-medical-llm-finetune--VmlldzoxNjM0NjAwMA?accessToken=2qhqj2j2up2qd98epspklrd3qo9v9cyzks3eqf8zs6csryfqv9fubw535j2ce8dc).

**SFT Training Loss** — The final run (3-epoch, USMLE data) shows a smooth and consistent loss decrease from ~2.0 → ~1.0, with no signs of instability or divergence. The cosine learning rate schedule produced the characteristic oscillation at lower loss values seen in the training curves.

**Eval Loss** — Validation loss curves across runs showed the final model achieved the lowest stable eval loss, with the best checkpoint loaded automatically at the end of training.

**DPO Metrics** — The `dpo/avg_loss` curve shows clean convergence from ~0.3 → near 0 across all runs, with the final configuration converging fastest due to the reduced learning rate and better-matched preference pairs from the USMLE data.

> **View the full interactive training dashboard →** [WandB Report](https://wandb.ai/kumarsarthakofficial-birla-institute-of-technology-mesra/medical-llm-finetune/reports/Projects-medical-llm-finetune--VmlldzoxNjM0NjAwMA?accessToken=2qhqj2j2up2qd98epspklrd3qo9v9cyzks3eqf8zs6csryfqv9fubw535j2ce8dc)

---

## 🚀 Deployment Architecture

The final model is served via a two-component production system:

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
│  Loads model in 4-bit, runs TextIteratorStreamer         │
│  Model: kumarsarthak98/medical-llama-3.2-3b-sft-dpo     │
└─────────────────────────────────────────────────────────┘
```

The backend uses `TextIteratorStreamer` from HuggingFace to stream tokens in real-time via Server-Sent Events (SSE), so the user sees the response appearing word-by-word rather than waiting for the full generation.

---

## 🛠️ Running Locally

### Prerequisites
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install unsloth transformers fastapi uvicorn gradio httpx pydantic
```

Or install from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Step 1 — Start the FastAPI backend
```bash
python app.py
# Server starts at http://localhost:8000
```

### Step 2 — Launch the Gradio frontend (in a separate terminal)
```bash
python gradio_app.py
# UI available at http://localhost:7860
```

### Step 3 — Run ROUGE Evaluation
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
├── Medical_LLM_Full_Pipeline_1st_Attempt.ipynb   # Baseline: toy dataset + PubMedQA
├── Medical_LLM_Full_Pipeline_2nd_Attempt.ipynb   # Refined formatting + DPO strategies
├── Medical_LLM_Full_Pipeline_Final.ipynb          # Final: USMLE data, 3 epochs, NEFTune
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

> The model explains the serotonin reuptake mechanism, synaptic effects, and the lag time to therapeutic effect — reflecting the biomedical depth acquired from PubMedQA training.

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

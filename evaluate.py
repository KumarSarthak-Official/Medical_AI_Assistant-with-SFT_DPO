import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer
from tqdm import tqdm

def main():
    # 1. Set up command line arguments
    parser = argparse.ArgumentParser(description="Evaluate Medical LLM with ROUGE scores")
    parser.add_argument("--model_id", type=str, required=True, help="Hugging Face model ID (e.g., kumarsarthak98/medical-llama-3.2-3b-sft-dpo)")
    parser.add_argument("--test_data", type=str, required=True, help="Path to JSON test file")
    args = parser.parse_args()

    print(f"📥 Loading model: {args.model_id} into VRAM...")
    
    # 2. Load the tokenizer and the merged model
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="auto" # Automatically puts the model on your GPU
    )

    print(f"📄 Loading test data from: {args.test_data}")
    with open(args.test_data, "r") as f:
        test_data = json.load(f)

    # 3. Set up the ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

    print("\n🚀 Starting evaluation loop...")
    
    # 4. Loop through every question in the test file
    for item in tqdm(test_data, desc="Evaluating"):
        question = item["instruction"]
        target_answer = item["output"]

        # Format the prompt using your exact chat template
        messages = [
            {"role": "system", "content": "You are a knowledgeable medical AI assistant."},
            {"role": "user", "content": question}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate the AI's answer
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=256, 
                temperature=0.1, # Low temperature for factual evaluation
                do_sample=False
            )
        
        # Decode only the newly generated text (ignoring the prompt)
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # 5. Calculate how closely the AI's answer matches the real doctor's answer
        score = scorer.score(target_answer, prediction)
        scores['rouge1'] += score['rouge1'].fmeasure
        scores['rouge2'] += score['rouge2'].fmeasure
        scores['rougeL'] += score['rougeL'].fmeasure

    # 6. Calculate the final averages
    num_samples = len(test_data)
    for key in scores:
        scores[key] /= num_samples

    print("\n" + "="*40)
    print("📊 FINAL EVALUATION SCORES 📊")
    print("="*40)
    print(f"ROUGE-1 (Single word overlap):  {scores['rouge1']:.4f}")
    print(f"ROUGE-2 (Two-word overlap):     {scores['rouge2']:.4f}")
    print(f"ROUGE-L (Longest sequence):     {scores['rougeL']:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
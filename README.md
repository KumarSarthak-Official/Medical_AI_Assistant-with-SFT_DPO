## 🧪 Standalone Evaluation
If you want to evaluate the model's accuracy locally using the ROUGE metric against a blind test set, you can run the included evaluation script:

```bash
pip install rouge-score nltk
python evaluate.py --model_id kumarsarthak98/medical-llama-3.2-3b-sft-dpo --test_data medqa_test.json

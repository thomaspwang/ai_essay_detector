import csv
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import torch

INPUT_CSV = 'detection_io/input.csv'
OUTPUT_CSV = 'detection_io/output.csv'
MODEL_NAME = "roberta-base-openai-detector"

def inference(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).detach().numpy()[0]
    return list(probabilities)

if __name__ == "__main__":
    print("Running GPT detector ...")
    model_name = MODEL_NAME
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    output = [] # [applicant_id, essay1_fake_prob, essay2_fake_prob, essay3_fake_prob, essay4_fake_prob]

    with open(INPUT_CSV, 'r') as f:
        reader = csv.reader(f)
        lines = list(reader)
        output.append(["Application ID", "Prompt #1 AI Probability", "Prompt #2 AI Probability", 
                       "Prompt #3 AI Probability","Prompt #3 AI Probability", "Prompt #4 AI Probability"])
        for line in tqdm(lines[1:]):
            output.append(
                [
                    line[0],
                    inference(line[19], tokenizer, model)[0],
                    inference(line[20], tokenizer, model)[0],
                    inference(line[21], tokenizer, model)[0],
                    inference(line[22], tokenizer, model)[0],
                ]
            )
            
    with open(OUTPUT_CSV, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(output)
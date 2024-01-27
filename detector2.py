from transformers import pipeline
from tqdm import tqdm
import csv


INPUT_CSV = 'detection_io/input.csv'
OUTPUT_CSV = 'detection_io/output.csv'

if __name__ == "__main__":
    print("Running GPT detector ...")
    pipe = pipeline("text-classification", model="roberta-base-openai-detector")

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
                    1 - pipe(line[19])[0]['score'],
                    1 - pipe(line[20])[0]['score'],
                    1 - pipe(line[21])[0]['score'],
                    1 - pipe(line[22])[0]['score'],
                ]
            )
            
    with open(OUTPUT_CSV, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(output)

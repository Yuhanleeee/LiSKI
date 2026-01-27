import pandas as pd
import json
import random


prompt = "Read the following case presentation, then output the reasoning for the diagnosis within the tags <think> ... </think> and the final diagnosis (just the name of the disease/entity) within the tags <answer> ... </answer>.\n"

df = pd.read_parquet("MedCaseReasoning/medcasereasoning_core.pqt")

jsons = []
for i in range(len(df)):
    if df.iloc[i]['split'] == 'train':
        instruction = prompt.format()
        inputs = "CASE PRESENTATION\n"+df.iloc[i]['case_prompt']+"\nYOUR ANSWER\n"
        answer = '<think>'+'\n'+df.iloc[i]['diagnostic_reasoning']+'\n'+'</think>'+'\n'+'<answer>'+df.iloc[i]['final_diagnosis']+'</answer>'
        jsons.append({"instruction": instruction, "input": inputs, "output": answer})

# jsons_sample = random.sample(jsons, 1000)

with open("MedCaseReasoning/medcasereasoning_train_reasoningdiagnostic.json", "w", encoding="utf-8") as f:
    json.dump(jsons, f, ensure_ascii=False, indent=2)


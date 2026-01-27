import glob
import json
import random


# # file matching
# files = glob.glob("*.txt")
# output = "all.txt"

# header_written = False

# with open(output, "w", encoding="utf-8") as fout:
#     for file in files:
#         with open(file, "r", encoding="utf-8") as fin:
#             lines = fin.readlines()
#             if not lines:
#                 continue
#             header = lines[0]
#             data = lines[1:]
#             if not header_written:
#                 fout.write(header)
#                 header_written = True
#             fout.writelines(data)

# print(f"Merge done， {len(files)} in total, output to {output}")

prompt = "You are an expert chemist. Given the molecule SMILES, your task is to provide the detailed description of the molecule using your experienced chemical knowledge.\nPlease strictly follow the format, no other information can be provided.\n"
jsons = []
with open('all.txt', 'r', encoding="utf-8") as f:
    lines = f.readlines()[1:]
    for line in lines:
        dict_temp = {}
        line_temp = line.split('\t')
        if len(line_temp) != 3:
            print(line)
        else:
            dict_temp["instruction"] = prompt
            dict_temp["input"] = "Molecule SMILES: {}\nDescription: ".format(line_temp[1])
            dict_temp["output"] = line_temp[2]
            jsons.append(dict_temp)

with open("CHEBI20_all.json", "w", encoding="utf-8") as f:
    # json.dump(random.sample(jsons, 300), f, ensure_ascii=False, indent=2)
    json.dump(jsons, f, ensure_ascii=False, indent=2)

import json

anno_file = "datasets/ytbvos/valid_ref.json"
dataset = json.load(open(anno_file, 'r'))
print('Load dataset...')

exp_list = []
for vid in dataset["videos"]:
    exp_list += vid["expressions"]

out_exp_file = anno_file.replace(".json", "_exp.txt")
with open(out_exp_file, "w") as output:
    for exp in exp_list:
        output.write(exp + '\n')

print("Done!")
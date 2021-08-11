import json
import os

with open("./acc_arranger_roberta_base_3epoch/is_test_true_eval_logits.txt", "r") as f:
    model_outputs = f.read().strip().split("\n")
    for i in range(len(model_outputs)):
        model_outputs[i] = model_outputs[i].split()
        for j in range(len(model_outputs[i])):
            model_outputs[i][j] = float(model_outputs[i][j])
        assert(len(model_outputs[i]) == 3)

cc_output_data = "./lm.output.test_seen.cc.txt"
with open(cc_output_data, "r") as f:
    data_cc = f.read()
data_cc = data_cc.split("[TransformerGenerator]:")[1:]
for i in range(len(data_cc)):
    data_cc[i] = data_cc[i].split("\n")[0].strip()

print("total chit chat:", len(data_cc))

def main():
    predict_cc = []
    stats = {0:0, 1:0, 2:0}
    for i in range(len(data_cc)):
        assert(len(model_outputs[i]) == 3)
        cc = {'start': "", 'end': "", 'mod': ""}
        o = 0
        for j in range(1, 3):
            if model_outputs[i][j] > model_outputs[i][o]:
                o = j
        stats[o] += 1
        if o == 0:
            predict_cc.append(cc)
        elif o == 1:
            cc["start"] = data_cc[i].strip()
            predict_cc.append(cc)
        else:
            cc["end"] = data_cc[i].strip()
            predict_cc.append(cc)

    print(stats)
    
    fns = os.listdir('data/test_seen')
    fns.sort()

    nlg = {}
    n = 0
    for fn in fns:
        with open("./data/test_seen/" + fn, "r") as f:
            data = json.load(f)
        for i in range(len(data)):
            gen_dial_output = {}
            for j in range(1, len(data[i]["turns"]), 2):
                tid = data[i]["turns"][j]['turn_id']
                gen_dial_output[tid] = predict_cc[n]
                n+=1
            nlg[data[i]['dialogue_id']] = gen_dial_output
            
    with open('nlg_output.json', 'w') as f:
        json.dump(nlg, f, indent=2)
    
if __name__ == "__main__":
    main()
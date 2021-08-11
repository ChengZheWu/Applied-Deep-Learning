import json
import re
import csv
import argparse


def write_csv(ans, output_path):
    ans = sorted(ans.items(), key=lambda x: x[0])
    with open(output_path, 'w') as f:
        f.write('id,state\n')
        for dialogue_id, states in ans:
            if len(states) == 0:  # no state ?
                str_state = 'None'
            else:
                states = sorted(states.items(), key=lambda x: x[0])
                str_state = ''
                for slot, value in states:
                    # NOTE: slot = "{}-{}".format(service_name, slot_name)
                    str_state += "{}={}|".format(
                            slot.lower(), value.replace(',', '_').lower())
                str_state = str_state[:-1]
            f.write('{},{}\n'.format(dialogue_id, str_state))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",type=str, help="QA output file")
    parser.add_argument("--seen",type=bool)
    args = parser.parse_args()
    with open(args.input, newline='') as jsonfile:
        data = json.load(jsonfile)


    dialogue_id=''
    slot_value={}
    slot_value_tmp={}
    for i in data.keys():
        dialogue_id_now=re.search('(.*?)-(.+)',i).group(1).split('.')[0]
        if dialogue_id=='':
            dialogue_id=dialogue_id_now
        if dialogue_id!=dialogue_id_now and slot_value_tmp!={}:
            slot_value[dialogue_id]=slot_value_tmp
            slot_value_tmp={}
            dialogue_id=dialogue_id_now
        
        slot=re.search('(.*?)-(.+)',i).group(2)
        slot_value_tmp[slot]=data[i]

    #print(slot_value)

    slot_value[dialogue_id]=slot_value_tmp

    #print(slot_value)


    if args.seen:
        output_path="./kaggle_seen.csv"
    else:
        output_path="./kaggle_unseen.csv"
    write_csv(slot_value,output_path)

if __name__ == '__main__':
    main() 
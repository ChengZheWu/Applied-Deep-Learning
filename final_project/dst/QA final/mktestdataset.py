import csv
import json
import re
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",type=str, help="last stage output file")
    parser.add_argument("--seen",type=bool)
    args = parser.parse_args()
    schema_path="./schema.json"
    test_seen_file="./test_seen_dials.json"
    test_unseen_file="./test_unseen_dials.json"
    test_output=args.input

    if args.seen==True:
        test_file=test_seen_file
        path="./test_seen.csv"
    
    else:
        test_file=test_unseen_file
        path="./test_unseen.csv"

    with open(schema_path) as jsonfile:
        schema = json.load(jsonfile)

    with open(test_file) as jsonfile:
        test_data = json.load(jsonfile)

    with open(test_output) as jsonfile:
        data = json.load(jsonfile)
    diaid_list=[]

    data_processed=[]
    slot_description=''
    for dial_data in data:
        diaid=dial_data.split('-')[0]
        if diaid not in diaid_list:
            diaid_list.append(diaid)
        service=dial_data.split('-')[1]
        slot=""
        for idx,i in enumerate(dial_data.split('-')):
            if idx!=0 and idx!=1 and idx!=len(dial_data.split('-'))-1:
                slot=slot+i+"-"
            if idx==len(dial_data.split('-'))-1:
                slot=slot+i
        #print(slot)
        utter=''
        for test_dial in test_data:
            if diaid==test_dial['dialogue_id']:
                for turn in test_dial['turns']:
                    utter=utter+turn["utterance"]
        for sch in schema:
            if sch["service_name"]==service:
                for s in sch['slots']:
                    if s['name']==slot:
                        slot_description=s['description']
                        data_list=[diaid,utter,service+'-'+slot,slot_description]
                        data_processed.append(data_list)
                        break
                break
        
        
        
        #data_list=[diaid,utter,service+'-'+slot,slot_description]
        #print(data_list)
        
        

    print(path)
    with open(path, 'w') as csvfile:

        writer = csv.writer(csvfile)


        #writer.writerow(['id','service-slot','question', 'context', 'answers'])
        writer.writerow(['context','question','id','start','text'])
        for idx,i in enumerate(data_processed):
                #writer.writerow([i[0],j,i[3][j], i[1], i[2][j][0]])
                writer.writerow([i[1],i[3],i[0]+"-"+i[2]])

    print(len(diaid_list))

if __name__ == '__main__':
    main() 
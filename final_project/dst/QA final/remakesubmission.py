import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",type=str, help="last stage output file")
    parser.add_argument("--seen",type=bool)
    args = parser.parse_args()
    if args.seen:
        empty=pd.read_csv('./seen_empty.csv', header=0)
        output_path='../kaggle_final_result_seen.csv'
    else:
        empty=pd.read_csv('./unseen_empty.csv', header=0)
        output_path='../kaggle_final_result_unseen.csv'

    submission=pd.read_csv(args.input,header=0)


    for idx, i in enumerate(submission['id']):
        sub_value=submission['state'][idx]
        empty_idx=empty.index[empty['id'] == i].tolist()
        new_column = pd.Series([sub_value], name='state', index=[empty_idx[0]])
        empty.update(new_column)


    empty.to_csv(output_path, index=False)
    print('test finished')

if __name__ == '__main__':
    main() 
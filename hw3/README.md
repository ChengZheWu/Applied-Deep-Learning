# Homework3

## How to run code

```shell
# download the weights，需放在跟 code 同一個位置
bash download.sh
unzip weights.zip
```

input data 需放在當前的目錄 input.jsonl
output data 也會存在當前的目錄 output.jsonl

### prediction
```shell
bash run.sh
or
# 使用 jupyter notebook 執行 pred.ipynb
# 修改 input_path =  "./input.jsonl" # 為 input data 的位置
# 修改 output_path = "./output.jsonl" # 為 output data 的位置 
```
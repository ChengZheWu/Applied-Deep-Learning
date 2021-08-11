# Chinese News Summarization

### Input: news content
從小就很會念書的李悅寧， 在眾人殷殷期盼下，以榜首之姿進入臺大醫學院， 但始終忘不了對天文的熱情。大學四年級一場遠行後，她決心遠赴法國攻讀天文博士。 從小沒想過當老師的她，再度跌破眾人眼鏡返台任教，......
### Output: news title
榜首進台大醫科卻休學 、27歲拿到法國天文博士 李悅寧跌破眾人眼鏡返台任教  


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

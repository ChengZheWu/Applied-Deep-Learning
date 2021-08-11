# Final Project
## Task 1 - DST
1. 將data放在team_3/data/ 當中。seen task 資料夾名稱為test_seen；unseen task 資料夾名稱為test_unseen
```shell 
cd dst
sh download.sh
# 下載完成後可以看到dst/QA final/output3 ，內有model config 與 model.bin
sh run_seen.sh
#執行完成後，輸出檔案:./kaggle_final_result_seen.csv

sh run_unseen.sh
# 執行完成後，輸出檔案:./kaggle_final_result_seen.csv
```

## training
### stage1-text classification
```
cd text_classification
```
1. open train_multiple_selection-Copy1.ipynb
2. run all script

### stage2-question answering

```
cd 'QA final'
```
1. run qa.py 



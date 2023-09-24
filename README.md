# Semantic-Health-NER
## 環境建置
下載`transformers`：
   ```shell
   pip install transformers
   ```

下載 `matplotlib`：

   ```shell
   pip install matplotlib
   ```

下載 `roBERTa`：

   ```shell
   mkdir 你的專案位置\berts\chinese-roberta-wwm-ext
   Invoke-WebRequest -Uri https://s3.amazonaws.com/models.huggingface.co/bert/hfl/chinese-roberta-wwm-ext/pytorch_model.bin -OutFile "你的專案位置\berts\chinese-roberta-wwm-ext\pytorch_model.bin"
   Invoke-WebRequest -Uri https://s3.amazonaws.com/models.huggingface.co/bert/hfl/chinese-roberta-wwm-ext/vocab.txt -OutFile "你的專案位置\berts\chinese-roberta-wwm-ext\vocab.txt"
   Invoke-WebRequest -Uri https://s3.amazonaws.com/models.huggingface.co/bert/hfl/chinese-roberta-wwm-ext/config.json -OutFile "你的專案位置\berts\chinese-roberta-wwm-ext\config.json"
   ```
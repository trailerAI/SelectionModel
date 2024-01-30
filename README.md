# Selection Model
질문에 대한 답변을 생성하는 과정에서, 답변을 보다 효율적으로 생성하기 위해 passage 추출을 위한 Selection Model을 구현했습니다.
모델은 [HuggingFace](https://huggingface.co/NHNDQ/SelectionModel)에 공개되어 있습니다.

## Setting
```
cd SelectionModel
poetry shell
poetry install
```

## Data
Dense Passage Retrieval 모델을 사용하여 question으로 부터 passage를 추출한 데이터를 활용합니다. Question과 가장 유사한 임베딩 벡터 값을 갖는 passage 상위 20개를 추출해서 데이터 셋을 구성했으며, 추출된 passage 내에 답변이 포함되어 있는지 여부를 기반으로 작동합니다. 답변이 포함되어 있으면 이를 '0'으로, 포함되어 있지 않으면 '1'로 라벨링 되어있습니다.

[DPR 모델에서 inference](https://github.com/trailerAI/KoDPR)를 통해 생성된 데이터 셋을 selection model을 학습시키기 위해 전처리 합니다.

```
python preprocess.py --fpath ./gold_32 --dtype train
```

```
python preprocess.py --fpath ./gold_32 --dtype valid
```

```
python preprocess.py --fpath ./gold_32 --dtype test
```

## Train
Selection Model 모델 학습을 진행하기 위해 top20.yaml 설정을 수정한 다음, `shell/train.sh`를 실행하면 됩니다.
```
chmod 755 ./shell/train.sh
./train.sh
```

## Inference
Selection Model에 대한 AUC 결과를 확인하고 question과 passage에 대한 Selection Model 결과 확률 값이 저장되어있는 파일을 만들기 위해 아래와 같이 실행합니다.
```
python inference.py -model_path ./output_top20/checkpoint-252960 -dpath ./gold_32/valid_top20_dataset.parquet -spath ./datasets/valid_top20_dataset_llm.csv
```

## Results
Accuracy about (True == Pred): 93.68%

`Retrival Model` vs `Retrival Model using Selection Model`

아래의 결과는 Selection Model을 통해 추출된 predict score를 내림차순으로 재정렬 후, 성능 측정
| Model  | Top@1 | Top@5 | Top@10 | Top@20 |
|----|-------|-------|--------|--------|
| Retrieval Model | 37.87%| 61.81%| 71.04% | 79.04% |
| Selection Model | 71.12%| 77.59%| 78.61% | 79.04% |

## Contributors
[Jisu, Kim](https://github.com/merry555), [TakSung Heo](https://github.com/HeoTaksung), [Minsu Jeong](https://github.com/skaeads12), and [Juhwan, Lee](https://github.com/juhwanlee-diquest)


## License
Apache License 2.0 lisence
...

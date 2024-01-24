# Selection Model
질문에 대한 답변을 생성하는 과정에서, 답변을 보다 효율적으로 생성하기 위해 Context 추출을 위한 Selection Model을 구현했습니다.
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


## Results


## Contributors
[Jisu, Kim](https://github.com/merry555), [TakSung Heo](https://github.com/HeoTaksung), [Minsu Jeong](https://github.com/skaeads12), and [Juhwan, Lee](https://github.com/juhwanlee-diquest)


## License
Apache License 2.0 lisence
...

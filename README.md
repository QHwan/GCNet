# GCNet

Graph Neural Network를 이용한 molecule 특성 예측 AI.
Baseline 작성이 완료되었으며, 현재는 freesolv dataset과 Graph Convolutional Network 만 지원됩니다.

## 향후 목표
- 다른 종류의 dataset 지원 가능.
- Network 추가. Graph attention, Graph isomorphism, LSTM...

## Example
data/ 폴더에 freesolv.csv 파일이 있어야 합니다. (다운로드 링크: http://moleculenet.ai/datasets-1)

우선 freesolv.csv 파일을 가공해서 .npz 파일로 변환합니다.
```
python data/data_convert.py
```

그 후에 학습시켜 줍니다.
```
python train.py --epoch 100
```

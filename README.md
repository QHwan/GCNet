# GCNet

Graph Neural Network를 이용한 molecule 특성 예측 AI.
Baseline 작성이 완료되었으며, 현재는 freesolv dataset과 Graph Convolutional Network 만 지원됩니다.

## 향후 목표
- 다른 종류의 dataset 지원 가능.
- Network 추가. Graph attention, Graph isomorphism, LSTM...

## Example
data/ 폴더에 freesolv.csv 파일이 있어야 합니다. (다운로드 링크: http://moleculenet.ai/datasets-1)
FreeSolv dataset은 642종류의 분자들의 solvation free energy를 담고 있습니다.
여기에서는 분자구조를 graphical data로 변환하고 graph convolutional network를 이용해 solvation free energy를 예측하는 ai를 개발합니다.

우선 freesolv.csv 파일을 가공해서 .npz 파일로 변환합니다.
```
python data/data_convert.py
```

그 후에 학습시켜 줍니다.
```
python train.py --epoch 100
```

다음과 같은 결과를 얻습니다.

![Figure_1](https://user-images.githubusercontent.com/30894739/80936944-19195b80-8e0e-11ea-825f-2f9d509f12c6.png)

결과의 RMSE 값은 1.48 kcal/mol 입니다. 가장 단순한 graph convolutional network만을 이용한 값이 이 정도고, 앞으로 더 좋은 network structure를 이용하면 더 정확한 결과를 얻을 수 있을 것입니다.

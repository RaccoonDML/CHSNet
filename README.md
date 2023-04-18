# This branch (fsc) is for CAC task
This branch provides baseline code for few-shot object counting task.

 VGG16Trans model is adopted for counting. 
 
 FSC147 dataset is released by [LearningToCountEverything](https://arxiv.org/pdf/2104.08391.pdf).

For more related works about class-agnostic counting(CAC), please refer this repository [Awesome-Class-Agnostic-Counting](https://github.com/RaccoonDML/Awesome-Class-Agnostic-Counting)

## to run
1. fork or clone this repository
2. pip install -r requirements.txt
3. download dataset, then put [images_384_VarV2](https://drive.google.com/file/d/1ymDYrGs9DSRicfZbSCDiOu0ikGDh5k6S/view?usp=sharing) and  [gt_density_map_adaptive_384_VarV2](https://archive.org/details/FSC147-GT) in ./datasets/FSC
4. sh run.sh

## baseline result
|MAE/MSE|FSC-test|FSC-val|
|:-:|:-:|:-:|
|FamNet|22.08/99.54|23.75/69.07|
|VGG16Trans without examplar (**baseline**)|17.51/132.62|19.21/77.43|

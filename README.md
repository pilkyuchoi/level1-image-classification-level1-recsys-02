![main](https://user-images.githubusercontent.com/50396533/147069300-5038c779-faa4-404b-b1fd-e9e3896f06b4.png)
# 마스크 착용 상태 분류
## 카메라로 촬영한 사람 얼굴 이미지의 마스크 착용 여부를 판단하는 Task
### Overview
**COVID-19**의 확산으로 우리나라는 물론 전 세계 사람들은 경제적, 생산적인 활동에 많은 제약을 가지게 되었습니다. 우리나라는 COVID-19 확산 방지를 위해 사회적 거리 두기를 단계적으로 시행하는 등의 많은 노력을 하고 있습니다. 과거 높은 사망률을 가진 사스(SARS)나 에볼라(Ebola)와는 달리 COVID-19의 치사율은 오히려 비교적 낮은 편에 속합니다. 그럼에도 불구하고, 이렇게 오랜 기간 동안 우리를 괴롭히고 있는 근본적인 이유는 바로 COVID-19의 강력한 전염력 때문입니다.  
감염자의 입, 호흡기로부터 나오는 비말, 침 등으로 인해 다른 사람에게 쉽게 전파가 될 수 있기 때문에 감염 확산 방지를 위해 무엇보다 중요한 것은 모든 사람이 마스크로 코와 입을 가려서 혹시 모를 감염자로부터의 전파 경로를 원천 차단하는 것입니다. 이를 위해 공공 장소에 있는 사람들은 반드시 마스크를 착용해야 할 필요가 있으며, 무엇 보다도 코와 입을 완전히 가릴 수 있도록 올바르게 착용하는 것이 중요합니다. 하지만 넓은 공공장소에서 모든 사람들의 **올바른 마스크 착용 상태를 검사**하기 위해서는 추가적인 인적자원이 필요할 것입니다.  
따라서, 우리는 카메라로 비춰진 사람 얼굴 이미지 만으로 이 사람이 마스크를 쓰고 있는지, 쓰지 않았는지, 정확히 쓴 것이 맞는지 자동으로 가려낼 수 있는 시스템이 필요합니다. 이 시스템이 공공장소 입구에 갖춰져 있다면 적은 인적자원으로도 충분히 검사가 가능할 것입니다.  
### Requirements
```bash
pip install -r requirements.txt
```
### Dataset (저작권 이슈로 깃헙 업로드가 불가능합니다)
마스크를 착용하는 건 COIVD-19의 확산을 방지하는데 중요한 역할을 합니다. 제공되는 이 데이터셋은 사람이 마스크를 착용하였는지 판별하는 모델을 학습할 수 있게 해줍니다. 모든 데이터셋은 아시아인 남녀로 구성되어 있고 나이는 20대부터 70대까지 다양하게 분포하고 있습니다. 간략한 통계는 다음과 같습니다.  
- 전체 사람 명 수 : 4,500  
- 한 사람당 사진의 개수: 7 [마스크 착용 5장, 이상하게 착용(코스크, 턱스크) 1장, 미착용 1장]  
- 이미지 크기: (384, 512)
 
전체 데이터셋 중에서 60%는 학습 데이터셋으로 활용됩니다.  
**입력값**: 마스크 착용 사진, 미착용 사진, 혹은 이상하게 착용한 사진(코스크, 턱스크)  
<img src="https://github.com/pilkyuchoi/images/blob/main/mask_classification/mask_example.png" width="250" height="300">  
**결과값**: 총 18개의 class를 예측해야합니다. 결과값으로 0~17에 해당되는 숫자가 각 이미지 당 하나씩 나와야합니다.  
**Class Description**: 마스크 착용여부, 성별, 나이를 기준으로 총 18개의 클래스가 있습니다.  
<img src="https://github.com/pilkyuchoi/images/blob/ee0bf9cda119c56b2340a5f04a875313cc9b2a33/mask_classification/class_description.png" width="700" height="600">   
### Pre-processing  
현재 이미지 데이터에는 학습에 불필요한 배경 데이터가 존재합니다. 따라서 얼굴 부분만을 잘라내 사용할 수 있도록 전처리를 진행하였습니다.  
사진에서 사람 얼굴을 Detection하여 Annotation 정보를 추출한 뒤 해당 영역에 맞추어 이미지를 잘라냈습니다.  
Detection에는 RetinaFace 라이브러리를 사용했습니다.(tensorflow==2.2.0 필요, 미처 detect 되지 못한 이미지 약 200장은 사람이 직접 annotate 했습니다)  
preprocessing 폴더 안의 retinaface.ipynb 파일을 실행하면 annotation 정보가 담긴 csv 파일이 생성되고,
해당 파일을 바탕으로 crop.ipynb 파일을 실행하면 기존 데이터에서 얼굴 영역을 잘라낸 파일들을 crop_images 폴더에 저장합니다.
### Model
예측해야하는 18개의 라벨이 결국 3개 Task 각 라벨의 조합(3 * 2 * 3)이어서 각 Task별로 모델 학습을 진행하였습니다.  
Mask와 Age Task에는 가중치가 다른 Focal Loss를, Gender Task에는 Cross Entropy를 사용했습니다.
학습시 Test 데이터에 대해서 Augementation(HorizontalFlip)을 진행하여 한 데이터에 대해 나온 2개 확률값의 평균을 해당 데이터의 예측 확률로 사용했습니다.  
3개 Task 모두 pretrained된 모델 classifier 부분에 간단한 MLP를 추가하여 모델링했습니다.  
사용된 pretrained 모델은 Resnet18, EfficientnetB1, EfficientnetB3입니다.  
각 모델 1개 별로 Out-Of-Fold Ensemble을 진행했습니다. 
이후 각 Task별로 서로 다른 모델에서 도출된 3개의 결과값을 Hard Voting 했습니다.  
마지막으로 각 Task의 예측값들의 조합을 구해 18개 라벨로 변환하였습니다.  
<img src="https://github.com/pilkyuchoi/images/blob/main/mask_classification/mask_classification_model.png">

### Result
Model, Task에 따라 5-Fold Cross Validation 했을 때 나온 Validation Accuracy의 평균값입니다.  
Data leakage를 방지하기 위해 동일한 사람에 대한 7장의 이미지들이 Train 데이터셋, Validation 데이터셋 둘 중 하나에만 포함되도록 하였습니다.
|Model \ Task|Mask|Gender|Age|
|---|---|---|---|
|Resnet18(5-Fold)|97.68|94.98|89.57|
|EfficientnetB1(5-Fold)|97.67|96.51|90.16|
|EfficientnetB3(5-Fold)|97.75|96.04|90.20|

### 시연 결과
학습된 모델을 실제 데이터에 적용해볼 수 있습니다. 예측하고 싶은 이미지들이 들어있는 데이터 경로를 아래 명령어에 적어주면 됩니다.
```bash
sh apply.sh 데이터 경로
```
실행하고 나면 preds 폴더에 각 이미지들의 예측 라벨이 담긴 preds.csv파일이 저장됩니다.  
새로운 데이터(작성자 본인)에 적용한 예시는 다음과 같습니다.
|<img src="https://github.com/pilkyuchoi/images/blob/main/mask_classification/normal.jpeg" width="100" height="150">|<img src="https://github.com/pilkyuchoi/images/blob/main/mask_classification/mask.jpeg" width="100" height="150">|<img src="https://github.com/pilkyuchoi/images/blob/main/mask_classification/incorrect.jpeg" width="100" height="150">|
|---|---|---|
|Not wear, Male, <30|Wear, Male, <30|Incorrect, Male, <30|

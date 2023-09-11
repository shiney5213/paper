# RCNN


## 개요
- paper :  [Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/abs/1311.2524)
- insight
    1. high-capacity CNN을  bottom-up방법으로  resion 제안에 적용하기 
    -> 코드 살펴보기
    2. superviesd pre-training/ domain-specific finetunig paradigm으로 data 부족문제에 효과적
    -> 데이터를 그냥 사용했을 때와 비교해보기

---
## code reference 
- [object-detection-algorithm/ R-CNN](https://github.com/object-detection-algorithm/R-CNN/)


## Ⅰ.  code 구성
<details>
<summary>code 구성</summary>
<div markdown="1">
    <pre>
    <code>
    - docs
    - imgs
    - py
        - utils 
            - data
                - create_bbox_regression_data.py          
                - create_classifier_data.py              
                - create_finetune_data.py                 
                - custom_batch_sampler.py                 
                - custom_bbox_regression_dataset.py       
                - custom_classifier_dataset.py            
                - custom_finetune_dataset.py              
                - custom_hard_negative_mining_dataset.py  
                - pascal_voc.py                           
                - pascal_voc_car.py
            - utils
        - bbox_regression.py                            
        - car_detector.py                              
        - finetune.py                                   
        - linear_svm.py                                 
        - selectivesearch.py    
        - data
        - utkls
    </code>
    </pre>    
</div>
</details>



## Ⅱ. code 분석
<details>
<summary><span style="font-size:150%">1. dataset 준비 및 car class data 추출</span> </summary>
<div markdown="1">

- py > utils> data > pascal_voc.py 실행 (dataset download)
- py > utils > data > pascal_voc_cal.py 실행 
- PASCAL_VOC_2007 dataset 이용
- ImageSets > main> cat_trainval.txt를 읽어 class가 car인 데이터를 train/ val, xml/jpg 파일로 나누어 데이터셋 구축 
- class가 car인 이미지만 추출

    | type | train   | val  |
    |:---:|:---:|:---:|
    |Annotations |  590  |  571   |
    | JPEGImages |  590  | 571  |

</div>
</details>

---
<details>
<summary><span style="font-size:150%">2. dataset 생성 및 저장</span> </summary>
<div markdown="1">


<details>
<summary><span style="font-size:120%">1) CNN 모델(AlexNet)을 finetune하기 위한 dataset</span> </summary>
<div markdown="1">

#### (1) selectivesearch
> - py > selectivesearch.py 실행
> - opencv에 구현된 cv2.ximgproc.segmentation.createSelectiveSearchSegmentation() 이용
> - 반환값: bounding box의 좌표값 array 반환
> - test image : lena.jpg

| 원본 이미지 | selectivesearch 이미지(초기 20개 box)|
|:---:|:---:|
| <img src = "./image/000012.jpg" width = 250> | <img src = "./image/000012_selectivesearch.jpg" width = 250>  

- 000012.jpg 이미지에서 찾은 box: 4648개

#### (2) CNN 모델(AlexNet)을 fine-tuning을 위한 annotation 데이터 생성
> - py > utils > data > create_finetune_data.py 실행
>- 이미지마다 selectivesearch로 찾은 predict box와 PASCAL_VOC dataset에 저장된 xml 파일의 Ground Truth를 비교
>- IoU가 0.5 이상이면 positive, 아니면 negative로 labeling
> | type | train   | val  |
> |:---:|:---:|:---:|
> |positive |  66122  |  64040   |
> | negative |  454839  | 407548  |
>- 상기 결과를 파일이름_0.csv, 파일이름_1.csv로 저장
> - 000012.jpg 이미지에서 selectivesearch한 4648개의 box중 IoU와 box area를 고려하여 positive 339개, negative 1116개 box를 선택

| positive_label | negative_label|
|:---:|:---:|
| <img src = "./image/000012_finetune_positive.png" width = 250> | <img src = "./image/000012_finetune_negative.png" width = 250>  


</div>
</details>

</div>
<details>
<summary><span style="font-size:120%">2) Linear SVM을 학습하기 위한 dataset</span> </summary>
<div markdown="1">

> - Linear SVM을 학습하기 위한 annotation 데이터 생성
> - py > utils > data > create_flassifier_data.py 실행
> - 상기 CNN모델을 fine-tuning하기 위한 과정과 비슷하나 data를 positive, negative로 labeling 하는 과정이 다름
> - groun truths의 box를 positive로, selective search가 찾은 box 중 IoU가 0.3보다 작고 면적을 고려하여 negative로 labeling
> - 상기 결과를 파일이름_0.csv, 파일이름_1.csv로 저장
> - 000012.jpg 이미지에서 selectivesearch한 4648개의 box중 IoU와 box area를 고려하여 positive 1개, negative 803개 box를 선택


| type | train   | val  |
|:---:|:---:|:---:|
|positive |  625  |  625   |
|negative |  366028  | 321474  |


 | positive_label | negative_label|
 |:---:|:---:|
 | <img src = "./image/000012_GT.png" width = 250> | <img src = "./image/000012_classifier_negative.png" width = 250> |
</div>
</details>


<details>
<summary><span style="font-size:120%">3) Bounding box Regressor 학습을 위한 dataset</span> </summary>
<div markdown="1">

> - Bounding box Regressor 학습을 위한 annotation 데이터 생성
> - py > utils > data > create_bbox_regression_data.py 실행
> - CNN을 fine-tuning하기 위해 생성한 positive box 중 IoU가 0.6 이상인 data만 사용
> -  000012.jpg 이미지에서 finetuning을 위한 positive 이미지  339개의 box중 IoU 고려하여 231개 box를 선택


| positive_bbox1 | positive_bbox2|positive_bbox3|
 |:---:|:---:|:---:|
 | <img src = "./image/000012_bbox_regression1.png" width = 250> | <img src = "./image/000012_bbox_regression2.png" width = 250> | <img src = "./image/000012_bbox_regression3.png" width = 250> | 

</div>
</details>
</details>

---

<details>
<summary><span style="font-size:150%">3. custom dataset 정의</span> </summary>
<div markdown="1">

>- 모델에 사용하기 위해 data type 정의

#### 1) CNN모델을 fine_tuning하기 위한 custom dataset
> - py > utils > data > custom_finetune_dataset.py 실행
> - 원본 이미지를 하나의 list에 저장
> - create_funetune_data.py에서 생성한 image annotation의 좌표값과 개수를 positive, negative로 나누어 각각 list에 저장
> - jpeg_image: 376장
> - positive_box : 66,122개
> - negative_box : 454,839개
> - total_box : 520961



</div>
</details>

 
















---
[참고]
- [Pytorch로 구현한 R-CNN 모델](https://herbwood.tistory.com/6)             


# RCNN
--

# 개요
- paper :  [Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/abs/1311.2524)
- insight
    1. high-capacity CNN을  bottom-up방법으로  resion 제안에 적용하기 
    -> 코드 살펴보기
    2. superviesd pre-training/ domain-specific finetunig paradigm으로 data 부족문제에 효과적
    -> 데이터를 그냥 사용했을 때와 비교해보기

- code reference : [object-detection-algorithm/ R-CNN](https://github.com/object-detection-algorithm/R-CNN/)

---

# code 구성
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








--
[참고]
- [Pytorch로 구현한 R-CNN 모델](https://herbwood.tistory.com/6)             


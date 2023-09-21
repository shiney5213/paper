# -*- coding: utf-8 -*-

"""
@date: 2020/3/2 上午8:07
@file: car_detector.py
@author: zj
@description: 车辆类别检测器
"""

import time
import copy
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import alexnet
import torchvision.transforms as transforms
import selectivesearch

import utils.util as util


def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_transform():
    # 数据转换
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform


def get_model(device=None):
    # 加载CNN模型
    model = alexnet()
    num_classes = 2
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)
    model.load_state_dict(torch.load('./models/linear_svm/best_linear_svm_alexnet_car.pt'))
    model.eval()

    # 取消梯度追踪
    for param in model.parameters():
        param.requires_grad = False
    if device:
        model = model.to(device)

    return model


def draw_box_with_text(img, rect_list, score_list):
    """
    경계 상자 및 분류 확률 그리기
    :param img:
    :param rect_list:
    :param score_list:
    :return:
    """
    for i in range(len(rect_list)):
        if i == 2:
            xmin, ymin, xmax, ymax = rect_list[i]
            score = score_list[i]

            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(0, 0, 255), thickness=3)
            cv2.putText(img, "{:.3f}".format(score), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            print('score', score)

def nms(rect_list, score_list):
    """
    Non maximum suppression
    :param rect_list: list，size[N, 4]
    :param score_list： list，size[N]
    """
    nms_rects = list()
    nms_scores = list()

    rect_array = np.array(rect_list)
    score_array = np.array(score_list)

    # 一次排序后即可
    # 분류 확률을 기준으로 내림차순 정렬
    idxs = np.argsort(score_array)[::-1]
    rect_array = rect_array[idxs]
    score_array = score_array[idxs]

    thresh = 0.3
    while len(score_array) > 0:
        # 가장 높은 positive 확률 저장
        nms_rects.append(rect_array[0])
        nms_scores.append(score_array[0])
        # 나머지 확률과 상자 저장
        rect_array = rect_array[1:]
        score_array = score_array[1:]

        length = len(score_array)
        if length <= 0:
            break

        # IoU 계산
        iou_scores = util.iou(np.array(nms_rects[len(nms_rects) - 1]), rect_array)
        # print(iou_scores)
        # 겹치는 부분이 thresh보다 작으면 제거
        idxs = np.where(iou_scores < thresh)[0]
        rect_array = rect_array[idxs]
        score_array = score_array[idxs]

    return nms_rects, nms_scores


if __name__ == '__main__':
    device = get_device()
    transform = get_transform()
    # load best_linear_svm
    model = get_model(device=device)   

    # selectivesearch
    gs = selectivesearch.get_selective_search()

    # 1) input data
    test_img_path = './imgs/000007.jpg'
    test_xml_path = './imgs/000007.xml'
    # test_img_path = './imgs/000012.jpg'
    # test_xml_path = './imgs/000012.xml'

    img = cv2.imread(test_img_path)
    
    dst = copy.deepcopy(img)
    selectivesearch_img = copy.deepcopy(img)

    bndboxs = util.parse_xml(test_xml_path)
    for bndbox in bndboxs:
        xmin, ymin, xmax, ymax = bndbox
        cv2.rectangle(dst, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=3)

    # 2) selective search
    selectivesearch.config(gs, img, strategy='f')
    rects = selectivesearch.get_rects(gs)
    print('제안된 후보 영역 수： %d' % len(rects))
    

    # softmax = torch.softmax()

    svm_thresh = 0.60

    # save positive sample box 
    score_list = list()
    positive_list = list()

    # tmp_score_list = list()
    # tmp_positive_list = list()
    start = time.time()
    for i, rect in enumerate(rects):
        xmin, ymin, xmax, ymax = rect
        rect_img = img[ymin:ymax, xmin:xmax]
        
       
        
        # transform size: (227, 227)
        rect_transform = transform(rect_img).to(device)
        wrap_img = transform(rect_img)
        
        
        # image visualize code
        # width = xmax - xmin
        # height = ymax - ymin
        
        # if width * height >= 3000:
        #     cv2.rectangle(selectivesearch_img, (xmin, ymin), (xmax, ymax), color=(0, 0, 255), thickness=3)
            
        #     wrap_img = wrap_img.detach().cpu().numpy()
            
        #     c, w, h = wrap_img[0], wrap_img[1], wrap_img[2]
        #     wrap_img = cv2.merge((h, w, c))
        #     cv2.imshow('wrap img', wrap_img)
        #     cv2.waitKey(0)
            
                
        # model output: [ negative, positive ]
        output = model(rect_transform.unsqueeze(0))[0]
        

        # positive > negative
        # argmax(): return index of big aixs
        if torch.argmax(output).item() == 1:
            """
            자동차에 대한 예측
            """
            probs = torch.softmax(output, dim=0).cpu().numpy()

            # tmp_score_list.append(probs[1])
            # tmp_positive_list.append(rect)

            if probs[1] >= svm_thresh:
                score_list.append(probs[1])
                positive_list.append(rect)
                # cv2.rectangle(dst, (xmin, ymin), (xmax, ymax), color=(0, 0, 255), thickness=2)
                print(f'{i}th box:', rect, output, probs)
    end = time.time()
    print('detect time: %d s' % (end - start))
    
    print('positive_list', len(positive_list), positive_list[0])
    print('score_list', len(score_list), score_list[0])

    # raise ValueError
    # tmp_img2 = copy.deepcopy(dst)
    # draw_box_with_text(tmp_img2, tmp_positive_list, tmp_score_list)
    # cv2.imshow('tmp', tmp_img2)
    #
    # tmp_img = copy.deepcopy(dst)
    # draw_box_with_text(tmp_img, positive_list, score_list)
    # cv2.imshow('tmp2', tmp_img)

    nms_rects, nms_scores = nms(positive_list, score_list)
    print('nms_rects', len(nms_rects), nms_rects[0])
    print('nms_scores', len(nms_scores), nms_scores[0])
    draw_box_with_text(dst, nms_rects, nms_scores)

    cv2.imshow('img', dst)
    cv2.waitKey(0)

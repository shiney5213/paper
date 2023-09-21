# -*- coding: utf-8 -*-

"""
@author: zj
@file:   selectivesearch.py
@time:   2020-02-25
"""

import sys, os
import cv2

rcnn_abspath = os.path.dirname(os.path.dirname( os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(rcnn_abspath)


def get_selective_search():
    gs = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    return gs


def config(gs, img, strategy='q'):
    gs.setBaseImage(img)

    if (strategy == 's'):
        gs.switchToSingleStrategy()
    elif (strategy == 'f'):
        gs.switchToSelectiveSearchFast()
    elif (strategy == 'q'):
        gs.switchToSelectiveSearchQuality()
    else:
        print(__doc__)
        sys.exit(1)


def get_rects(gs):
    rects = gs.process()
    rects[:, 2] += rects[:, 0]
    rects[:, 3] += rects[:, 1]

    return rects


if __name__ == '__main__':
    """
    选择性搜索算法操作
    """
    gs = get_selective_search()

    # img = cv2.imread('./data/lena.jpg', cv2.IMREAD_COLOR)
    img = cv2.imread('./imgs/000012.jpg', cv2.IMREAD_COLOR)
    img = cv2.imread('./imgs/000007.jpg', cv2.IMREAD_COLOR)
    
    
    config(gs, img, strategy='q')

    rects = get_rects(gs)
    print('rects', rects.shape)
    
    cv2.imshow("img", img)
    cv2.waitKey()
    
    
    img_selectivesearch = img.copy()
    for i, rect in enumerate(rects):
        if i <= 5:
            cv2.rectangle(img_selectivesearch,
                        (rect[0], rect[2]) ,
                        (rect[1], rect[3]),
                        thickness= 1,
                        color = (255, 0, 0)
                    )
    cv2.imshow("selectivesearch", img_selectivesearch)
    cv2.waitKey()
    # print(rects)

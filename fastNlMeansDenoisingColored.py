import numpy as np
import cv2
import os
import argparse
from PIL import Image

def fastNlMeasnDenoising(input_path, output_path):
    for file in os.listdir(input_path):
        img = Image.open(input_path+file)
        img = np.array(img)
        img = cv2.fastNlMeansDenoisingColored(img, None, 6, 10, 7, 21)
        '''
        fastNlMeansDenoisingColored(
            src ---------- InputArray、カラー入力画像
            dst ---------- OutputArray、入力画像と同じサイズ、同じタイプ
            h = 3 -------- float、輝度成分のフィルタの平滑化の度合い、大きいとノイズが減少するが、エッジ部にも影響する
            hColor = 3 ---- float、色成分のフィルタの平滑化の度合い、10にしておけば十分
            templateWindowSize = 7 ---- int、重みを計算するのに使うテンプレート・パッチの辺の長さ（奇数に限る）、7を推奨
            searchWindowSize = 21 ---- int、加重平均を取るウィンドウの辺の長さ（奇数に限る）、21を推奨
        ）
        '''
        img = Image.fromarray(img)
        img.save(output_path+file)

def fastNlMeasnDenoisingMulti(input_path, output_path):
    imgs = []
    files = sorted(os.listdir(input_path))
    for file in files[:10]:
        img = Image.open(input_path+file)
        img = np.array(img)
        imgs.append(img)
    for index, (target_file, append_file) in enumerate(zip(files[5:-5], files[10:])):
        img = cv2.fastNlMeansDenoisingColoredMulti(imgs, 5, 5, None, 6, 10, 7, 21)
        img = Image.fromarray(img)
        img.save(output_path+target_file)
        img = Image.open(input_path+append_file)
        img = np.array(img)
        imgs.append(img)
        imgs.pop(0)

if __name__ == '__main__':
    parser = parser = argparse.ArgumentParser(description='This script is remove noiz filter from images')
    parser.add_argument('-i', '--input', type=str, default='')
    parser.add_argument('-o', '--output', type=str, default='')
    parser.add_argument('-m', '--multi', default=False, action='store_true')

    args = parser.parse_args()
    if args.multi:
        fastNlMeasnDenoisingMulti(args.input, args.output)
    else:
        fastNlMeasnDenoising(args.input, args.output)




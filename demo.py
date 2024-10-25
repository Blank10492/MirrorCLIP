import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from mirrorclip import MirrorCLIP


def demo():
    parser = argparse.ArgumentParser()
    parser.add_argument('-encoder', type=str, default="ViT-B/32")
    parser.add_argument('-feat', type=str, default="visual")
    parser.add_argument('-image', type=str, default="./dog_cat.jpg")
    parser.add_argument('-task', type=str, default="image")
    args = parser.parse_args()

    mirror_clip = MirrorCLIP(args)

    if args.task == "image":
        cls_list = ["a photo of dog", "a photo of cat", "a photo of egg", "a photo of eraser"]
    elif args.task == "text":
        cls_list = [ "text of egg", "text of eraser", "text of dog", "text of cat"]
    else:
        raise ValueError

    attack_score = list(mirror_clip.inference(args.image, cls_list, feat='origin').detach().cpu().tolist())
    attack_score_disentangled = list(mirror_clip.inference(args.image, cls_list, feat=args.feat).detach().cpu().tolist())

    print(f'original results: {attack_score}')
    print(f'{args.feat} results: {attack_score_disentangled}')

    list1 = np.array([round(x, 3) for x in attack_score])
    list2 = np.array([round(x, 3) for x in attack_score_disentangled])
    length = len(list1)
    x = np.arange(length)
    listDate = cls_list

    plt.figure()
    total_width, n = 0.8, 2
    width = total_width / n
    x1 = x - width / 2
    x2 = x1 + width 

    plt.title("")
    plt.ylabel("score")
    plt.xlabel('text prompt')
    bar1 = plt.bar(x1, list1, width=width, label="image features", color=["#FFBE7D"])
    if args.task == "image":
        bar2 = plt.bar(x2, list2, width=width, label="visual features", color=["#F1CE63"])
    elif args.task == "text":
        bar2 = plt.bar(x2, list2, width=width, label="textual features", color=["#86BCB6"])

    plt.xticks(x, listDate)
    plt.legend()
    plt.bar_label(bar1)
    plt.bar_label(bar2)
    plt.show()

if __name__ == '__main__':
    demo()
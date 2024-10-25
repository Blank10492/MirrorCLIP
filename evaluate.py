import argparse
from mirrorclip import MirrorCLIP


def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, default='real-world')
    parser.add_argument('-t', '--evaluate_on_TA', action='store_true')
    parser.add_argument('-encoder', type=str, default="ViT-B/32")
    parser.add_argument('-batch', type=int, default=1024)
    parser.add_argument('-feat', type=str, default="origin")
    args = parser.parse_args()

    dataset_name = [args.dataset]
    if args.dataset == 'synthetic':
        dataset_name = ['imagenet', 'caltech', 'pets', 'cars', 'flowers', 'food', 'aircraft', 'dtd', 'sun', 'eurosat']
    elif args.dataset == 'real-world':
        dataset_name = ['disentangling', 'rta-100', 'paint']
    elif args.dataset == 'text':
        dataset_name = ['disentanglingtypo', 'rta-100typo', 'painttypo', 'imagenettypo', 'flowerstypo', 'foodtypo']

    mirror_clip = MirrorCLIP(args)

    acc_list = []
    for dataset_i in dataset_name:
        print(f'Evaluating on {dataset_i}')
        args.dataset = dataset_i
        accuracy = mirror_clip.evaluate(args)
        print(f'Accuracy: {accuracy}')
        acc_list.append(round(accuracy*100, 2))

    acc_list.append(round((sum(acc_list) / len(acc_list)), 2))
    
    print("Results: ", '& '.join(list(map(str, acc_list))))

if __name__ == '__main__':
    evaluate()
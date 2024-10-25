from datasets.caltech101 import Caltech101
from datasets.disentangling_ta import DisentanglingDataset
from datasets.disentangling_ta_typo import DisentanglingDatasetTypo
from datasets.dtd import DTD
from datasets.eurosat import EuroSAT
from datasets.fgvcaircraft import FGVCAircraft
from datasets.flowers102 import Flowers102
from datasets.flowers102_typo import Flowers102Typo
from datasets.food101 import Food101
from datasets.food101_typo import Food101Typo
from datasets.ImageNetV2 import ImageNetValDataset
from datasets.ImageNetV2_typo import ImageNetValDatasetTypo
from datasets.oxford_pets import OxfordIIITPet
from datasets.paint_ta import PAINTDataset
from datasets.paint_ta_typo import PAINTDatasetTypo
from datasets.rta100 import RTA100
from datasets.rta100_typo import RTA100Typo
from datasets.stanford_cars import StanfordCars
from datasets.sun397 import SUN397


def dataset(args, preprocess):
    # synthesized data
    if args.dataset == 'imagenet':
        data = ImageNetValDataset(location='data', transform=preprocess)
    elif args.dataset == 'imagenettypo':
        data = ImageNetValDatasetTypo(location='data', transform=preprocess)
    elif args.dataset == 'caltech':
        data = Caltech101(root='data', transform=preprocess, download=True, make_typographic_dataset=True)
    elif args.dataset == 'pets':
        data = OxfordIIITPet(root='data', split='test', transform=preprocess, download=True, make_typographic_dataset=True)
    elif args.dataset == 'cars':
        data = StanfordCars(root='data', split='test', transform=preprocess, download=True, make_typographic_dataset=True)
    elif args.dataset == 'flowers':
        data = Flowers102(root='data', split='test', transform=preprocess, download=True, make_typographic_dataset=True)
    elif args.dataset == 'flowerstypo':
        data = Flowers102Typo(root='data', split='test', transform=preprocess, download=True, make_typographic_dataset=True)
    elif args.dataset == 'food':
        data = Food101(root='data', split='test', transform=preprocess, download=True)
    elif args.dataset == 'foodtypo':
        data = Food101Typo(root='data', split='test', transform=preprocess, download=True)
    elif args.dataset == 'aircraft':
        data = FGVCAircraft(root='data', split='test', transform=preprocess, download=True)
    elif args.dataset == 'dtd':
        data = DTD(root='data', split='test', transform=preprocess, download=True)
    elif args.dataset == 'eurosat':
        data = EuroSAT(root='data', split='test', transform=preprocess, download=True)
    elif args.dataset == 'sun':
        data = SUN397(root='data', split='test', transform=preprocess, download=True)
    # real world data
    elif args.dataset == 'paint':
        data = PAINTDataset(root='data', transform=preprocess)
    elif args.dataset == 'disentangling':
        data = DisentanglingDataset(root='data', transform=preprocess)
    elif args.dataset == 'rta-100':
        data = RTA100(root='data', transform=preprocess)
    elif args.dataset == 'painttypo':
        data = PAINTDatasetTypo(root='data', transform=preprocess)
    elif args.dataset == 'disentanglingtypo':
        data = DisentanglingDatasetTypo(root='data', transform=preprocess)
    elif args.dataset == 'rta-100typo':
        data = RTA100Typo(root='data', transform=preprocess)
    else:
        raise ValueError
    return data

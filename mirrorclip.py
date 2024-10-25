import clip
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

from dataset import dataset

class MirrorCLIP():
    def __init__(self, args):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model, self.preprocess = clip.load(args.encoder, device=self.device)
        self.model.eval()

    def disentanglement(self, image, feat='visual'):
        image_f = torch.flip(image, dims=[-1])
        img_features_f = self.model.encode_image(image_f)
        img_features = self.model.encode_image(image)
        
        img_features /= img_features.norm(dim=-1, keepdim=True)
        img_features_f /= img_features_f.norm(dim=-1, keepdim=True)
        
        sim_of = img_features * img_features_f
        nontypo_mask = sim_of > 0
        typo_mask = sim_of < 0
        
        typo_feat = img_features.clone().detach()
        nontypo_feat = img_features.clone().detach()
        typo_feat_f = img_features_f.clone().detach()
        nontypo_feat_f = img_features_f.clone().detach()

        typo_feat_clone = typo_feat.clone()
        nontypo_feat_clone = nontypo_feat.clone()

        typo_feat[nontypo_mask] -= typo_feat_f[nontypo_mask]
        nontypo_feat[nontypo_mask] += nontypo_feat_f[nontypo_mask]
        typo_feat_f[nontypo_mask] -= typo_feat_clone[nontypo_mask]
        nontypo_feat_f[nontypo_mask] += nontypo_feat_clone[nontypo_mask]

        typo_feat /= typo_feat.norm(dim=-1, keepdim=True)
        nontypo_feat /= nontypo_feat.norm(dim=-1, keepdim=True)
        typo_feat_f /= typo_feat_f.norm(dim=-1, keepdim=True)
        nontypo_feat_f /= nontypo_feat_f.norm(dim=-1, keepdim=True)

        if feat=='visual':
            return nontypo_feat_f
        elif feat=='textual':
            return typo_feat
        else:
            raise ValueError
        
    def inference(self, image, prompt, feat):
        image = self.preprocess(Image.open(image)).unsqueeze(0).to(self.device)
        text = clip.tokenize(prompt).to(self.device)
        text_features = self.model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        if feat=='origin':
            features = self.model.encode_image(image)
            features /= features.norm(dim=-1, keepdim=True)
        else:
            features = self.disentanglement(image, feat)

        probs = (self.model.logit_scale.exp() * (features @ text_features.T)).softmax(dim=-1)[0]

        return probs


    def evaluate(self, args):
        test_data = dataset(args, self.preprocess)
        test_loader = DataLoader(test_data, batch_size=args.batch, shuffle=False, num_workers=8, pin_memory=True)

        text_inputs = torch.cat([clip.tokenize(test_data.templates[0].format(c)) for c in test_data.classes]).to(self.device)
        text_inputs = text_inputs.cuda()
        acc, total = 0, 0
        with torch.no_grad():
            text_feature = self.model.encode_text(text_inputs)
            text_feature /= text_feature.norm(dim=-1, keepdim=True)
            
            for i, (original_image, typographic_image, target) in enumerate(test_loader):
                print(f'{i}/{len(test_loader)}')
                if args.evaluate_on_TA:
                    image = typographic_image.to(self.device)
                else:
                    image = original_image.to(self.device)
                target = target.to(self.device)      

                if args.feat=='origin':
                    features = self.model.encode_image(image)
                    features /= features.norm(dim=-1, keepdim=True)
                else:
                    features = self.disentanglement(image, args.feat)
                
                similarity = 100. * (features @ text_feature.T)

                probs = F.softmax(similarity, dim=-1).max(-1)[1]

                acc += probs.eq(target).sum().item()
                total += target.size(0)

        return acc / total
import argparse
import os
from PIL import Image
import torch
from torchvision import transforms
from resnet_cf import resnetcf18
from lib import *

def main(args):
    model = resnetcf18(imagenet_pretrained=True)
    if args.pretrained_checkpoint_path is not None:
        checkpoint = torch.load(args.pretrained_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(10, Image.BILINEAR),
            transforms.RandomResizedCrop(
                (256, 256), scale=((0.95)**2, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # Follow https://pytorch.org/vision/stable/models.html
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            # Follow https://pytorch.org/vision/stable/models.html
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }

    dataset = {x: ImageFolderWithCf(args.cf_csv_path,
                                    os.path.join(args.image_folder_path, x),
                                    data_transforms[x])
               for x in ['train', 'val']}

    dataloaders = {
        'train': torch.utils.data.DataLoader(dataset['train'],
                                             batch_size=8,
                                             shuffle=True,
                                             num_workers=1,
                                             pin_memory=True),
        'val':   torch.utils.data.DataLoader(dataset['val'],
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=1,
                                             pin_memory=True)
    }

    train_eval(args.log_dir_path, dataloaders, model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The training script of HCC MVI.')
    parser.add_argument('image_folder_path', help='')
    parser.add_argument('cf_csv_path', help='')
    parser.add_argument('log_dir_path', help='')
    parser.add_argument('--pretrained_checkpoint_path', help='')
    args = parser.parse_args()
    main(args)

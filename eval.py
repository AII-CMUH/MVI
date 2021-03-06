import argparse
import os
import torch
from torchvision import transforms
from resnet_cf import resnetcf18
from lib import *

def main(args):
    model = resnetcf18()
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    data_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # Follow https://pytorch.org/vision/stable/models.html
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageFolderWithCf(args.cf_csv_path,
                                os.path.join(args.image_folder_path, 'val'),
                                data_transform)

    dataloaders = {
        'val':   torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=1,
                                             pin_memory=True)
    }

    train_eval(args.log_dir_path, dataloaders, model, eval_only=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The evaluation script of HCC MVI.')
    parser.add_argument('image_folder_path', help='the path of the folder that contains input images')
    parser.add_argument('cf_csv_path', help='the path of the file that contains clinical factors in csv format')
    parser.add_argument('log_dir_path', help='log directory path')
    parser.add_argument('checkpoint_path', help='the path of trained model file')
    args = parser.parse_args()
    main(args)

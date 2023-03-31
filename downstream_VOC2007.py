import data
from tqdm import tqdm

import models

import torch
from torch.nn import functional as F

import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

from sklearn.metrics import average_precision_score

import argparse
import os

def extract_features(model, dataloader):
    features = []
    ys = []

    model.eval()

    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X = X.to('cuda', non_blocking=True)
            features.append(F.normalize(model.extract_visual_features(X).mean(dim=[-1,-2]), p=2, dim=-1))
            ys.append(y)

    features = torch.cat(features, dim=0).cpu().numpy()
    ys = torch.cat(ys, dim=0).numpy()
    return features, ys

def main(args):
    # Construct the dataloaders
    trainset = data.VOC2007(root=args.data_dir, split='trainval', transforms=data.default_test_transform)
    testset = data.VOC2007(root=args.data_dir, split='test', transforms=data.default_test_transform)

    trainloader = torch.utils.data.DataLoader(trainset, 
                                          batch_size=64, 
                                          num_workers=4, 
                                          drop_last=False,
                                          shuffle=False, pin_memory=True)

    testloader = torch.utils.data.DataLoader(testset, 
                                          batch_size=64, 
                                          num_workers=4, 
                                          drop_last=False,
                                          shuffle=False, pin_memory=True)

    # load the model
    model = models.BiDirectional_Captioning_Model().load_from_checkpoint(args.model_path).to("cuda")
    
    # Extract features 
    features_train, ys_train = extract_features(model, trainloader)
    features_test, ys_test = extract_features(model, testloader)

    # Create linear SVC
    svc = LinearSVC(penalty='l2', 
                loss='squared_hinge', 
                dual=True,
                tol=1e-4,
                max_iter=2000,
                class_weight={1: 2, -1: 1}, 
                random_state=args.seed)

    clf = GridSearchCV(
        svc, param_grid={'C': [0.01, 0.1, 1.0, 10.0]}, 
        scoring="average_precision", 
        n_jobs=-1, cv=3
    ) 

    scores = []
    for cl in tqdm(range(len(trainset.classes))):
        # set ignored as negative to increase training set size
        label_train = ys_train[:, cl].copy()
        label_train[label_train == 0] = -1
        clf.fit(features_train, label_train)
        
        # remove ignored for testing
        idx_test = (ys_test[:, cl] != 0)
        label_test = ys_test[idx_test, cl]
        scores.append(average_precision_score(label_test, clf.decision_function(features_test[idx_test]).reshape(-1, 1)))

    mAP = np.mean(scores)
    print(f"Model: {args.model_path}")
    print("Average precision score: ", mAP)

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        np.save(os.path.join(args.save_dir, f"mAP_{args.model_path.split('/')[-1].split('.')[0]}.npy"), {
            "mAP": mAP,
            "APs": scores,
            "model_path": args.model_path,
            "classes": trainset.classes
        })



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/scratch/datasets/cp598/VOC2007")
    parser.add_argument("--model_path", type=str, default="testrun_5k/epoch=108-step=50000.ckpt")
    parser.add_argument("--save_dir", type=str, default="downstream_VOC2007")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args()
    main(args)
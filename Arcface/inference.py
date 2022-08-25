from sklearn.neighbors import NearestNeighbors
import os
import torch
from arcface import *
import torch.nn as nn
from logger import get_logger
import torch.optim as optim
import torch.autograd as autograd
from torch.utils.data import DataLoader
import pandas as pd
from torchvision import transforms
from whale_dataset import *

class Inference:
    def __init__(self, data_path = "/kaggle/input/happy-whale-and-dolphin", csv_name = 'train.csv', test = False, number_classes = 2000, path="efficientnet_b5.pkl"):
        """
        Init Dataset, Model and others
        """
        self.data_path = data_path
        self.save_path = path
        self.test = test
        self.num_classes = number_classes
        train_transform = transforms.Compose([

            transforms.Resize([256,256]),
            #transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            # 归一化 均值 方差待修正
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            # [0.656,0.487,0.411], [1., 1., 1.]
            ])
        val_transform = transforms.Compose([
            transforms.Resize([256,256]),
            
            # 归一化 均值 方差待修正
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            # [0.656,0.487,0.411], [1., 1., 1.]
            ]) 
        #self.cacd_dataset = ImageData(root_path=root_path, label_path="data/label.npy", name_path="data/name.npy", train_mode = "train")
        train_ds, self.unique_id, self.unique_species = get_train_pd(data_path, csv_name)
        self.ds_train = WhaleDataset(train_ds, data_path, transform=train_transform)
        test_ds = get_test_pd(data_path)
        self.ds_test = WhaleDataset(test_ds, data_path, test = self.test, transform=val_transform)

        self.model = Effnet(num_classes = number_classes, pretrained = True)
        
        if torch.cuda.device_count() > 1:
            print("There are ", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
        elif torch.cuda.device_count() == 1:
            print("There is only one GPU")
        else:
            print("Only use CPU")
        if torch.cuda.is_available():
           self.model.cuda()
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, 'arcface.pth')))           

    @torch.no_grad()
    def neighbor(self):
        train_targets = []
        train_embeddings = []
        train_loader = DataLoader(self.ds_train, batch_size=self.batch_size, num_workers=8)
            
        for i_batch, sample_batch in enumerate(train_loader):
            images_batch, labels_batch = sample_batch['image'], sample_batch['label']
                                   
            labels_batch = torch.LongTensor(labels_batch.view(-1).numpy())
            if torch.cuda.is_available():
                input_image = autograd.Variable(images_batch.cuda())
                target_label = autograd.Variable(labels_batch.cuda())
            else:
                input_image = autograd.Variable(images_batch)
                target_label = autograd.Variable(labels_batch)
            # Step.2 calculate loss
            feature = self.model(input_image)
            
            train_targets.append(target_label.data.cpu())
            train_embeddings.append(feature.data.cpu())
        train_targets = np.concatenate(train_targets)
        train_embeddings = np.concatenate(train_embeddings)
        neigh = NearestNeighbors(n_neighbors=50,metric='cosine')
        neigh.fit(train_embeddings)
        print('neigh build success!')
        return neigh, train_targets

    @torch.no_grad()  
    def knn_inference(self, batch_size=32, threshold = 0.4):
        self.batch_size = batch_size
        neigh, train_targets = self.neighbor()
        test_df = pd.read_csv(os.path.join(self.data_path, 'sample_submission.csv'))
        preds = []
        test_loader = DataLoader(self.ds_test, batch_size=batch_size, num_workers=8)            
        for i_batch, sample_batch in enumerate(test_loader):
            # Step.1 Load data and label
            images_batch = sample_batch['image']

            if torch.cuda.is_available():
                input_image = autograd.Variable(images_batch.cuda())
            else:
                input_image = autograd.Variable(images_batch)
                # Step.2 calculate loss
            feature = self.model(input_image)
            distances,idxs = neigh.kneighbors(feature.data.cpu(), return_distance=True)
            confidences = distances
            for i in len(confidences):
                top5 = []
                pred = ''
                count = 0
                j = 0
                while count < 5 and j < len(confidences[i]):
                    if 'new_individual' in top5 or confidences[i][j] <= threshold:
                        target = self.unique_id.loc[int(train_targets[idxs[i][j]])]['individual_id']
                        if target not in top5:
                            top5.append(target)
                            count += 1
                        j += 1
                    elif 'new_individual' not in top5 and confidences[i][j] > threshold:
                        top5.append('new_individual')
                        count += 1
                pred = ' '.join(top5)
                preds.append(pred)      

        test_df['predictions'] = pd.Series(preds)
        if not os.path.exists('./submission'):
            os.mkdir('./submission')
        test_df.to_csv('./submission/acrface_submission.csv', index=False)
        print('success')

    @torch.no_grad()
    def inference(self, batch_size = 32, threshold = 0.4):
        self.batch_size = batch_size
        neigh, train_targets = self.neighbor()
        test_df = pd.read_csv(os.path.join(self.data_path, 'sample_submission.csv'))
        preds = []
        test_loader = DataLoader(self.ds_test, batch_size=batch_size, num_workers=8)            
        for i_batch, sample_batch in enumerate(test_loader):
            # Step.1 Load data and label
            images_batch = sample_batch['image']

            if torch.cuda.is_available():
                input_image = autograd.Variable(images_batch.cuda())
            else:
                input_image = autograd.Variable(images_batch)
                # Step.2 calculate loss
            feature = self.model(input_image)
            distances,idxs = torch.topk(feature, k=5, dim=1, largest=True)
            confidences = distances
            for i in len(confidences):
                top5 = []
                pred = ''
                count = 0
                j = 0
                while count < 5 and j < len(confidences[i]):
                    if 'new_individual' in top5 or confidences[i][j] >= threshold:
                        target = self.unique_id.loc[int(train_targets[idxs[i][j]])]['individual_id']
                        if target not in top5:
                            top5.append(target)
                            count += 1
                        j += 1
                    elif 'new_individual' not in top5 and confidences[i][j] < threshold:
                        top5.append('new_individual')
                        count += 1
                pred = ' '.join(top5)
                preds.append(pred)      

        test_df['predictions'] = pd.Series(preds)
        if not os.path.exists('./submission'):
            os.mkdir('./submission')
        test_df.to_csv('./submission/acrface_submission.csv', index=False)
        print('success')


inference = Inference(data_path = "./datasets", csv_name = 'train.csv', test = True, number_classes = 15587, path="./model")

inference.inference(64)
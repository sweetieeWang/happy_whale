from bdb import effective
from builtins import type
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
class Train:
    def __init__(self, data_path = "/kaggle/input/happy-whale-and-dolphin", csv_name = 'train.csv', test = False, s = 10.0, m = 0.5, number_classes = 2000, species = 30, special_id = 300, train_percentage=0.7, path="model.pkl",  loadPretrain=0):
        """
        Init Dataset, Model and others
        """
        self.data_path = data_path
        self.save_path = path
        self.test = test
        self.num_classes = number_classes
        self.s = s
        self.m = m
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
        if self.test:
            train_ds, self.unique_id, self.unique_species = get_train_pd(data_path, csv_name)
            self.ds_train = WhaleDataset(train_ds, data_path, transform=train_transform)
            self.ds_val = None
            test_ds = get_test_pd(data_path)
            self.ds_test = WhaleDataset(test_ds, data_path, test = self.test, transform=val_transform)
        else:
            train_ds, val_ds = file_split(data_path, csv_name, number_classes, species, special_id, train_percentage)
            self.ds_train = WhaleDataset(train_ds, data_path, transform=train_transform)
            self.ds_val = WhaleDataset(val_ds, data_path, transform=val_transform)
            self.ds_test = None

        self.model = Effnet(num_classes = number_classes, pretrained = True)
        self.arcface = ArcFace(num_classes = number_classes,
                            s = 30.0,
                            m = 0.5,
                            easy_margin = False,
                            ls_eps = 0.0)
        if torch.cuda.device_count() > 1:
            print("There are ", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
        elif torch.cuda.device_count() == 1:
            print("There is only one GPU")
        else:
            print("Only use CPU")

        if torch.cuda.is_available():
           self.model.cuda()
           self.arcface.cuda()
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, 'arcface.pth')))

    def start_train(self, epoch=10, logger = None, batch_size=32, learning_rate=0.001, weight_decay = 1e-4, batch_display=50, loss = 'FocalLoss', save_freq=1):
        """
        Detail of training
        """
        self.epoch_num = epoch
        self.batch_size = batch_size
        self.lr = learning_rate
        self.weight_decay = weight_decay

        loss_function = FocalLoss(class_num=self.num_classes, 
                                alpha=None, 
                                gamma=2, 
                                size_average=True).cuda()
        #loss_function = FocalLoss(class_num =self.num_classes).cuda()
        optimizer = optim.Adam([{"params":self.model.parameters()},{"params":self.arcface.parameters()}], 
                            lr=self.lr, 
                            weight_decay=self.weight_decay )

        for epoch in range(self.epoch_num):
            batch_count = 0
            total_loss = 0
            accuracy = 0.0
            train_loader = DataLoader(self.ds_train, batch_size=self.batch_size, num_workers=8)
            
            for i_batch, sample_batch in enumerate(train_loader):
                # Step.output1 Load data and label
                images_batch, labels_batch = sample_batch['image'], sample_batch['label']
                                   
                labels_batch = torch.LongTensor(labels_batch.view(-1).numpy())
                if torch.cuda.is_available():
                    input_image = autograd.Variable(images_batch.cuda())
                    target_label = autograd.Variable(labels_batch.cuda())
                else:
                    input_image = autograd.Variable(images_batch)
                    target_label = autograd.Variable(labels_batch)
                # Step.2 calculate loss
                consine = self.model(input_image)
                output = self.arcface(consine, target_label)
                loss = loss_function(output, target_label)
                batch_count += 1
                total_loss += loss
                # Step.3 Update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pred_label = torch.argmax(output, dim=1)
                #print("***************",pred_label[:4], target_label[:4])
                batch_correct =  (pred_label == target_label).sum().data * 1.0 / self.batch_size
                accuracy += batch_correct
                # Check Result
                if (i_batch+1) % batch_display == 0:                    
                    print("Epoch : {}, Batch : {}, Loss : {:.4f}, Batch Accuracy {:.2%}".format(epoch, i_batch, loss, batch_correct))

            """
            Save model
            """
            if not self.test:
                #neigh, train_targets = self.neighbor()
                accuracy_val = self.evaluate(dataset=self.ds_val, batch_size = self.batch_size)
                logger.info('Epoch:[{}/{}]\t loss={:.5f}\t traing acc={:.3%} val acc={:.3%}'.format(epoch , 100, total_loss / batch_count, accuracy / batch_count,accuracy_val ))
                #print("Epoch {} train accuracy: {:.2%} val accuracy: {:.2%}".format(epoch, accuracy/batch_count, accuracy_val))
            else:
                logger.info('Epoch:[{}/{}]\t loss={:.5f}\t traing acc={:.3%}'.format(epoch , 100, total_loss / batch_count, accuracy / batch_count))
            if (epoch+1) % save_freq == 0:
                torch.save(self.model.state_dict(), os.path.join(self.save_path,'arcface.pth'))

    def evaluate(self, dataset =None, batch_size=32):
        """
        Detail of evaluate
        """
        with torch.no_grad():
            correct_num = 0
            eval_loader = DataLoader(dataset, batch_size=batch_size, num_workers=8)
            for i_batch, sample_batch in enumerate(eval_loader):
                # Step.1 Load data and label
                images_batch, labels_batch = sample_batch['image'], sample_batch['label']
                """
                for i in range(images_batch.shape[0]):
                    img_tmp = transforms.ToPILImage()(images_batch[i]).convert('RGB')
                    plt.imshow(img_tmp)
                    plt.pause(0.001)
                """
                labels_batch = torch.LongTensor(labels_batch.view(-1).numpy())
                if torch.cuda.is_available():
                    input_image = autograd.Variable(images_batch.cuda())
                    target_label = autograd.Variable(labels_batch.cuda())
                else:
                    input_image = autograd.Variable(images_batch)
                    target_label = autograd.Variable(labels_batch)
                # Step.2 calculate loss
                output = self.model(input_image)
                # print(output[:10])
                pred_prob, pred_label = torch.max(output, dim=1)
                # print(pred_label)
                correct_num += (pred_label == target_label).sum().data

            accuracy = correct_num * 1.0 / len(dataset)
            return accuracy    




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
        return neigh, train_targets

    
    def knn_evaluate(self, neigh, train_targets, dataset=None, batch_size=32):
        """
        Detail of evaluate
        """
        with torch.no_grad():
            correct_num = 0
            eval_loader = DataLoader(dataset, batch_size=batch_size, num_workers=8)            
            for i_batch, sample_batch in enumerate(eval_loader):
                # Step.1 Load data and label
                images_batch, labels_batch = sample_batch['image'], sample_batch['label']
                           
                labels_batch = torch.LongTensor(labels_batch.view(-1).numpy())
                target_label = labels_batch.numpy()
                if torch.cuda.is_available():
                    input_image = autograd.Variable(images_batch.cuda())
                else:
                    input_image = autograd.Variable(images_batch)
                # Step.2 calculate loss
                output = self.model(input_image)

                pred_label = []
                idxs = neigh.kneighbors(output.data.cpu(), 1, return_distance=False)
                for i in idxs:
                    pred_label.append(train_targets[i])
                # pred_label = torch.tensor(pred_label)
                pred_label = np.array(pred_label)

                correct_num += (pred_label == target_label).sum()
            accuracy = correct_num * 1.0 / len(dataset)
            return accuracy
 
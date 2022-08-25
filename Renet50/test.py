import torch
import torchvision
import os
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.utils.data import DataLoader
from torchvision import transforms
from whale_dataset import WhaleDataset, file_split
from torchvision import models
from logger import get_logger
import logging
import torch.optim as optim
from torch.optim import lr_scheduler
class Train:
    def __init__(self, data_path = "/kaggle/input/happy-whale-and-dolphin", csv_name = 'train.csv', model_name = "resnet50", number_classes = 800, species = 30, special_id = 22, train_percentage=0.7, path="model.pkl",  loadPretrain=0):
        """
        Init Dataset, Model and others
        """
        self.save_path = path
        train_ds, val_ds = file_split(data_path, csv_name, number_classes, species, special_id, train_percentage)
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

        val_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            # 归一化 均值 方差待修正
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        self.ds_train = WhaleDataset(train_ds, data_path, transform=train_transform)
        self.ds_val = WhaleDataset(val_ds, data_path, transform=val_transform)
        if model_name == "resnet50":

            # self.model = models.resnet50(pretrained = True, num_classes = number_classes)
            self.model = models.resnet50(pretrained=True)
            numFit = self.model.fc.in_features
            self.model.fc = nn.Linear(numFit, number_classes)

        if torch.cuda.device_count() > 1:
            print("There are ", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
        elif torch.cuda.device_count() == 1:
            print("There is only one GPU")
        else:
            print("Only use CPU")

        if torch.cuda.is_available():
           self.model.cuda()

    def start_train(self, epoch=10, batch_size=32, learning_rate=0.001, batch_display=50, save_freq=1):
        """
        Detail of training
        """
        self.epoch_num = epoch
        self.batch_size = batch_size
        self.lr = learning_rate
        
        loss_function = nn.CrossEntropyLoss().cuda()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-2)

        logger = get_logger('./train_label_lr%.4f_bc%d.log' % (self.lr, self.batch_size))
        scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
        for epoch in range(self.epoch_num):
            scheduler.step()
            batch_count = 0
            total_loss = 0
            accuracy = 0.0
            train_loader = DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True, num_workers=4)
            for i_batch, sample_batch in enumerate(train_loader):
                # Step.1 Load data and label
                images_batch, labels_batch = sample_batch['image'], sample_batch['label']
                
                                   
                labels_batch = torch.LongTensor(labels_batch.view(-1).numpy())
                if torch.cuda.is_available():
                    input_image = autograd.Variable(images_batch.cuda())
                    target_label = autograd.Variable(labels_batch.cuda())
                else:
                    input_image = autograd.Variable(images_batch)
                    target_label = autograd.Variable(labels_batch)
                # Step.2 calculate loss
                # print('******************************************', input_image.shape)
                output = self.model(input_image)
                loss = loss_function(output, target_label)
                batch_count += 1
                total_loss += loss
                # Step.3 Update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pred_prob, pred_label = torch.max(output, dim=1)
                batch_correct =  (pred_label == target_label).sum().data * 1.0 / self.batch_size
                accuracy += batch_correct
                # Check Result
                if i_batch % batch_display == 0:
                    print("Epoch : {}, Batch : {}, Loss : {:.4f}, Batch Accuracy {:.2%}".format(epoch, i_batch, loss, batch_correct))
            """
            Save model
            """
            accuracy = accuracy/batch_count
            accuracy_val = self.evaluate(self.batch_size)
            logger.info('Epoch:[{}/{}]\t loss={:.5f}\t traing acc={:.3%} val acc={:.3%}'.format(epoch , 100, total_loss / batch_count, accuracy,accuracy_val ))
            print("Epoch {} Average Loss : {:.4f} train accuracy: {:.2%} val accuracy: {:.2%}".format(epoch, total_loss / batch_count, accuracy, accuracy_val))

            if epoch % save_freq == 0:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, 'model.pth'))
        logger.info('finish training!')


    def evaluate(self, batch_size=32):
        """
        Detail of evaluate
        """
        with torch.no_grad():
            correct_num = 0
            eval_loader = DataLoader(self.ds_val, batch_size=batch_size, num_workers=8)
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

                pred_prob, pred_label = torch.max(output, dim=1)
                correct_num += (pred_label == target_label).sum().data

            accuracy = correct_num * 1.0 / len(self.ds_val)
            return accuracy
            
            
            
            
            
            
            
import torch
import torchvision
import os
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.utils.data import DataLoader
from torchvision import transforms
from whale_dataset import WhaleDataset, file_split
from torchvision import models
from logger import get_logger
import logging
import torch.optim as optim
from torch.optim import lr_scheduler
class Train:
    def __init__(self, data_path = "/kaggle/input/happy-whale-and-dolphin", csv_name = 'train.csv', model_name = "resnet50", number_classes = 800, species = 30, special_id = 22, train_percentage=0.7, path="model.pkl",  loadPretrain=0):
        """
        Init Dataset, Model and others
        """
        self.save_path = path
        train_ds, val_ds = file_split(data_path, csv_name, number_classes, species, special_id, train_percentage)
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

        val_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            # 归一化 均值 方差待修正
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        self.ds_train = WhaleDataset(train_ds, data_path, transform=train_transform)
        self.ds_val = WhaleDataset(val_ds, data_path, transform=val_transform)
        if model_name == "resnet50":

            # self.model = models.resnet50(pretrained = True, num_classes = number_classes)
            self.model = models.resnet50(pretrained=True)
            numFit = self.model.fc.in_features
            self.model.fc = nn.Linear(numFit, number_classes)

        if torch.cuda.device_count() > 1:
            print("There are ", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
        elif torch.cuda.device_count() == 1:
            print("There is only one GPU")
        else:
            print("Only use CPU")

        if torch.cuda.is_available():
           self.model.cuda()

    def start_train(self, epoch=10, batch_size=32, learning_rate=0.001, batch_display=50, save_freq=1):
        """
        Detail of training
        """
        self.epoch_num = epoch
        self.batch_size = batch_size
        self.lr = learning_rate
        
        loss_function = nn.CrossEntropyLoss().cuda()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-2)

        logger = get_logger('./train_label_lr%.4f_bc%d.log' % (self.lr, self.batch_size))
        scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
        for epoch in range(self.epoch_num):
            scheduler.step()
            batch_count = 0
            total_loss = 0
            accuracy = 0.0
            train_loader = DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True, num_workers=4)
            for i_batch, sample_batch in enumerate(train_loader):
                # Step.1 Load data and label
                images_batch, labels_batch = sample_batch['image'], sample_batch['label']
                
                                   
                labels_batch = torch.LongTensor(labels_batch.view(-1).numpy())
                if torch.cuda.is_available():
                    input_image = autograd.Variable(images_batch.cuda())
                    target_label = autograd.Variable(labels_batch.cuda())
                else:
                    input_image = autograd.Variable(images_batch)
                    target_label = autograd.Variable(labels_batch)
                # Step.2 calculate loss
                # print('******************************************', input_image.shape)
                output = self.model(input_image)
                loss = loss_function(output, target_label)
                batch_count += 1
                total_loss += loss
                # Step.3 Update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pred_prob, pred_label = torch.max(output, dim=1)
                batch_correct =  (pred_label == target_label).sum().data * 1.0 / self.batch_size
                accuracy += batch_correct
                # Check Result
                if i_batch % batch_display == 0:
                    print("Epoch : {}, Batch : {}, Loss : {:.4f}, Batch Accuracy {:.2%}".format(epoch, i_batch, loss, batch_correct))
            """
            Save model
            """
            accuracy = accuracy/batch_count
            accuracy_val = self.evaluate(self.batch_size)
            logger.info('Epoch:[{}/{}]\t loss={:.5f}\t traing acc={:.3%} val acc={:.3%}'.format(epoch , 100, total_loss / batch_count, accuracy,accuracy_val ))
            print("Epoch {} Average Loss : {:.4f} train accuracy: {:.2%} val accuracy: {:.2%}".format(epoch, total_loss / batch_count, accuracy, accuracy_val))

            if epoch % save_freq == 0:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, 'model.pth'))
        logger.info('finish training!')


    def evaluate(self, batch_size=32):
        """
        Detail of evaluate
        """
        with torch.no_grad():
            correct_num = 0
            eval_loader = DataLoader(self.ds_val, batch_size=batch_size, num_workers=8)
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

                pred_prob, pred_label = torch.max(output, dim=1)
                correct_num += (pred_label == target_label).sum().data

            accuracy = correct_num * 1.0 / len(self.ds_val)
            return accuracy






import torch
import torchvision
import os
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.utils.data import DataLoader
from torchvision import transforms
from whale_dataset import WhaleDataset, file_split
from torchvision import models
from logger import get_logger
import logging
import torch.optim as optim
from torch.optim import lr_scheduler
class Train:
    def __init__(self, data_path = "/kaggle/input/happy-whale-and-dolphin", csv_name = 'train.csv', model_name = "resnet50", number_classes = 800, species = 30, special_id = 22, train_percentage=0.7, path="model.pkl",  loadPretrain=0):
        """
        Init Dataset, Model and others
        """
        self.save_path = path
        train_ds, val_ds = file_split(data_path, csv_name, number_classes, species, special_id, train_percentage)
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

        val_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            # 归一化 均值 方差待修正
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        self.ds_train = WhaleDataset(train_ds, data_path, transform=train_transform)
        self.ds_val = WhaleDataset(val_ds, data_path, transform=val_transform)
        if model_name == "resnet50":

            # self.model = models.resnet50(pretrained = True, num_classes = number_classes)
            self.model = models.resnet50(pretrained=True)
            numFit = self.model.fc.in_features
            self.model.fc = nn.Linear(numFit, number_classes)

        if torch.cuda.device_count() > 1:
            print("There are ", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
        elif torch.cuda.device_count() == 1:
            print("There is only one GPU")
        else:
            print("Only use CPU")

        if torch.cuda.is_available():
           self.model.cuda()

    def start_train(self, epoch=10, batch_size=32, learning_rate=0.001, batch_display=50, save_freq=1):
        """
        Detail of training
        """
        self.epoch_num = epoch
        self.batch_size = batch_size
        self.lr = learning_rate
        
        loss_function = nn.CrossEntropyLoss().cuda()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-2)

        logger = get_logger('./train_label_lr%.4f_bc%d.log' % (self.lr, self.batch_size))
        scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
        for epoch in range(self.epoch_num):
            scheduler.step()
            batch_count = 0
            total_loss = 0
            accuracy = 0.0
            train_loader = DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True, num_workers=4)
            for i_batch, sample_batch in enumerate(train_loader):
                # Step.1 Load data and label
                images_batch, labels_batch = sample_batch['image'], sample_batch['label']
                
                                   
                labels_batch = torch.LongTensor(labels_batch.view(-1).numpy())
                if torch.cuda.is_available():
                    input_image = autograd.Variable(images_batch.cuda())
                    target_label = autograd.Variable(labels_batch.cuda())
                else:
                    input_image = autograd.Variable(images_batch)
                    target_label = autograd.Variable(labels_batch)
                # Step.2 calculate loss
                # print('******************************************', input_image.shape)
                output = self.model(input_image)
                loss = loss_function(output, target_label)
                batch_count += 1
                total_loss += loss
                # Step.3 Update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pred_prob, pred_label = torch.max(output, dim=1)
                batch_correct =  (pred_label == target_label).sum().data * 1.0 / self.batch_size
                accuracy += batch_correct
                # Check Result
                if i_batch % batch_display == 0:
                    print("Epoch : {}, Batch : {}, Loss : {:.4f}, Batch Accuracy {:.2%}".format(epoch, i_batch, loss, batch_correct))
            """
            Save model
            """
            accuracy = accuracy/batch_count
            accuracy_val = self.evaluate(self.batch_size)
            logger.info('Epoch:[{}/{}]\t loss={:.5f}\t traing acc={:.3%} val acc={:.3%}'.format(epoch , 100, total_loss / batch_count, accuracy,accuracy_val ))
            print("Epoch {} Average Loss : {:.4f} train accuracy: {:.2%} val accuracy: {:.2%}".format(epoch, total_loss / batch_count, accuracy, accuracy_val))

            if epoch % save_freq == 0:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, 'model.pth'))
        logger.info('finish training!')


    def evaluate(self, batch_size=32):
        """
        Detail of evaluate
        """
        with torch.no_grad():
            correct_num = 0
            eval_loader = DataLoader(self.ds_val, batch_size=batch_size, num_workers=8)
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

                pred_prob, pred_label = torch.max(output, dim=1)
                correct_num += (pred_label == target_label).sum().data

            accuracy = correct_num * 1.0 / len(self.ds_val)
            return accuracy
            
            
            
            
import torch
import torchvision
import os
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.utils.data import DataLoader
from torchvision import transforms
from whale_dataset import WhaleDataset, file_split
from torchvision import models
from logger import get_logger
import logging
import torch.optim as optim
from torch.optim import lr_scheduler
class Train:
    def __init__(self, data_path = "/kaggle/input/happy-whale-and-dolphin", csv_name = 'train.csv', model_name = "resnet50", number_classes = 800, species = 30, special_id = 22, train_percentage=0.7, path="model.pkl",  loadPretrain=0):
        """
        Init Dataset, Model and others
        """
        self.save_path = path
        train_ds, val_ds = file_split(data_path, csv_name, number_classes, species, special_id, train_percentage)
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

        val_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            # 归一化 均值 方差待修正
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        self.ds_train = WhaleDataset(train_ds, data_path, transform=train_transform)
        self.ds_val = WhaleDataset(val_ds, data_path, transform=val_transform)
        if model_name == "resnet50":

            # self.model = models.resnet50(pretrained = True, num_classes = number_classes)
            self.model = models.resnet50(pretrained=True)
            numFit = self.model.fc.in_features
            self.model.fc = nn.Linear(numFit, number_classes)

        if torch.cuda.device_count() > 1:
            print("There are ", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
        elif torch.cuda.device_count() == 1:
            print("There is only one GPU")
        else:
            print("Only use CPU")

        if torch.cuda.is_available():
           self.model.cuda()

    def start_train(self, epoch=10, batch_size=32, learning_rate=0.001, batch_display=50, save_freq=1):
        """
        Detail of training
        """
        self.epoch_num = epoch
        self.batch_size = batch_size
        self.lr = learning_rate
        
        loss_function = nn.CrossEntropyLoss().cuda()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-2)

        logger = get_logger('./train_label_lr%.4f_bc%d.log' % (self.lr, self.batch_size))
        scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
        for epoch in range(self.epoch_num):
            scheduler.step()
            batch_count = 0
            total_loss = 0
            accuracy = 0.0
            train_loader = DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True, num_workers=4)
            for i_batch, sample_batch in enumerate(train_loader):
                # Step.1 Load data and label
                images_batch, labels_batch = sample_batch['image'], sample_batch['label']
                
                                   
                labels_batch = torch.LongTensor(labels_batch.view(-1).numpy())
                if torch.cuda.is_available():
                    input_image = autograd.Variable(images_batch.cuda())
                    target_label = autograd.Variable(labels_batch.cuda())
                else:
                    input_image = autograd.Variable(images_batch)
                    target_label = autograd.Variable(labels_batch)
                # Step.2 calculate loss
                # print('******************************************', input_image.shape)
                output = self.model(input_image)
                loss = loss_function(output, target_label)
                batch_count += 1
                total_loss += loss
                # Step.3 Update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pred_prob, pred_label = torch.max(output, dim=1)
                batch_correct =  (pred_label == target_label).sum().data * 1.0 / self.batch_size
                accuracy += batch_correct
                # Check Result
                if i_batch % batch_display == 0:
                    print("Epoch : {}, Batch : {}, Loss : {:.4f}, Batch Accuracy {:.2%}".format(epoch, i_batch, loss, batch_correct))
            """
            Save model
            """
            accuracy = accuracy/batch_count
            accuracy_val = self.evaluate(self.batch_size)
            logger.info('Epoch:[{}/{}]\t loss={:.5f}\t traing acc={:.3%} val acc={:.3%}'.format(epoch , 100, total_loss / batch_count, accuracy,accuracy_val ))
            print("Epoch {} Average Loss : {:.4f} train accuracy: {:.2%} val accuracy: {:.2%}".format(epoch, total_loss / batch_count, accuracy, accuracy_val))

            if epoch % save_freq == 0:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, 'model.pth'))
        logger.info('finish training!')


    def evaluate(self, batch_size=32):
        """
        Detail of evaluate
        """
        with torch.no_grad():
            correct_num = 0
            eval_loader = DataLoader(self.ds_val, batch_size=batch_size, num_workers=8)
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

                pred_prob, pred_label = torch.max(output, dim=1)
                correct_num += (pred_label == target_label).sum().data

            accuracy = correct_num * 1.0 / len(self.ds_val)
            return accuracy
            
            
            
import pandas as pd
df = pd.read_csv('../train.csv')
df

# df_mini = df.sample(n=1230, frac=None, replace=False, weights=None, random_state=None, axis=None)

condition = False
k = 1000000000
while(k > 0 and condition == False):
    print(k)
    df_mini = df.sample(n=1230, frac=None, replace=False, weights=None, random_state=None, axis=None)
    condition = df_mini['individual_id'].value_counts().shape[0] == 1000
    k -= 1
    print(k)

print(df_mini)
print(df_mini.describe())
# df_mini.to_csv("train_1000.csv")


print(df_mini.describe())
df_mini.to_csv("train_1000.csv", index = False)



import logging

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger
    

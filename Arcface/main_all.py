from train import *
import argparse
import os
import time
from logger import get_logger


parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, choices=["resnet50, resnet101, vgg16", "effnet_arcface"],
                    default="effnet_arcface", help="model name")
parser.add_argument("--test", type=bool, default=False,
                    help="test model or not")
parser.add_argument("--num-classes", type=int, default=15587,
                    help="number of classes")
parser.add_argument("--model-path", type=str, default="./model",
                    help="path to save and load model")
parser.add_argument("--loss", type=str, choices=["cross_entropy_loss", "focal_loss"], 
                    default="focal_loss", help="loss function")
parser.add_argument("--num-epoch", type=int, default=30,
                    help="number of epoch")
parser.add_argument("--batch-size", type=int, default=32,
                    help="batch size")
parser.add_argument("--lr", type=float, default=1e-5,
                    help="learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-6,
                    help="learning rate")
parser.add_argument("--s", type=float, default=30.0,
                    help="sphere")
parser.add_argument("--m", type=float, default=0.5,
                    help="margin")
parser.add_argument("--batch-display", type=int, default=10,
                    help="frequency of batch to display result")
parser.add_argument("--save-freq", type=int, default=1,
                    help="frequency to save model")
parser.add_argument("--pretrained", type=bool, default=False,
                    help="Load pretrained model or not")
parser.add_argument("--data_path", type=str, default="./datasets/",
                    help="root path of images")
parser.add_argument("--special_id", type=int, default=0,
                    help="the num of special individual_id")
parser.add_argument("--species", type=int, default=100,
                    help="the num of train species")
parser.add_argument("--csv_name", type=str, default="train.csv",
                    help="the name of csv")
parser.add_argument("--train_percentage", type=float, default=0.7,
                    help="the percentage of train images")


args = parser.parse_args()



if __name__ == '__main__':
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)
    
    if not os.path.exists('./Logs'):
        os.mkdir('./Logs')

    localtime = time.asctime( time.localtime(time.time()) )
    logger = get_logger('./Logs/%s.log' % localtime)
    logger.info(args._get_args)

    train_whale = Train(data_path = args.data_path, s = args.s, m = args.m, csv_name = args.csv_name, test = args.test, number_classes = args.num_classes, species = args.species, special_id = args.special_id, train_percentage=args.train_percentage, path=args.model_path, loadPretrain=args.pretrained)
    
    train_whale.start_train(epoch=args.num_epoch,logger = logger, batch_size=args.batch_size, loss=args.loss,
                          learning_rate=args.lr, weight_decay = args.weight_decay, batch_display=args.batch_display, save_freq=args.save_freq)

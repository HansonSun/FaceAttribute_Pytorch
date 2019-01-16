import sys, os, argparse, time
sys.path.append("model")

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import datasets
import torch.utils.model_zoo as model_zoo
import importlib
from torch.optim.lr_scheduler import MultiStepLR

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
          default=5, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=32, type=int)
    parser.add_argument('--lr', dest='lr', help='Base learning rate.',
          default=0.0001, type=float)
    parser.add_argument('--output_string', dest='output_string', help='String appended to output snapshots.', default = 'test', type=str)
    parser.add_argument('--alpha', dest='alpha', help='Regression loss coefficient.',
          default=0.001, type=float)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
          default='', type=str)

    args = parser.parse_args()
    return args

def get_ignored_params(model):
    # Generator function that yields ignored params.
    b = [model.conv1, model.bn1]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_non_ignored_params(model):
    # Generator function that yields params that will be optimized.
    b = [model.layer1, model.layer2, model.layer3, model.layer4]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_fc_params(model):
    # Generator function that yields fc layer params.
    b = [model.fc_yaw, model.fc_pitch, model.fc_roll]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param

def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = args.gpu_id

    if not os.path.exists('output/snapshots'):
        os.makedirs('output/snapshots')

    model=importlib.import_module("resnet").inference()

    '''
    if args.snapshot == '':
        print("lodel pretrain model")
        load_filtered_state_dict(model, model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth'))
    else:
        saved_state_dict = torch.load(args.snapshot)
        model.load_state_dict(saved_state_dict)
    '''

    transformations = transforms.Compose([transforms.Resize(128),
    transforms.RandomCrop(112), transforms.ToTensor(),
    transforms.Normalize(mean=[0.4, 0.4, 0.4], std=[0.2, 0.2, 0.2])])


    pose_dataset = datasets.Pose_300W_LP(transformations)



    train_loader = torch.utils.data.DataLoader(dataset=pose_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=2)

    model.cuda(gpu)
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    reg_criterion = nn.MSELoss().cuda(gpu)
    # Regression loss coefficient
    alpha = args.alpha

    softmax = nn.Softmax(dim=1).cuda(gpu)
    idx_tensor = [idx for idx in range(67)]
    idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(gpu)

    optimizer = torch.optim.Adam([{'params': get_ignored_params(model), 'lr': 0},
                                  {'params': get_non_ignored_params(model), 'lr': args.lr},
                                  {'params': get_fc_params(model), 'lr': args.lr * 5}],
                                   lr = args.lr)

    lr_scheduler=MultiStepLR(optimizer, [30,80], gamma=0.1, last_epoch=-1)

    print ('Ready to train network.')
    for epoch in range(num_epochs):
        lr_scheduler.step()

        for i, (images, labels, cont_labels, name) in enumerate(train_loader):
            images = Variable(images).cuda(gpu)

            # Binned labels
            label_yaw = Variable(labels[:,0]).cuda(gpu)
            label_pitch = Variable(labels[:,1]).cuda(gpu)
            label_roll = Variable(labels[:,2]).cuda(gpu)

            # Continuous labels
            label_yaw_cont = Variable(cont_labels[:,0]).cuda(gpu)
            label_pitch_cont = Variable(cont_labels[:,1]).cuda(gpu)
            label_roll_cont = Variable(cont_labels[:,2]).cuda(gpu)

            # Forward pass
            yaw, pitch, roll = model(images)

            yaw=yaw.view(-1,yaw.size(1))
            pitch=pitch.view(-1,pitch.size(1))
            roll=roll.view(-1,roll.size(1))


            yaw_acc  = yaw.argmax(dim=1).eq(label_yaw).float().mean()
            pitch_acc= pitch.argmax(dim=1).eq(label_pitch).float().mean()
            roll_acc = roll.argmax(dim=1).eq(label_roll).float().mean()

            # Cross entropy loss
            loss_yaw = criterion(yaw, label_yaw)
            #print (label_pitch)
            loss_pitch = criterion(pitch, label_pitch)
            #print (label_roll)
            loss_roll = criterion(roll, label_roll)

            # MSE loss
            yaw_predicted = softmax(yaw)
            pitch_predicted = softmax(pitch)
            roll_predicted = softmax(roll)

            yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 99
            pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 99
            roll_predicted = torch.sum(roll_predicted * idx_tensor, 1) * 3 - 99

            loss_reg_yaw = reg_criterion(yaw_predicted, label_yaw_cont)
            loss_reg_pitch = reg_criterion(pitch_predicted, label_pitch_cont)
            loss_reg_roll = reg_criterion(roll_predicted, label_roll_cont)

            # Total loss
            loss_yaw += alpha * loss_reg_yaw
            loss_pitch += alpha * loss_reg_pitch
            loss_roll += alpha * loss_reg_roll

            loss_seq = [loss_yaw, loss_pitch, loss_roll]
            grad_seq = [torch.ones(1).cuda(gpu) for _ in range(len(loss_seq))]
            optimizer.zero_grad()
            torch.autograd.backward(loss_seq, grad_seq)
            optimizer.step()
    
            if (i+1) % 100 == 0:
                print ('Epoch [%d/%d], Iter [%d/%d] Losses: Yaw %.4f, Pitch %.4f, Roll %.4f Acc: Yaw %f Pitch %f Roll %f'
                       %(epoch+1, num_epochs, i+1, len(pose_dataset)//batch_size, loss_yaw.data.item(), loss_pitch.data.item(), loss_roll.data.item(),
                        (yaw_acc),(pitch_acc),(roll_acc) ))

        # Save models at numbered epochs.
        if epoch % 1 == 0 and epoch < num_epochs:
            print ('Taking snapshot...')
            torch.save(model.state_dict(),
            'output/snapshots/' + args.output_string + '_epoch_'+ str(epoch+1) + '.pkl')

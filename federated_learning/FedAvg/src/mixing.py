#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from utils import get_weight_dataset,reshape_weights
from models import CNNCifar
import random
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
# class MLP_Mixer(nn.Module):
#     def __init__(self, dim_in,dim_out):
#         super(MLP, self).__init__()
#         self.layer_input = nn.Linear(dim_in, dim_out)
#         self.relu = nn.ReLU()

#         #######LSTM
#         # self.layer_hidden = nn.Linear(dim_hidden, dim_in)
#         # self.sigmoid = nn.Sigmoid(dim=1)

#     def forward(self, x):
#         x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
#         x = self.layer_input(x)
#         # x = self.relu(x)
#         return self.relu(x)

class MLP_Mixer(nn.Module):
    def __init__(self, dim_in,dim_out):
        super(MLP_Mixer, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_out)
        # self.sigmoid = nn.Sigmoid()
        self.sigmoid = nn.ReLU()

        #######LSTM
        # self.layer_hidden = nn.Linear(dim_hidden, dim_in)
        # self.sigmoid = nn.Sigmoid(dim=1)

    def forward(self, x):
        # x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        # print(x.size())
        x = self.layer_input(x)
        # x = self.relu(x)
        return self.sigmoid(x)


def loss_compare(p1,p2):
    loss=0
    for pclient in p1:
        compared_loss = torch.nn.KLDivLoss(reduction='batchmean')(pclient,p2)#-pclient*F.log(p2)
        loss+=compared_loss
    return loss

def loss_distile(q,student_out):
    loss = torch.sum(-q * F.log_softmax(student_out, dim=-1), dim=-1)


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        # for iq, q in enumerate(teacher_out):
        for v in range(len(student_out)):
            if v == iq:
                # we skip cases where student and teacher operate on the same view
                continue
            loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
            total_loss += loss.mean()
            n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)



def mixer(dataset,idxs_users,d_0,args):
    device = 'cuda' 
    concat_dataset=get_weight_dataset(dataset)
    # print('^^^^^^^^^^^^^^^^^^^^^^^^^',concat_dataset.keys(),len([concat_dataset.values()][0]))
    input_size = len(idxs_users)
    # print(input_size)
    mixer_model = MLP_Mixer(dim_in=input_size,dim_out=1)
    server_model = CNNCifar(args=args)

    mixer_model.to(device)
    server_model.to(device)

    optimizer1 = torch.optim.SGD(mixer_model.parameters(), lr=args.lr,
                                        momentum=0.5)
    optimizer2 = torch.optim.SGD(server_model.parameters(), lr=args.lr,
                                        momentum=0.5)
    # criterion=loss_compare
    w={}
    w_agu={}
    train_loss=0
    loss_list=[]
    mixer_model.train()
    server_model.train()
    for epoch in range(1):
        print(f'\n | Mixer Training Round : {epoch+1} |\n')

    for batch_idx, (images, labels) in enumerate(d_0):
        for key in concat_dataset.keys():
            output_parameter= []
            mixer_model.zero_grad()
            server_model.zero_grad()
            inputs=concat_dataset[key]
            
            inputs=inputs.to(device) 
            # print('^^^^^^^^^^^^^^^^^^^^^^^^^',inputs.size())
            for i in range(inputs.size(1)):
                each_input=inputs[:,i].to(device)

                each_input_agu=each_input+ torch.randn(each_input.size()) * 0.1.to(device) + 0.2.to(device)
                ## print(each_input.size())

                outputs = mixer_model(each_input)
                output_parameter.append(outputs)

                outputs_agu = mixer_model(each_input)
                output_parameter_agu.append(outputs_agu)

            output_parameter=torch.cat(output_parameter, dim=0)     
            output_parameter_agu=torch.cat(output_parameter_agu, dim=0) 

            w[key]=output_parameter
            w_agu[key]=output_parameter_agu

        
        
        images=images.to(device)

        server_model.load_state_dict(reshape_weights(w,dataset[0]))
        q=server_model(images)
        server_model.load_state_dict(reshape_weights(w_agu,dataset[0]))
        student_out=server_model(images)
        loss2=loss_distile(q,student_out)
        loss2.backward()
        optimizer2.step() 

        for key in concat_dataset.keys():

            loss1 = loss_compare(concat_dataset[key].to(device) ,w_agu[key].to(device))
        
        print(loss)
        loss1.backward()
        optimizer1.step()     
        train_loss += loss1.item()
        print(train_loss)
    loss_list.append(loss1)
    print(train_loss)

    plt.figure()
    plt.title('Server Model Training Loss')
    plt.plot(range(len(train_loss)), train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Epochs')
    plt.savefig('../save/loss------------@@@@@@@@@@@@@@'+str(epoch)+'.png')
    
    return reshape_weights(w,dataset[0])

def mixer_server(dataset,idxs_users,d_0,args):
    device = 'cuda' 
    concat_dataset=get_weight_dataset(dataset)
    # print('^^^^^^^^^^^^^^^^^^^^^^^^^',concat_dataset.keys(),len([concat_dataset.values()][0]))
    input_size = len(idxs_users)
    # print(input_size)
    mixer_model = MLP_Mixer(dim_in=input_size,dim_out=1)
    server_model = CNNCifar(args=args)

    mixer_model.to(device)
    server_model.to(device)

    optimizer1 = torch.optim.SGD(mixer_model.parameters(), lr=args.lr,
                                        momentum=0.5)
    optimizer2 = torch.optim.SGD(server_model.parameters(), lr=args.lr,
                                        momentum=0.5)
    criterion = nn.CrossEntropyLoss().to(device)
    criterion1 = nn.MSELoss().to(device)
    # criterion=loss_compare
    w={}
    w_agu={}
    train_loss=0
    loss_list=[]
    mixer_model.train()
    server_model.train()
    for epoch in range(1):
        print(f'\n | Mixer Training Round : {epoch+1} |\n')
    
        for key in concat_dataset.keys():
            output_parameter= []
            mixer_model.zero_grad()
            inputs=concat_dataset[key]
            
            inputs=inputs.to(device) 
            # print('^^^^^^^^^^^^^^^^^^^^^^^^^',inputs.size())
            for i in range(inputs.size(1)):
                each_input=inputs[:,i]
                ## print(each_input.size())

                outputs = mixer_model(each_input)
                output_parameter.append(outputs)

            output_parameter=torch.cat(output_parameter, dim=0)     

            w[key]=output_parameter

        server_model.load_state_dict(reshape_weights(w,dataset[0]))

        for batch_idx, (images, labels) in enumerate(d_0):
            print(labels)
            images,labels=images.to(device),labels.to(device)

            output_im=server_model(images)

            loss2=criterion(labels,output_im)
            loss2.backward()
            optimizer2.step() 
        

        for key in concat_dataset.keys():

            loss1 = criterion1(concat_dataset[key].to(device) ,w[key].to(device))

            
            print(loss)
            loss1.backward()
            optimizer1.step()     
            train_loss += loss1.item()
            print(train_loss)
        loss_list.append(loss1)
    print(train_loss)

    plt.figure()
    plt.title('Server Model Training Loss')
    plt.plot(range(len(train_loss)), train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Epochs')
    plt.savefig('../save/loss------------'+str(epoch)+'.png')
    
    return reshape_weights(w,dataset[0])



def mixer_old(dataset,idxs_users,d_0,ind,args):
    device = 'cuda' 
    concat_dataset=get_weight_dataset(dataset)
    # print('^^^^^^^^^^^^^^^^^^^^^^^^^',concat_dataset.keys(),len([concat_dataset.values()][0]))
    input_size = len(idxs_users)
    # print(input_size)
    mixer_model = MLP_Mixer(dim_in=input_size,dim_out=1)

    mixer_model.to(device)

    optimizer = torch.optim.SGD(mixer_model.parameters(), lr=args.lr,
                                        momentum=0.5)
    
    # criterion=loss_compare
    w={}
    train_loss=0
    loss_list=[]
    mixer_model.train()
    for epoch in range(5):
        print(f'\n | Mixer Training Round : {epoch+1} |\n')
        for key in concat_dataset.keys():
            output_parameter= []
            mixer_model.zero_grad()
            inputs=concat_dataset[key]
            
            inputs=inputs.to(device) 
            # print('^^^^^^^^^^^^^^^^^^^^^^^^^',inputs.size())
            for i in range(inputs.size(1)):
                each_input=inputs[:,i]
                ## print(each_input.size())

                outputs = mixer_model(each_input)
                output_parameter.append(outputs)

                outputs_agu = mixer_model(each_input)

            output_parameter=torch.cat(output_parameter, dim=0)     

            w[key]=output_parameter

        loss = loss_compare(inputs ,output_parameter)
        loss.backward()
        optimizer.step()     
        train_loss += loss.item()
        print(train_loss)
        loss_list.append(loss.item())
        print(loss_list)
        plt.figure()
        plt.title('Server Model Training Loss')
        plt.plot(range(len(loss_list)), loss_list, color='r')
        plt.ylabel('Training loss')
        plt.xlabel('Epochs')
        plt.savefig('../save/loss------------'+str(ind)+'.png')
    
    
    return reshape_weights(w,dataset[0]),train_loss

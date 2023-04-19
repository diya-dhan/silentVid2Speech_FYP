# comments added by diya

import torch
from torch import nn
import numpy as np
from models.visual_encoder import Visual_Encoder
from models.audio_encoder import Audio_Encoder
from models.memory import Memory
from models.decoder import Decoder
import os
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torch.nn.parallel
import time


def train_net(args):

    f_optimizer = torch.optim.AdamW(lr=0.0001)

    train_data = DataLoader(
        grid = "C:\Users\diyad\DIYA\FYP\Project\silentVid2Speech_FYP\dataset",
        mode='train'
    )

    v_front = Visual_Encoder()
    a_front = Audio_Encoder()
    mem = Memory()
    dec = Decoder()

    checkpoint = torch.load("C:\Users\diyad\DIYA\FYP\Project\silentVid2Speech_FYP\checkpoint")
    a_front.load_state_dict(checkpoint['a_front_state_dict'])
    v_front.load_state_dict(checkpoint['v_front_state_dict'])
    mem.load_state_dict(checkpoint['mem_state_dict'])
    dec.load_state_dict(checkpoint['dec_state_dict'])


    v_front.cuda()
    a_front.cuda()
    mem.cuda()
    dec.cuda()

    train(v_front, a_front, mem, dec, train_data, 50, optimizer=f_optimizer)

def train(v_front, a_front, mem, dec, train_data, epochs, optimizer):
    criterion = nn.MSELoss().cuda()

    v_front.train()
    a_front.train()
    mem.train()
    dec.train()

    dataloader = DataLoader(
        train_data,
        batch_size=12
    )

    samples = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    step = 0
    for epoch in range(epochs):
        loss_list = []

        prev_time = time.time()
        for i, batch in enumerate(dataloader):
            step += 1
            iter_time = (time.time() - prev_time) / 100
            prev_time = time.time()
            print("******** Training [%d / %d] : %d / %d, Iter Time : %.3f sec, Learning Rate of %s: %f ********" % (
            epoch, epochs, (i + 1) * batch_size, samples, iter_time, optimizer.param_groups[0]['name'], optimizer.param_groups[0]['lr']))
            a_in, v_in, target = batch

            v_feat = v_front(v_in.cuda())  
            a_feat = a_front(a_in.cuda()) 
            te_fusion, tr_fusion, recon_loss, add_loss = mem(v_feat, a_feat, inference=False)

            te_mem_pred = dec(te_fusion, a_feat, infer=False)
            tr_mem_pred = dec(tr_fusion, a_feat, infer=False)

            te_mem_loss = criterion(te_mem_pred, target.cuda())
            tr_mem_loss = criterion(tr_mem_pred, target.cuda())

            tot_loss = 1.0 * (te_mem_loss + tr_mem_loss) + recon_loss + add_loss

            loss_list.append(tot_loss.cpu().item())

            tot_loss.backward()
            
            print('Saving checkpoint: %d' % epoch)
            print('Loss : ', tot_loss)
            
            v_state_dict = v_front.state_dict()
            a_state_dict = a_front.state_dict()
            mem_state_dict = mem.state_dict()
            dec_state_dict = dec.state_dict()


            torch.save({'v_front_state_dict': v_state_dict,
                        'a_front_state_dict': a_state_dict,
                        'mem_state_dict': mem_state_dict,
                        'tcn_state_dict': dec_state_dict},
                       os.path.join("C:\Users\diyad\DIYA\FYP\Project\silentVid2Speech_FYP\checkpoint", 'Epoch_%04d_acc_%.5f.ckpt' % (epoch, logs[1])))

if __name__ == "__main__":
    train_net()


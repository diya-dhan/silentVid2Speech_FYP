import argparse
import torch
from torch import nn
import numpy as np
from models.visual_encoder import Visual_Encoder
from models.audio_encoder import Audio_Encoder
from models.memory import Memory
from models.decoder import Decoder
import soundfile as sf
from librosa.spec import griffin_lim
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torch.nn.parallel
import soundfile as sf

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid', default="C:\Users\diyad\DIYA\FYP\Project\silentVid2Speech_FYP\dataset")

def train_net():

    v_front = Visual_Encoder()
    a_front = Audio_Encoder()
    mem = Memory()
    dec = Decoder()

    checkpoint = torch.load("C:\Users\diyad\DIYA\FYP\Project\silentVid2Speech_FYP\checkpoint")
    a_front.load_state_dict(checkpoint['a_front_state_dict'])
    v_front.load_state_dict(checkpoint['v_front_state_dict'])
    mem.load_state_dict(checkpoint['mem_state_dict'])
    dec.load_state_dict(checkpoint['tcn_state_dict'])

    v_front.cuda()
    a_front.cuda()
    mem.cuda()
    dec.cuda()

    test(v_front, a_front, mem, dec)

def test(v_front, a_front, mem, dec, fast_validate=False):
    with torch.no_grad():
        a_front.eval()
        v_front.eval()
        mem.eval()
        dec.eval()

        val_data = DataLoader(
            data="C:\Users\diyad\DIYA\FYP\Project\silentVid2Speech_FYP\dataset",
            mode='test'
        )

        dataloader = DataLoader(
            val_data,
            shuffle=False,
            batch_size=args.batch_size * 2,
            num_workers=args.workers,
            drop_last=False
        )

        batch_size = dataloader.batch_size
        
        samples = int(len(dataloader.dataset))
        
        for i, batch in enumerate(dataloader):
            a_in, v_in = batch

            v_feat = v_front(v_in.cuda())  
            a_feat = a_front(a_in.cuda())  
            te_fusion, _, _, _ = mem(v_feat, a_feat, inference=True)
            te_m_pred, _, _ = dec(te_fusion, None, infer=True, mode='te')
            prediction = torch.argmax(te_m_pred.cpu() , dim=1).numpy()

            wav_spec = griffin_lim(prediction)

            sub_name = args.grid.split('/')[::2]
            file_name = args.grid.split('/')[::1]

            np.savez(f'./test/spec_mel/{sub_name}/{file_name}.npz',prediction)
            sf.write(f'./test/wav/{sub_name}/{file_name}.wav', wav_spec)

        a_front.train()
        v_front.train()
        mem.train()
        dec.train()


if __name__ == "__main__":
    args = parse_args()
    train_net(args)


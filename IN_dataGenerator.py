from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd.variable import *
import torch.optim as optim
import os
import numpy as np
import pandas as pd
import util
import setGPU
import glob
import sys
import tqdm
import argparse

#sys.path.insert(0, '/nfshome/jduarte/DL4Jets/mpi_learn/mpi_learn/train')
print(torch.__version__)

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
test_path = '/storage/group/gpu/bigdata/BumbleB/convert_20181121_ak8_80x_deepDoubleB_db_pf_cpf_sv_dl4jets_test/'
train_path = '/storage/group/gpu/bigdata/BumbleB/convert_20181121_ak8_80x_deepDoubleB_db_pf_cpf_sv_dl4jets_train_val/'
NBINS = 40 # number of bins for loss function
MMAX = 200. # max value
MMIN = 40. # min value

N = 60 # number of charged particles
N_neu = 100 # number of neutral particles
N_sv = 5 # number of SVs 
n_targets = 2 # number of classes

params_1 = ['pfcand_ptrel',
          'pfcand_erel',
          'pfcand_phirel',
          'pfcand_etarel',
          'pfcand_deltaR',
          'pfcand_puppiw',
          'pfcand_drminsv',
          'pfcand_drsubjet1',
          'pfcand_drsubjet2',
          'pfcand_hcalFrac'
         ]

params_2 = ['track_ptrel',     
          'track_erel',     
          'track_phirel',     
          'track_etarel',     
          'track_deltaR',
          'track_drminsv',     
          'track_drsubjet1',     
          'track_drsubjet2',
          'track_dz',     
          'track_dzsig',     
          'track_dxy',     
          'track_dxysig',     
          'track_normchi2',     
          'track_quality',     
          'track_dptdpt',     
          'track_detadeta',     
          'track_dphidphi',     
          'track_dxydxy',     
          'track_dzdz',     
          'track_dxydz',     
          'track_dphidxy',     
          'track_dlambdadz',     
          'trackBTag_EtaRel',     
          'trackBTag_PtRatio',     
          'trackBTag_PParRatio',     
          'trackBTag_Sip2dVal',     
          'trackBTag_Sip2dSig',     
          'trackBTag_Sip3dVal',     
          'trackBTag_Sip3dSig',     
          'trackBTag_JetDistVal'
         ]

params_3 = ['sv_ptrel',
          'sv_erel',
          'sv_phirel',
          'sv_etarel',
          'sv_deltaR',
          'sv_pt',
          'sv_mass',
          'sv_ntracks',
          'sv_normchi2',
          'sv_dxy',
          'sv_dxysig',
          'sv_d3d',
          'sv_d3dsig',
          'sv_costhetasvpv'
         ]



'''
#Deep double-b features 
params_2 = params_2[22:]
params_3 = params_2[11:13]
'''

def main(args):
    """ Main entry point of the app """
    
    #Convert two sets into two branch with one set in both and one set in only one (Use for this file)
    params_neu = params_1
    params = params_2
    params_sv = params_3
    
    from data import H5Data
    files = glob.glob(train_path + "/newdata_*.h5")
    files_val = files[:5] # take first 5 for validation
    files_train = files[5:] # take rest for training
    
    label = 'new'
    outdir = args.outdir
    vv_branch = args.vv_branch
    sv_branch = args.sv_branch
    os.system('mkdir -p %s'%outdir)

    batch_size = 128
    data_train = H5Data(batch_size = batch_size,
                        cache = None,
                        preloading=0,
                        features_name='training_subgroup', 
                        labels_name='target_subgroup',
                        spectators_name='spectator_subgroup')
    data_train.set_file_names(files_train)
    data_val = H5Data(batch_size = batch_size,
                      cache = None,
                      preloading=0,
                      features_name='training_subgroup', 
                      labels_name='target_subgroup',
                      spectators_name='spectator_subgroup')
    data_val.set_file_names(files_val)

    n_val=data_val.count_data()
    n_train=data_train.count_data()
    print("val data:", n_val)
    print("train data:", n_train)

    # Three implementations of IN with GraphNet being default
    from gnn import GraphNetnoSV
    from gnn import GraphNet
    from gnn import GraphNetAllParticle
    
    if sv_branch: 
        gnn = GraphNet(N, n_targets, len(params), args.hidden, N_sv, len(params_sv),
                   vv_branch=int(vv_branch),
                   De=args.De,
                   Do=args.Do)
    else: 
        gnn = GraphNetnoSV(N, n_targets, len(params), args.hidden,
                       De=args.De,
                       Do=args.Do)
    
    #Architecture with all-particles
    #gnn = GraphNetAllParticle(N, N_neu, n_targets, len(params), len(params_neu), args.hidden, N_sv, len(params_sv),vv_branch=int(vv_branch), De=args.De, Do=args.Do) 
    
    # pre load best model
    #gnn.load_state_dict(torch.load('out/gnn_new_best.pth'))

    n_epochs = 200
    
    loss = nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.Adam(gnn.parameters(), lr = 0.0001)
    loss_vals_training = np.zeros(n_epochs)
    loss_std_training = np.zeros(n_epochs)
    loss_vals_validation = np.zeros(n_epochs)
    loss_std_validation = np.zeros(n_epochs)

    acc_vals_training = np.zeros(n_epochs)
    acc_vals_validation = np.zeros(n_epochs)
    acc_std_training = np.zeros(n_epochs)
    acc_std_validation = np.zeros(n_epochs)

    final_epoch = 0
    l_val_best = 99999
    
    from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
    softmax = torch.nn.Softmax(dim=1)
    import time

    for m in range(n_epochs):
        print("Epoch %s\n" % m)
        #torch.cuda.empty_cache()
        final_epoch = m
        lst = []
        loss_val = []
        loss_training = []
        correct = []
        tic = time.perf_counter()
        for sub_X,sub_Y,sub_Z in tqdm.tqdm(data_train.generate_data(),total=n_train/batch_size):
            training = sub_X[2]
            #training_neu = sub_X[1]
            training_sv = sub_X[3]
            target = sub_Y[0]
            spec = sub_Z[0]
            trainingv = (torch.FloatTensor(training)).cuda()
            #trainingv_neu = (torch.FloatTensor(training_neu)).cuda()
            trainingv_sv = (torch.FloatTensor(training_sv)).cuda()
            targetv = (torch.from_numpy(np.argmax(target, axis = 1)).long()).cuda()
            optimizer.zero_grad()
            #out = gnn(trainingv.cuda(), trainingv_sv.cuda())
            
            #Input training dataset 
            if sv_branch: 
                out = gnn(trainingv.cuda(), trainingv_sv.cuda())
            else: 
                out = gnn(trainingv.cuda())
                
            l = loss(out, targetv.cuda())
            loss_training.append(l.item())
            l.backward()
            optimizer.step()
            loss_string = "Loss: %s" % "{0:.5f}".format(l.item())
            del trainingv, trainingv_sv, targetv
        toc = time.perf_counter()
        print(f"Training done in {toc - tic:0.4f} seconds")
        tic = time.perf_counter()
        for sub_X,sub_Y,sub_Z in tqdm.tqdm(data_val.generate_data(),total=n_val/batch_size):
            training = sub_X[2]
            #training_neu = sub_X[1]
            training_sv = sub_X[3]
            target = sub_Y[0]
            spec = sub_Z[0]
            trainingv = (torch.FloatTensor(training)).cuda()
            #trainingv_neu = (torch.FloatTensor(training_neu)).cuda()
            trainingv_sv = (torch.FloatTensor(training_sv)).cuda()
            targetv = (torch.from_numpy(np.argmax(target, axis = 1)).long()).cuda()
            
            #Input validation dataset 
            if sv_branch: 
                out = gnn(trainingv.cuda(), trainingv_sv.cuda())
            else: 
                out = gnn(trainingv.cuda())
                
            lst.append(softmax(out).cpu().data.numpy())
            l_val = loss(out, targetv.cuda())
            loss_val.append(l_val.item())
            
            targetv_cpu = targetv.cpu().data.numpy()
        
            correct.append(target)
            del trainingv, trainingv_sv, targetv
        toc = time.perf_counter()
        print(f"Evaluation done in {toc - tic:0.4f} seconds")
        l_val = np.mean(np.array(loss_val))
    
        predicted = np.concatenate(lst) #(torch.FloatTensor(np.concatenate(lst))).to(device)
        print('\nValidation Loss: ', l_val)

        l_training = np.mean(np.array(loss_training))
        print('Training Loss: ', l_training)
        val_targetv = np.concatenate(correct) #torch.FloatTensor(np.array(correct)).cuda()
        
        torch.save(gnn.state_dict(), '%s/gnn_%s_last.pth'%(outdir,label))
        if l_val < l_val_best:
            print("new best model")
            l_val_best = l_val
            torch.save(gnn.state_dict(), '%s/gnn_%s_best.pth'%(outdir,label))
            
        print(val_targetv.shape, predicted.shape)
        print(val_targetv, predicted)
        acc_vals_validation[m] = accuracy_score(val_targetv[:,0],predicted[:,0]>0.5)
        print("Validation Accuracy: ", acc_vals_validation[m])
        loss_vals_training[m] = l_training
        loss_vals_validation[m] = l_val
        loss_std_validation[m] = np.std(np.array(loss_val))
        loss_std_training[m] = np.std(np.array(loss_training))
        if m > 8 and all(loss_vals_validation[max(0, m - 8):m] > min(np.append(loss_vals_validation[0:max(0, m - 8)], 200))):
            print('Early Stopping...')
            print(loss_vals_training, '\n', np.diff(loss_vals_training))
            break
        print()

    acc_vals_validation = acc_vals_validation[:(final_epoch + 1)]
    loss_vals_training = loss_vals_training[:(final_epoch + 1)]
    loss_vals_validation = loss_vals_validation[:(final_epoch + 1)]
    loss_std_validation = loss_std_validation[:(final_epoch + 1)]
    loss_std_training = loss_std_training[:(final_epoch)]
    np.save('%s/acc_vals_validation_%s.npy'%(outdir,label),acc_vals_validation)
    np.save('%s/loss_vals_training_%s.npy'%(outdir,label),loss_vals_training)
    np.save('%s/loss_vals_validation_%s.npy'%(outdir,label),loss_vals_validation)
    np.save('%s/loss_std_validation_%s.npy'%(outdir,label),loss_std_validation)
    np.save('%s/loss_std_training_%s.npy'%(outdir,label),loss_std_training)

if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()
    
    # Required positional arguments
    parser.add_argument("outdir", help="Required output directory")
    parser.add_argument("sv_branch", help="Required positional argument")
    parser.add_argument("vv_branch", help="Required positional argument")
    
    # Optional arguments
    parser.add_argument("--De", type=int, action='store', dest='De', default = 5, help="De")
    parser.add_argument("--Do", type=int, action='store', dest='Do', default = 6, help="Do")
    parser.add_argument("--hidden", type=int, action='store', dest='hidden', default = 15, help="hidden")

    args = parser.parse_args()
    main(args)

import os
import shutil
import torch
from collections import OrderedDict
import glob
import numpy as np
import scipy.io as scio

class Saver(object):

    def __init__(self, args):
        self.args = args
        self.directory = os.path.join('./run', args.dataset, args.backbone+'_'+str(args.groups))
        self.runs = glob.glob(os.path.join(self.directory, 'experiment_*'))
        self.run_id = int(len(self.runs)) if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(self.run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, val_vote_acc, is_best, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename, _use_new_zipfile_serialization=False)

        if 'WHUHi' in self.args.dataset:
            val_vote_acc = np.array(val_vote_acc)
            scio.savemat(os.path.join(self.directory, 'experiment_{}'.format(str(self.run_id)), 'val_vote_acc.mat'),{'data':val_vote_acc})

        if is_best:
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))
            # if self.runs:
            #     previous_miou = [0.0]
            #     for run in self.runs:
            #         run_id = run.split('_')[-1]
            #         path = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_pred.txt')
            #         if os.path.exists(path):
            #             with open(path, 'r') as f:
            #                 miou = float(f.readline())
            #                 previous_miou.append(miou)
            #         else:
            #             continue
            #     #max_miou = 0#max(previous_miou)
            #     # if best_pred > max_miou:
            #     #     max_miou = best_pred
            #     shutil.copyfile(filename, os.path.join(self.directory, 'experiment_{}'.format(str(self.run_id)),'model_best.pth.tar'))
            # else:
            shutil.copyfile(filename, os.path.join(self.directory, 'experiment_{}'.format(str(self.run_id)), 'model_best.pth.tar'))
        else:
            shutil.copyfile(filename, os.path.join(self.directory, 'experiment_{}'.format(str(self.run_id)), 'model_last.pth.tar'))

    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        p = OrderedDict()
        p['datset'] = self.args.dataset
        p['backbone'] = self.args.backbone
        p['lr'] = self.args.lr
        p['lr_scheduler'] = self.args.lr_scheduler
        p['epoch'] = self.args.epochs
        p['base_size'] = self.args.base_size
        p['crop_size'] = self.args.crop_size

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()
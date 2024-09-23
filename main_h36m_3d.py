from utils import h36motion3d as datasets
from model import AttModel
from utils.opt import Options
from utils import util
from utils import log
from utils import data_utils
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import time
import csv
import h5py
import torch.optim as optim

class Loader:
    def __init__(self, filename):
        with open(filename, "r") as fp:
            reader = csv.reader(fp)
            self.rawvals = np.array([[float(c) for c in row] for row in reader])
            self.nvals = self.rawvals.reshape([self.rawvals.shape[0], -1, 3])

def calculate_mpjpe(p3d_out, p3d_h36, in_n, out_n):
    """Calculate the Mean Per Joint Position Error (MPJPE)."""
    return torch.mean(torch.norm(p3d_h36[:, in_n:in_n + out_n] - p3d_out, dim=3))

def calculate_mpjpe_sum(p3d_out, p3d_h36, in_n, out_n):
    """Calculate the sum of the Mean Per Joint Position Error (MPJPE)."""
    return torch.sum(torch.mean(torch.norm(p3d_h36[:, in_n:] - p3d_out, dim=3), dim=2), dim=0)

def calculate_mpjae(p3d_out, p3d_h36, in_n, out_n):
    """Calculate the Mean Per Joint Angle Error (MPJAE)."""
    out_n = min(out_n, p3d_out.size(1))

    # Ensure correct slicing
    p3d_out = p3d_out[:, :out_n, :, :]
    p3d_h36 = p3d_h36[:, in_n:in_n + out_n, :, :]

    # Calculate the vector differences
    vec_pred = p3d_out[:, :, 1:, :] - p3d_out[:, :, :-1, :]
    vec_gt = p3d_h36[:, :, 1:, :] - p3d_h36[:, :, :-1, :]

    # Normalize vectors, avoiding division by zero
    vec_pred_norm = vec_pred / (torch.norm(vec_pred, dim=-1, keepdim=True) + 1e-8)
    vec_gt_norm = vec_gt / (torch.norm(vec_gt, dim=-1, keepdim=True) + 1e-8)

    #print(vec_pred_norm.shape)
    #print(vec_gt_norm.shape)
    #exit(0)

    # Calculate dot products
    dot_product = torch.sum(vec_pred_norm * vec_gt_norm, dim=-1)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)

    # Calculate angle differences in degrees
    angles = torch.acos(dot_product) * (180.0 / np.pi)
    angles = torch.nan_to_num(angles, nan=0.0)  # Replace NaNs with zeros if needed

    return torch.mean(angles)

def calculate_mpjae_sum(p3d_out, p3d_h36, in_n, out_n):
    """Calculate the sum of the Mean Per Joint Angle Error (MPJAE)."""
    return torch.sum(calculate_mpjae(p3d_out, p3d_h36, in_n, out_n))


def run_model(net_pred, optimizer=None, is_train=0, data_loader=None, epo=1, opt=None):
    if is_train == 0:
        net_pred.train()
    else:
        net_pred.eval()

    l_p3d = 0
    if is_train <= 1:
        m_p3d_h36 = 0
        m_p3d_h36_angle = 0
    else:
        m_p3d_h36 = np.zeros(opt.output_n)  # Initialize as zeros for the number of output frames
        m_p3d_h36_angle = np.zeros(opt.output_n)

    n = 0
    in_n = opt.input_n
    out_n = opt.output_n
    dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                         26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                         46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                         75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
    seq_in = opt.kernel_size
    joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
    index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
    joint_equal = np.array([13, 19, 22, 13, 27, 30])
    index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

    itera = 1
    idx = np.expand_dims(np.arange(seq_in + out_n), axis=1) + (
            out_n - seq_in + np.expand_dims(np.arange(itera), axis=0))

    # Titles initialization
    titles = np.array(range(out_n)) + 1
    st = time.time()

    for i, (p3d_h36, ano, aug) in enumerate(data_loader):

        batch_size, seq_n, _ = p3d_h36.shape
        if batch_size == 1 and is_train == 0:
            continue
        n += batch_size
        bt = time.time()
        p3d_h36 = p3d_h36.float().cuda()

        p3d_sup = p3d_h36.clone()[:, :, dim_used][:, -out_n - seq_in:].reshape(
            [-1, seq_in + out_n, len(dim_used) // 3, 3])
        p3d_src = p3d_h36.clone()[:, :, dim_used]

        p3d_out_all = net_pred(p3d_src, input_n=in_n, output_n=out_n, itera=itera)
        p3d_out = p3d_h36.clone()[:, in_n:in_n + out_n]
        p3d_out[:, :, dim_used] = p3d_out_all[:, seq_in:, 0]
        p3d_out[:, :, index_to_ignore] = p3d_out[:, :, index_to_equal]

        p3d_out = p3d_out.reshape([-1, out_n, 32, 3])
        p3d_h36 = p3d_h36.reshape([-1, in_n + out_n, 32, 3])

        p3d_out_all = p3d_out_all.reshape([batch_size, seq_in + out_n, itera, len(dim_used) // 3, 3])

        grad_norm = 0
        if is_train == 0:
            loss_p3d = torch.mean(torch.norm(p3d_out_all[:, :, 0] - p3d_sup, dim=3))
            loss_all = loss_p3d
            optimizer.zero_grad()
            loss_all.backward()
            nn.utils.clip_grad_norm_(list(net_pred.parameters()), max_norm=opt.max_norm)
            optimizer.step()
            l_p3d += loss_p3d.cpu().data.numpy() * batch_size

        if is_train <= 1:
            mpjpe_p3d_h36 = calculate_mpjpe(p3d_out, p3d_h36, in_n, out_n)
            mpjae_p3d_h36 = calculate_mpjae(p3d_out, p3d_h36, in_n, out_n)

            m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy() * batch_size
            m_p3d_h36_angle += mpjae_p3d_h36.cpu().data.numpy() * batch_size

        else:
            mpjpe_p3d_h36 = torch.mean(torch.norm(p3d_h36[:, in_n:in_n + out_n] - p3d_out, dim=3), dim=2).mean(0)
            mpjae_p3d_h36 = calculate_mpjae(p3d_out, p3d_h36, in_n, out_n)

            m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy()
            m_p3d_h36_angle += mpjae_p3d_h36.cpu().data.numpy()

        if i % 1000 == 0:
            print('{}/{}|bt {:.3f}s|tt{:.0f}s|gn{}'.format(i + 1, len(data_loader), time.time() - bt,
                                                           time.time() - st, grad_norm))
    ret = {}
    if is_train == 0:
        ret["l_p3d"] = l_p3d / n

    if is_train <= 1:
        ret["m_p3d_h36"] = m_p3d_h36 / n
        ret["m_p3d_h36_angle"] = m_p3d_h36_angle / n
    else:
        m_p3d_h36 = m_p3d_h36 / n
        m_p3d_h36_angle = m_p3d_h36_angle / n
        for j in range(out_n):
            ret["#{:d}".format(titles[j])] = m_p3d_h36[j]
            ret["#{:d}_angle".format(titles[j])] = m_p3d_h36_angle[j]
    return ret



def main(opt):
    lr_now = opt.lr_now
    start_epoch = 1
    print('>>> create models')
    in_features = opt.in_features
    d_model = opt.d_model
    kernel_size = opt.kernel_size

    if opt.model_fold:
        net_pred = AttModel.AttModel_fold(in_features=in_features, kernel_size=kernel_size, d_model=d_model,
                                          num_stage=opt.num_stage, dct_n=opt.dct_n)
    else:
        net_pred = AttModel.AttModel(in_features=in_features, kernel_size=kernel_size, d_model=d_model,
                                     num_stage=opt.num_stage, dct_n=opt.dct_n)

    net_pred.cuda()

    optimizer = optim.Adam(filter(lambda x: x.requires_grad, net_pred.parameters()), lr=opt.lr_now)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in net_pred.parameters()) / 1000000.0))

    if opt.is_load or opt.is_eval:
        model_path_len = './{}/ckpt_best.pth.tar'.format(opt.ckpt)
        print(">>> loading ckpt len from '{}'".format(model_path_len))
        ckpt = torch.load(model_path_len)
        start_epoch = ckpt['epoch'] + 1
        err_best = ckpt['err']
        err_best_angle = ckpt.get('err_angle', float('inf'))  # Load best angle error if available
        lr_now = ckpt['lr']
        net_pred.load_state_dict(ckpt['state_dict'])
        print(">>> ckpt len loaded (epoch: {} | err: {} | err_angle: {})".format(
            ckpt['epoch'], ckpt['err'], ckpt.get('err_angle', 'N/A')))

    print('>>> loading datasets')

    if not opt.is_eval:
        dataset = datasets.Datasets(opt, split=0)
        print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
        data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=False)
        valid_dataset = datasets.Datasets(opt, split=1)
        print('>>> Validation dataset length: {:d}'.format(valid_dataset.__len__()))
        valid_loader = DataLoader(valid_dataset, batch_size=opt.test_batch_size, shuffle=True, num_workers=0,
                                  pin_memory=False)

    test_dataset = datasets.Datasets(opt, split=2)
    print('>>> Testing dataset length: {:d}'.format(test_dataset.__len__()))
    test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0,
                             pin_memory=False)

    if opt.is_eval:
        ret_test = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt)
        ret_log = np.array([])
        head = np.array([])
        for k in ret_test.keys():
            ret_log = np.append(ret_log, [ret_test[k]])
            head = np.append(head, [k])
        log.save_csv_log(opt, head, ret_log, is_create=True, file_name='test_walking')

    if opt.is_visualize:
        single = single_run_model(net_pred, opt)
        return single

    if opt.is_visualize_fold:
        single_fold = single_run_model_fold(net_pred, opt)
        return single_fold

    if opt.is_mpjpe:
        mpjpe = mpjpe_dump(opt)
        return

    if opt.model_fold:
        err_best = 1000
        err_best_angle = float('inf')  # Initialize the best angle error
        for epo in range(start_epoch, opt.epoch + 1):
            is_best = False
            is_best_angle = False
            lr_now = util.lr_decay_mine(optimizer, lr_now, 0.2 ** (1 / opt.epoch))
            print('>>> training epoch: {:d}'.format(epo))
            ret_train = run_model_fold(net_pred, optimizer, is_train=0, data_loader=data_loader, epo=epo, opt=opt,
                                       offset=10)
            print('train error: {:.3f}'.format(ret_train['m_p3d_h36']))
            print('train angle error: {:.3f}'.format(ret_train['m_p3d_h36_angle']))
            ret_valid = run_model_fold(net_pred, is_train=1, data_loader=valid_loader, opt=opt, epo=epo, offset=10)
            print('validation error: {:.3f}'.format(ret_valid['m_p3d_h36']))
            print('validation angle error: {:.3f}'.format(ret_valid['m_p3d_h36_angle']))
            ret_test = run_model_fold(net_pred, is_train=3, data_loader=test_loader, opt=opt, epo=epo, offset=10)
            print('testing error: {:.3f}'.format(ret_test['#1']))
            print('testing angle error: {:.3f}'.format(ret_test['#1_angle']))
            ret_log = np.array([epo, lr_now])
            head = np.array(['epoch', 'lr'])
            for k in ret_train.keys():
                ret_log = np.append(ret_log, [ret_train[k]])
                head = np.append(head, [k])
            for k in ret_valid.keys():
                ret_log = np.append(ret_log, [ret_valid[k]])
                head = np.append(head, ['valid_' + k])
            for k in ret_test.keys():
                ret_log = np.append(ret_log, [ret_test[k]])
                head = np.append(head, ['test_' + k])
            log.save_csv_log(opt, head, ret_log, is_create=(epo == 1))

            if ret_valid['m_p3d_h36'] < err_best:
                err_best = ret_valid['m_p3d_h36']
                is_best = True

            if ret_valid['m_p3d_h36_angle'] < err_best_angle:
                err_best_angle = ret_valid['m_p3d_h36_angle']
                is_best_angle = True

            log.save_ckpt({'epoch': epo,
                           'lr': lr_now,
                           'err': ret_valid['m_p3d_h36'],
                           'err_angle': ret_valid['m_p3d_h36_angle'],
                           'state_dict': net_pred.state_dict(),
                           'optimizer': optimizer.state_dict()},
                          is_best=is_best or is_best_angle, opt=opt)

    if not opt.is_eval:
        err_best = 1000
        err_best_angle = float('inf')  # Initialize the best angle error
        for epo in range(start_epoch, opt.epoch + 1):
            is_best = False
            is_best_angle = False
            lr_now = util.lr_decay_mine(optimizer, lr_now, 0.2 ** (1 / opt.epoch))
            print('>>> training epoch: {:d}'.format(epo))
            ret_train = run_model(net_pred, optimizer, is_train=0, data_loader=data_loader, epo=epo, opt=opt)
            print('train error: {:.3f}'.format(ret_train['m_p3d_h36']))
            print('train angle error: {:.3f}'.format(ret_train['m_p3d_h36_angle']))
            ret_valid = run_model(net_pred, is_train=1, data_loader=valid_loader, opt=opt, epo=epo)
            print('validation error: {:.3f}'.format(ret_valid['m_p3d_h36']))
            print('validation angle error: {:.3f}'.format(ret_valid['m_p3d_h36_angle']))
            ret_test = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt, epo=epo)
            print('testing error: {:.3f}'.format(ret_test['#1']))
            print('testing angle error: {:.3f}'.format(ret_test['#1_angle']))
            ret_log = np.array([epo, lr_now])
            head = np.array(['epoch', 'lr'])
            for k in ret_train.keys():
                ret_log = np.append(ret_log, [ret_train[k]])
                head = np.append(head, [k])
            for k in ret_valid.keys():
                ret_log = np.append(ret_log, [ret_valid[k]])
                head = np.append(head, ['valid_' + k])
            for k in ret_test.keys():
                ret_log = np.append(ret_log, [ret_test[k]])
                head = np.append(head, ['test_' + k])
            log.save_csv_log(opt, head, ret_log, is_create=(epo == 1))

            if ret_valid['m_p3d_h36'] < err_best:
                err_best = ret_valid['m_p3d_h36']
                is_best = True

            if ret_valid['m_p3d_h36_angle'] < err_best_angle:
                err_best_angle = ret_valid['m_p3d_h36_angle']
                is_best_angle = True

            log.save_ckpt({'epoch': epo,
                           'lr': lr_now,
                           'err': ret_valid['m_p3d_h36'],
                           'err_angle': ret_valid['m_p3d_h36_angle'],
                           'state_dict': net_pred.state_dict(),
                           'optimizer': optimizer.state_dict()},
                          is_best=is_best or is_best_angle, opt=opt)

if __name__ == '__main__':
    option = Options().parse()
    main(option)

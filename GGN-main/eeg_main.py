# -*- coding: utf-8 -*

import random
import collections
import time
from os import walk
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import f1_score
import numpy as np
from sklearn.metrics import confusion_matrix

from torch.utils.tensorboard import SummaryWriter
from torch import optim
from torch import nn

from models.ggn import ST_HGGN,GGN,EnhancedGGN
from eeg_util import *
import eeg_util
from models.baseline_models import *


import networkx as nx
import json


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

seizure_type_data = collections.namedtuple('seizure_type_data', ['patient_id','seizure_type', 'data'])


# NOTE: FIXED, Order cannot be changed!!!!!!!!!
# s_types = {'FNSZ': 1836,'GNSZ': 583, 'CPSZ': 367,'ABSZ': 99,  'TNSZ': 62, 'SPSZ': 52, 'TCSZ': 48}
s_types = {'GNSZ': 1448, 'CPSZ': 20, 'FNSZ': 310, 'ABSZ': 2177, 'TNSZ': 388, 'BCKG': 3680, 'TCSZ': 37}
label_dict = {}
number_label_dict = {}
for i, k in enumerate(s_types.keys()):
    label_dict[k] = i
    number_label_dict[i] = k
print('labels:', label_dict)

def load_tuh_data(args, feature_name=""):

    feature = np.load(os.path.join(args.data_path, f"seizure_x{feature_name}_from_begin.npy"))
    label = np.load(os.path.join(args.data_path, f"seizure_y{feature_name}_from_begin.npy"))
    print('load seizure data, shape:', feature.shape, label.shape)


    if args.testing:
        print('loading shuffled index!!!!!!')
        shuffled_index = np.load('shuffled_index.npy')
    else:
        # shuffle:
        shuffled_index = np.random.permutation(np.arange(feature.shape[0]))
    print('shuffled_index:', shuffled_index)
    
    
    feature = feature[shuffled_index]
    label = label[shuffled_index]


    # train, test:

    label_dict = {}
    for i, l in enumerate(label):
        if l not in label_dict:
            label_dict[l] = []
        label_dict[l].append(i)


    # Filter the MYSZ:

    # take 1/3 as test set for each seizure type.
    train_x, train_y, test_x, test_y = [],[],[],[]
    for k, v in label_dict.items():
        test_size = int(len(v)/3)
        train_x.append(feature[v[test_size:]])
        train_y.append(label[v[test_size:]])
        test_x.append(feature[v[:test_size]])
        test_y.append(label[v[:test_size]])
    train_x = np.concatenate(train_x)
    train_y = np.concatenate(train_y)
    test_x = np.concatenate(test_x)
    test_y = np.concatenate(test_y)
    print('before trans:', train_x.shape,train_y.shape,test_x.shape,test_y.shape)
    # reshape to B, C, N, T:
    B, T, N, C = train_x.shape
    train_x =  train_x.transpose(0, 3, 2, 1)
    test_x = test_x.transpose(0, 3, 2, 1)

    print('after trans:', train_x.shape,train_y.shape,test_x.shape,test_y.shape)

    # load to dataloader:
    return [train_x, test_x], [train_y, test_y]


def generate_tuh_data(args, file_name=""):
    """ generate data for training or ploting functional connectivity.
    """
    data_path = args.data_path

    freqs = [12]
    x_data = []
    y_data = []

    types_dict = {}
    for freq in freqs:
        x_f_data = []
        y_f_data = []
        min_len = 10000
        freq_file_name = f"fft_seizures_wl1_ws_0.25_sf_250_fft_min_1_fft_max_{freq}"
        dir, _, files = next(walk(os.path.join(data_path, freq_file_name)))
        for i, name in enumerate(files):
            fft_data = pickle.load(open(os.path.join(dir,name), 'rb'))
            if fft_data.seizure_type == 'MYSZ':
                continue
            if fft_data.data.shape[0] < 34:
                continue
            if fft_data.data.shape[0] < min_len:
                min_len = fft_data.data.shape[0]
                
            x_f_data.append(fft_data.data)
            y_f_data.append(label_dict[fft_data.seizure_type])
        print('min len:', min_len)
        x_f_data = [d[:min_len,...] for d in x_f_data]
        x_f_data = np.stack(x_f_data, axis=0)
        print(x_f_data.shape)
        y_f_data = np.stack(y_f_data, axis=0)
        print(y_f_data.shape)
        x_data.append(x_f_data)
        y_data.append(y_f_data)

    # check each y_f_data:
    print('prepare save!')
    x_data = np.concatenate(x_data, axis=3)
    print('x data shape:', x_data.shape)
    np.save(f'seizure_x_{file_name}.npy', x_data)
    np.save(f'seizure_y_{file_name}.npy', y_data[0])
    print('y data shape:', y_data[0].shape)
    print('save done!')

def normalize_seizure_features(features):
    """inplace-norm
    Args:
        features (list of tensors): train,test,val
    """
    for i in range(len(features)):
        # (B, F, N, T)
        for j in range(features[i].shape[-1]):
            features[i][..., j] = normalize(features[i][..., j])
    
def generate_dataloader_seizure(features, labels, args):
    """
     features: [train, test, val], if val is empty then val == test
     train: B, T, N, F(12,24,48,64,96)
    """
    cates = ['train', 'test', 'val']
    datasets = dict()
    # normalize over feature dimension

    for i in range(len(features)):
        datasets[cates[i] + '_loader'] = SeqDataLoader(features[i], labels[i], args.batch_size, cuda=args.cuda)

    if len(features) < 3: # take test as validation.
        datasets['val_loader'] = SeqDataLoader(features[-1], labels[-1], args.batch_size, cuda=args.cuda)

    return datasets


def init_adjs(args, index=0):
    adjs = []
   
    if args.adj_type == 'rand10':
        adj_mx = eeg_util.generate_rand_adj(0.1*(index+1), N=20)
    elif args.adj_type == 'er':
        adj_mx = nx.to_numpy_array(nx.erdos_renyi_graph(20, 0.1*(index+1)))
    else:
        adj_mx = load_eeg_adj(args.adj_file, args.adj_type)
    adjs.append(adj_mx)

    # #     model = EEGEncoder(adj_mx, args, is_gpu=args.cuda)
    # adj = torch.from_numpy(adjs[0]).float().cuda()
    # adjs[0] = adj
    # return adjs

    # 根据 args.cuda 决定设备
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    adj = torch.from_numpy(adjs[0]).float().to(device)
    adjs[0] = adj
    return adjs


def chose_model(args, adjs):
    if args.task.upper() == 'GGN':
        adj = adjs[0]
        model = GGN(adj, args)
    elif args.task == 'EnhancedGGN':
        adj = adjs[0]
        model = EnhancedGGN(adj, args)
    elif args.task == 'ST-HGGN':
        adj = adjs[0]
        model = ST_HGGN(adj, args)
    elif args.task == 'transformer':
        DEVICE = torch.device("cuda:0" if args.cuda else "cpu")  
        print(f'use device: {DEVICE}')
        models = 512
        hiddens = 1024
        q = 8
        v = 8
        h = 8
        N = 8
        dropout = 0.2
        pe = True  # # 设置的是双塔中 score=pe score=channel默认没有pe
        mask = True  # 设置的是双塔中 score=input的mask score=channel默认没有mask

        # inputs = 34
        inputs = 139
        channels = 20
        outputs = args.predict_class_num  # 分类类别
        hz = args.feature_len
        model = Transformer(d_model=models, d_input=inputs, d_channel=channels, d_hz = hz, d_output=outputs, d_hidden=hiddens,
                        q=q, v=v, h=h, N=N, dropout=dropout, pe=pe, mask=mask, device=DEVICE)
    elif args.task == 'gnnnet':
        model = DCRNNModel_classification(
        args, adjs, adjs[0].shape[0], args.predict_class_num, args.feature_len, device='cuda')
    elif args.task == 'cnnnet':
        model = CNNNet(args)
    else:
        model = None
        print('No model found!!!!')
    return model



def init_trainer(model, args):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    def lr_adjust(epoch):
        if epoch < 20:
            return 1
        
        return args.lr_decay_rate ** ((epoch - 19) / 3 + 1)
    
    lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_adjust)

    c={0: 1448, 1: 20, 2: 310, 3: 2177, 4: 388, 5: 3680, 6: 37}
    w = np.array([c[i] for i in range(7)])
    m = np.median(w)
    total = np.sum(w)
    weights = None
    if args.weighted_ce == 'prop':
        weights =1 - w/total
    elif args.weighted_ce == 'rand':
        weights = np.random.rand(7)*10
    elif args.weighted_ce == 'median':
        weights = m/w
    if weights is not None:
        # weights =  torch.from_numpy(weights).float().cuda()

        device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
        weights = torch.from_numpy(weights).float().to(device)
    print('weights:', weights)

    if args.focalloss:
        crite = FocalLoss(nn.CrossEntropyLoss(weight=weights, reduce=False), alpha=0.9, gamma=args.focal_gamma)
    else:
        crite = nn.CrossEntropyLoss(weight=weights)
        
    trainer = Trainer(args, model, optimizer, criterion=crite, sched=lr_sched)
    return trainer

def train_eeg(args, datasets, index=0):
    # SummaryWriter

    import os
    dt = time.strftime("%m_%d_%H_%M", time.localtime())
    log_dir = "./tfboard/"+args.server_tag+"/" + dt
    print('tensorboard path:', log_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = SummaryWriter(log_dir)
    
    adjs = init_adjs(args, index)

    model = chose_model(args, adjs)

    print('args_cuda:', args.cuda)
    if args.cuda:
        print('rnn_train RNNBlock to cuda!')
        model.cuda()
    else:
        print('rnn_train RNNBlock to cpu!')

    # add scheduler.
    trainer = init_trainer(model, args)
    
    best_val_acc = 0
    best_unchanged_threshold = 100  # accumulated epochs of best val_mae unchanged
    best_count = 0
    best_index = -1
    train_val_metrics = []
    start_time = time.time()
    basedir, file_tag = os.path.split(args.best_model_save_path)
    model_save_path = os.path.join(basedir, f'{index}_{file_tag}')
    
    for e in range(args.epochs):
        datasets['train_loader'].shuffle()
        train_loss, train_preds = [], []

        for i, (input_data, target) in enumerate(datasets['train_loader'].get_iterator()):
            loss, preds = trainer.train(input_data, target)
            # training metrics
            train_loss.append(loss)
            train_preds.append(preds)
        # validation metrics
        val_loss, val_preds = [], []
    
        for j, (input_data, target) in enumerate(datasets['val_loader'].get_iterator()):
            loss, preds  = trainer.eval(input_data, target)
            # add metrics
            val_loss.append(loss)
            val_preds.append(preds)

        # cal metrics as a whole:
        # reshape:
        train_preds = torch.cat(train_preds, dim=0)
        val_preds = torch.cat(val_preds, dim=0)
        
        train_acc = eeg_util.calc_eeg_accuracy(train_preds, datasets['train_loader'].ys)
        val_acc = eeg_util.calc_eeg_accuracy(val_preds, datasets['val_loader'].ys)

        m = dict(train_loss=np.mean(train_loss), train_acc=train_acc,
                 val_loss=np.mean(val_loss), val_acc=val_acc)

        m = pd.Series(m)

        if e % 20 == 0:
            print('epoch:', e)
            print(m)
        # write to tensorboard:
        writer.add_scalars(f'epoch/loss', {'train': m['train_loss'], 'val': m['val_loss']}, e)
        writer.add_scalars(f'epoch/acc', {'train': m['train_acc'], 'val': m['val_acc']}, e)

        train_val_metrics.append(m)
        if m['val_acc'] > best_val_acc:
            best_val_acc = m['val_acc']
            best_count = 0
            print("update best model, epoch: ", e)

            # 确保保存目录存在
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

            torch.save(trainer.model.state_dict(), model_save_path)
            print(m)
            best_index = e
        else:
            best_count += 1
        if best_count > best_unchanged_threshold:
            print('Got best')
            break

        trainer.lr_schedule()
    print('training: :')
    if args.lgg:
        print('after training adj_fix', trainer.model.LGG.adj_fix[0])
    print('best_epoch:', best_index)

    # test_model = chose_model(args, adjs)
    # test_model.load_state_dict(torch.load(model_save_path))
    # test_model.cuda()
    # trainer.model = test_model
    test_model = chose_model(args, adjs)
    # test_model.load_state_dict(torch.load(model_save_path))
    test_model.load_state_dict(torch.load(model_save_path), strict=False)   #这里是为了忽略不匹配的键
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    test_model.to(device)


    trainer.model = test_model
    if args.lgg:
        print('after load best model adj_fix', trainer.model.LGG.adj_fix[0])
    
    test_metrics = []
    test_loss, test_preds = [], []


    for i, (input_data, target) in enumerate(datasets['test_loader'].get_iterator()):
        loss, preds = trainer.eval(input_data, target)
        # add metrics
        test_loss.append(loss)
        test_preds.append(preds)
    # cal metrics as a whole:

    # reshape:
    test_preds = torch.cat(test_preds, dim=0)
    test_preds = torch.softmax(test_preds, dim=1)

    test_acc = eeg_util.calc_eeg_accuracy(test_preds, datasets['test_loader'].ys)

    m = dict(test_acc=test_acc, test_loss=np.mean(test_loss))
    m = pd.Series(m)
    print("test:")
    print(m)
    test_metrics.append(m)
    preds_b = test_preds.argmax(dim=1)
    
    basedir, file_tag = os.path.split(args.fig_filename)
    date_dir = time.strftime('%Y%m%d', time.localtime(time.time()))
    fig_save_dir = os.path.join(basedir, date_dir)
    if not os.path.exists(fig_save_dir):
        os.makedirs(fig_save_dir)
    confused_fig_dir = os.path.join(fig_save_dir, f'{file_tag}_{index}_confusion.png')
    loss_fig_dir = os.path.join(basedir, date_dir, f'{file_tag}_{index}_loss.png')
    
    plot_confused_cal_f1(preds_b, datasets['test_loader'].ys, fig_dir=confused_fig_dir)
    plot(train_val_metrics, test_metrics, loss_fig_dir)
    #####################################################
    # 在 plot_confused_cal_f1(...) 之后追加
    y_true = datasets['test_loader'].ys.cpu().numpy()
    proba = test_preds.cpu().numpy()

    rel = plot_top1_top2_reliability(
        y_true, proba, fig_save_dir, n_bins=10, binning='quantile',
        prefix=f'{file_tag}_{index}_'  # 让输出图名带上模型和轮次
    )
    print("[Reliability] ECE (Top-1 / Top-2):", rel['ece_top1'], rel['ece_top2'])
    #####################################################

    print('finish rnn_train!, time cost:', time.time() - start_time)
    return train_val_metrics, test_metrics

############################################
# Standalone t-SNE visualization (works even in testing mode)
############################################
def tsne_visualization(args, datasets, model, fig_save_dir):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    def plot_tsne(x, y, title, save_path):
        tsne = TSNE(
            n_components=2,
            learning_rate='auto',
            init='pca',
            perplexity=30
        )
        x_2d = tsne.fit_transform(x)

        plt.figure(figsize=(6, 5))
        scatter = plt.scatter(x_2d[:,0], x_2d[:,1], c=y, cmap="tab10", s=10)
        plt.legend(*scatter.legend_elements(), title="Class", fontsize=8, loc='lower left')
        # plt.title(title)

        plt.xlabel(title, fontsize=12, fontweight='normal',
                   fontfamily='Times New Roman', ha='center')

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"[t-SNE] saved: {save_path}")

    ############################################
    # 1. Raw features (initial state)
    ############################################
    print("[t-SNE] Generating RAW feature embedding...")

    raw_feats = []
    raw_labels = []

    for batch_x, batch_y in datasets["test_loader"].get_iterator():
        batch_x_np = batch_x.cpu().numpy()   # (B, C, N, T)
        B, C, N, T = batch_x_np.shape
        raw_feats.append(batch_x_np.reshape(B, -1))   # flatten
        raw_labels.append(batch_y.cpu().numpy())

    raw_feats = np.concatenate(raw_feats, axis=0)
    raw_labels = np.concatenate(raw_labels, axis=0)

    plot_tsne(
        raw_feats,
        raw_labels,
        "(a)",
        os.path.join(fig_save_dir, "tsne_raw.png")
    )

    ############################################
    # 2. Trained embedding from model (trained state)
    ############################################
    print("[t-SNE] Generating TRAINED embedding...")

    emb_list = []
    lab_list = []

    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        for batch_x, batch_y in datasets["test_loader"].get_iterator():
            batch_x = batch_x.to(device)
            encoded = model.encode(batch_x)      # (B, C', N, T')
            emb = encoded.mean(dim=[2,3])        # global avg → (B, C')
            emb_list.append(emb.cpu().numpy())
            lab_list.append(batch_y.cpu().numpy())

    emb_feats = np.concatenate(emb_list, axis=0)
    emb_labels = np.concatenate(lab_list, axis=0)

    plot_tsne(
        emb_feats,
        emb_labels,
        "(b)",
        os.path.join(fig_save_dir, "tsne_trained.png")
    )

    print("[t-SNE] All done.")


def cal_f1(preds, labels):

    mi_f1 = f1_score(labels, preds, average='micro')
    ma_f1 = f1_score(labels, preds, average='macro')
    weighted_f1 = f1_score(labels, preds, average='weighted')


    return mi_f1, ma_f1, weighted_f1

def plot_confused_cal_f1(preds, labels, fig_dir):
    preds = preds.cpu()
    labels = labels.cpu()
    
    ori_preds = preds
    sns.set()
    fig = plt.figure(figsize=(5, 4), dpi=100)
    ax = fig.gca()
    gts = [number_label_dict[int(l)][:-2] for l in labels]
    preds = [number_label_dict[int(l)][:-2] for l in preds]
    
    label_names = [v[:-2] for v in number_label_dict.values()]
    print(label_names)
    C2= np.around(confusion_matrix(gts, preds, labels=label_names, normalize='true'), decimals=2)

    # from confusion to ACC, micro-F1, macro-F1, weighted-f1.
    print('Confusion:', C2)
    mi_f1, ma_f1, w_f1 = cal_f1(ori_preds, labels)
    print(f'micro f1: {mi_f1}, macro f1: {ma_f1}, weighted f1: {w_f1}')

    sns.heatmap(C2, cbar=True, annot=True, ax=ax, cmap="YlGnBu", square=True,annot_kws={"size":9},
        yticklabels=label_names,xticklabels=label_names)

    ax.figure.savefig(fig_dir, transparent=False, bbox_inches='tight')


def plot(train_val_metrics, test_metrics, fig_filename='mae'):
    epochs = len(train_val_metrics)
    x = range(epochs)
    train_loss = [m['train_loss'] for m in train_val_metrics]
    val_loss = [m['val_loss'] for m in train_val_metrics]

    plt.figure(figsize=(8, 6))
    plt.plot(x, train_loss, '', label='train_loss')
    plt.plot(x, val_loss, '', label='val_loss')
    plt.title('loss')
    plt.legend(loc='upper right')  # 设置label标记的显示位置
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.tight_layout()
    plt.savefig(fig_filename)
    

def multi_train(args, tags="", runs=10):
    '''
    train multiple times, analyze the results, get the mean and variance.
    '''

    test_loss = []
    test_acc = []
    xs, ys = load_tuh_data(args)
    normalize_seizure_features(xs)
    datasets = generate_dataloader_seizure(xs,ys,args)
    
    for i in range(runs):
        
        tr, te = train_eeg(args, datasets, i)
        test_loss.append(te[0]['test_loss'])
        test_acc.append(te[0]['test_acc'])

    # Analysis:
    test_loss_m = np.mean(test_loss)
    test_loss_v = np.std(test_loss)

    test_acc_m = np.mean(test_acc)
    test_acc_v = np.std(test_acc)

    print('%s,trials: %s, t loss mean/std: %f/%f, t acc mean/std: %f%s/%f \n' % (
        tags, runs, test_loss_m, test_loss_v, test_acc_m, '%', test_acc_v))


#######################################################################################
# 依赖：sklearn & matplotlib & numpy & seaborn
from sklearn.metrics import (
    precision_recall_fscore_support, roc_auc_score, average_precision_score,
    cohen_kappa_score, matthews_corrcoef, balanced_accuracy_score,
    RocCurveDisplay, PrecisionRecallDisplay, confusion_matrix
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
import time

from reliability_utils import plot_top1_top2_reliability

def compute_all_metrics(y_true, proba, label_names):
    """
    y_true: 1D tensor/ndarray of ints
    proba: (N, C) ndarray/tensor, softmax 后概率
    """
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba)
    y_pred = proba.argmax(axis=1)

    # 基础与稳健指标
    acc = (y_pred == y_true).mean()
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    # F1/Prec/Rec（宏/加权 + 按类）
    prec_m, rec_m, f1_m, sup = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    prec_w, rec_w, f1_w, _   = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    prec_c, rec_c, f1_c, sup_c = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

    # top-2 acc（可选）
    top2 = (np.take_along_axis(proba, np.argsort(-proba, axis=1)[:, :2], axis=1).sum(axis=1))
    top2_pred_hit = []
    top2_idx = np.argsort(-proba, axis=1)[:, :2]
    for i, t in enumerate(y_true):
        top2_pred_hit.append(int(t in top2_idx[i]))
    top2_acc = np.mean(top2_pred_hit)

    # ROC/PR（OvR）
    C = proba.shape[1]
    y_bin = label_binarize(y_true, classes=list(range(C)))
    try:
        roc_macro = roc_auc_score(y_bin, proba, average='macro', multi_class='ovr')
        pr_macro  = average_precision_score(y_bin, proba, average='macro')
    except Exception:
        roc_macro, pr_macro = np.nan, np.nan

    metrics = {
        "acc": acc, "balanced_acc": bal_acc, "kappa": kappa, "mcc": mcc,
        "macro_f1": f1_m, "macro_prec": prec_m, "macro_rec": rec_m,
        "weighted_f1": f1_w, "weighted_prec": prec_w, "weighted_rec": rec_w,
        "top2_acc": top2_acc, "roc_auc_macro": roc_macro, "pr_auc_macro": pr_macro,
        "per_class": [
            {"label": label_names[i], "precision": float(prec_c[i]), "recall": float(rec_c[i]),
             "f1": float(f1_c[i]), "support": int(sup_c[i])} for i in range(C)
        ]
    }
    return metrics

def plot_compare_bars(save_dir, metrics_dict, title="Model Comparison (macro)"):
    """
    metrics_dict: {"GGN": {...}, "ST-HGGN": {...}, "NeuroDynamicNet": {...}}
    画 ACC / Macro-F1 / BalancedAcc / Kappa / MCC / ROC-AUC(macro) / PR-AUC(macro)
    """
    os.makedirs(save_dir, exist_ok=True)
    keys = ["acc","macro_f1","balanced_acc","kappa","mcc","roc_auc_macro","pr_auc_macro","top2_acc"]
    disp = ["ACC","Macro-F1","BalancedAcc","Kappa","MCC","ROC-AUC(m)","PR-AUC(m)","Top-2"]
    # models = list(metrics_dict.keys())
    # models = ['GGN', 'HGGN', 'ST-HGGN', 'NeuroDynamicNet']  # 自定义顺序
    models = ['cnnnet', 'gnnnet', 'transformer', 'NeuroDynamicNet']  # 自定义顺序
    vals = np.array([[metrics_dict[m].get(k, np.nan) for k in keys] for m in models])

    plt.figure(figsize=(10,5.5))
    x = np.arange(len(keys))
    w = 0.8/len(models)
    colors = ['#dff1f8', '#334a52', '#7db3c6', '#fbe8ff']  # Customize bar colors
    # for i,m in enumerate(models):
    #     plt.bar(x + i*w, vals[i], width=w, label=m, color=colors[i])
    for i, m in enumerate(models):
        bars = plt.bar(x + i * w, vals[i], width=w, label=m, color=colors[i])

        # Add numbers on top of the bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2),
                     ha='center', va='bottom', fontsize=8, fontstyle='italic', rotation=45)  # Add number above each bar
    plt.xticks(x + w*(len(models)-1)/2, disp, rotation=15)
    plt.ylabel("score")
    # plt.title(title)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "compare_macro_bars.png"))

def plot_per_class_f1(save_dir, metrics_dict):
    # order = ['GGN', 'HGGN', 'ST-HGGN', 'NeuroDynamicNet']  # 自定义顺序
    order = ['cnnnet', 'gnnnet', 'transformer', 'NeuroDynamicNet']
    metrics_dict = {key: metrics_dict[key] for key in order}

    os.makedirs(save_dir, exist_ok=True)
    # 用第一个模型的类名作为顺序
    first = next(iter(metrics_dict.values()))
    labels = [pc["label"] for pc in first["per_class"]]
    x = np.arange(len(labels))
    plt.figure(figsize=(12,4.5))
    w = 0.8/len(metrics_dict)
    colors = ['#dff1f8', '#334a52', '#7db3c6', '#fbe8ff']  # Customize bar colors
    # for i,(name,md) in enumerate(metrics_dict.items()):
    #     f1s = [pc["f1"] for pc in md["per_class"]]
    #     plt.bar(x + i*w, f1s, width=w, label=name,  color=colors[i])
    for i, (name, md) in enumerate(metrics_dict.items()):
        f1s = [pc["f1"] for pc in md["per_class"]]
        bars = plt.bar(x + i * w, f1s, width=w, label=name, color=colors[i])

        # Add numbers on top of the bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2),
                     ha='center', va='bottom', fontsize=8, fontstyle='italic')  # Add number above each bar
    plt.xticks(x + w*(len(metrics_dict)-1)/2, labels, rotation=30, ha='right')
    plt.ylabel("F1")
    # plt.title("Per-class F1 comparison")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "compare_per_class_f1.png"))

# def plot_macro_roc_pr(save_dir, y_true, proba_dict, label_names):
#     """每个模型一条 macro ROC 与 macro PR（OvR）"""
#     os.makedirs(save_dir, exist_ok=True)
#     from sklearn.preprocessing import label_binarize
#     y_true = np.asarray(y_true).astype(int)
#     classes = list(range(len(label_names)))
#     y_bin = label_binarize(y_true, classes=classes)
#
#     # ROC
#     plt.figure(figsize=(6,5))
#     for name,proba in proba_dict.items():
#         try:
#             macro_auc = roc_auc_score(y_bin, proba, average='macro', multi_class='ovr')
#             RocCurveDisplay.from_predictions(y_bin.ravel(), proba.ravel(), name=f"{name} ({macro_auc:.3f})")
#         except Exception:
#             pass
#     plt.title("Macro ROC (OvR)")
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, "macro_roc_ovr.png"))
#
#     # PR
#     plt.figure(figsize=(6,5))
#     for name,proba in proba_dict.items():
#         try:
#             macro_ap = average_precision_score(y_bin, proba, average='macro')
#             PrecisionRecallDisplay.from_predictions(y_bin.ravel(), proba.ravel(), name=f"{name} ({macro_ap:.3f})")
#         except Exception:
#             pass
#     plt.title("Macro PR (OvR)")
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, "macro_pr_ovr.png"))

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

def plot_macro_roc_pr(save_dir, y_true, proba_dict, label_names):
    """所有模型在一张图中绘制 macro ROC 与 macro PR（OvR）对比"""
    os.makedirs(save_dir, exist_ok=True)
    y_true = np.asarray(y_true).astype(int)
    classes = list(range(len(label_names)))
    y_bin = label_binarize(y_true, classes=classes)

    # ROC
    plt.figure(figsize=(6,5))
    for name, proba in proba_dict.items():
        try:
            # 计算 macro AUC
            macro_auc = roc_auc_score(y_bin, proba, average='macro', multi_class='ovr')
            # 绘制 ROC 曲线并标注 AUC 分数
            RocCurveDisplay.from_predictions(y_bin.ravel(), proba.ravel(), name=f"{name} ({macro_auc:.3f})", ax=plt.gca())
        except Exception as e:
            print(f"Error in ROC for model {name}: {e}")
    plt.title("Macro ROC (OvR)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "macro_roc_ovr_comparison.png"))
    plt.close()

    # PR
    plt.figure(figsize=(6,5))
    for name, proba in proba_dict.items():
        try:
            # 计算 macro AP
            macro_ap = average_precision_score(y_bin, proba, average='macro')
            # 绘制 PR 曲线并标注 AP 分数
            PrecisionRecallDisplay.from_predictions(y_bin.ravel(), proba.ravel(), name=f"{name} ({macro_ap:.3f})", ax=plt.gca())
        except Exception as e:
            print(f"Error in PR for model {name}: {e}")
    plt.title("Macro PR (OvR)")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "macro_pr_ovr_comparison.png"))
    plt.close()

def plot_calibration(save_dir, y_true, proba_dict):
    """可靠性图 + Brier score（按“预测的最大类概率”做分箱）"""
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(6,5))
    for name,proba in proba_dict.items():
        conf = proba.max(axis=1)
        hit  = (proba.argmax(axis=1) == y_true).astype(int)
        frac_pos, mean_pred = calibration_curve(hit, conf, n_bins=10, strategy='uniform')
        plt.plot(mean_pred, frac_pos, marker="o", label=name)
    plt.plot([0,1],[0,1],'--',lw=1)
    plt.xlabel("Predicted probability (max class)")
    plt.ylabel("Empirical accuracy")
    plt.title("Reliability diagram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "calibration_reliability.png"))
############################################################################################


def testing(args, dataloaders, test_model, batch=False):
    torch.cuda.empty_cache()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    test_model.to(device)
    preds = []
    for x, y in dataloaders['test_loader'].get_iterator():
        p = test_model(x)
        preds.append(p.detach().cpu())
        del p
        torch.cuda.empty_cache()

    preds = torch.cat(preds, dim=0)
        
    print('preds shape:', preds.shape)
    preds = torch.softmax(preds, dim=1)
    
    basedir, file_tag = os.path.split(args.fig_filename)
    date_dir = time.strftime('%Y%m%d', time.localtime(time.time()))
    fig_save_dir = os.path.join(basedir, date_dir)
    if not os.path.exists(fig_save_dir):
        os.makedirs(fig_save_dir)
    confused_fig_dir = os.path.join(fig_save_dir, f'testing_confusion_map_{file_tag}.png')
    
    preds_b = preds.argmax(dim=1)
    plot_confused_cal_f1(preds_b, datasets['test_loader'].ys, fig_dir=confused_fig_dir)

    ##############################################################################
    # 在 plot_confused_cal_f1(...) 之后追加
    y_true = datasets['test_loader'].ys.cpu().numpy()  # 如果你想更严谨，也可以用 dataloaders
    proba = preds.cpu().numpy()

    rel = plot_top1_top2_reliability(
        y_true, proba, fig_save_dir, n_bins=10, binning='quantile',
        prefix=f'{file_tag}_'  # 单次测试可不带 index
    )
    print("[Reliability] ECE (Top-1 / Top-2):", rel['ece_top1'], rel['ece_top2'])

    # 1) 取得 label_names（你在 plot_confused_cal_f1 里已有 number_label_dict -> names）
    label_names = [v[:-2] for v in number_label_dict.values()]  # 复用你的命名

    # 2) 计算并保存
    y_true = datasets['test_loader'].ys.cpu().numpy()
    proba = preds.cpu().numpy()

    metrics = compute_all_metrics(y_true, proba, label_names)

    # 组织输出目录（你已有按日期建目录的逻辑）
    basedir, file_tag = os.path.split(args.fig_filename)
    date_dir = time.strftime('%Y%m%d', time.localtime(time.time()))
    fig_save_dir = os.path.join(basedir, date_dir)
    os.makedirs(fig_save_dir, exist_ok=True)

    # 模型名可用 args.task 或文件名标识
    model_tag = os.path.splitext(os.path.basename(args.best_model_save_path))[0]

    np.savez_compressed(os.path.join(fig_save_dir, f"{model_tag}_test_proba_labels.npz"),
                        proba=proba, y_true=y_true)
    with open(os.path.join(fig_save_dir, f"{model_tag}_metrics.json"), "wt") as f:
        json.dump(metrics, f, indent=2)
######################################################################################################################

    return preds


if __name__ == "__main__":
    
    start_t = time.time()
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    args = eeg_util.get_common_args()
    args = args.parse_args()
    eeg_util.DLog.init(args)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.testing:
        print('Unit_test!!!!!!!!!!!!!')
        if args.arg_file != 'None':
            args_dict = vars(args)
            print(args_dict.keys())
            print('testing args:')
            with open(args.arg_file, 'rt') as f:
                args_dict.update(json.load(f))
            print('args_dict keys after update:', args_dict.keys())
            args.testing = True
            
        xs, ys = load_tuh_data(args)
        normalize_seizure_features(xs)
        datasets = generate_dataloader_seizure(xs,ys,args)
        adjs = init_adjs(args)
        test_model = chose_model(args, adjs)
        test_model.load_state_dict(torch.load(args.best_model_save_path), strict=False)
        device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
        test_model.to(device)
        test_model.eval()

        # 5. t-SNE visual output directory
        basedir, _ = os.path.split(args.fig_filename)
        fig_save_dir = os.path.join(basedir, "tsne_output")
        os.makedirs(fig_save_dir, exist_ok=True)

        # 6. run t-SNE
        tsne_visualization(args, datasets, test_model, fig_save_dir)
        exit()

        DLog.log('args is : by DLOG:', args)
        testing(args, datasets, test_model)
        
    elif args.task == 'generate_data':
        generate_tuh_data(args, file_name="from_begin")
    else:
        dt = time.strftime('%Y%m%d', time.localtime(time.time()))
        model_used = "basic model"
        
        tags = "type:" + model_used + str(dt)
        # Save the args:
        _, file_tag = os.path.split(args.fig_filename)
        args_path = f'./args/{dt}/'
        if not os.path.exists(args_path):
            os.makedirs(args_path)
        with open(os.path.join(args_path, f'{file_tag}.json'), 'wt') as f:
            json.dump(vars(args), f, indent=4)
            
        DLog.log('------------ Args Saved! -------------')
        DLog.log('args is : by DLOG:', args)
        multi_train(args, tags=tags, runs=args.runs)
        
    print('Main running Over, total time spent:',time.time() - start_t)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

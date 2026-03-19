import os

import torch
import torch.nn.functional as F
from torch import  nn
# from torch_scatter import scatter_mean, scatter, scatter_add, scatter_max
from torch_geometric.nn.conv import MessagePassing

import eeg_util
from eeg_util import DLog
from models.baseline_models import GAT
from models.graph_conv_layer import *
from models.encoder_decoder import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class GGN(nn.Module):
    def __init__(self, adj, args, out_mid_features=False):
        super(GGN, self).__init__()
        self.args = args
        self.log = False
        self.adj_eps = 0.1
        self.adj = adj
        self.adj_x = adj

        self.N = adj.shape[0]
        print('N:', self.N)
        en_hid_dim = args.encoder_hid_dim
        en_out_dim = 16
        self.out_mid_features = out_mid_features
        
        
        if args.encoder == 'rnn':
            self.encoder = RNNEncoder(args, args.feature_len, en_hid_dim, en_out_dim, args.bidirect)
            decoder_in_dim = en_hid_dim
            if args.bidirect:
                decoder_in_dim *= 2

            de_out_dim = args.decoder_out_dim
            encoder_out_width = 34
        elif args.encoder == 'lstm':
            self.encoder = LSTMEncoder(args, args.feature_len, en_hid_dim, en_out_dim, args.bidirect)
            decoder_in_dim = en_hid_dim
            if args.bidirect:
                decoder_in_dim *= 2
            de_out_dim = args.decoder_out_dim
            
        elif args.encoder == 'cnn2d':
            cnn = CNN2d(in_dim=args.feature_len, 
                               hid_dim=en_hid_dim, 
                               out_dim=args.decoder_out_dim, 
                               width=34, height=self.N, stride=2, layers=3, dropout=args.dropout)
            self.encoder = cnn
            de_out_dim = args.decoder_out_dim
        else:
            self.encoder = MultiEncoders(args, args.feature_len, en_hid_dim, en_out_dim)
            decoder_in_dim = en_out_dim * 2
            de_out_dim = args.decoder_out_dim + decoder_in_dim

        if args.gnn_adj_type == 'rand':
            self.adj = None
            self.adj_tensor = None

        if args.lgg:
            self.LGG = LatentGraphGenerator(args, adj, args.lgg_tau, decoder_in_dim, args.lgg_hid_dim,
                                        args.lgg_k)

        if args.decoder == 'gnn':
            if args.cut_encoder_dim > 0:
                decoder_in_dim *= args.cut_encoder_dim
            self.decoder = GNNDecoder(self.N, args, decoder_in_dim, de_out_dim)
            if args.agg_type == 'cat':
                de_out_dim *= self.N
        elif args.decoder == 'gat_cnn':
            # adj_coo = eeg_util.torch_dense_to_coo_sparse(adj)
            self.adj_x =  torch.ones((self.N, self.N)).float().cuda()
            print('gat adj_x: ', self.adj_x)
            g_pooling = GateGraphPooling(args, self.N)
            gnn = GAT(decoder_in_dim, args.gnn_hid_dim, de_out_dim, 
                               dropout=args.dropout, pooling=g_pooling)
            cnn_in_dim = decoder_in_dim
            cnn = CNN2d(cnn_in_dim, args.decoder_hid_dim, de_out_dim,
                            width=34, height=self.N, stride=2, layers=3, dropout=args.dropout)
            self.decoder = SpatialDecoder(args, gnn, cnn)
            de_out_dim *= 2
            
        elif args.decoder == 'lgg_cnn':
            gnn = GNNDecoder(self.N, args, decoder_in_dim, args.gnn_out_dim)
            cnn_in_dim = decoder_in_dim
            cnn = CNN2d(cnn_in_dim, args.decoder_hid_dim, de_out_dim,
                            width=34, height=self.N, stride=2, layers=3, dropout=args.dropout)
            self.decoder = SpatialDecoder(args, gnn, cnn)
            if args.agg_type == 'cat':
                de_out_dim += args.gnn_out_dim * self.N
            else:
                if args.lgg and args.lgg_time:
                    de_out_dim += args.gnn_out_dim * 34
                else:
                    de_out_dim += args.gnn_out_dim
        else:
            self.decoder = None
            de_out_dim = decoder_in_dim * self.N
            
            
        self.predictor = ClassPredictor(de_out_dim, hidden_channels=args.predictor_hid_dim,
                                class_num=args.predict_class_num, num_layers=args.predictor_num, dropout=args.dropout)

        self.warmup = args.lgg_warmup
        self.epoch = 0
        
        DLog.log('-------- ecoder: -----------\n', self.encoder)
        DLog.log('-------- decoder: -----------\n', self.decoder)
        
        self.reset_parameters()

    def adj_to_coo_longTensor(self, adj):
        """adj is cuda tensor
        """
        DLog.debug(adj)
        adj[adj > self.adj_eps] = 1
        adj[adj <= self.adj_eps] = 0

        idx = torch.nonzero(adj).T.long() # (row, col)
        DLog.debug('idx shape:', idx.shape)
        return idx

    def encode(self, x):
        # B,C,N,T = x.shape
        x = self.encoder(x)
        return x

    def fake_decoder(self, adj, x):
        DLog.debug('fake decoder in shape:', x.shape)
        # trans to BC:
        if len(x.shape) == 4:
            x = x[:,-1,...]
            
        if len(x.shape) > 2:
            x = torch.flatten(x, start_dim=1)
        
        return x

    def decode(self, x, B, N, adj):

        if self.decoder is None:
            x = self.fake_decoder(adj, x)
            DLog.debug('decoder out shape:', x.shape)
            return x

        if self.args.cut_encoder_dim > 0:
            x = x[:,:,:,-self.args.cut_encoder_dim:]

        x = self.decoder(adj, x)
        
        DLog.debug('decoder out shape:', x.shape)
        return x

    def alternative_freeze_grad(self, epoch):
        self.epoch = epoch
        if self.epoch > self.warmup:
            if epoch % 2==0:
                # freeze LGG
                eeg_util.freeze_module(self.LGG)
                eeg_util.unfreeze_module(self.encoder)
            else:
                # freeze encoder
                eeg_util.freeze_module(self.encoder)
                eeg_util.unfreeze_module(self.LGG)
                

    def forward(self, x, *options):
        """
        input x shape: B, C, N, T
        output x: class
        """
        B,C,N,T = x.shape

        # (1) encoder:
        x = self.encode(x)

        # before: BNCT
        x = x.permute(0, 3, 1, 2)
        # permute to BTNC

        # (2) adj selection:

        # LGG, latent graph generator:
        if self.args.lgg:
            if self.args.lgg_time:
                adj_x_times = []
                for t in range(T):
                    x_t = x[:, t, ...]
                    if self.training:
                        if self.epoch < self.warmup:
                            adj_x = self.LGG(x_t, self.adj)
                        else:
                            adj_x = self.LGG(x_t, self.adj)
                    else:
                        adj_x = self.LGG(x_t, self.adj)
                        DLog.debug('Model is Eval!!!!!!!!!!!!!!!!!')
                    adj_x_times.append(adj_x)
                self.adj_x = adj_x_times
            else:
                x_t = x[:, -1, ...]  # NOTE: take last time step.
                if self.training and self.epoch < self.warmup:
                    self.adj_x = self.LGG(x_t, self.adj)
                else:
                    self.adj_x = self.LGG(x_t, self.adj)
                    DLog.debug('Model is Eval!!!!!!!!!!!!!!!!!')

        # (3) decoder:
        DLog.debug('decoder input shape:', x.shape)
        x = self.decode(x, B, N, self.adj_x)
        DLog.debug('decoder output shape:', x.shape)

        if self.out_mid_features:
            return x
        
        # (4) predictor:
        x = self.predictor(x)
        return x

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.count=0

    # ===== 兼容 ST_HGGN 的轻量适配方法（BEGIN）=====
    def encode_only(self, x, time_info=None):
        """
        适配 ST_HGGN：仅做编码，返回 (encoded_features, meta)
        - 自动把常见的 [B, C_feat, T, N] 转成 [B, C_feat, N, T] 再走 self.encode(...)
        - meta 至少包含 B/N/T，供 decode_only 使用
        """
        if x.dim() != 4:
            raise ValueError(f"encode_only expects 4D input, got {x.shape}")

        # 你的数据常是 [B, F, T, N] = [*, 48, 139, 20]，而底座一般用 [B, F, N, T]
        # 这里尽量鲁棒判断：如果最后一维是通道数 N，就认为是 [B, F, T, N]，转成 [B, F, N, T]
        B, Cfeat, D2, D3 = x.shape
        N_expected = getattr(self, "N", None)
        if N_expected is not None and D3 == N_expected:
            # 输入是 [B, F, T, N] -> 转为 [B, F, N, T]
            x = x.transpose(2, 3)  # 现在是 [B, F, N, T]
            N, T = x.shape[2], x.shape[3]
        else:
            # 假定输入已经是 [B, F, N, T]
            N, T = x.shape[2], x.shape[3]

        if N_expected is not None and N != N_expected:
            raise ValueError(f"[encode_only] N mismatch: got {N}, expected {N_expected}")

        encoded = self.encode(x)  # 复用你现有的 encode
        meta = {"B": B, "N": N, "T": T}
        return encoded, meta

    def decode_only(self, encoded_features, adj, meta: dict = None):
        """
        适配 ST_HGGN：仅做解码，返回“解码后的特征”（不做分类头）
        - 复用你现有的 decode(encoded, B, N, adj)
        """
        if meta is None:
            B = encoded_features.shape[0]
            N = getattr(self, "N", None)
            if N is None:
                raise ValueError("[decode_only] cannot infer N; pass meta={'N': ...}.")
        else:
            B = meta.get("B", encoded_features.shape[0])
            N = meta.get("N", getattr(self, "N", None))
            if N is None:
                raise ValueError("[decode_only] meta must contain 'N' or model must have self.N.")

        decoded = self.decode(encoded_features, B, N, adj)  # 复用你现有的 decode
        return decoded
    # ===== 兼容 ST_HGGN 的轻量适配方法（END）=====


class ClassPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, class_num, num_layers,
                 dropout=0.5):
        super(ClassPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        DLog.log('Predictor in channel:', in_channels)
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, class_num))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        DLog.debug('input prediction x shape:', x.shape)
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class LatentGraphGenerator(nn.Module):
    def __init__(self, args, A_0, tau, in_dim, hid_dim, K=10):
        super(LatentGraphGenerator,self).__init__()
        self.N = A_0.shape[0] # num of nodes.
        self.args = args
        self.A_0 = A_0
        self.args = args

        if args.gnn_pooling == 'att':
            pooling = AttGraphPooling(args, self.N, in_dim, 64)
        elif args.gnn_pooling == 'cpool':
            pooling = CompactPooling(args, 3, self.N)
        elif args.gnn_pooling.upper() == 'NONE':
            pooling = None
        else:
            pooling = GateGraphPooling(args, self.N)
            
        self.gumbel_tau = tau
        self.mu_nn = MultilayerGNN(self.N, 2, pooling, in_dim, hid_dim, K, args.dropout)
        self.sig_nn = MultilayerGNN(self.N, 2, pooling, in_dim, hid_dim, K, args.dropout)
        self.pi_nn = MultilayerGNN(self.N, 2, pooling, in_dim, hid_dim, K, args.dropout)
        
        self.adj_fix = nn.Parameter(self.A_0)

        print('adj_fix', self.adj_fix.shape)

        self.init_norm()


    def init_norm(self):
        # self.Norm = torch.randn(size=(1000, self.args.batch_size, self.N)).cuda()
        device = torch.device("cuda" if self.args.cuda and torch.cuda.is_available() else "cpu")
        self.Norm = torch.randn(size=(1000, self.args.batch_size, self.N)).to(device)
        self.norm_index = 0

    def get_norm_noise(self, size):
        if self.norm_index >= 999:
            self.init_norm()

        if size == self.args.batch_size:
            self.norm_index += 1
            return self.Norm[self.norm_index].squeeze()
        else:
            return torch.randn((size, self.N)).cuda()
        
    def update_A(self, mu, sig, pi):
        """ mu, sig, pi, shape: (B, N, K)
        update A, 
        """
        # cal prob of pi:
        DLog.debug('pi Has Nan:', torch.isnan(pi).any())
        logits = torch.log(torch.softmax(pi, dim=-1))
        DLog.debug('logits Has Nan:', torch.isnan(logits).any())

        pi_onehot = F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=False, dim=-1)

        # select one component of mu, sig via pi for each node:

        mu_k = torch.sum(mu * pi_onehot, dim=-1) # BN
        sig_k = torch.sum(sig * pi_onehot, dim=-1) #BN

        n = self.get_norm_noise(mu_k.shape[0]) # BN
        DLog.debug('mu shape:', mu_k.shape)
        DLog.debug('sig_k shape:', sig_k.shape)
        DLog.debug('n shape:', n.shape)

        S = mu_k + n*sig_k
        S = S.unsqueeze(dim=-1)
        # change to gumbel softmax, discrete sampling.
        # DLog.debug('S Has Nan:', torch.isnan(S).any())
        Sim = torch.einsum('bnc, bcm -> bnm', S, S.transpose(2, 1)) # need to be softmax

        P = torch.sigmoid(Sim)

        pp = torch.stack((P+0.01, 1-P + 0.01), dim=3)
        DLog.debug('min:', torch.min(pp))
        # DLog.debug('max',torch.max(pp))
        pp_logits = torch.log(pp)
        DLog.debug('Has Nan:', torch.isnan(pp_logits).any())
        pp_onehot = F.gumbel_softmax(pp_logits, tau=self.gumbel_tau, hard=False, dim=-1)
        A = pp_onehot[:,:,:,0]
        A = torch.mean(A, dim=0)

        return A

    def forward(self, x, adj_t=None):
        if adj_t is None:
            adj_t = self.adj_fix
            DLog.debug('LGG: adj_t shape', adj_t.shape)
        
        mu = self.mu_nn(x, adj_t)
        sig = self.sig_nn(x, adj_t)
        pi = self.pi_nn(x, adj_t)

        A = self.update_A(mu, sig, pi)

        return A


#####################################################################################
class SpatioTemporalDecayGraphGenerator(nn.Module):
    """
    将神经电活动的时空衰减机制转成可学习的图演化；输入期望为 [B,N,C] 或 [B,T,N,C]
    """
    def __init__(self, args, adj, tau, in_dim, hid_dim, K=10, decay_factor=0.1):
        super().__init__()
        self.args = args
        self.N = adj.shape[0]
        self.A_0 = adj
        self.gumbel_tau = tau
        self.input_dim = in_dim        # 节点特征维度（应与编码器输出一致，通常是32）
        self.hid_dim = hid_dim         # LGG 的中间隐藏维（通常很小，默认3）
        self.K = K

        print(f"SpatioTemporalDecayGraphGenerator: N={self.N}, in_dim={in_dim}, hid_dim={hid_dim}, K={K}")

        # ✅ 关键字参数，避免传反
        self.graph_generator = LatentGraphGenerator(
            args=args,
            A_0=adj,
            tau=self.gumbel_tau,
            in_dim=self.input_dim,
            hid_dim=self.hid_dim,
            K=self.K,
        )

        # 时空衰减相关参数（保持你原逻辑）
        self.temporal_decay = nn.Parameter(torch.ones(self.N) * decay_factor)
        self.spatial_decay = nn.Parameter(torch.ones(self.N, self.N) * decay_factor)

        # 基于邻接权重构造距离矩阵（与原实现一致）
        self._build_distance_from_adj(adj)

        # 自适应融合系数
        self.adaptive_factor = nn.Parameter(torch.tensor(0.9))
        # 缓存上一步邻接
        self.prev_adjs = {}
        self.time_step = 0

        # 🔧 当外部误传 C 维时，懒加载一个线性投影把 C→self.input_dim
        self._proj = None

    def _build_distance_from_adj(self, adj: torch.Tensor):
        device = adj.device
        N = adj.shape[0]
        A = adj.detach().float()
        eps = 1e-6
        # 权重大的连边距离更近：d = 1/(w+eps)，无连接给一个较大距离
        inv = 1.0 / (A + eps)
        inv[A <= eps] = 5.0
        # 对角为0
        inv = inv * (1.0 - torch.eye(N, device=device)) + 0.0 * torch.eye(N, device=device)
        self.register_buffer("distance_matrix", inv)

    def _maybe_project(self, x: torch.Tensor) -> torch.Tensor:
        # x:[B,N,C] → 确保 C == self.input_dim
        C = x.size(-1)
        if C != self.input_dim:
            if self._proj is None:
                self._proj = nn.Linear(C, self.input_dim, bias=False).to(x.device)
                print(f"[STDGG] 自动建立通道投影: {C}→{self.input_dim}")
            x = self._proj(x)
        return x

    def forward(self, x, adj_t=None, batch_id: int = None):
        """
        x: [B,N,C] 或 [B,T,N,C]
        adj_t: 当前时间步先验邻接（可为 None）
        """
        # 取 [B,N,C]
        if x.dim() == 4:     # [B,T,N,C]
            node_feat = x[:, -1]      # 取最后一个时间步
        elif x.dim() == 3:   # [B,N,C]
            node_feat = x
        else:
            raise ValueError(f"STDGG 期望 x 为 [B,N,C] 或 [B,T,N,C]，收到 {tuple(x.shape)}")

        node_feat = self._maybe_project(node_feat)

        if adj_t is None:
            adj_t = self.A_0

        # 生成当前时间步图
        cur_adj = self.graph_generator(node_feat, adj_t)

        # 取上一时刻图
        if batch_id is None:
            batch_id = 0
        prev_adj = self.prev_adjs.get(batch_id, adj_t)

        # 计算时空衰减系数
        # 时序：按节点独立衰减
        t_decay = torch.sigmoid(self.temporal_decay).view(1, -1, 1)   # [1,N,1]
        # 空间：按(i,j)对衰减，结合距离
        s_decay = torch.sigmoid(self.spatial_decay)                   # [N,N]
        dist = self.distance_matrix
        s_decay = s_decay * torch.exp(-dist)                          # 距离越远越小

        # 融合
        cur_adj = t_decay.squeeze(-1) * cur_adj
        cur_adj = s_decay * cur_adj

        alpha = torch.sigmoid(self.adaptive_factor)
        final_adj = alpha * cur_adj + (1 - alpha) * prev_adj

        # 缓存
        self.prev_adjs[batch_id] = final_adj.detach()
        if len(self.prev_adjs) > 100:
            for k in list(self.prev_adjs.keys())[:50]:
                del self.prev_adjs[k]

        return final_adj

    def reset_states(self):
        self.prev_adjs = {}
        self.time_step = 0



# 层次化图生成网络（安全版：关键字传参 + 自动维度对齐）
class HierarchicalGraphGenerator(nn.Module):
    """
    将局部/区域/全局三个尺度的图组合；输入期望为 [B,N,C] 或 [B,T,N,C]
    """
    def __init__(self, args, adj, tau, in_dim, hid_dim, K=10):
        super().__init__()
        self.args = args
        self.N = adj.shape[0]
        self.A_0 = adj
        self.input_dim = in_dim
        self.hid_dim = hid_dim
        self.K = K

        # ✅ 关键字参数，确保不传反
        k_local = max(1, K // 2)
        self.local_generator = LatentGraphGenerator(
            args=args, A_0=adj, tau=tau, in_dim=self.input_dim, hid_dim=self.hid_dim, K=k_local
        )
        self.regional_generator = LatentGraphGenerator(
            args=args, A_0=adj, tau=tau, in_dim=self.input_dim, hid_dim=self.hid_dim, K=k_local
        )
        self.global_generator = LatentGraphGenerator(
            args=args, A_0=adj, tau=tau, in_dim=self.input_dim, hid_dim=self.hid_dim, K=k_local
        )

        self.level_weights = nn.Parameter(torch.ones(3))

        # 🔧 自动维度对齐（外部若误传C，做一次线性投影）
        self._proj = None

    def _maybe_project(self, x: torch.Tensor) -> torch.Tensor:
        C = x.size(-1)
        if C != self.input_dim:
            if self._proj is None:
                self._proj = nn.Linear(C, self.input_dim, bias=False).to(x.device)
                print(f"[HGGN] 自动建立通道投影: {C}→{self.input_dim}")
            x = self._proj(x)
        return x

    def forward(self, x, adj_t=None):
        # 取 [B,N,C]
        if x.dim() == 4:
            node_feat = x[:, -1]
        elif x.dim() == 3:
            node_feat = x
        else:
            raise ValueError(f"HGGN 期望 x 为 [B,N,C] 或 [B,T,N,C]，收到 {tuple(x.shape)}")

        node_feat = self._maybe_project(node_feat)

        if adj_t is None:
            adj_t = self.A_0

        A_local = self.local_generator(node_feat, adj_t)
        A_reg   = self.regional_generator(node_feat, adj_t)
        A_glob  = self.global_generator(node_feat, adj_t)

        w = F.softmax(self.level_weights, dim=0)
        A = w[0] * A_local + w[1] * A_reg + w[2] * A_glob
        return A



# 动态连接重要性加权（安全版：分类器输入维自动对齐）
class DynamicConnectionWeighting(nn.Module):
    """
    A_final = σ(λ)*A_gen + (1-σ(λ))*A_class(node_features)
    """
    def __init__(self, num_nodes, num_classes, hidden_dim=64, lambda_balance=0.9):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_classes = num_classes

        # 类别连接模式参数 [C,N,N]
        self.class_connection_weights = nn.Parameter(torch.randn(num_classes, num_nodes, num_nodes) * 0.01)
        self.class_importances = nn.Parameter(torch.zeros(num_classes))  # β_c
        self.lambda_balance = nn.Parameter(torch.tensor(lambda_balance))

        # 分类头（输入维在第一次 forward 时根据 node_features 自动对齐）
        self._clf_in = None
        self.class_predictor = None

        self._init_spectral()

    def _init_spectral(self):
        with torch.no_grad():
            for c in range(self.num_classes):
                for i in range(self.num_nodes):
                    for j in range(self.num_nodes):
                        freq = 1.0 + 0.5 * c
                        dist = abs(i - j)
                        self.class_connection_weights[c, i, j] = math.cos(freq * dist * math.pi / self.num_nodes)
                self.class_connection_weights[c] += torch.randn_like(self.class_connection_weights[c]) * 0.01

    def _ensure_classifier(self, in_dim, hidden_dim=64):
        if (self.class_predictor is None) or (self._clf_in != in_dim):
            self._clf_in = in_dim
            self.class_predictor = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, self.num_classes)
            )

    def forward(self, adj_gen: torch.Tensor, node_features: torch.Tensor):
        """
        adj_gen: [N,N] 或 [B,N,N]
        node_features: [B,N,C]（将用作类别判别的输入）
        """
        B, N, C = node_features.shape
        self._ensure_classifier(in_dim=C, hidden_dim=C)  # 分类器输入与 C 对齐

        # 类别概率
        pooled = node_features.mean(dim=1)              # [B,C]
        class_logits = self.class_predictor(pooled)     # [B,C_classes]
        class_probs = torch.softmax(class_logits, dim=-1)  # [B,C_classes]

        # 重要性权重 β_c
        importance = torch.softmax(self.class_importances, dim=0)      # [C_classes]
        scaled = class_probs * importance.unsqueeze(0)                 # [B,C_classes]
        scaled = scaled / (scaled.sum(dim=1, keepdim=True) + 1e-10)

        # [B,N,N]
        if adj_gen.dim() == 2:
            adj_gen = adj_gen.unsqueeze(0).expand(B, -1, -1)

        # 按类别加权叠加类别图
        batch_cls_adj = torch.zeros_like(adj_gen)
        for c in range(self.num_classes):
            batch_cls_adj += scaled[:, c].view(B, 1, 1) * self.class_connection_weights[c]

        batch_cls_adj = torch.sigmoid(batch_cls_adj)
        lam = torch.sigmoid(self.lambda_balance)
        A = lam * adj_gen + (1 - lam) * batch_cls_adj
        A = 0.5 * (A + A.transpose(1, 2))  # 对称化

        return A, class_logits



# 增强型图生成网络(GGN)，集成动态连接重要性加权机制
class EnhancedGGN(nn.Module):
    """
    增强型图生成网络(GGN)，集成动态连接重要性加权机制
    用于改进癫痫发作检测和分类。
    """

    def __init__(self, adj, args, out_mid_features=False):
        super(EnhancedGGN, self).__init__()
        self.args = args
        self.log = False
        self.adj_eps = 0.1
        self.adj = adj
        self.adj_x = adj

        self.N = adj.shape[0]
        print('N:', self.N)
        en_hid_dim = args.encoder_hid_dim
        en_out_dim = 16
        self.out_mid_features = out_mid_features

        # 从args获取时间维度，默认为139
        time_dim = getattr(args, 'eeg_seq_len', 139)

        # 根据args初始化编码器
        if args.encoder == 'rnn':
            from models.encoder_decoder import RNNEncoder
            self.encoder = RNNEncoder(args, args.feature_len, en_hid_dim, en_out_dim, args.bidirect)
            decoder_in_dim = en_hid_dim
            if args.bidirect:
                decoder_in_dim *= 2
            de_out_dim = args.decoder_out_dim
            encoder_out_width = 34
        elif args.encoder == 'lstm':
            from models.encoder_decoder import LSTMEncoder
            self.encoder = LSTMEncoder(args, args.feature_len, en_hid_dim, en_out_dim, args.bidirect)
            decoder_in_dim = en_hid_dim
            if args.bidirect:
                decoder_in_dim *= 2
            de_out_dim = args.decoder_out_dim
        elif args.encoder == 'cnn2d':
            from models.encoder_decoder import CNN2d
            cnn = CNN2d(in_dim=args.feature_len,
                        hid_dim=en_hid_dim,
                        out_dim=args.decoder_out_dim,
                        width=34, height=self.N, stride=2, layers=3, dropout=args.dropout)
            self.encoder = cnn
            de_out_dim = args.decoder_out_dim
        else:
            # 使用正确的时间维度的多编码器
            from models.encoder_decoder import MultiEncoders
            self.encoder = MultiEncoders(args, args.feature_len, en_hid_dim, en_out_dim, time_dim=time_dim)
            decoder_in_dim = en_out_dim * 2
            de_out_dim = args.decoder_out_dim + decoder_in_dim

        # 动态连接重要性加权机制
        lambda_balance = getattr(args, 'lambda_balance', 0.5)
        self.dynamic_connection = DynamicConnectionWeighting(
            num_nodes=self.N,
            num_classes=args.predict_class_num,
            hidden_dim=decoder_in_dim,
            lambda_balance=lambda_balance
        )

        # 需要时初始化潜在图生成器
        if args.gnn_adj_type == 'rand':
            self.adj = None
            self.adj_tensor = None

        if args.lgg:
            self.LGG = LatentGraphGenerator(args, adj, args.lgg_tau, decoder_in_dim, args.lgg_hid_dim,
                                            args.lgg_k)

        # 初始化解码器
        if args.decoder == 'gnn':
            from models.encoder_decoder import GNNDecoder
            if args.cut_encoder_dim > 0:
                decoder_in_dim *= args.cut_encoder_dim
            self.decoder = GNNDecoder(self.N, args, decoder_in_dim, de_out_dim)
            if args.agg_type == 'cat':
                de_out_dim *= self.N
        elif args.decoder == 'gat_cnn':
            from models.encoder_decoder import SpatialDecoder
            from models.baseline_models import GAT
            from models.encoder_decoder import CNN2d
            from models.graph_conv_layer import GateGraphPooling

            device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
            self.adj_x = torch.ones((self.N, self.N)).float().to(device)
            print('gat adj_x: ', self.adj_x)
            g_pooling = GateGraphPooling(args, self.N)
            gnn = GAT(decoder_in_dim, args.gnn_hid_dim, de_out_dim,
                      dropout=args.dropout, pooling=g_pooling)
            cnn_in_dim = decoder_in_dim
            cnn = CNN2d(cnn_in_dim, args.decoder_hid_dim, de_out_dim,
                        width=34, height=self.N, stride=2, layers=3, dropout=args.dropout)
            self.decoder = SpatialDecoder(args, gnn, cnn)
            de_out_dim *= 2
        elif args.decoder == 'lgg_cnn':
            from models.encoder_decoder import SpatialDecoder, GNNDecoder
            from models.encoder_decoder import CNN2d

            gnn = GNNDecoder(self.N, args, decoder_in_dim, args.gnn_out_dim)
            cnn_in_dim = decoder_in_dim
            cnn = CNN2d(cnn_in_dim, args.decoder_hid_dim, de_out_dim,
                        width=34, height=self.N, stride=2, layers=3, dropout=args.dropout)
            self.decoder = SpatialDecoder(args, gnn, cnn)
            if args.agg_type == 'cat':
                de_out_dim += args.gnn_out_dim * self.N
            else:
                if args.lgg and args.lgg_time:
                    de_out_dim += args.gnn_out_dim * 34
                else:
                    de_out_dim += args.gnn_out_dim
        else:
            self.decoder = None
            de_out_dim = decoder_in_dim * self.N

        # 分类预测器
        self.predictor = ClassPredictor(de_out_dim, hidden_channels=args.predictor_hid_dim,
                                        class_num=args.predict_class_num, num_layers=args.predictor_num,
                                        dropout=args.dropout)

        # 保存辅助变量
        self.aux_class_logits = None  # 用于存储类别logits以供后续分析
        self.warmup = args.lgg_warmup
        self.epoch = 0

        print('-------- encoder: -----------\n', self.encoder)
        print('-------- decoder: -----------\n', self.decoder)

        self.reset_parameters()

    def adj_to_coo_longTensor(self, adj):
        """将邻接矩阵转换为COO格式
        """
        adj[adj > self.adj_eps] = 1
        adj[adj <= self.adj_eps] = 0

        idx = torch.nonzero(adj).T.long()  # (row, col)
        print('idx shape:', idx.shape)
        return idx

    def encode(self, x):
        """编码输入数据
        """
        return self.encoder(x)

    def fake_decoder(self, adj, x):
        """在没有实际解码器时使用的简单解码器
        """
        print('fake decoder in shape:', x.shape)
        # 转换为BC格式:
        if len(x.shape) == 4:
            x = x[:, -1, ...]

        if len(x.shape) > 2:
            x = torch.flatten(x, start_dim=1)

        return x

    def decode(self, x, B, N, adj):
        """解码编码后的特征
        """
        if self.decoder is None:
            x = self.fake_decoder(adj, x)
            print('decoder out shape:', x.shape)
            return x

        if self.args.cut_encoder_dim > 0:
            x = x[:, :, :, -self.args.cut_encoder_dim:]

        x = self.decoder(adj, x)

        # print('decoder out shape:', x.shape)
        return x

    def alternative_freeze_grad(self, epoch):
        """训练期间交替冻结梯度
        """
        self.epoch = epoch
        if self.epoch > self.warmup:
            if epoch % 2 == 0:
                # 冻结LGG
                from eeg_util import freeze_module, unfreeze_module
                freeze_module(self.LGG)
                unfreeze_module(self.encoder)
            else:
                # 冻结编码器
                from eeg_util import freeze_module, unfreeze_module
                freeze_module(self.encoder)
                unfreeze_module(self.LGG)

    def forward(self, x, *options):
        """
        增强的前向传播，使用动态连接重要性加权

        参数:
            x: 形状为[B, C, N, T]的输入数据(预期)

        返回:
            x: 输出预测
        """
        # 检查形状并在必要时进行转置
        if x.shape[2] == 139 and x.shape[3] == 20:
            x = x.transpose(2, 3)  # 将[B, C, T, N]转换为[B, C, N, T]

        B, C, N, T = x.shape
        if N != self.N:
            raise ValueError(f"预期N={self.N}，实际为N={N}")

        # 检查维度
        if x.dim() != 4:
            raise ValueError(f"预期4D输入(B, C, N, T)，实际得到{x.shape}")

        # (1) 编码输入:
        x_encoded = self.encode(x)

        # (2) 生成邻接矩阵:
        class_logits = None  # 用于存储类别logits

        if self.args.lgg:
            if self.args.lgg_time:
                adj_x_times = []
                for t in range(T):
                    x_t = x_encoded[:, t, ...]
                    if self.training:
                        if self.epoch < self.warmup:
                            adj_x = self.LGG(x_t, self.adj)
                        else:
                            adj_x = self.LGG(x_t, self.adj)
                    else:
                        adj_x = self.LGG(x_t, self.adj)
                        print('Model is Eval!!')
                    adj_x_times.append(adj_x)
                self.adj_x = adj_x_times

                # 应用动态连接重要性加权
                # 获取节点特征
                node_features = x_encoded[:, -1, :, :]
                # 处理最后一个时间步的邻接矩阵
                self.adj_x[-1], class_logits = self.dynamic_connection(self.adj_x[-1], node_features)
                # print("动态连接重要性加权机制已应用")

            else:
                x_t = x_encoded[:, -1, ...]  # 获取最后一个时间步
                if self.training and self.epoch < self.warmup:
                    self.adj_x = self.LGG(x_t, self.adj)
                else:
                    self.adj_x = self.LGG(x_t, self.adj)
                    print('Model is Eval!!')

                # 应用动态连接重要性加权
                node_features = x_encoded[:, -1, :, :]
                self.adj_x, class_logits = self.dynamic_connection(self.adj_x, node_features)
                # print("动态连接重要性加权机制已应用")

        # 保存类别logits以用于后续分析和损失计算
        self.aux_class_logits = class_logits

        # (3) 解码:
        x = self.decode(x_encoded, B, N, self.adj_x)

        if self.out_mid_features:
            return x

        # (4) 预测:
        x = self.predictor(x)

        # 如果有辅助类别logits，可以将其与最终预测结合
        if self.aux_class_logits is not None and self.training:
            # 在训练期间，将动态权重预测与最终预测结合
            # 这创建了更强的梯度流，并鼓励动态权重机制对分类有更直接的贡献
            alpha = 0.2  # 辅助预测的权重
            x = x + alpha * self.aux_class_logits

        return x

    def get_class_connection_patterns(self):
        """
        返回学习的类别特定连接模式，用于可视化和分析
        """
        if hasattr(self, 'dynamic_connection'):
            # print("动态连接重要性加权机制已应用")
            return self.dynamic_connection.class_connection_weights.detach()
        return None

    def get_class_importances(self):
        """
        返回学习的类别重要性权重，用于分析哪些类别需要更多自定义连接
        """
        if hasattr(self, 'dynamic_connection'):
            # print("动态连接重要性加权机制已应用")
            return F.softmax(self.dynamic_connection.class_importances, dim=0).detach()
        return None

    def reset_parameters(self):
        """重置所有参数
        """
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.count = 0

    # ===== 兼容 ST_HGGN 的轻量适配方法（BEGIN）=====
    def encode_only(self, x, time_info=None):
        """
        适配 ST_HGGN：仅做编码，返回 (encoded_features, meta)
        - 自动把常见的 [B, C_feat, T, N] 转成 [B, C_feat, N, T] 再走 self.encode(...)
        - meta 至少包含 B/N/T，供 decode_only 使用
        """
        if x.dim() != 4:
            raise ValueError(f"encode_only expects 4D input, got {x.shape}")

        # 你的数据常是 [B, F, T, N] = [*, 48, 139, 20]，而底座一般用 [B, F, N, T]
        # 这里尽量鲁棒判断：如果最后一维是通道数 N，就认为是 [B, F, T, N]，转成 [B, F, N, T]
        B, Cfeat, D2, D3 = x.shape
        N_expected = getattr(self, "N", None)
        if N_expected is not None and D3 == N_expected:
            # 输入是 [B, F, T, N] -> 转为 [B, F, N, T]
            x = x.transpose(2, 3)  # 现在是 [B, F, N, T]
            N, T = x.shape[2], x.shape[3]
        else:
            # 假定输入已经是 [B, F, N, T]
            N, T = x.shape[2], x.shape[3]

        if N_expected is not None and N != N_expected:
            raise ValueError(f"[encode_only] N mismatch: got {N}, expected {N_expected}")

        encoded = self.encode(x)  # 复用你现有的 encode
        meta = {"B": B, "N": N, "T": T}
        return encoded, meta

    def decode_only(self, encoded_features, adj, meta: dict = None):
        """
        适配 ST_HGGN：仅做解码，返回“解码后的特征”（不做分类头）
        - 复用你现有的 decode(encoded, B, N, adj)
        """
        if meta is None:
            B = encoded_features.shape[0]
            N = getattr(self, "N", None)
            if N is None:
                raise ValueError("[decode_only] cannot infer N; pass meta={'N': ...}.")
        else:
            B = meta.get("B", encoded_features.shape[0])
            N = meta.get("N", getattr(self, "N", None))
            if N is None:
                raise ValueError("[decode_only] meta must contain 'N' or model must have self.N.")

        decoded = self.decode(encoded_features, B, N, adj)  # 复用你现有的 decode
        return decoded
    # ===== 兼容 ST_HGGN 的轻量适配方法（END）=====



# 完整的ST-HGGN模型
class ST_HGGN(nn.Module):
    """
    目标顺序：
      GGN(基线特征/解码) → 时空衰减图生成器(STDGG) → 层次化图生成网络(HGGN) → [可选] 动态连接重要性加权(DCW)
    其中 DCW 通过 args.use_dynamic_in_st_hggn 开关控制（默认 False）
    """
    def __init__(self, adj, args):
        super(ST_HGGN, self).__init__()
        self.args = args
        self.N = adj.shape[-1] if isinstance(adj, torch.Tensor) else adj.size(-1)

        # 1) 基线：只用 GGN 做编码/解码（不包含动态加权）
        # 注意：需要确保 GGN 支持 out_mid_features=True 返回中间编码特征
        self.base_model = GGN(adj, args, out_mid_features=True)

        # 2) 维度声明（与你现有编码器输出对齐；如你项目中是 32 就保持 32）
        # ✅ 以解码侧实际通道为准：优先用 gnn_out_dim（日志里是 64），退化再用 encoder_out_dim
        self.input_dim = getattr(args, "gnn_out_dim", getattr(args, "encoder_out_dim", 32))

        # 时空衰减图 & 层次化图都吃 self.input_dim
        self.st_graph_generator = SpatioTemporalDecayGraphGenerator(
            args, adj, args.lgg_tau,
            self.input_dim, args.lgg_hid_dim, args.lgg_k
        )
        self.hierarchical_generator = HierarchicalGraphGenerator(
            args, adj, args.lgg_tau,
            self.input_dim, args.lgg_hid_dim, args.lgg_k
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))
        # （可选）最后再做 DCW：输入维度必须与编码特征一致
        self.use_dynamic = getattr(args, "use_dynamic_in_st_hggn", False)
        if self.use_dynamic:
            dcw_hidden = self.input_dim  # ← 关键：用同一通道数（通常=64）
            self.dynamic_connection = DynamicConnectionWeighting(
                num_nodes=self.N,
                num_classes=args.predict_class_num,
                hidden_dim=dcw_hidden,
                lambda_balance=getattr(args, "dcw_lambda_balance", 0.9)
            )

    def reset_states(self):
        if hasattr(self.base_model, "reset_states"):
            self.base_model.reset_states()
        if hasattr(self.st_graph_generator, "reset_states"):
            self.st_graph_generator.reset_states()
        if hasattr(self.hierarchical_generator, "reset_states"):
            self.hierarchical_generator.reset_states()

    def forward(self, x, time_info=None):
        # 1) 编码特征
        encoded_features, _ = self.base_model.encode_only(x, time_info=time_info)

        # 2) 并联邻接
        st_adj = self.st_graph_generator(encoded_features)   # 时空图
        h_adj  = self.hierarchical_generator(encoded_features, st_adj.detach())
        # 注意这里用 st_adj.detach()，避免两个模块梯度冲突

        # 3) 融合（残差+α加权）
        final_adj = self.alpha * h_adj + (1 - self.alpha) * st_adj

        # 4) 可选 DCW（轻量 re-weight）
        if self.use_dynamic:
            final_adj, _ = self.dynamic_connection(final_adj, encoded_features)

        # 5) 解码
        logits = self.base_model.decode_only(encoded_features, final_adj)
        return logits
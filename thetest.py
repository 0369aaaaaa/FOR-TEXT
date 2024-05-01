import os
import glob
import time
import torch
import random
import argparse
import numpy as np
import pandas as pd
from torch import nn
import scipy.sparse as sp
import torch.optim as optim
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch.autograd import Variable
from matplotlib.font_manager import FontProperties
from torch_geometric.nn import GAE, GCNConv, GATConv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
class GAT3re(torch.nn.Module):
    def __init__(self, in_channels, hidden1, hidden2, out_channels, dropout):
        super(GAT3re, self).__init__()
        self.gat1 = GATConv(in_channels, hidden1, heads=7)
        self.gat2 = GATConv(hidden1 * 7, hidden2, heads=7)
        self.gat3 = GATConv(hidden2 * 7, out_channels, heads=1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, edge_index):
        x = self.dropout(x)
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.gat2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.gat3(x, edge_index)
        return F.log_softmax(x, dim=1)
class GAT3e(torch.nn.Module):
    def __init__(self, in_channels, hidden1, hidden2, out_channels, dropout):
        super(GAT3e, self).__init__()
        self.gat1 = GATConv(in_channels, hidden1, heads=7)
        self.gat2 = GATConv(hidden1 * 7, hidden2, heads=7)
        self.gat3 = GATConv(hidden2 * 7, out_channels, heads=1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, edge_index):
        x = self.dropout(x)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.gat3(x, edge_index)
        return F.log_softmax(x, dim=1)
class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, dropout):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, out_channels)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, edge_index):
        x = self.dropout(x)
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
def args_go():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='不使用gpu')
    parser.add_argument('--seed', type=int, default=60, help='随机数种子')
    parser.add_argument('--epochs', type=int, default=300, help='迭代次数')
    parser.add_argument('--lr', type=float, default=0.005, help='初始学习率')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='L2参数损失')
    parser.add_argument('--dropout', type=float, default=0.5, help='衰落')
    parser.add_argument('--patience', type=int, default=30, help='Patience')
    return parser.parse_args()
def initialize_data(path, dataset, seed):
    print('Loading {} dataset...'.format(dataset))
    random.seed(seed)  # 随机数设置
    np.random.seed(seed)
    torch.manual_seed(seed)
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.str_)
    feature = sp.csr_matrix(idx_features_labels[:, 1:-1],
                            dtype=np.float32)  # 按行压缩的矩阵为每行中元素不为0个数的计数，值得注意的是这个计数是累加的（0,0） 0，第一为行数，第二为列数 后面为值
    label = pd.get_dummies(idx_features_labels[:, -1])  # 标签用onehot方式表示
    idx = np.array(idx_features_labels[:, 0], dtype=np.str_)  # 获取id号
    idx_map = {j: i for i, j in enumerate(idx)}  # 从id号到一个唯一的整数索引的映射
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.str_)  # 读取边索引
    edge = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    rev_edges = np.array(edge[:, [1, 0]])
    rev_edges = rev_edges.reshape(1, rev_edges.shape[0] * 2)
    edge = edge.reshape(1, edge.shape[0] * 2)
    full_edges = np.concatenate((edge, rev_edges), axis=0)
    full_edges = torch.LongTensor(full_edges)
    ad = sp.coo_matrix((np.ones(full_edges.shape[0]), (full_edges[:, 0], full_edges[:, 1])),
                       shape=(label.shape[0], label.shape[0]), dtype=np.float32)  # 生成邻接矩阵,类型与csr相同,有向图型
    ad = ad + ad.T.multiply(ad.T > ad) - ad.multiply(ad.T > ad)  # 生成邻接矩阵,类型与csr相同，无向图型
    feature = normalize_features(feature)  # 归一化
    ad = normalize_adj(ad + sp.eye(ad.shape[0]))  # 归一化
    idx_trai, idx_va, idx_tes = split_dataset(np.arange(len(idx_map)))
    ad = torch.FloatTensor(np.array(ad.todense()))
    feature = torch.FloatTensor(np.array(feature.todense()))
    label = torch.LongTensor(np.where(label)[1])
    idx_trai = torch.LongTensor(idx_trai)
    idx_va = torch.LongTensor(idx_va)
    idx_tes = torch.LongTensor(idx_tes)
    return ad, feature, label, idx_trai, idx_va, idx_tes, full_edges
def normalize_adj(mx):
    rowsum = np.array(mx.sum(1))  # 计算每行的和
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()  # 每行的平方根的倒数
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.  # 无穷大替换为0
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)  # 构建对角矩阵，对角线元素为r_inv_sqrt
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
def normalize_features(mx):
    rowsum = np.array(mx.sum(1))  # 计算每行的和
    r_inv = np.power(rowsum, -1).flatten()  # 每行的倒数
    r_inv[np.isinf(r_inv)] = 0.  # 无穷大替换为0
    r_mat_inv = sp.diags(r_inv)  # 构建对角矩阵，对角线元素为r_inv
    mx = r_mat_inv.dot(mx)
    return mx
def split_dataset(dataset):
    num_samples = len(dataset)
    train_size = int(0.4 * num_samples)
    val_size = int(0.2 * num_samples)
    test_size = num_samples - train_size - val_size
    indices = list(range(num_samples))  # 生成索引列表
    random.shuffle(indices)  # 随机打乱索引列表
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    train_set = [dataset[i] for i in train_indices]  # 根据索引提取样本
    val_set = [dataset[i] for i in val_indices]
    test_set = [dataset[i] for i in test_indices]
    return train_set, val_set, test_set
def train(model, ep, op):
    t = time.time()
    model.train()  # 设置GAT模型为训练模式
    op.zero_grad()  # 清空之前参数的梯度
    output = model(features, edges)  # 对特征和边进行前向传播，得到输出output
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])  # 计算训练集上的损失值，负对数似然损失函数F.nll_loss
    acc_train = accuracy_score(output[idx_train].max(1)[1].type_as(labels[idx_train]), labels[idx_train])  # 计算训练集上的准确率
    loss_train.backward()  # 进行反向传播计算梯度
    op.step()  # 更新模型参数
    model.eval()
    output = model(features, edges)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])  # 计算验证集上的损失值
    acc_val = accuracy_score(output[idx_val].max(1)[1].type_as(labels[idx_val]), labels[idx_val])  # 计算验证集上的准确率
    k = output[idx_test].max(1)[1].type_as(labels[idx_test])
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy_score(k, labels[idx_test])  # 计算验证集上的准确率
    pre_test = precision_score(k, labels[idx_test], average='macro')
    re_test = recall_score(k, labels[idx_test], average='macro')
    f1_test = f1_score(k, labels[idx_test], average='macro')
    print('迭代次数: {:03d}'.format(ep + 1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val),
          '耗时: {:.4f}s'.format(time.time() - t))
    return loss_val.data.item(), loss_test.data.item(), acc_test, pre_test, re_test, f1_test
def train1(model, title, name, op):
    t_total = time.time()
    loss_values = []  # 初始化一个空列表，用于存储每个epoch的训练损失值。
    ls_values = []
    acc_values = []
    pre_values = []
    re_values = []
    f1_values = []
    bad_counter = 0  # 用于跟踪连续多少个epoch的损失值没有改善
    best = args.epochs + 1
    best_epoch = 0
    for epoch in range(args.epochs):
        ls, ls1, ac, pr, re, f1 = train(model, epoch, op)
        loss_values.append(ls)
        if epoch % 5 == 0:
            ls_values.append(ls1)
            acc_values.append(ac)
            pre_values.append(pr)
            re_values.append(re)
            f1_values.append(f1)
        torch.save(model.state_dict(), '{}.pkl'.format(epoch))
        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch
        #    bad_counter = 0
        #else:
            #bad_counter += 1
        #if bad_counter == args.patience:
            #break
        files = glob.glob('*.pkl')
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(file)
    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(file)
    print("迭代结束!")
    print("总耗时: {:.4f}s".format(time.time() - t_total))
    print('读取{}次迭代'.format(best_epoch + 1))
    model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))
    visualize_tsne_embeddings(model, title, name, ls_values, acc_values, pre_values, re_values, f1_values)
    return 0
def visualize_tsne_embeddings(model, title, name, ls, ac, pr, re, f1):
    model.eval()
    output = model(features, edges)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy_score(output[idx_val].max(1)[1].type_as(labels[idx_val]), labels[idx_val])
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    k = output[idx_test].max(1)[1].type_as(labels[idx_test])
    acc_test = accuracy_score(k, labels[idx_test])
    print("Test set results:",
          "loss_val= {:.4f}".format(loss_val.data.item()),
          "acc_val= {:.4f}".format(acc_val))
    ts = TSNE(n_components=2, learning_rate='auto', init='random')  # 降维到二维空间，并设置学习率为自动调整，初始化方式为随机
    ts.fit_transform(output[idx_test].to('cpu').detach().numpy())
    x = ts.embedding_  # 获取t-SNE转换后的二维数据
    y = labels[idx_test].to('cpu').detach().numpy()  # 从原始数据中获取标签信息
    xi = []
    for i in range(7):
        xi.append(x[np.where(y == i)])
    colors = ['blue', 'green', 'red', 'yellow', 'cyan', 'violet', 'springgreen']
    plt.figure(figsize=(12, 10))
    for i in range(7):
        plt.scatter(xi[i][:, 0], xi[i][:, 1], s=30, color=colors[i], marker='.', alpha=1)
    font = FontProperties(fname='C:/Windows/Fonts/simsun.ttc', size=12)
    plt.title(title, fontproperties=font)
    text = ["loss_test= {:.4f}".format(loss_test.data.item()), "acc_test= {:.4f}".format(acc_test),
            "precision_test= {:.4f}".format(precision_score(k, labels[idx_test], average='macro')),
            "recall_test= {:.4f}".format(recall_score(k, labels[idx_test], average='macro')),
            "f1_test= {:.4f}".format(f1_score(k, labels[idx_test], average='macro'))]
    plt.text(0, 0, text, fontproperties=font, color='black', ha='left', va='bottom', transform=plt.gca().transAxes)
    plt.savefig(name)
    x = np.arange(len(ls))
    fig, ax1 = plt.subplots(figsize=(15, 10))
    plt.title(title, fontproperties=font)
    ax1.plot(x*5, ls, color='black', label='loss', marker='*')
    ax1.set_xlabel('epochs_test')
    ax1.set_ylabel('loss', color='black')
    ax1.tick_params('y', colors='black')
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))
    # 创建第二个坐标轴
    ax2 = ax1.twinx()
    ax2.plot(x*5, ac, color='red', label='accuracy', marker='.')
    ax2.plot(x*5, pr, color='green', label='precision', marker='+')
    ax2.plot(x*5, re, color='purple', label='recall', marker='^')
    ax2.plot(x*5, f1, color='cyan', label='f1', marker='o')
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax2.tick_params('y', colors='blue')
    # 设置每个坐标轴的范围
    ax1.set_ylim(0.2, 2)
    ax2.set_ylim(0, 1)
    plt.legend(loc='upper left')
    ax1.set_xlim(-1, x[-1]*5+1)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(10))
    name = "loss_" + name
    plt.savefig(name)
    # plt.show()
def program_1():
    adj, features, labels, idx_train, idx_val, idx_test, edges = initialize_data("./data/cora/", "cora", args.seed)
    gae_model = GAE(GCNEncoder(features.shape[1], 512, int(labels.max()) + 1, args.dropout))
    print(gae_model)
    optimizer = optim.Adam(gae_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # 使用Adam优化器来优化模型的参数
    if args.cuda:
        gae_model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
    features, adj, labels = Variable(features), Variable(adj), Variable(labels)
    train1(gae_model, "gae2", "cora1,gae2,ep300,lr0.005,pa30.png", optimizer)
def program_2():
    adj, features, labels, idx_train, idx_val, idx_test, edges = initialize_data("./data/cora/", "cora", args.seed)
    gat_model = GAE(GAT3re(features.shape[1], 256, 128, int(labels.max()) + 1, args.dropout))
    print(gat_model)
    optimizer = optim.Adam(gat_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # 使用Adam优化器来优化模型的参数
    if args.cuda:
        gat_model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
    features, adj, labels = Variable(features), Variable(adj), Variable(labels)
    train1(gat_model, "gat3re", "cora2,gat3re,ep300,lr0.005,pa30.png", optimizer)
def program_3():
    adj, features, labels, idx_train, idx_val, idx_test, edges = initialize_data("./data/cora/", "cora", args.seed)
    gat_model = GAE(GAT3e(features.shape[1], 256, 128, int(labels.max()) + 1, args.dropout))
    print(gat_model)
    optimizer = optim.Adam(gat_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # 使用Adam优化器来优化模型的参数
    if args.cuda:
        gat_model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
    features, adj, labels = Variable(features), Variable(adj), Variable(labels)
    train1(gat_model, "gat3e", "cora3,gat3e,ep300,lr0.005,pa30.png", optimizer)
def program_4():
    adj, features, labels, idx_train, idx_val, idx_test, edges = initialize_data("./data/citeseer/", "citeseer",
                                                                                 args.seed)
    gae_model = GAE(GCNEncoder(features.shape[1], 512, int(labels.max()) + 1, args.dropout))
    print(gae_model)
    optimizer = optim.Adam(gae_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # 使用Adam优化器来优化模型的参数
    if args.cuda:
        gae_model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
    features, adj, labels = Variable(features), Variable(adj), Variable(labels)
    train1(gae_model, "gae2", "citeseer1,gae2,ep300,lr0.005,pa30.png", optimizer)
def program_5():
    adj, features, labels, idx_train, idx_val, idx_test, edges = initialize_data("./data/citeseer/", "citeseer",
                                                                                 args.seed)
    gat_model = GAE(GAT3re(features.shape[1], 256, 128, int(labels.max()) + 1, args.dropout))
    print(gat_model)
    optimizer = optim.Adam(gat_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # 使用Adam优化器来优化模型的参数
    if args.cuda:
        gat_model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
    features, adj, labels = Variable(features), Variable(adj), Variable(labels)
    train1(gat_model, "gat3re", "citeseer2,gat3re,ep300,lr0.005,pa30.png", optimizer)
def program_6():
    adj, features, labels, idx_train, idx_val, idx_test, edges = initialize_data("./data/citeseer/", "citeseer",
                                                                                 args.seed)
    gat_model = GAE(GAT3e(features.shape[1], 256, 128, int(labels.max()) + 1, args.dropout))
    print(gat_model)
    optimizer = optim.Adam(gat_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # 使用Adam优化器来优化模型的参数
    if args.cuda:
        gat_model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
    features, adj, labels = Variable(features), Variable(adj), Variable(labels)
    train1(gat_model, "gat3e", "citeseer3,gat3e,ep300,lr0.005,pa30.png", optimizer)
if __name__ == "__main__":
    args = args_go()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    random.seed(args.seed)  # 随机数设置
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    print("请选择要执行的程序：")
    print("1. gae2,ep300,lr0.01,pa30,cora")
    print("2. gat3relu,ep300,lr0.01,pa30,cora")
    print("3. gat3elu,ep300,lr0.01,pa30,cora")
    print("4. gae2,ep300,lr0.01,pa30,citeseer")
    print("5. gat3relu,ep300,lr0.01,pa30,citeseer")
    print("6. gat3elu,ep300,lr0.01,pa30,citeseer")
    choice = input("输入数字选择程序：")
    if choice == '1':
        adj, features, labels, idx_train, idx_val, idx_test, edges = initialize_data("./data/cora/", "cora", args.seed)
        program_1()
    elif choice == '2':
        adj, features, labels, idx_train, idx_val, idx_test, edges = initialize_data("./data/cora/", "cora", args.seed)
        program_2()
    elif choice == '3':
        adj, features, labels, idx_train, idx_val, idx_test, edges = initialize_data("./data/cora/", "cora", args.seed)
        program_3()
    elif choice == '4':
        adj, features, labels, idx_train, idx_val, idx_test, edges = initialize_data("./data/citeseer/", "citeseer",
                                                                                     args.seed)
        program_4()
    elif choice == '5':
        adj, features, labels, idx_train, idx_val, idx_test, edges = initialize_data("./data/citeseer/", "citeseer",
                                                                                     args.seed)
        program_5()
    elif choice == '6':
        adj, features, labels, idx_train, idx_val, idx_test, edges = initialize_data("./data/citeseer/", "citeseer",
                                                                                     args.seed)
        program_6()
    else:
        print("输入无效，请输入1、2、3、4、5、6来选择程序。")
import numpy as np
import torch
import pdb
import time
import sys, os


from torch.distributions import Bernoulli, Categorical

import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import  to_dense_adj
import src.Settings

OP_PRIMITIVES = [
    'output', 
    'input'
]

if __name__ == "__main__":
    sys.path.insert(1, os.path.join(os.getcwd(),'AG_Net/'))


def adj_to_scores(adj, l = 21):
    row_idx, col_idx = np.tril_indices(adj.T.shape[1], k=-1)
    row_idx = torch.LongTensor(row_idx).to(adj.device)
    col_idx = torch.LongTensor(col_idx).to(adj.device)
    scores =  adj.T[row_idx, col_idx]
    if len(scores) != l:
        scores = F.pad(scores, pad=(0,l-len(scores)), value=0)
    return scores.float()


def scores_to_adj(scores, num_nodes=7):
    tril_indices = torch.tril_indices(row=num_nodes, col=num_nodes, offset=-1)
    adj = torch.zeros(num_nodes, num_nodes).to(scores.device)
    adj[tril_indices[0], tril_indices[1]] = scores
    return adj.T

##############################################################################
#
#                         Generator
#
##############################################################################


class GNNLayer_forward(MessagePassing):
    def __init__(self, ndim):
        super(GNNLayer_forward, self).__init__(aggr='add')
        self.msg = nn.Linear(ndim, ndim)
        self.upd = nn.GRUCell(ndim, ndim//2)

    def forward(self, edge_index, h):
        return self.propagate(edge_index, h=h)

    def message(self, h_j, h_i):
        m = torch.cat([h_j, h_i], dim=1)
        a=self.msg(m)
        return a

    def update(self, aggr_out, h):
        h = self.upd(aggr_out, h)
        return h

class GNNLayer_backward(MessagePassing):
    def __init__(self, ndim):
        super(GNNLayer_backward, self).__init__(aggr='add')
        self.msg = nn.Linear(ndim, ndim)
        self.upd = nn.GRUCell(ndim, ndim//2)

    def forward(self, edge_index, h):
        return self.propagate(edge_index, h=h)

    def message(self, h_j, h_i):
        m = torch.cat([h_j, h_i], dim=1)
        a=self.msg(m)
        return a

    def update(self, aggr_out, h):
        h = self.upd(aggr_out, h)
        return h



class NodeEmbUpd(nn.Module):
    def __init__(self,ndim, num_layers, model_config):
        super().__init__()
        self.ndim = ndim
        self.num_layers = num_layers
        self.dropout = model_config['node_dropout']
        self.GNNLayers_forward = nn.ModuleList([GNNLayer_forward(ndim) for _ in range(num_layers)])
        self.GNNLayers_backward = nn.ModuleList([GNNLayer_backward(ndim) for _ in range(num_layers)])


    def forward(self, h, edge_index):
        if h.size(1)==self.ndim:
            h_forward, h_backward = torch.split(h, h.size(1)//2, 1)
        else:
            h_forward=h.clone()
            h_backward=h.clone()

        for layer in self.GNNLayers_forward:
            h_forward = F.dropout(h_forward, p=self.dropout, training=self.training)
            h_forward = layer(edge_index, h_forward)

        edge_index=edge_index[[1,0]]
        for layer in self.GNNLayers_backward:
            h_backward = F.dropout(h_backward, p=self.dropout, training=self.training)
            h_backward = layer(edge_index, h_backward)

        h=torch.cat([h_forward,h_backward],1)
        return h


class GraphAggr(nn.Module):
    def __init__(self, ndim, gdim, aggr='gsum'):
        super().__init__()
        self.ndim = ndim
        self.gdim = gdim
        self.aggr = aggr
        self.f_m = nn.Linear(ndim, gdim)
        if aggr == 'gsum':
            self.g_m = nn.Linear(ndim, 1)
            self.sigm = nn.Sigmoid()

    def forward(self, h, idx):
        if self.aggr == 'mean':
            h = self.f_m(h).view(-1, idx, self.gdim)
            return torch.mean(h, 1)
        elif self.aggr == 'gsum':
            h_vG = self.f_m(h)
            g_vG = self.sigm(self.g_m(h))
            h = torch.mul(h_vG, g_vG).view(-1, idx, self.gdim)
            return torch.sum(h, 1)



class GraphEmbed(nn.Module):
    def __init__(self, ndim, gdim, num_gnn_layers, model_config, nb301):
        super().__init__()
        self.ndim = ndim
        self.gdim = gdim
        self.num_gnn_layers=num_gnn_layers
        self.model_config=model_config
        self.NodeEmb = NodeEmbUpd(ndim, num_gnn_layers, model_config)
        self.GraphEmb = GraphAggr(ndim, gdim)
        self.GraphEmb_init = GraphAggr(ndim, gdim)
        self.nb301 = nb301

    def forward(self, h, edge_index):
        idx = h.size(1)
        h = h.view(-1, self.ndim)
        if idx == 1:
            return h.unsqueeze(1), self.GraphEmb.f_m(h), self.GraphEmb_init.f_m(h)
        elif idx ==2 and self.nb301:
            h = h.view(-1, idx, self.ndim)[:,1]
            return h.unsqueeze(1), self.GraphEmb.f_m(h), self.GraphEmb_init.f_m(h)
        else:
            h = self.NodeEmb(h, edge_index)
            h_G = self.GraphEmb(h, idx)
            h_G_init = self.GraphEmb_init(h, idx)
            return h.view(-1, idx, self.ndim), h_G, h_G_init


class NodeAdd(nn.Module):
    def __init__(self, gdim, num_node_atts):
        super().__init__()
        self.gdim = gdim
        self.num_node_atts=num_node_atts
        self.f_an = nn.Linear(gdim*2, gdim)
        self.f_an_2 = nn.Linear(gdim, num_node_atts)

    def forward(self, h_G, c):
        s = self.f_an(torch.cat([h_G, c], 1))
        return self.f_an_2(F.relu(s))

class Embedding_Network(nn.Module):
    def __init__(self, in_channels = 5, hidden_channels = 32, out_channels = 32, num_layers = 2):
        super().__init__()

        net = torch.nn.Sequential()

        for i in range(num_layers):
            net.add_module(
                f"Linear{i}",
                torch.nn.Linear(
                    in_features  = in_channels if i==0 else hidden_channels,
                    out_features = out_channels if i==num_layers-1 else hidden_channels
                )
            )
            if i < num_layers-1:
                net.add_module(
                    f"ReLU{i}",
                    torch.nn.ReLU()
                )

        self.lin_layers = net

    def forward(self, x):
        x = self.lin_layers(x)
        return x


class NodeInit(nn.Module):
    def __init__(self, ndim, gdim, num_node_atts):
        super().__init__()
        self.ndim = ndim
        self.gdim = gdim
        self.num_node_atts=num_node_atts
        self.NodeInits = Embedding_Network(in_channels = num_node_atts, hidden_channels = ndim, out_channels = ndim, num_layers = 2)
        self.f_init = nn.Linear(ndim+gdim*2, ndim+gdim)
        self.f_init_2 = nn.Linear(ndim+gdim, ndim)
        self.f_start = nn.Linear(ndim+gdim, ndim+gdim)
        self.f_start_2 = nn.Linear(ndim+gdim, ndim)

    def forward(self, h_G_init, node_atts, c):
        e = self.NodeInits(node_atts)
        if h_G_init==None:
            return e
        if isinstance(h_G_init, str):
            h_inp = self.f_start(torch.cat([e, c], 1))
            return self.f_start_2(F.relu(h_inp))
        h_v = self.f_init(torch.cat([e, h_G_init, c], 1))
        return self.f_init_2(F.relu(h_v))



class Nodes(nn.Module):
    def __init__(self, ndim, gdim):
        super().__init__()
        self.ndim = ndim
        self.gdim = gdim
        self.f_s_1 = nn.Linear(ndim*2+gdim*2, ndim+gdim)
        self.f_s_2 = nn.Linear(ndim+gdim, 1)

    def forward(self, h, h_v, h_G, c):
        idx = h.size(1)
        s = self.f_s_1(torch.cat([h.view(-1, self.ndim),
                                  h_v.unsqueeze(1).repeat(1, idx, 1).view(-1, self.ndim),
                                  h_G.repeat(idx, 1),
                                  c.repeat(idx, 1)], dim=1))
        return self.f_s_2(F.relu(s)).view(-1, idx)


class Generator(nn.Module):
    def __init__(self, ndim, gdim, num_gnn_layers, num_node_atts,max_n, model_config, nb301, alpha=.5, stop=10):
        super().__init__()
        self.ndim = ndim
        self.gdim = gdim
        self.num_gnn_layers=num_gnn_layers
        self.num_node_atts=num_node_atts
        self.alpha = alpha
        self.model_config=model_config
        self.prop = GraphEmbed(ndim, gdim,num_gnn_layers, model_config, nb301)
        self.nodeAdd = NodeAdd(gdim, num_node_atts)
        self.nodeInit = NodeInit(ndim, gdim, num_node_atts)
        self.nodes = Nodes(ndim, gdim)
        self.stop = stop
        self.max_n = max_n
        self.nb301 = nb301

    def forward(self, h, c, edge_index):
        idx = h.size(1)
        if self.nb301:
            second_input_h = h
        h, h_G, h_G_init = self.prop(h, edge_index)
        node_score = self.nodeAdd(h_G, c)
        h_v = self.nodeInit(h_G_init, node_score, c)  

        if h.size(1) == 1 and idx == 1:
            h = torch.cat([h, h_v.unsqueeze(1)], 1)
            if self.nb301:
                return h,  node_score  , torch.empty_like(c[:,:1]).fill_(-1) 
            return h,  node_score  , torch.empty_like(c[:,:1]).fill_(20)  
        if h.size(1) == 1 and idx == 2:
            h = torch.cat([second_input_h, h_v.unsqueeze(1)], 1)
            return h, node_score,  torch.cat((torch.empty_like(c[:,:1]).fill_(1), torch.empty_like(c[:,:1]).fill_(-1)),1)

        edge_score = self.nodes(h, h_v, h_G, c)

        h = torch.cat([h, h_v.unsqueeze(1)], 1)

        return h, node_score , edge_score  


class GNNDecoder(nn.Module):
    def __init__(self, ndim, gdim, num_gnn_layers, num_node_atts, max_n, model_config, nb301, list_all_lost=False):
        super().__init__()
        self.ndim = ndim
        self.gdim = gdim
        self.num_gnn_layers = num_gnn_layers
        self.num_node_atts = num_node_atts
        self.max_n = max_n
        self.model_config = model_config
        self.nb301 = nb301
        self.generator = Generator(ndim, gdim, num_gnn_layers, num_node_atts, max_n, model_config,nb301)
        self.list_all_lost = list_all_lost

    def forward(self, c, t=0, generate_nb301=False):
        def scores_to_index(edge_scores_list, num_nodes, device, t=0.5): ########
            edge_index = torch.LongTensor().to(device)
            edge_list = [torch.nonzero(scores_to_adj(s,num_nodes) > t, as_tuple=False).T for s in edge_scores_list]
            for n,edge in enumerate(edge_list):
                edge = edge.to(device)
                if n == 0:
                    edge_index = torch.cat((edge_index, edge),1)
                else:
                    edge_index = torch.cat((edge_index, edge + (torch.max(edge_index).item()+1) ),1)

            return edge_index

        input_one_hot = torch.cat(c.size(0)*[torch.tensor(np.eye(self.num_node_atts)[OP_PRIMITIVES.index('input')]).unsqueeze(0)]).float().to(c.device) 
        h = self.generator.nodeInit('start',input_one_hot, c).unsqueeze(1)  

        edge_index = 0
        edge_scores =  torch.tensor([]).to(c.device)
        node_scores = 20*input_one_hot#input_one_hot
        for i in range(self.max_n-1):
            h, node_score, edges = self.generator(h,
                                    c,
                                    edge_index,
                                    )
            edge_scores = torch.cat([edge_scores, edges],1)
            node_scores = torch.cat([node_scores, node_score],1)

            if i == 0 and self.nb301:
                edge_index = 0
            else:
                edge_index = scores_to_index(edge_scores, i+2, c.device)

        return node_scores, edge_scores


class MLP_predictor(nn.Module):
    def __init__(self, layers, in_channels, hidden_channels, out_channels=1, condition=0):
        self.pars = locals()
        del self.pars["self"]
        del self.pars["__class__"]
        super().__init__()

        self.net_mlp = nn.ModuleList([torch.nn.Linear(in_features = in_channels,out_features=hidden_channels )])
        self.net_mlp.extend([torch.nn.Linear(in_features=hidden_channels+condition, out_features=hidden_channels) for i in range(layers-1)])
        self.net_mlp.append(torch.nn.Linear(in_features=hidden_channels+condition, out_features=out_channels))

    def forward(self, y, mlp_condition=0):
        if type(mlp_condition) == int:
            for layer in self.net_mlp[:-1]:
                y = F.relu(layer(y))
        else:
            for layer in self.net_mlp[:-1]:
                y = F.relu(layer(y))
                y = torch.cat((y, mlp_condition),1)

        y = self.net_mlp[-1](y)
        return y

    ###########################################################################
    def to_checkpoint(self):
        chkpt = {}
        chkpt["state"] = self.state_dict()
        chkpt["pars"] = {**self.pars}
        return chkpt

class Generator_Decoder(nn.Module):
    def __init__(self, model_config, data_config, acc_prediction=False, generator_conditions=1, regression_conditions=0, list_all_lost=False, nb301=False):
        self.pars = locals()
        del self.pars["self"]
        del self.pars["__class__"]

        super().__init__()
        self.ndim = model_config['node_embedding_dim']
        self.gdim = model_config['graph_embedding_dim']
        self.num_gnn_layers = model_config['gnn_iteration_layers']
        self.num_node_atts = data_config['num_node_atts']
        self.model_config = model_config
        self.max_n = data_config['max_num_nodes']
        self.acc_prediction = acc_prediction
        self.list_all_lost = list_all_lost
        self.generator_conditions = generator_conditions    
        self.nb301 = nb301
        if self.list_all_lost:
            self.node_criterion = torch.nn.CrossEntropyLoss(reduction='none')
            self.edge_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
            self.acc_criterion = torch.nn.MSELoss(reduction = 'none')
        else:
            self.node_criterion = torch.nn.CrossEntropyLoss(reduction='mean')
            self.edge_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
            self.acc_criterion = torch.nn.MSELoss()
        
        
        self.Decoder = GNNDecoder(self.ndim, model_config['graph_embedding_dim'], self.num_gnn_layers, self.num_node_atts, self.max_n, model_config, nb301, list_all_lost)

        
        if self.acc_prediction:
            self.regression_output = data_config['regression_output']
            self.Predictor = MLP_predictor(layers=model_config['num_regression_layers'], in_channels=data_config['regression_input'], hidden_channels=data_config['regression_input'], out_channels=data_config['regression_output'], condition=regression_conditions)

    ###########################################################################
    def generate(
            self,
            instances : int,
            device : str,
            conditions = None, 
            Dataset = None ,
            generate_nb301 = False,
    ): 
        r' Not applicable so far for HW Nasbench search'

        noise = torch.randn(    
                instances, self.gdim,
                device = device)

        fake = self(noise, Dataset=Dataset, generate_nb301=generate_nb301)

        return fake

    def forward(self, x, mlp_condition=0,  Dataset=None, generate_nb301=False):
        node_scores, edge_scores = self.Decoder(x, generate_nb301=generate_nb301)

        data = []
        for i, graph in enumerate(zip(node_scores, edge_scores)):
            g = self.convert_to_pg_data(graph)
            data.append(g)
            data[-1].z = x[i]
            data[-1].g = torch.cat((graph[0], graph[1]))
            if Dataset is not None:
                generated = torch.cat((graph[0], graph[1])).unsqueeze(0)
                path = Dataset.encode_gt_computational_data(generated)
                data[-1].path = path 
            if self.acc_prediction:
                generated = torch.cat((graph[0], graph[1]))
                if not type(mlp_condition) == int:
                    generated = torch.cat((generated.unsqueeze(0),mlp_condition),1)
                    output = self.Predictor(generated, mlp_condition)
                    pred_acc = output[:,0]
                    pred_hw = output[:,1]
                    data[-1].latency = pred_hw
                else:
                    pred_acc = self.Predictor(generated)
                
                data[-1].val_acc = pred_acc

        return data

    def loss(self, batch_list, instances, device_latency=None):
        
        noise = torch.randn(
            instances, self.gdim,
            device = batch_list.x.device
            )

        nodes, edges = self.Decoder(noise)
        
        generated = torch.cat((nodes, edges),1)
        ln = self.node_criterion(nodes.view(-1,self.num_node_atts),  torch.argmax(batch_list.x_binary, dim=1).view(instances,-1).flatten())
        le = self.edge_criterion(edges, batch_list.scores.view(instances, -1))

        if self.list_all_lost:
            ln = torch.mean(ln.view(instances, -1),1)
            le = torch.mean(le,1) 
        loss = 2*(0.5*ln+ 0.5*le)

        generated = (nodes, edges)

        mse =  torch.tensor(0)

        if self.acc_prediction:
            ind = torch.where(batch_list.val_acc > 0)[0]
            if ind.numel() == 0:
                mse =  torch.tensor(0)
            else:
                generated = torch.cat((nodes, edges),1)
                if device_latency is not None:
                    generated = torch.cat((nodes, edges),1)
                    generated = torch.cat((generated,device_latency[:,:-1]),1)
                    output  = self.Predictor(generated,  device_latency[:,:-1])
                    acc = output[:,0]
                    latency = output[:,1]
                    mse_a = self.acc_criterion(acc.view(-1), batch_list.val_acc)
                    mse_l = self.acc_criterion(latency.view(-1), device_latency[:,-1])
                    mse = mse_a , mse_l
                else:
                    acc  = self.Predictor(generated[ind])
                    mse = self.acc_criterion(acc.view(-1), batch_list.val_acc[ind])
        
        return generated, loss, mse


    def convert_to_pg_data(self, data):
        edge_scores = data[1]
        
        # Node scores until second last layer (last layer always output)
        node_scores = data[0].view(-1, self.num_node_atts)[:-1]

        true_nodes =  torch.cat([node_scores,20*torch.tensor([np.eye(self.num_node_atts)[OP_PRIMITIVES.index('output')]]).float().to(node_scores.device)]) 
            
        node_atts  = Categorical(logits=true_nodes).sample().long()

        # Prune node atts until first predicted output layer 
        num_zeros = (node_atts == 0).nonzero()
        if num_zeros.size(0) > 1:
            num_zeros = num_zeros[0]
        node_atts = node_atts[:num_zeros+1]

        num_nodes = len(node_atts)

        edges = Bernoulli(logits=edge_scores).sample().float()

        adjacency_matrix = scores_to_adj(edges, self.max_n)[:num_nodes, :num_nodes]

        edge_index = torch.nonzero(adjacency_matrix, as_tuple=False).T
        scores = adj_to_scores(adjacency_matrix, l =(self.max_n*(self.max_n-1)//2))

        y_nodes = node_atts
        if len(y_nodes) != self.max_n:
            y_nodes = F.pad(y_nodes, pad=(0,self.max_n-len(y_nodes)), value=0)

        x_binary = torch.nn.functional.one_hot(y_nodes, num_classes=self.num_node_atts)

        y = torch.cat((x_binary.reshape(-1).float(), scores.float()))


        return Data(edge_index=edge_index.long(), x=node_atts, num_nodes=num_nodes, x_binary=x_binary, y=y, scores=scores)

    def number_of_parameters(self):
        return(sum(p.numel() for p in self.parameters() if p.requires_grad))

    ###########################################################################
    def to_checkpoint(self):
        chkpt = {}
        chkpt["state"] = self.state_dict()
        chkpt["pars"] = {**self.pars}
        return chkpt


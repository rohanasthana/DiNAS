import math

import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn import functional as F
from torch import Tensor
import tensorflow.compat.v1 as tf
from src import utils
from src.diffusion import diffusion_utils
from src.models.layers import Xtoy, Etoy, masked_softmax
from scipy.sparse import csgraph
#tf.enable_eager_execution()

def laplacian_positional_encoding(A):
    """
        Graph positional encoding v/ Laplacian eigenvectors
        
    """
    A=csgraph.laplacian(A,normed=True)
    #print(A.shape)
    return A

def get_timing_signal_1d(length,
                         channels,
                         min_timescale=1.0,
                         max_timescale=1.0e4,
                         start_index=0):
    """Gets a bunch of sinusoids of different frequencies.

    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase.

    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.

    The use of relative position is possible because sin(x+y) and cos(x+y) can be
    expressed in terms of y, sin(x) and cos(x).

    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.

    Args:
    length: scalar, length of timing signal sequence.
    channels: scalar, size of timing embeddings to create. The number of
        different timescales is equal to channels / 2.
    min_timescale: a float
    max_timescale: a float
    start_index: index of first position

    Returns:
    a Tensor of timing signals [1, length, channels]
    """
    tf.enable_eager_execution()
    position = tf.cast((tf.range(length) + start_index),tf.float32)
    num_timescales = channels // 2
    log_timescale_increment = (
      math.log(float(max_timescale) / float(min_timescale)) /
      (tf.cast(num_timescales,tf.float32) - 1))
    inv_timescales = min_timescale * tf.exp(tf.cast(tf.range(num_timescales),tf.float32) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.math.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels])
    #print(tf.__version__)
    np_tensor = signal.numpy()
    signal=torch.from_numpy(np_tensor).cuda()
    return signal

class XEyTransformerLayer(nn.Module):
    """ Transformer that updates node, edge and global features
        d_x: node features
        d_e: edge features
        dz : global features
        n_head: the number of heads in the multi_head_attention
        dim_feedforward: the dimension of the feedforward network model after self-attention
        dropout: dropout probablility. 0 to disable
        layer_norm_eps: eps value in layer normalizations.
    """
    def __init__(self, dx: int, de: int, dy: int, n_head: int, dim_ffX: int = 2048,
                 dim_ffE: int = 128, dim_ffy: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, device=None, dtype=None) -> None:
        kw = {'device': device, 'dtype': dtype}
        super().__init__()

        self.self_attn = NodeEdgeBlock(dx, de, dy, n_head, **kw)

        self.linX1 = Linear(dx, dim_ffX, **kw)
        self.linX2 = Linear(dim_ffX, dx, **kw)
        self.normX1 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.normX2 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.dropoutX1 = Dropout(dropout)
        self.dropoutX2 = Dropout(dropout)
        self.dropoutX3 = Dropout(dropout)
        self.embedding_lap_pos_enc = nn.Linear(dx, dx)
        self.posencdim=dx

        self.linE1 = Linear(de, dim_ffE, **kw)
        self.linE2 = Linear(dim_ffE, de, **kw)
        self.normE1 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.normE2 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.dropoutE1 = Dropout(dropout)
        self.dropoutE2 = Dropout(dropout)
        self.dropoutE3 = Dropout(dropout)

        self.lin_y1 = Linear(dy, dim_ffy, **kw)
        self.lin_y2 = Linear(dim_ffy, dy, **kw)
        self.norm_y1 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.norm_y2 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.dropout_y1 = Dropout(dropout)
        self.dropout_y2 = Dropout(dropout)
        self.dropout_y3 = Dropout(dropout)
        self.LSTM= nn.LSTM(dx, dx,num_layers=2)
        self.RNN= nn.RNN(dx, dx,num_layers=2)
        self.activation = F.relu

    def forward(self, X: Tensor, E: Tensor, y, node_mask: Tensor):
        """ Pass the input through the encoder layer.
            X: (bs, n, d)
            E: (bs, n, n, d)
            y: (bs, dy)
            node_mask: (bs, n) Mask for the src keys per batch (optional)
            Output: newX, newE, new_y with the same shape.
        """
        #print(X.shape)
        #posenc=laplacian_positional_encoding(X)
        #X_lap_pos_enc = self.embedding_lap_pos_enc(X.float()) 
        #X = X + X_lap_pos_enc
        #print(X.shape)

        newX, newE, new_y = self.self_attn(X, E, y, node_mask=node_mask)
        
        
        newX_d = self.dropoutX1(newX)
        X = self.normX1(X + newX_d)
        #X,_=self.RNN(X)

        newE_d = self.dropoutE1(newE)
        E = self.normE1(E + newE_d)
        

        new_y_d = self.dropout_y1(new_y)
        y = self.norm_y1(y + new_y_d)

        ff_outputX = self.linX2(self.dropoutX2(self.activation(self.linX1(X))))
        ff_outputX = self.dropoutX3(ff_outputX)
        X = self.normX2(X + ff_outputX)

        ff_outputE = self.linE2(self.dropoutE2(self.activation(self.linE1(E))))
        ff_outputE = self.dropoutE3(ff_outputE)
        E = self.normE2(E + ff_outputE)

        ff_output_y = self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))
        ff_output_y = self.dropout_y3(ff_output_y)
        y = self.norm_y2(y + ff_output_y)

        return X, E, y


class NodeEdgeBlock(nn.Module):
    """ Self attention layer that also updates the representations on the edges. """
    def __init__(self, dx, de, dy, n_head, **kwargs):
        super().__init__()
        assert dx % n_head == 0, f"dx: {dx} -- nhead: {n_head}"
        self.dx = dx
        self.de = de
        
        self.dy = dy
        self.df = int(dx / n_head)
        self.n_head = n_head

        # Attention
        self.q = Linear(dx, dx)
        self.k = Linear(dx, dx)
        self.v = Linear(dx, dx)

        # FiLM E to X
        self.e_add = Linear(de, dx)
        self.e_mul = Linear(de, dx)

        # FiLM y to E
        self.y_e_mul = Linear(dy, dx)           # Warning: here it's dx and not de
        self.y_e_add = Linear(dy, dx)

        # FiLM y to X
        self.y_x_mul = Linear(dy, dx)
        self.y_x_add = Linear(dy, dx)

        # Process y
        self.y_y = Linear(dy, dy)
        self.x_y = Xtoy(dx, dy)
        self.e_y = Etoy(de, dy)

        # Output layers
        self.x_out = Linear(dx, dx)
        self.e_out = Linear(dx, de)
        self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

    def forward(self, X, E, y, node_mask):
        """
        :param X: bs, n, d        node features
        :param E: bs, n, n, d     edge features
        :param y: bs, dz           global features
        :param node_mask: bs, n
        :return: newX, newE, new_y with the same shape.
        """
        bs, n, _ = X.shape
        x_mask = node_mask.unsqueeze(-1)        # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)           # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)           # bs, 1, n, 1

        # 1. Map X to keys and queries
        Q = self.q(X) * x_mask           # (bs, n, dx)
        K = self.k(X) * x_mask           # (bs, n, dx)
        diffusion_utils.assert_correctly_masked(Q, x_mask)
        # 2. Reshape to (bs, n, n_head, df) with dx = n_head * df

        Q = Q.reshape((Q.size(0), Q.size(1), self.n_head, self.df))
        K = K.reshape((K.size(0), K.size(1), self.n_head, self.df))

        Q = Q.unsqueeze(2)                              # (bs, 1, n, n_head, df)
        K = K.unsqueeze(1)                              # (bs, n, 1, n head, df)

        # Compute unnormalized attentions. Y is (bs, n, n, n_head, df)
        Y = Q * K
        Y = Y / math.sqrt(Y.size(-1))
        diffusion_utils.assert_correctly_masked(Y, (e_mask1 * e_mask2).unsqueeze(-1))

        E1 = self.e_mul(E) * e_mask1 * e_mask2                        # bs, n, n, dx
        E1 = E1.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        E2 = self.e_add(E) * e_mask1 * e_mask2                        # bs, n, n, dx
        E2 = E2.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        # Incorporate edge features to the self attention scores.
        Y = Y * (E1 + 1) + E2                  # (bs, n, n, n_head, df)

        # Incorporate y to E
        newE = Y.flatten(start_dim=3)                      # bs, n, n, dx
        #print(newE.shape)
        #print(self.de)
        ye1 = self.y_e_add(y).unsqueeze(1).unsqueeze(1)  # bs, 1, 1, de
        ye2 = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        #print(ye1.shape,ye2.shape,newE.shape)
        newE = ye1 + (ye2 + 1) * newE

        # Output E
        newE = self.e_out(newE) * e_mask1 * e_mask2      # bs, n, n, de
        diffusion_utils.assert_correctly_masked(newE, e_mask1 * e_mask2)

        # Compute attentions. attn is still (bs, n, n, n_head, df)
        softmax_mask = e_mask2.expand(-1, n, -1, self.n_head)    # bs, 1, n, 1
        attn = masked_softmax(Y, softmax_mask, dim=2)  # bs, n, n, n_head

        V = self.v(X) * x_mask                        # bs, n, dx
        V = V.reshape((V.size(0), V.size(1), self.n_head, self.df))
        V = V.unsqueeze(1)                                     # (bs, 1, n, n_head, df)

        # Compute weighted values
        weighted_V = attn * V
        weighted_V = weighted_V.sum(dim=2)

        # Send output to input dim
        weighted_V = weighted_V.flatten(start_dim=2)            # bs, n, dx

        # Incorporate y to X
        yx1 = self.y_x_add(y).unsqueeze(1)
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = yx1 + (yx2 + 1) * weighted_V

        # Output X
        newX = self.x_out(newX) * x_mask
        diffusion_utils.assert_correctly_masked(newX, x_mask)

        # Process y based on X axnd E
        y = self.y_y(y)
        e_y = self.e_y(E)
        x_y = self.x_y(X)
        new_y = y + x_y + e_y
        new_y = self.y_out(new_y)               # bs, dy

        return newX, newE, new_y


class GraphTransformer(nn.Module):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict, hidden_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU()):
        super().__init__()
        self.n_layers = n_layers
        self.out_dim_X = output_dims['X']
        self.out_dim_E = output_dims['E']
        self.out_dim_y = output_dims['y']
        self.hidden_dimsX=hidden_dims['dx']
        self.de=hidden_dims['de']
        self.in2hidden = nn.Linear(hidden_dims['dx'], hidden_dims['dx'])
        self.in2output = nn.Linear(hidden_dims['dx'], hidden_dims['dx'])
        self.Y_dims=(input_dims['y'],hidden_mlp_dims['y'],hidden_dims['dy'],output_dims['y'])
        self.X_dims=(input_dims['X'],hidden_mlp_dims['X'],hidden_dims['dx'],output_dims['X'])
        self.E_dims=(input_dims['E'],hidden_mlp_dims['E'],hidden_dims['de'],output_dims['E'])

        self.mlp_in_X = nn.Sequential(nn.Linear(input_dims['X'], hidden_mlp_dims['X']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['X'], hidden_dims['dx']), act_fn_in)

        self.mlp_in_E = nn.Sequential(nn.Linear(input_dims['E'], hidden_mlp_dims['E']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['E'], hidden_dims['de']), act_fn_in)

        self.mlp_in_y = nn.Sequential(nn.Linear(input_dims['y'], hidden_mlp_dims['y']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['y'], hidden_dims['dy']), act_fn_in)
        self.LSTM= nn.LSTM(hidden_dims['dx'], hidden_dims['dx'],num_layers=2,bidirectional=True)
        self.RNN= nn.RNN(hidden_dims['dx'], hidden_dims['dx'],num_layers=2,bidirectional=True)
        self.Linear=nn.Linear(self.hidden_dimsX * 2, self.hidden_dimsX)
        
        #self.LSTM=nn.Sequential(nn.LSTM(hidden_dims['dx'], hidden_dims['dx'],num_layers=2,bidirectional=True),
                               #nn.Linear(self.hidden_dimsX * 2, self.hidden_dimsX),act_fn_in)


        self.tf_layers = nn.ModuleList([XEyTransformerLayer(dx=2*hidden_dims['dx'],
                                                            de=2*hidden_dims['de'],
                                                            dy=hidden_dims['dy'],
                                                            n_head=hidden_dims['n_head'],
                                                            dim_ffX=hidden_dims['dim_ffX'],
                                                            dim_ffE=hidden_dims['dim_ffE'])
                                        for i in range(n_layers)])
        #self.h = nn.Embedding(len(X), 2)

        self.mlp_out_X = nn.Sequential(nn.Linear(2*hidden_dims['dx'], hidden_mlp_dims['X']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['X'], output_dims['X']))

        self.mlp_out_E = nn.Sequential(nn.Linear(2*hidden_dims['de'], hidden_mlp_dims['E']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['E'], output_dims['E']))

        self.mlp_out_y = nn.Sequential(nn.Linear(hidden_dims['dy'], hidden_mlp_dims['y']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['y'], output_dims['y']))
        
        self.linear_map_class_X = nn.Sequential(
            nn.Linear(input_dims['y'], hidden_dims['dx']),
            nn.ReLU(),
            nn.Linear(hidden_dims['dx'], int(hidden_dims['dx'])),


        )
        self.linear_map_class_E = nn.Sequential(
            nn.Linear(input_dims['y'], hidden_dims['de']),
            nn.ReLU(),
            nn.Linear(hidden_dims['de'], int(hidden_dims['de'])),

        )
        #self.linear_map_class_X_lat = nn.Sequential(
         #   nn.Linear(input_dims['y'], hidden_dims['dx']),
          #  nn.ReLU(),
           # nn.Linear(hidden_dims['dx'], int(hidden_dims['dx'])),


        #)
        #self.linear_map_class_E_lat = nn.Sequential(
         #   nn.Linear(input_dims['y'], hidden_dims['de']),
          #  nn.ReLU(),
           # nn.Linear(hidden_dims['de'], int(hidden_dims['de'])),

        #)

    def forward(self, X, E, y, node_mask):
        bs, n = X.shape[0], X.shape[1]
        #print(self.de)
        #print(E.shape)
        #print(self.Y_dims)
        #print(self.X_dims)
        #print(self.E_dims)
        #print('YSHAPE',y.shape)
        #if (y.shape[-1]>1):
         #   y=y[:,-1]
          #  y=torch.unsqueeze(y,dim=-1)
           # print(y.shape)
            
        #print(X.shape)
        #y=y.reshape(2,self.Y_dims[0])
        
        #print(X.shape,E.shape,y.shape)

        diag_mask = torch.eye(n)
        diag_mask = ~diag_mask.type_as(E).bool()
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        X_to_out = X[..., :self.out_dim_X]
        E_to_out = E[..., :self.out_dim_E]
        y_to_out = y[..., :self.out_dim_y]
        #y=torch.sigmoid(y)


        new_E = self.mlp_in_E(E)
        new_E = (new_E + new_E.transpose(1, 2)) / 2
        #y_t=y.transpose(0,1)
        
        #Y_t=self.mlp_in_y(y)
        #print('X_t',X_tt.shape)
        #y=torch.unsqueeze(y,dim=-1)
        #print(y.shape)
        #y=torch.unsqueeze(y,dim=-1)
        #print('Y',y)
        
        new_y=self.mlp_in_y(y)

        #print(X.shape,E.shape,y_X.shape,y_E.shape)
        after_in = utils.PlaceHolder(X=self.mlp_in_X(X), E=new_E, y=new_y).mask(node_mask)
        X, E, y_new = after_in.X, after_in.E, after_in.y
        
        X=X+get_timing_signal_1d(X.shape[1],X.shape[2])
        #print(X.shape,E.shape,y.shape)
        #print(y_new.shape)
        

        #X=torch.tensor(X)
        #combined = torch.cat((X, hidden_state), 1)
        #hidden = torch.sigmoid(self.in2hidden(combined))
        #X = self.in2output(combined)
        #X,hidden=self.RNN(X) #For indexing information
        #print(X.shape)
        #h0 = torch.zeros(4,X.size(1),self.hidden_dimsX).requires_grad_().cuda()
        #c0 = torch.zeros(4, X.size(1), self.hidden_dimsX).requires_grad_().cuda()
        #X, (hn, cn) = self.LSTM(X, (h0.detach(), c0.detach()))
        #X,hidden=self.RNN(X,h0.detach())
        #X = self.Linear(X)
        #X=nn.ReLU()(X)
        #X=X.cuda()
        #E=E.cuda()
        #y=y.cuda()
        #node_mask=node_mask.cuda()
        
        ##y_acc=y[:,:,0].reshape((8,2))
        ##y_lat=y[:,:,1].reshape((8,2))
        
        ##y_X_acc = self.linear_map_class_X(y_acc)
        ##y_E_acc = self.linear_map_class_E(y_acc)
        ##y_X_lat = self.linear_map_class_X_lat(y_lat)
        ##y_E_lat = self.linear_map_class_E_lat(y_lat)
        #print('Y_X',y_X.shape)
        #print('Y_E',y_E.shape)
        #print(y.shape)
        ##y_X_lat = y_X_lat.view(int(y_X_lat.shape[0]),1,y_X_lat.shape[1])

        ##y_X_lat = y_X_lat.expand(-1, X.shape[1], -1)
        #print(y_X.shape)
        ##y_E_lat = y_E_lat.view(int(y_E_lat.shape[0]),1,1,y_E_lat.shape[1])

        ##y_E_lat = y_E_lat.expand(-1, E.shape[1],E.shape[2], -1)
        y_X = self.linear_map_class_X(y)
        y_E= self.linear_map_class_E(y)
        y_X = y_X.view(int(y_X.shape[0]),1,y_X.shape[1])

        y_X = y_X.expand(-1, X.shape[1], -1)
        #print(y_X.shape)
        y_E = y_E.view(int(y_E.shape[0]),1,1,y_E.shape[1])

        y_E = y_E.expand(-1, E.shape[1],E.shape[2], -1)
        
        ##y_X_acc = y_X_acc.view(int(y_X_acc.shape[0]),1,y_X_acc.shape[1])

        ##y_X_acc = y_X_acc.expand(-1, X.shape[1], -1)
        #print(y_X.shape)
        ##y_E_acc = y_E_acc.view(int(y_E_acc.shape[0]),1,1,y_E_acc.shape[1])

        ##y_E_acc = y_E_acc.expand(-1, E.shape[1],E.shape[2], -1)
        #print(y_E.shape)

        #print('Y_X',y_X)
        #print('Y_E',y_E)
        
        #y_X=y_X.unsqueeze(1).repeat(1, X.shape[2])
        #y_E=y_E.unsqueeze(1).unsqueeze(-1).repeat(1,E.shape[1], 1, E.shape[3])
        
        #print(X.shape,E.shape,y_X.shape, y_E.shape)
        #y_E=y_E.reshape(E.shape[0], E.shape[2], E.shape[1], y.shape[1], E.shape[3])
        
        #print(X.shape)
        #print(E.shape)
        
        #X=X+y_X
        #E=E+y_E
        
        #print(X.shape)
        #print(E.shape)
        #print(y_X_lat.shape)
        #print(y_E_lat.shape)
        #print(y_X_acc.shape)
        #print(y_E_acc.shape)
        ##y_X=torch.cat([y_X_acc,y_X_lat],dim=-1)
        #print(y_X.shape)
        ##y_E=torch.cat([y_E_acc,y_E_lat],dim=-1)
        

        #print(y_X.shape,y_E.shape)
        
        
        
        X=torch.cat([X,y_X],dim=-1)
        E=torch.cat([E,y_E],dim=-1)
        #print(X.shape,E.shape)
        #print(X,E)

        for layer in self.tf_layers:
            X, E, y = layer(X, E, y_new, node_mask)

        X = self.mlp_out_X(X)
        E = self.mlp_out_E(E)
        #print(y.shape)
        y = self.mlp_out_y(y)

        X = (X + X_to_out)
        E = (E + E_to_out) * diag_mask
        y = y + y_to_out

        E = 1/2 * (E + torch.transpose(E, 1, 2))
        
        #print(y.shape)

        return utils.PlaceHolder(X=X, E=E, y=y).mask(node_mask)
    
    
'''class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(RNNModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # RNN layers
        self.rnn = nn.RNN(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, h0 = self.rnn(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)
        return out
        
        
class RNNTransformer(nn.Module):
    def __init__(self, RNN, Transformer):
        super(RNNTransformer, self).__init__()
        self.RNN = RNN
        self.Transformer = Transformer
        
    def forward(self, X,E,y,node_mask):
        E = self.RNN(E)
        print(E.shape)
        print(hidden.shape)
        out = self.Transformer(X,E,y,node_mask)
        return out '''

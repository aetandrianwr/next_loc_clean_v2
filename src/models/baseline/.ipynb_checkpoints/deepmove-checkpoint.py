# coding: utf-8
"""
DeepMove: Predicting Human Mobility with Attentional Recurrent Networks

This module implements the DeepMove model as described in the WWW'18 paper:
"DeepMove: Predicting Human Mobility with Attentional Recurrent Networks"
by Jie Feng, Yong Li, Chao Zhang, Funing Sun, Fanchao Meng, Ang Guo, Depeng Jin

Paper: https://dl.acm.org/doi/10.1145/3178876.3186058

The key contributions of DeepMove:
1. Multi-modal embedding: Location + Time embeddings
2. Attentional RNN with historical attention mechanism
3. User embedding for personalized prediction

Architecture variants implemented:
- TrajPreSimple: Basic RNN model without attention
- TrajPreAttnAvgLongUser: RNN with long-term history attention and user embedding
- TrajPreLocalAttnLong: RNN with local attention on long history

Usage:
    from src.models.baseline.deepmove import DeepMoveModel
    
    model = DeepMoveModel(config)
    scores = model(loc, tim, history_loc, history_tim, history_count, uid, target_len)
"""

from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Attn(nn.Module):
    """
    Attention Module for computing attention weights over history.
    
    Supports three attention methods:
    - dot: Simple dot product attention
    - general: Linear transformation then dot product
    - concat: Concatenate and feed through linear layer
    
    Based on Practical PyTorch seq2seq implementation.
    
    Args:
        method (str): Attention method ('dot', 'general', 'concat')
        hidden_size (int): Hidden dimension size
    """

    def __init__(self, method, hidden_size, use_cuda=True):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size
        self.use_cuda = use_cuda

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(self.hidden_size))

    def forward(self, out_state, history):
        """
        Compute attention weights over history.
        
        Args:
            out_state: Current decoder state [state_len, hidden_size]
            history: Historical states [seq_len, hidden_size]
        
        Returns:
            Attention weights [state_len, seq_len]
        """
        seq_len = history.size()[0]
        state_len = out_state.size()[0]
        
        if self.use_cuda:
            attn_energies = Variable(torch.zeros(state_len, seq_len)).cuda()
        else:
            attn_energies = Variable(torch.zeros(state_len, seq_len))
            
        for i in range(state_len):
            for j in range(seq_len):
                attn_energies[i, j] = self.score(out_state[i], history[j])
        return F.softmax(attn_energies, dim=-1)

    def score(self, hidden, encoder_output):
        """Compute attention score between hidden state and encoder output."""
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output)))
            energy = self.other.dot(energy)
            return energy


class TrajPreSimple(nn.Module):
    """
    Basic RNN trajectory prediction model (baseline).
    
    Simple architecture without attention:
    1. Location embedding
    2. Time embedding  
    3. RNN encoder (GRU/LSTM/RNN)
    4. Linear output layer
    
    Args:
        parameters: Config object containing model hyperparameters
    """

    def __init__(self, parameters):
        super(TrajPreSimple, self).__init__()
        self.loc_size = parameters.loc_size
        self.loc_emb_size = parameters.loc_emb_size
        self.tim_size = parameters.tim_size
        self.tim_emb_size = parameters.tim_emb_size
        self.hidden_size = parameters.hidden_size
        self.use_cuda = parameters.use_cuda
        self.rnn_type = parameters.rnn_type

        self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)
        self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)

        input_size = self.loc_emb_size + self.tim_emb_size

        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size, self.hidden_size, 1)
        self.init_weights()

        self.fc = nn.Linear(self.hidden_size, self.loc_size)
        self.dropout = nn.Dropout(p=parameters.dropout_p)

    def init_weights(self):
        """Initialize weights following Keras defaults for reproducibility."""
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)

    def forward(self, loc, tim):
        h1 = Variable(torch.zeros(1, 1, self.hidden_size))
        c1 = Variable(torch.zeros(1, 1, self.hidden_size))
        if self.use_cuda:
            h1 = h1.cuda()
            c1 = c1.cuda()

        loc_emb = self.emb_loc(loc)
        tim_emb = self.emb_tim(tim)
        x = torch.cat((loc_emb, tim_emb), 2)
        x = self.dropout(x)

        if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
            out, h1 = self.rnn(x, h1)
        elif self.rnn_type == 'LSTM':
            out, (h1, c1) = self.rnn(x, (h1, c1))
        out = out.squeeze(1)
        out = F.selu(out)
        out = self.dropout(out)

        y = self.fc(out)
        score = F.log_softmax(y, dim=-1)
        return score


class TrajPreAttnAvgLongUser(nn.Module):
    """
    DeepMove main model: RNN with long-term history attention and user embedding.
    
    This is the primary DeepMove architecture featuring:
    1. Location + Time embeddings
    2. User embedding for personalization
    3. RNN encoder for current trajectory
    4. Attention over averaged historical embeddings
    5. Context vector combined with RNN output
    
    The model captures both recurrent patterns (via RNN) and long-term
    mobility patterns (via attention over history).
    
    Args:
        parameters: Config object with hyperparameters:
            - loc_size: Number of unique locations
            - loc_emb_size: Location embedding dimension
            - tim_size: Number of time slots
            - tim_emb_size: Time embedding dimension
            - uid_size: Number of users
            - uid_emb_size: User embedding dimension
            - hidden_size: RNN hidden dimension
            - attn_type: Attention type ('dot', 'general', 'concat')
            - rnn_type: RNN type ('GRU', 'LSTM', 'RNN')
            - use_cuda: Whether to use GPU
            - dropout_p: Dropout probability
    """

    def __init__(self, parameters):
        super(TrajPreAttnAvgLongUser, self).__init__()
        self.loc_size = parameters.loc_size
        self.loc_emb_size = parameters.loc_emb_size
        self.tim_size = parameters.tim_size
        self.tim_emb_size = parameters.tim_emb_size
        self.uid_size = parameters.uid_size
        self.uid_emb_size = parameters.uid_emb_size
        self.hidden_size = parameters.hidden_size
        self.attn_type = parameters.attn_type
        self.rnn_type = parameters.rnn_type
        self.use_cuda = parameters.use_cuda

        # Embeddings
        self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)
        self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)
        self.emb_uid = nn.Embedding(self.uid_size, self.uid_emb_size)

        input_size = self.loc_emb_size + self.tim_emb_size
        
        # Attention mechanism
        self.attn = Attn(self.attn_type, self.hidden_size, self.use_cuda)
        self.fc_attn = nn.Linear(input_size, self.hidden_size)

        # RNN encoder
        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size, self.hidden_size, 1)

        # Output layer: combines RNN output (hidden), context, and user embedding
        self.fc_final = nn.Linear(2 * self.hidden_size + self.uid_emb_size, self.loc_size)
        self.dropout = nn.Dropout(p=parameters.dropout_p)
        self.init_weights()

    def init_weights(self):
        """Initialize weights following Keras defaults."""
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)

    def forward(self, loc, tim, history_loc, history_tim, history_count, uid, target_len):
        """
        Forward pass with history attention.
        
        Args:
            loc: Current trajectory locations [seq_len, 1]
            tim: Current trajectory times [seq_len, 1]
            history_loc: Historical locations [history_len, 1]
            history_tim: Historical times [history_len, 1]
            history_count: Count of locations per time slot for averaging
            uid: User ID tensor
            target_len: Number of predictions to make
        
        Returns:
            Log softmax scores [target_len, loc_size]
        """
        h1 = Variable(torch.zeros(1, 1, self.hidden_size))
        c1 = Variable(torch.zeros(1, 1, self.hidden_size))
        if self.use_cuda:
            h1 = h1.cuda()
            c1 = c1.cuda()

        # Current trajectory embedding
        loc_emb = self.emb_loc(loc)
        tim_emb = self.emb_tim(tim)
        x = torch.cat((loc_emb, tim_emb), 2)
        x = self.dropout(x)

        # Historical embeddings with time-based averaging
        loc_emb_history = self.emb_loc(history_loc).squeeze(1)
        tim_emb_history = self.emb_tim(history_tim).squeeze(1)
        
        # Average embeddings within same time slot
        if self.use_cuda:
            loc_emb_history2 = Variable(torch.zeros(len(history_count), loc_emb_history.size()[-1])).cuda()
            tim_emb_history2 = Variable(torch.zeros(len(history_count), tim_emb_history.size()[-1])).cuda()
        else:
            loc_emb_history2 = Variable(torch.zeros(len(history_count), loc_emb_history.size()[-1]))
            tim_emb_history2 = Variable(torch.zeros(len(history_count), tim_emb_history.size()[-1]))
            
        count = 0
        for i, c in enumerate(history_count):
            if c == 1:
                tmp = loc_emb_history[count].unsqueeze(0)
            else:
                tmp = torch.mean(loc_emb_history[count:count + c, :], dim=0, keepdim=True)
            loc_emb_history2[i, :] = tmp
            tim_emb_history2[i, :] = tim_emb_history[count, :].unsqueeze(0)
            count += c

        # Project history to attention space
        history = torch.cat((loc_emb_history2, tim_emb_history2), 1)
        history = torch.tanh(self.fc_attn(history))

        # RNN encoding
        if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
            out_state, h1 = self.rnn(x, h1)
        elif self.rnn_type == 'LSTM':
            out_state, (h1, c1) = self.rnn(x, (h1, c1))
        out_state = out_state.squeeze(1)

        # Attention over history
        attn_weights = self.attn(out_state[-target_len:], history).unsqueeze(0)
        context = attn_weights.bmm(history.unsqueeze(0)).squeeze(0)
        
        # Combine RNN output with context
        out = torch.cat((out_state[-target_len:], context), 1)

        # Add user embedding
        uid_emb = self.emb_uid(uid).repeat(target_len, 1)
        out = torch.cat((out, uid_emb), 1)
        out = self.dropout(out)

        # Final prediction
        y = self.fc_final(out)
        score = F.log_softmax(y, dim=-1)

        return score


class TrajPreLocalAttnLong(nn.Module):
    """
    RNN with local attention over long history.
    
    Uses separate encoder and decoder RNNs:
    - Encoder: Processes historical trajectory
    - Decoder: Processes current trajectory
    - Attention: Decoder attends to encoder outputs
    
    Args:
        parameters: Config object with hyperparameters
    """

    def __init__(self, parameters):
        super(TrajPreLocalAttnLong, self).__init__()
        self.loc_size = parameters.loc_size
        self.loc_emb_size = parameters.loc_emb_size
        self.tim_size = parameters.tim_size
        self.tim_emb_size = parameters.tim_emb_size
        self.hidden_size = parameters.hidden_size
        self.attn_type = parameters.attn_type
        self.use_cuda = parameters.use_cuda
        self.rnn_type = parameters.rnn_type

        self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)
        self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)

        input_size = self.loc_emb_size + self.tim_emb_size
        self.attn = Attn(self.attn_type, self.hidden_size, self.use_cuda)
        self.fc_attn = nn.Linear(input_size, self.hidden_size)

        if self.rnn_type == 'GRU':
            self.rnn_encoder = nn.GRU(input_size, self.hidden_size, 1)
            self.rnn_decoder = nn.GRU(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'LSTM':
            self.rnn_encoder = nn.LSTM(input_size, self.hidden_size, 1)
            self.rnn_decoder = nn.LSTM(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'RNN':
            self.rnn_encoder = nn.RNN(input_size, self.hidden_size, 1)
            self.rnn_decoder = nn.LSTM(input_size, self.hidden_size, 1)

        self.fc_final = nn.Linear(2 * self.hidden_size, self.loc_size)
        self.dropout = nn.Dropout(p=parameters.dropout_p)
        self.init_weights()

    def init_weights(self):
        """Initialize weights following Keras defaults."""
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)

    def forward(self, loc, tim, target_len):
        h1 = Variable(torch.zeros(1, 1, self.hidden_size))
        h2 = Variable(torch.zeros(1, 1, self.hidden_size))
        c1 = Variable(torch.zeros(1, 1, self.hidden_size))
        c2 = Variable(torch.zeros(1, 1, self.hidden_size))
        if self.use_cuda:
            h1 = h1.cuda()
            h2 = h2.cuda()
            c1 = c1.cuda()
            c2 = c2.cuda()

        loc_emb = self.emb_loc(loc)
        tim_emb = self.emb_tim(tim)
        x = torch.cat((loc_emb, tim_emb), 2)
        x = self.dropout(x)

        if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
            hidden_history, h1 = self.rnn_encoder(x[:-target_len], h1)
            hidden_state, h2 = self.rnn_decoder(x[-target_len:], h2)
        elif self.rnn_type == 'LSTM':
            hidden_history, (h1, c1) = self.rnn_encoder(x[:-target_len], (h1, c1))
            hidden_state, (h2, c2) = self.rnn_decoder(x[-target_len:], (h2, c2))

        hidden_history = hidden_history.squeeze(1)
        hidden_state = hidden_state.squeeze(1)
        attn_weights = self.attn(hidden_state, hidden_history).unsqueeze(0)
        context = attn_weights.bmm(hidden_history.unsqueeze(0)).squeeze(0)
        out = torch.cat((hidden_state, context), 1)
        out = self.dropout(out)

        y = self.fc_final(out)
        score = F.log_softmax(y, dim=-1)

        return score


class DeepMoveModel(nn.Module):
    """
    Unified DeepMove model wrapper for training with the next_loc_clean_v2 data format.
    
    This wrapper adapts the original DeepMove implementation to work with the
    preprocessed data format used in next_loc_clean_v2 project.
    
    The model supports the main DeepMove variant (attn_avg_long_user) which includes:
    - Location and time embeddings
    - User embedding for personalization
    - RNN encoder for sequential patterns
    - Historical attention mechanism for long-term patterns
    
    Args:
        config: Configuration object containing:
            - loc_size: Number of unique locations
            - loc_emb_size: Location embedding dimension
            - tim_size: Number of time slots (48 for half-hour slots)
            - tim_emb_size: Time embedding dimension
            - uid_size: Number of users
            - uid_emb_size: User embedding dimension
            - hidden_size: RNN hidden dimension
            - attn_type: Attention type ('dot', 'general', 'concat')
            - rnn_type: RNN type ('GRU', 'LSTM')
            - dropout_p: Dropout probability
            - model_mode: Model variant ('attn_avg_long_user', 'simple', etc.)
            - use_cuda: Whether to use GPU
    """
    
    def __init__(self, config):
        super(DeepMoveModel, self).__init__()
        
        self.config = config
        self.model_mode = getattr(config, 'model_mode', 'attn_avg_long_user')
        self.use_cuda = getattr(config, 'use_cuda', torch.cuda.is_available())
        
        # Create the appropriate model variant
        if self.model_mode == 'simple':
            self.model = TrajPreSimple(config)
        elif self.model_mode == 'attn_avg_long_user':
            self.model = TrajPreAttnAvgLongUser(config)
        elif self.model_mode == 'attn_local_long':
            self.model = TrajPreLocalAttnLong(config)
        else:
            # Default to attn_avg_long_user
            self.model = TrajPreAttnAvgLongUser(config)
    
    def forward(self, loc, tim, history_loc, history_tim, history_count, uid, target_len):
        """
        Forward pass.
        
        Args:
            loc: Current trajectory locations [seq_len, 1]
            tim: Current trajectory times [seq_len, 1]
            history_loc: Historical locations [history_len, 1]
            history_tim: Historical times [history_len, 1]
            history_count: Count of locations per time slot
            uid: User ID tensor
            target_len: Number of predictions to make
        
        Returns:
            Log softmax scores [target_len, loc_size]
        """
        if self.model_mode == 'simple':
            return self.model(loc, tim)
        elif self.model_mode == 'attn_local_long':
            return self.model(loc, tim, target_len)
        else:
            return self.model(loc, tim, history_loc, history_tim, history_count, uid, target_len)
    
    def get_num_parameters(self):
        """Get the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_deepmove_config(loc_size, uid_size, loc_emb_size=500, tim_emb_size=10,
                          uid_emb_size=40, hidden_size=500, dropout_p=0.3,
                          rnn_type='GRU', attn_type='dot', model_mode='attn_avg_long_user',
                          use_cuda=True, tim_size=48):
    """
    Create a configuration object for DeepMove model.
    
    Args:
        loc_size: Number of unique locations
        uid_size: Number of unique users
        loc_emb_size: Location embedding dimension (default: 500)
        tim_emb_size: Time embedding dimension (default: 10)
        uid_emb_size: User embedding dimension (default: 40)
        hidden_size: RNN hidden dimension (default: 500)
        dropout_p: Dropout probability (default: 0.3)
        rnn_type: RNN type ('GRU', 'LSTM', 'RNN') (default: 'GRU')
        attn_type: Attention type ('dot', 'general', 'concat') (default: 'dot')
        model_mode: Model variant (default: 'attn_avg_long_user')
        use_cuda: Whether to use GPU (default: True)
        tim_size: Number of time slots (default: 48 for half-hour)
    
    Returns:
        Configuration object (SimpleNamespace)
    """
    from types import SimpleNamespace
    
    config = SimpleNamespace(
        loc_size=loc_size,
        loc_emb_size=loc_emb_size,
        tim_size=tim_size,
        tim_emb_size=tim_emb_size,
        uid_size=uid_size,
        uid_emb_size=uid_emb_size,
        hidden_size=hidden_size,
        dropout_p=dropout_p,
        rnn_type=rnn_type,
        attn_type=attn_type,
        model_mode=model_mode,
        use_cuda=use_cuda
    )
    
    return config

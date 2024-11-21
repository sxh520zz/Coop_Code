import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import HubertModel


class MatchingAttention(nn.Module):

    def __init__(self, mem_dim, cand_dim, alpha_dim=None, att_type='general'):
        super(MatchingAttention, self).__init__()
        assert att_type!='concat' or alpha_dim!=None
        assert att_type!='dot' or mem_dim==cand_dim
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        if att_type=='general':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=False)
        if att_type=='general2':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=True)
            #torch.nn.init.normal_(self.transform.weight,std=0.01)
        elif att_type=='concat':
            self.transform = nn.Linear(cand_dim+mem_dim, alpha_dim, bias=False)
            self.vector_prod = nn.Linear(alpha_dim, 1, bias=False)

    def forward(self, M, x, mask=None):
        """
        M -> (seq_len, batch, mem_dim)
        x -> (batch, cand_dim)
        mask -> (batch, seq_len)
        """
        if type(mask)==type(None):
            mask = torch.ones(M.size(1), M.size(0)).type(M.type())

        if self.att_type=='dot':
            # vector = cand_dim = mem_dim
            M_ = M.permute(1,2,0) # batch, vector, seqlen
            x_ = x.unsqueeze(1) # batch, 1, vector
            alpha = F.softmax(torch.bmm(x_, M_), dim=2) # batch, 1, seqlen
        elif self.att_type=='general':
            M_ = M.permute(1,2,0) # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1) # batch, 1, mem_dim
            alpha = F.softmax(torch.bmm(x_, M_), dim=2) # batch, 1, seqlen
        elif self.att_type=='general2':
            M_ = M.permute(1,2,0) # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1) # batch, 1, mem_dim
            mask_ = mask.unsqueeze(2).repeat(1, 1, self.mem_dim).transpose(1, 2) # batch, seq_len, mem_dim
            M_ = M_ * mask_
            alpha_ = torch.bmm(x_, M_)*mask.unsqueeze(1)
            alpha_ = torch.tanh(alpha_)
            alpha_ = F.softmax(alpha_, dim=2)
            # alpha_ = F.softmax((torch.bmm(x_, M_))*mask.unsqueeze(1), dim=2) # batch, 1, seqlen
            alpha_masked = alpha_*mask.unsqueeze(1) # batch, 1, seqlen
            alpha_sum = torch.sum(alpha_masked, dim=2, keepdim=True) # batch, 1, 1
            alpha = alpha_masked/alpha_sum # batch, 1, 1 ; normalized
            #import ipdb;ipdb.set_trace()
        else:
            M_ = M.transpose(0,1) # batch, seqlen, mem_dim
            x_ = x.unsqueeze(1).expand(-1,M.size()[0],-1) # batch, seqlen, cand_dim
            M_x_ = torch.cat([M_,x_],2) # batch, seqlen, mem_dim+cand_dim
            mx_a = F.tanh(self.transform(M_x_)) # batch, seqlen, alpha_dim
            alpha = F.softmax(self.vector_prod(mx_a),1).transpose(1,2) # batch, 1, seqlen

        attn_pool = torch.bmm(alpha, M.transpose(0,1))[:,0,:] # batch, mem_dim
        return attn_pool, alpha

class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()
        self.input_dim = 17
        self.query = nn.Linear(17, 17)
        self.key = nn.Linear(17, 17)
        self.value = nn.Linear(17, 17)

    def forward(self, x):
        # Calculate Query, Key, and Value
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        X = torch.tensor(self.input_dim)
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(1, 0)) / torch.sqrt(X)

        # Compute attention weights using softmax
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply attention weights to the value
        attended_value = torch.matmul(attn_weights, V)

        return attended_value
    
class GRUModel(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,args):
        super(GRUModel, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.self_attention = SelfAttention()
        # linear
        self.hidden2label = nn.Linear(17, output_size)
        self.relu = nn.LeakyReLU()

    def forward(self, U):
        #attended_value = self.self_attention(U)
        U = self.dropout(U)
        Out_in = self.relu(U)
        Out_in = self.dropout(Out_in)
        Out_out = self.hidden2label(Out_in)
        return Out_out

class GRUModel_Mid_1(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,args):
        super(GRUModel_Mid_1, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.attention = args.attention
        self.num_layers = args.dia_layers
        self.bigru = nn.GRU(input_size, hidden_size,batch_first=True, num_layers=self.num_layers, bidirectional=True)
        if self.attention:
            self.matchatt = MatchingAttention(2 * hidden_size, 2 * hidden_size, att_type='dot')

        # linear
        self.input2hidden = nn.Linear(512, hidden_size * 2)
        self.hidden2label = nn.Linear(hidden_size * 2, output_size)
        self.relu = nn.LeakyReLU()

    def forward(self, U):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        U = self.dropout(U)
        emotions, hidden = self.bigru(U)
        if self.attention:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions, t, mask=None)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:, 0, :])
            att_emotions = torch.cat(att_emotions, dim=0)
            emotions = att_emotions
            #hidden = F.relu(self.linear(att_emotions))
        else:
            #hidden = F.relu(self.linear(emotions))
            emotions = emotions
        gru_out = torch.transpose(emotions, 1, 2)
        gru_out = F.tanh(gru_out)
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        Out_in = self.relu(gru_out)
        Out_in = self.dropout(Out_in)
        Out_out = self.hidden2label(Out_in)
        return Out_out
class GRUModel_Mid_2(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,args):
        super(GRUModel_Mid_2, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.attention = args.attention
        self.num_layers = args.dia_layers
        self.bigru = nn.GRU(input_size, hidden_size,batch_first=True, num_layers=self.num_layers, bidirectional=True)
        if self.attention:
            self.matchatt = MatchingAttention(2 * hidden_size, 2 * hidden_size, att_type='dot')

        # linear
        self.input2hidden = nn.Linear(512, hidden_size * 2)
        self.hidden2label = nn.Linear(hidden_size * 2, output_size)
        self.relu = nn.LeakyReLU()

    def forward(self, U):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        U = self.dropout(U)
        emotions, hidden = self.bigru(U)
        if self.attention:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions, t, mask=None)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:, 0, :])
            att_emotions = torch.cat(att_emotions, dim=0)
            emotions = att_emotions
            #hidden = F.relu(self.linear(att_emotions))
        else:
            #hidden = F.relu(self.linear(emotions))
            emotions = emotions
        gru_out = torch.transpose(emotions, 1, 2)
        gru_out = F.tanh(gru_out)
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        Out_in = self.relu(gru_out)
        Out_in = self.dropout(Out_in)
        Out_out = self.hidden2label(Out_in)
        return Out_out
class GRUModel_Mid_3(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,args):
        super(GRUModel_Mid_3, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.attention = args.attention
        self.num_layers = args.dia_layers
        self.bigru = nn.GRU(input_size, hidden_size,batch_first=True, num_layers=self.num_layers, bidirectional=True)
        if self.attention:
            self.matchatt = MatchingAttention(2 * hidden_size, 2 * hidden_size, att_type='dot')

        # linear
        self.input2hidden = nn.Linear(512, hidden_size * 2)
        self.hidden2label = nn.Linear(hidden_size * 2, output_size)
        self.relu = nn.LeakyReLU()

    def forward(self, U):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        U = self.dropout(U)
        emotions, hidden = self.bigru(U)
        if self.attention:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions, t, mask=None)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:, 0, :])
            att_emotions = torch.cat(att_emotions, dim=0)
            emotions = att_emotions
            #hidden = F.relu(self.linear(att_emotions))
        else:
            #hidden = F.relu(self.linear(emotions))
            emotions = emotions
        gru_out = torch.transpose(emotions, 1, 2)
        gru_out = F.tanh(gru_out)
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        Out_in = self.relu(gru_out)
        Out_in = self.dropout(Out_in)
        Out_out = self.hidden2label(Out_in)
        return Out_out
class GRUModel_Mid_4(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,args):
        super(GRUModel_Mid_4, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.attention = args.attention
        self.num_layers = args.dia_layers
        self.bigru = nn.GRU(input_size, hidden_size,batch_first=True, num_layers=self.num_layers, bidirectional=True)
        if self.attention:
            self.matchatt = MatchingAttention(2 * hidden_size, 2 * hidden_size, att_type='dot')

        # linear
        self.input2hidden = nn.Linear(512, hidden_size * 2)
        self.hidden2label = nn.Linear(hidden_size * 2, output_size)
        self.relu = nn.LeakyReLU()

    def forward(self, U):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        U = self.dropout(U)
        emotions, hidden = self.bigru(U)
        if self.attention:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions, t, mask=None)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:, 0, :])
            att_emotions = torch.cat(att_emotions, dim=0)
            emotions = att_emotions
            #hidden = F.relu(self.linear(att_emotions))
        else:
            #hidden = F.relu(self.linear(emotions))
            emotions = emotions
        gru_out = torch.transpose(emotions, 1, 2)
        gru_out = F.tanh(gru_out)
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        Out_in = self.relu(gru_out)
        Out_in = self.dropout(Out_in)
        Out_out = self.hidden2label(Out_in)
        return Out_out
class GRUModel_Mid_5(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,args):
        super(GRUModel_Mid_5, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.attention = args.attention
        self.num_layers = args.dia_layers
        self.bigru = nn.GRU(input_size, hidden_size,batch_first=True, num_layers=self.num_layers, bidirectional=True)
        if self.attention:
            self.matchatt = MatchingAttention(2 * hidden_size, 2 * hidden_size, att_type='dot')

        # linear
        self.input2hidden = nn.Linear(512, hidden_size * 2)
        self.hidden2label = nn.Linear(hidden_size * 2, output_size)
        self.relu = nn.LeakyReLU()

    def forward(self, U):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        U = self.dropout(U)
        emotions, hidden = self.bigru(U)
        if self.attention:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions, t, mask=None)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:, 0, :])
            att_emotions = torch.cat(att_emotions, dim=0)
            emotions = att_emotions
            #hidden = F.relu(self.linear(att_emotions))
        else:
            #hidden = F.relu(self.linear(emotions))
            emotions = emotions
        gru_out = torch.transpose(emotions, 1, 2)
        gru_out = F.tanh(gru_out)
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        Out_in = self.relu(gru_out)
        Out_in = self.dropout(Out_in)
        Out_out = self.hidden2label(Out_in)
        return Out_out
class GRUModel_Mid_6(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,args):
        super(GRUModel_Mid_6, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.attention = args.attention
        self.num_layers = args.dia_layers
        self.bigru = nn.GRU(input_size, hidden_size,batch_first=True, num_layers=self.num_layers, bidirectional=True)
        if self.attention:
            self.matchatt = MatchingAttention(2 * hidden_size, 2 * hidden_size, att_type='dot')

        # linear
        self.input2hidden = nn.Linear(512, hidden_size * 2)
        self.hidden2label = nn.Linear(hidden_size * 2, output_size)
        self.relu = nn.LeakyReLU()

    def forward(self, U):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        U = self.dropout(U)
        emotions, hidden = self.bigru(U)
        if self.attention:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions, t, mask=None)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:, 0, :])
            att_emotions = torch.cat(att_emotions, dim=0)
            emotions = att_emotions
            #hidden = F.relu(self.linear(att_emotions))
        else:
            #hidden = F.relu(self.linear(emotions))
            emotions = emotions
        gru_out = torch.transpose(emotions, 1, 2)
        gru_out = F.tanh(gru_out)
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        Out_in = self.relu(gru_out)
        Out_in = self.dropout(Out_in)
        Out_out = self.hidden2label(Out_in)
        return Out_out
class GRUModel_Mid_7(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,args):
        super(GRUModel_Mid_7, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.attention = args.attention
        self.num_layers = args.dia_layers
        self.bigru = nn.GRU(input_size, hidden_size,batch_first=True, num_layers=self.num_layers, bidirectional=True)
        if self.attention:
            self.matchatt = MatchingAttention(2 * hidden_size, 2 * hidden_size, att_type='dot')

        # linear
        self.input2hidden = nn.Linear(512, hidden_size * 2)
        self.hidden2label = nn.Linear(hidden_size * 2, output_size)
        self.relu = nn.LeakyReLU()

    def forward(self, U):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        U = self.dropout(U)
        emotions, hidden = self.bigru(U)
        if self.attention:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions, t, mask=None)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:, 0, :])
            att_emotions = torch.cat(att_emotions, dim=0)
            emotions = att_emotions
            #hidden = F.relu(self.linear(att_emotions))
        else:
            #hidden = F.relu(self.linear(emotions))
            emotions = emotions
        gru_out = torch.transpose(emotions, 1, 2)
        gru_out = F.tanh(gru_out)
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        Out_in = self.relu(gru_out)
        Out_in = self.dropout(Out_in)
        Out_out = self.hidden2label(Out_in)
        return Out_out
class GRUModel_Mid_8(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,args):
        super(GRUModel_Mid_8, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.attention = args.attention
        self.num_layers = args.dia_layers
        self.bigru = nn.GRU(input_size, hidden_size,batch_first=True, num_layers=self.num_layers, bidirectional=True)
        if self.attention:
            self.matchatt = MatchingAttention(2 * hidden_size, 2 * hidden_size, att_type='dot')

        # linear
        self.input2hidden = nn.Linear(512, hidden_size * 2)
        self.hidden2label = nn.Linear(hidden_size * 2, output_size)
        self.relu = nn.LeakyReLU()

    def forward(self, U):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        U = self.dropout(U)
        emotions, hidden = self.bigru(U)
        if self.attention:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions, t, mask=None)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:, 0, :])
            att_emotions = torch.cat(att_emotions, dim=0)
            emotions = att_emotions
            #hidden = F.relu(self.linear(att_emotions))
        else:
            #hidden = F.relu(self.linear(emotions))
            emotions = emotions
        gru_out = torch.transpose(emotions, 1, 2)
        gru_out = F.tanh(gru_out)
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        Out_in = self.relu(gru_out)
        Out_in = self.dropout(Out_in)
        Out_out = self.hidden2label(Out_in)
        return Out_out
class GRUModel_Mid_9(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,args):
        super(GRUModel_Mid_9, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.attention = args.attention
        self.num_layers = args.dia_layers
        self.bigru = nn.GRU(input_size, hidden_size,batch_first=True, num_layers=self.num_layers, bidirectional=True)
        if self.attention:
            self.matchatt = MatchingAttention(2 * hidden_size, 2 * hidden_size, att_type='dot')

        # linear
        self.input2hidden = nn.Linear(512, hidden_size * 2)
        self.hidden2label = nn.Linear(hidden_size * 2, output_size)
        self.relu = nn.LeakyReLU()

    def forward(self, U):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        U = self.dropout(U)
        emotions, hidden = self.bigru(U)
        if self.attention:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions, t, mask=None)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:, 0, :])
            att_emotions = torch.cat(att_emotions, dim=0)
            emotions = att_emotions
            #hidden = F.relu(self.linear(att_emotions))
        else:
            #hidden = F.relu(self.linear(emotions))
            emotions = emotions
        gru_out = torch.transpose(emotions, 1, 2)
        gru_out = F.tanh(gru_out)
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        Out_in = self.relu(gru_out)
        Out_in = self.dropout(Out_in)
        Out_out = self.hidden2label(Out_in)
        return Out_out
class GRUModel_Mid_10(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,args):
        super(GRUModel_Mid_10, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.attention = args.attention
        self.num_layers = args.dia_layers
        self.bigru = nn.GRU(input_size, hidden_size,batch_first=True, num_layers=self.num_layers, bidirectional=True)
        if self.attention:
            self.matchatt = MatchingAttention(2 * hidden_size, 2 * hidden_size, att_type='dot')

        # linear
        self.input2hidden = nn.Linear(512, hidden_size * 2)
        self.hidden2label = nn.Linear(hidden_size * 2, output_size)
        self.relu = nn.LeakyReLU()

    def forward(self, U):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        U = self.dropout(U)
        emotions, hidden = self.bigru(U)
        if self.attention:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions, t, mask=None)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:, 0, :])
            att_emotions = torch.cat(att_emotions, dim=0)
            emotions = att_emotions
            #hidden = F.relu(self.linear(att_emotions))
        else:
            #hidden = F.relu(self.linear(emotions))
            emotions = emotions
        gru_out = torch.transpose(emotions, 1, 2)
        gru_out = F.tanh(gru_out)
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        Out_in = self.relu(gru_out)
        Out_in = self.dropout(Out_in)
        Out_out = self.hidden2label(Out_in)
        return Out_out
class GRUModel_Mid_11(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,args):
        super(GRUModel_Mid_11, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.attention = args.attention
        self.num_layers = args.dia_layers
        self.bigru = nn.GRU(input_size, hidden_size,batch_first=True, num_layers=self.num_layers, bidirectional=True)
        if self.attention:
            self.matchatt = MatchingAttention(2 * hidden_size, 2 * hidden_size, att_type='dot')

        # linear
        self.input2hidden = nn.Linear(512, hidden_size * 2)
        self.hidden2label = nn.Linear(hidden_size * 2, output_size)
        self.relu = nn.LeakyReLU()

    def forward(self, U):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        U = self.dropout(U)
        emotions, hidden = self.bigru(U)
        if self.attention:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions, t, mask=None)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:, 0, :])
            att_emotions = torch.cat(att_emotions, dim=0)
            emotions = att_emotions
            #hidden = F.relu(self.linear(att_emotions))
        else:
            #hidden = F.relu(self.linear(emotions))
            emotions = emotions
        gru_out = torch.transpose(emotions, 1, 2)
        gru_out = F.tanh(gru_out)
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        Out_in = self.relu(gru_out)
        Out_in = self.dropout(Out_in)
        Out_out = self.hidden2label(Out_in)
        return Out_out
class GRUModel_Mid_12(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,args):
        super(GRUModel_Mid_12, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.attention = args.attention
        self.num_layers = args.dia_layers
        self.bigru = nn.GRU(input_size, hidden_size,batch_first=True, num_layers=self.num_layers, bidirectional=True)
        if self.attention:
            self.matchatt = MatchingAttention(2 * hidden_size, 2 * hidden_size, att_type='dot')

        # linear
        self.input2hidden = nn.Linear(512, hidden_size * 2)
        self.hidden2label = nn.Linear(hidden_size * 2, output_size)
        self.relu = nn.LeakyReLU()

    def forward(self, U):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        U = self.dropout(U)
        emotions, hidden = self.bigru(U)
        if self.attention:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions, t, mask=None)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:, 0, :])
            att_emotions = torch.cat(att_emotions, dim=0)
            emotions = att_emotions
            #hidden = F.relu(self.linear(att_emotions))
        else:
            #hidden = F.relu(self.linear(emotions))
            emotions = emotions
        gru_out = torch.transpose(emotions, 1, 2)
        gru_out = F.tanh(gru_out)
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        Out_in = self.relu(gru_out)
        Out_in = self.dropout(Out_in)
        Out_out = self.hidden2label(Out_in)
        return Out_out
class GRUModel_Mid_13(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,args):
        super(GRUModel_Mid_13, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.attention = args.attention
        self.num_layers = args.dia_layers
        self.bigru = nn.GRU(input_size, hidden_size,batch_first=True, num_layers=self.num_layers, bidirectional=True)
        if self.attention:
            self.matchatt = MatchingAttention(2 * hidden_size, 2 * hidden_size, att_type='dot')

        # linear
        self.input2hidden = nn.Linear(512, hidden_size * 2)
        self.hidden2label = nn.Linear(hidden_size * 2, output_size)
        self.relu = nn.LeakyReLU()

    def forward(self, U):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        U = self.dropout(U)
        emotions, hidden = self.bigru(U)
        if self.attention:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions, t, mask=None)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:, 0, :])
            att_emotions = torch.cat(att_emotions, dim=0)
            emotions = att_emotions
            #hidden = F.relu(self.linear(att_emotions))
        else:
            #hidden = F.relu(self.linear(emotions))
            emotions = emotions
        gru_out = torch.transpose(emotions, 1, 2)
        gru_out = F.tanh(gru_out)
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        Out_in = self.relu(gru_out)
        Out_in = self.dropout(Out_in)
        Out_out = self.hidden2label(Out_in)
        return Out_out
class GRUModel_Mid_14(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,args):
        super(GRUModel_Mid_14, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.attention = args.attention
        self.num_layers = args.dia_layers
        self.bigru = nn.GRU(input_size, hidden_size,batch_first=True, num_layers=self.num_layers, bidirectional=True)
        if self.attention:
            self.matchatt = MatchingAttention(2 * hidden_size, 2 * hidden_size, att_type='dot')

        # linear
        self.input2hidden = nn.Linear(512, hidden_size * 2)
        self.hidden2label = nn.Linear(hidden_size * 2, output_size)
        self.relu = nn.LeakyReLU()

    def forward(self, U):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        U = self.dropout(U)
        emotions, hidden = self.bigru(U)
        if self.attention:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions, t, mask=None)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:, 0, :])
            att_emotions = torch.cat(att_emotions, dim=0)
            emotions = att_emotions
            #hidden = F.relu(self.linear(att_emotions))
        else:
            #hidden = F.relu(self.linear(emotions))
            emotions = emotions
        gru_out = torch.transpose(emotions, 1, 2)
        gru_out = F.tanh(gru_out)
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        Out_in = self.relu(gru_out)
        Out_in = self.dropout(Out_in)
        Out_out = self.hidden2label(Out_in)
        return Out_out
class GRUModel_Mid_15(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,args):
        super(GRUModel_Mid_15, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.attention = args.attention
        self.num_layers = args.dia_layers
        self.bigru = nn.GRU(input_size, hidden_size,batch_first=True, num_layers=self.num_layers, bidirectional=True)
        if self.attention:
            self.matchatt = MatchingAttention(2 * hidden_size, 2 * hidden_size, att_type='dot')

        # linear
        self.input2hidden = nn.Linear(512, hidden_size * 2)
        self.hidden2label = nn.Linear(hidden_size * 2, output_size)
        self.relu = nn.LeakyReLU()

    def forward(self, U):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        U = self.dropout(U)
        emotions, hidden = self.bigru(U)
        if self.attention:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions, t, mask=None)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:, 0, :])
            att_emotions = torch.cat(att_emotions, dim=0)
            emotions = att_emotions
            #hidden = F.relu(self.linear(att_emotions))
        else:
            #hidden = F.relu(self.linear(emotions))
            emotions = emotions
        gru_out = torch.transpose(emotions, 1, 2)
        gru_out = F.tanh(gru_out)
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        Out_in = self.relu(gru_out)
        Out_in = self.dropout(Out_in)
        Out_out = self.hidden2label(Out_in)
        return Out_out
class GRUModel_Mid_16(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,args):
        super(GRUModel_Mid_16, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.attention = args.attention
        self.num_layers = args.dia_layers
        self.bigru = nn.GRU(input_size, hidden_size,batch_first=True, num_layers=self.num_layers, bidirectional=True)
        if self.attention:
            self.matchatt = MatchingAttention(2 * hidden_size, 2 * hidden_size, att_type='dot')

        # linear
        self.input2hidden = nn.Linear(512, hidden_size * 2)
        self.hidden2label = nn.Linear(hidden_size * 2, output_size)
        self.relu = nn.LeakyReLU()

    def forward(self, U):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        U = self.dropout(U)
        emotions, hidden = self.bigru(U)
        if self.attention:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions, t, mask=None)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:, 0, :])
            att_emotions = torch.cat(att_emotions, dim=0)
            emotions = att_emotions
            #hidden = F.relu(self.linear(att_emotions))
        else:
            #hidden = F.relu(self.linear(emotions))
            emotions = emotions
        gru_out = torch.transpose(emotions, 1, 2)
        gru_out = F.tanh(gru_out)
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        Out_in = self.relu(gru_out)
        Out_in = self.dropout(Out_in)
        Out_out = self.hidden2label(Out_in)
        return Out_out
class GRUModel_Mid_17(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,args):
        super(GRUModel_Mid_17, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.attention = args.attention
        self.num_layers = args.dia_layers
        self.bigru = nn.GRU(input_size, hidden_size,batch_first=True, num_layers=self.num_layers, bidirectional=True)
        if self.attention:
            self.matchatt = MatchingAttention(2 * hidden_size, 2 * hidden_size, att_type='dot')

        # linear
        self.input2hidden = nn.Linear(512, hidden_size * 2)
        self.hidden2label = nn.Linear(hidden_size * 2, output_size)
        self.relu = nn.LeakyReLU()

    def forward(self, U):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        U = self.dropout(U)
        emotions, hidden = self.bigru(U)
        if self.attention:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions, t, mask=None)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:, 0, :])
            att_emotions = torch.cat(att_emotions, dim=0)
            emotions = att_emotions
            #hidden = F.relu(self.linear(att_emotions))
        else:
            #hidden = F.relu(self.linear(emotions))
            emotions = emotions
        gru_out = torch.transpose(emotions, 1, 2)
        gru_out = F.tanh(gru_out)
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        Out_in = self.relu(gru_out)
        Out_in = self.dropout(Out_in)
        Out_out = self.hidden2label(Out_in)
        return Out_out
         
class SpeechRecognitionModel(nn.Module):
    def __init__(self, args):
        super(SpeechRecognitionModel, self).__init__()
        self.feature_extractor = HubertModel.from_pretrained("/mnt/data1/liyongwei/SSL_Models/facebook/hubert-base-ls960")
        self.layer_norm = nn.LayerNorm(17)
        self.classifier = nn.Linear(self.feature_extractor.config.hidden_size, args.hidden_layer)
        self.out_layer = nn.Linear(args.hidden_layer, args.out_class)
        self.hidden_dim = args.hidden_layer
        self.out_class = args.out_class
        self.dropout = nn.Dropout(args.dropout)
        self.attention = args.attention
        self.num_layers = args.dia_layers

        self.Utterance_net_1 = GRUModel_Mid_1(args.utt_insize, args.hidden_layer, args.mid_class, args)
        self.Utterance_net_2 = GRUModel_Mid_2(args.utt_insize, args.hidden_layer, args.mid_class, args)
        self.Utterance_net_3 = GRUModel_Mid_3(args.utt_insize, args.hidden_layer, args.mid_class, args)
        self.Utterance_net_4 = GRUModel_Mid_4(args.utt_insize, args.hidden_layer, args.mid_class, args)
        self.Utterance_net_5 = GRUModel_Mid_5(args.utt_insize, args.hidden_layer, args.mid_class, args)
        self.Utterance_net_6 = GRUModel_Mid_6(args.utt_insize, args.hidden_layer, args.mid_class, args)
        self.Utterance_net_7 = GRUModel_Mid_7(args.utt_insize, args.hidden_layer, args.mid_class, args)
        self.Utterance_net_8 = GRUModel_Mid_8(args.utt_insize, args.hidden_layer, args.mid_class, args)
        self.Utterance_net_9 = GRUModel_Mid_9(args.utt_insize, args.hidden_layer, args.mid_class, args)
        self.Utterance_net_10 = GRUModel_Mid_10(args.utt_insize, args.hidden_layer, args.mid_class, args)
        self.Utterance_net_11 = GRUModel_Mid_11(args.utt_insize, args.hidden_layer, args.mid_class, args)
        self.Utterance_net_12 = GRUModel_Mid_12(args.utt_insize, args.hidden_layer, args.mid_class, args)
        self.Utterance_net_13 = GRUModel_Mid_13(args.utt_insize, args.hidden_layer, args.mid_class, args)
        self.Utterance_net_14 = GRUModel_Mid_14(args.utt_insize, args.hidden_layer, args.mid_class, args)
        self.Utterance_net_15 = GRUModel_Mid_15(args.utt_insize, args.hidden_layer, args.mid_class, args)
        self.Utterance_net_16 = GRUModel_Mid_16(args.utt_insize, args.hidden_layer, args.mid_class, args)
        self.Utterance_net_17 = GRUModel_Mid_17(args.utt_insize, args.hidden_layer, args.mid_class, args)

        self.Out_net = GRUModel(args.utt_insize, args.hidden_layer, args.out_class, args)

    def forward(self, input_waveform):
        features = self.feature_extractor(input_waveform).last_hidden_state
        logits_1 = self.Utterance_net_1(features)
        logits_2 = self.Utterance_net_2(features)
        logits_3 = self.Utterance_net_3(features)
        logits_4 = self.Utterance_net_4(features)
        logits_5 = self.Utterance_net_5(features)
        logits_6 = self.Utterance_net_6(features)
        logits_7 = self.Utterance_net_7(features)
        logits_8 = self.Utterance_net_8(features)
        logits_9 = self.Utterance_net_9(features)
        logits_10 = self.Utterance_net_10(features)
        logits_11 = self.Utterance_net_11(features)
        logits_12 = self.Utterance_net_12(features)
        logits_13 = self.Utterance_net_13(features)
        logits_14 = self.Utterance_net_14(features)
        logits_15 = self.Utterance_net_15(features)
        logits_16 = self.Utterance_net_16(features)
        logits_17 = self.Utterance_net_17(features)
        Out_input= torch.cat((logits_1,logits_2,logits_3,logits_4,logits_5,logits_6,logits_7,logits_8,logits_9,logits_10,logits_11,logits_12,logits_13,logits_14,logits_15,logits_16,logits_17),axis = 1)
        Out_input = self.layer_norm(Out_input)
        logits = self.Out_net(Out_input)
        return logits,logits_1,logits_2,logits_3,logits_4,logits_5,logits_6,logits_7,logits_8,logits_9,logits_10,logits_11,logits_12,logits_13,logits_14,logits_15,logits_16,logits_17
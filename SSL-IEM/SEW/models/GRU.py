import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SEWModel

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

class GRUModel(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,args):
        super(GRUModel, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.attention = args.attention
        self.num_layers = args.dia_layers
        self.bigru = nn.GRU(input_size, hidden_size,batch_first=True, num_layers=self.num_layers, bidirectional=True)
        if self.attention:
            self.matchatt = MatchingAttention(2 * hidden_size, 2 * hidden_size, att_type='dot')

        # linear
        #self.input2hidden = nn.Linear(hidden_size, hidden_size * 2)
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
        #gru_out = self.input2hidden(gru_out)
        Out_in = self.relu(gru_out)
        Out_in = self.dropout(Out_in)
        Out_out = self.hidden2label(Out_in)
        return Out_out
        
class SpeechRecognitionModel(nn.Module):
    def __init__(self, args):
        super(SpeechRecognitionModel, self).__init__()
        self.feature_extractor = SEWModel.from_pretrained("/mnt/data1/liyongwei/SSL_Models/asapp/sew-tiny-100k-ft-ls100h")
        self.classifier = nn.Linear(self.feature_extractor.config.hidden_size, args.hidden_layer)
        self.out_layer = nn.Linear(args.hidden_layer, args.out_class)
        self.hidden_dim = args.hidden_layer
        self.out_class = args.out_class
        self.dropout = nn.Dropout(args.dropout)
        self.attention = args.attention
        self.num_layers = args.dia_layers
        self.Utterance_net = GRUModel(args.utt_insize, args.hidden_layer, args.out_class, args)

    def forward(self, input_waveform):
        features = self.feature_extractor(input_waveform).last_hidden_state
        logits = self.Utterance_net(features)
        return logits
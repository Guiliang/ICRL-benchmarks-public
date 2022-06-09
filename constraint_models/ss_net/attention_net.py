import torch
import numpy as np

from utils.model_utils import masked_softmax


class Sparse_grad_attention(torch.autograd.Function):
    # def __init__(self, top_k):
    #     super(Sparse_grad_attention,self).__init__()
    #
    #     self.sa = Sparse_attention(top_k=top_k)

    @staticmethod
    def forward(ctx, inp, sa):
        sparsified = sa(inp)
        ctx.save_for_backward(inp, sparsified)

        return inp

    @staticmethod
    def backward(ctx, grad_output):
        inp, sparsified = ctx.saved_tensors
        # print('sparsified', sparsified)
        return (grad_output) * (sparsified > 0.0).float()


class SparseAttention(torch.nn.Module):
    """
    SparseAttention from https://github.com/anirudh9119/RIMs/blob/master/event_based/sparse_attn.py
    """

    def __init__(self, top_k=5):
        super(SparseAttention, self).__init__()
        top_k += 1
        self.top_k = top_k

    def forward(self, attn_s):

        # normalize the attention weights using piece-wise Linear function
        # only top k should
        attn_plot = []
        # torch.max() returns both value and location
        # attn_s_max = torch.max(attn_s, dim = 1)[0]
        # attn_w = torch.clamp(attn_s_max, min = 0, max = attn_s_max)
        eps = 10e-8
        time_step = attn_s.size()[1]
        if time_step <= self.top_k:
            # just make everything greater than 0, and return it
            # delta = torch.min(attn_s, dim = 1)[0]
            return attn_s
        else:
            # get top k and return it
            # bottom_k = attn_s.size()[1] - self.top_k
            # value of the top k elements
            # delta = torch.kthvalue(attn_s, bottm_k, dim= 1 )[0]
            delta = torch.topk(attn_s, self.top_k, dim=1)[0][:, -1] + eps
            # delta = attn_s_max - torch.topk(attn_s, self.top_k, dim= 1)[0][:,-1] + eps
            # normalize
            delta = delta.reshape((delta.shape[0], 1))

        attn_w = attn_s - delta.repeat(1, time_step)
        attn_w = torch.clamp(attn_w, min=0)
        attn_w_sum = torch.sum(attn_w, dim=1, keepdim=True)
        attn_w_sum = attn_w_sum + eps
        attn_w_normalize = attn_w / attn_w_sum.repeat(1, time_step)

        # print('attn', attn_w_normalize)

        return attn_w_normalize


class ScaledDotProductAttention(torch.nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, topk=None, grad_sparse=False, attn_dropout=0.0):
        super().__init__()
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=2)
        self.grad_sparse = grad_sparse
        self.topk = topk
        if self.topk is not None:
            self.sa = SparseAttention(top_k=topk)  # k=2
        # self.sga = Sparse_grad_attention(top_k=2)
        self.dropout = torch.nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask, use_sparse=False):
        attn = torch.bmm(q, k.transpose(1, 2))  # [batch_size, len, dim] x [batch_size, dim, len]
        attn = attn / self.temperature
        # if mask is not None:
        #     attn = attn.masked_fill(mask, -np.inf)
        #     # attn = self.dropout(attn)
        # attn = self.softmax(attn)
        attn = masked_softmax(attn, mask, 2)
        # use_sparse = True  # False
        if use_sparse:
            mb, ins, outs = attn.shape[0], attn.shape[1], attn.shape[2]
            sparse_attn = attn.reshape((mb * ins, outs))
            # print('sparse attn shape 1', sparse_attn.shape)
            # sga = Sparse_grad_attention(2)
            if self.grad_sparse:
                sga = Sparse_grad_attention(self.topk)
                sparse_attn = sga(sparse_attn)
            else:
                sparse_attn = self.sa(sparse_attn)
            sparse_attn = sparse_attn.reshape((mb, ins, outs))
            attn = sparse_attn * 1.0

        attn = self.dropout(attn)
        # tmp = torch.sum(attn, dim=2)
        output = torch.bmm(attn, v)

        return output, attn


class SelfAttention(torch.nn.Module):
    ''' From Multi-Head Attention module
    https://github.com/jadore801120/attention-is-all-you-need-pytorch'''

    def __init__(self, block_hidden_dim, n_head, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.block_hidden_dim = block_hidden_dim
        self.w_qs = torch.nn.Linear(block_hidden_dim, n_head * block_hidden_dim, bias=False)
        self.w_ks = torch.nn.Linear(block_hidden_dim, n_head * block_hidden_dim, bias=False)
        self.w_vs = torch.nn.Linear(block_hidden_dim, n_head * block_hidden_dim, bias=False)
        torch.nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (block_hidden_dim * 2)))
        torch.nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (block_hidden_dim * 2)))
        torch.nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (block_hidden_dim * 2)))
        self.attention = ScaledDotProductAttention(temperature=np.power(block_hidden_dim, 0.5))
        self.fc = torch.nn.Linear(n_head * block_hidden_dim, block_hidden_dim)
        self.layer_norm = torch.nn.LayerNorm(self.block_hidden_dim)
        torch.nn.init.xavier_normal_(self.fc.weight)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, q, mask, k, v):
        # q: batch x len_q x hid
        # k: batch x len_k x hid
        # v: batch x len_v x hid
        # mask: batch x len_q x len_k
        batch_size, len_q = q.size(0), q.size(1)
        len_k, len_v = k.size(1), v.size(1)
        assert mask.size(1) == len_q
        assert mask.size(2) == len_k
        residual = q

        q = self.w_qs(q).view(batch_size, len_q, self.n_head, self.block_hidden_dim)
        k = self.w_ks(k).view(batch_size, len_k, self.n_head, self.block_hidden_dim)
        v = self.w_vs(v).view(batch_size, len_v, self.n_head, self.block_hidden_dim)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, self.block_hidden_dim)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, self.block_hidden_dim)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, self.block_hidden_dim)  # (n*b) x lv x dv

        mask = mask.repeat(self.n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(self.n_head, batch_size, len_q, self.block_hidden_dim)
        output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, len_q, -1)  # b x lq x (n*dv)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn

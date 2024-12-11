# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 11:33
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

"""
SASRec
################################################

Reference:
    Wang-Cheng Kang et al. "Self-Attentive Sequential Recommendation." in ICDM 2018.

Reference:
    https://github.com/kang205/SASRec

"""
import math
import random
import torch
from torch import nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import FeedForward
from recbole.model.loss import BPRLoss
from IPython import embed

class FMLPEncoder(nn.Module):
    r"""One TransformerEncoder consists of several TransformerLayers.

    Args:
        n_layers(num): num of transformer layers in transformer encoder. Default: 2
        n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        hidden_size(num): the input and output hidden size. Default: 64
        inner_size(num): the dimensionality in feed-forward layer. Default: 256
        hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12

    """

    def __init__(
        self,
        config,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
    ):
        super(FMLPEncoder, self).__init__()
        self.dropout = nn.Dropout(config['freq_dropout_prob'])
        self.layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.n_layers = config['n_layers']
        self.layer = nn.ModuleList()
        for n in range(self.n_layers):
            self.fmblock = FMLPLayer(
                config,
                n_heads,
                hidden_size,
                inner_size,
                hidden_dropout_prob,
                attn_dropout_prob,
                hidden_act,
                layer_norm_eps,
                n
            )
            self.layer.append(self.fmblock)
        self.frequency_nce=config['frequency_nce']
        self.c = config['c']
        self.random_drop = config['random_drop']

    def forward(self, hidden_states, item_seq_len, timestamp, attn_mask, output_all_encoded_layers=True):

        all_encoder_layers = []
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, item_seq_len, timestamp, attn_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)

        all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        
        return all_encoder_layers

class FMLPLayer(nn.Module):
    def __init__(
        self,
        config,
        n_heads,
        hidden_size,
        intermediate_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
        i
    ):
        super(FMLPLayer, self).__init__()
        # self.self_attention = MultiHeadAttention(n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps)
        self.filter_layer = FilterLayer(config,i)
        self.feed_forward = FeedForward(
            hidden_size,
            intermediate_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,
            config
        )

    def forward(self, hidden_states, item_seq_len, timestamp, seq_mask):
        # hidden_states = self.self_attention(hidden_states, attn_mask)
        filter_output = self.filter_layer(hidden_states, item_seq_len, seq_mask)
        feedforward_output = self.feed_forward(filter_output)
        return feedforward_output



class FilterLayer(nn.Module):
    def __init__(self, config,i):
        super(FilterLayer, self).__init__()
    
        self.out_dropout = nn.Dropout(config['freq_dropout_prob'])
        self.N = config['N']
        self.LayerNorm = nn.LayerNorm(config['hidden_size'], eps=1e-12)
        self.use_noise = config['use_noise']
        self.threshold = config['c']

        self.useful_weight = nn.Parameter(torch.randn(1, config['MAX_ITEM_LIST_LENGTH']//2 + 1, config['hidden_size'], 2, dtype=torch.float32) * 0.02)

    def forward(self, input_tensor, item_seq_len,seq_mask, aug=False, rand=False):
        # [batch, seq_len, hidden]
        
        batch, max_len, hidden = input_tensor.shape
        # input_signal = input_tensor*seq_mask
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        
        useful_weight = torch.view_as_complex(self.useful_weight)
        use_emb = x * useful_weight    
        sequence_signal = torch.fft.irfft(use_emb, n=max_len, dim=1, norm='ortho') 
        # from IPython import embed; embed()
        hidden_states = self.out_dropout(sequence_signal)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class FMLPRec(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(FMLPRec, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.lmd_sem = config['lmd_sem']
        self.inner_size = config[
            "inner_size"
        ]  # the dimensionality in feed-forward layer
        self.use_supaug=config['use_supaug']
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]       
        
        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.mask_default = self.mask_correlated_samples(batch_size=config['train_batch_size'])
        
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = FMLPEncoder(
            config = config,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )
        self.aug_nce_fct = nn.CrossEntropyLoss()
        self.masked_aug_loss = nn.MSELoss()
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        # self.tau = config['tau']
        # self.sim = config['sim']
        self.train_batch_size = config['train_batch_size']
        self.ssl = config['contrast']
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)
        

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


    
    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask
    
    def forward(self, item_seq, item_seq_len, timestamp, aug=False, aug_emb=None):
        if aug:
            input_emb = aug_emb
        else:
            position_ids = torch.arange(
                item_seq.size(1), dtype=torch.long, device=item_seq.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
            position_embedding = self.position_embedding(position_ids)

            item_emb = self.item_embedding(item_seq)
            input_emb = item_emb + position_embedding
            sequence_mask = self.sequence_mask(item_seq)
            # input_emb *= sequence_mask
            input_emb = self.LayerNorm(input_emb)
            input_emb = self.dropout(input_emb)
            
        

        trm_output = self.trm_encoder(
                input_emb, item_seq_len, timestamp, sequence_mask, output_all_encoded_layers=True
                )
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  


    @staticmethod
    def alignment(seq_output, target_item_emb, alpha=2):
        seq_output = F.normalize(seq_output, dim=-1)
        target_item_emb = F.normalize(target_item_emb, dim=-1)
        return (seq_output-target_item_emb).norm(p=2, dim=1).pow(alpha).mean()
    
    @staticmethod
    def uniformity(x):
        x = F.normalize(x, dim=-1)
        loss = torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()
        return loss
    
    def sequence_mask(self, input_ids):
        mask = (input_ids != 0) * 1
        return mask.unsqueeze(-1) 
        
    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_seq_ts = interaction['timestamp_list']
        seq_output = self.forward(item_seq, item_seq_len, item_seq_ts)
            
        # seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))     
            loss = self.loss_fct(logits, pos_items)

        # if self.ssl == 'us_x':
        #     aug_seq_output = self.forward(item_seq, item_seq_len, item_seq_ts)

        #     sem_aug, sem_aug_lengths = interaction['sem_aug'], interaction['sem_aug_lengths']
        #     sem_aug_seq_output = self.forward(sem_aug, sem_aug_lengths,item_seq_ts)

        #     sem_nce_logits, sem_nce_labels = self.info_nce(aug_seq_output, sem_aug_seq_output, temp=self.tau,
        #                                                    batch_size=item_seq_len.shape[0], sim=self.sim)

        #     loss += self.lmd_sem * self.aug_nce_fct(sem_nce_logits, sem_nce_labels)
            
        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_seq_ts = interaction['timestamp_list']
        test_item = interaction[self.ITEM_ID]
        try:
            item_seq = item_seq.reshape(-1,101,item_seq.size()[-1])[:,0,:]
            item_seq_len = item_seq_len.reshape(-1,101)[:,1]
            seq_output = self.forward(item_seq, item_seq_len, item_seq_ts)
            test_item_emb = self.item_embedding(test_item)
            seq_output = seq_output.reshape(-1,1,self.hidden_size)
            test_item_emb = test_item_emb.reshape(-1,101,self.hidden_size)
            scores = torch.mul(seq_output, test_item_emb).sum(dim=-1).reshape(-1)  # [B]
        except:
            seq_output = self.forward(item_seq, item_seq_len,item_seq_ts)
            seq_output = seq_output[0]
            test_item_emb = self.item_embedding(test_item)
            scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]

        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_seq_ts = interaction['timestamp_list']
        seq_output = self.forward(item_seq, item_seq_len, item_seq_ts)
        test_items_emb = self.item_embedding.weight

        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
    
    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask
    
    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot'):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)
           
        if sim == 'cos':
            sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.T) / temp

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        if batch_size != self.train_batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels
    

    def forward2(self, item_seq, item_seq_len, timestamp, aug=False, aug_emb=None):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        if aug:
            input_emb = aug_emb
        else:
            position_ids = torch.arange(
                item_seq.size(1), dtype=torch.long, device=item_seq.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
            position_embedding = self.position_embedding(position_ids)

            item_emb = self.item_embedding(item_seq)
            input_emb = item_emb + position_embedding
            input_emb = self.LayerNorm(input_emb)
            input_emb = self.dropout(input_emb)

        
        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(
                input_emb, item_seq_len, timestamp, extended_attention_mask, output_all_encoded_layers=True
                )
        if self.use_frequency_nce:
            output, aug1, aug2 = trm_output[-1]
            aug1 = self.gather_indexes(aug1, item_seq_len-1)
            aug2 = self.gather_indexes(aug2, item_seq_len-1)
        else:
            output = trm_output[-1]
        # output = self.gather_indexes(output, item_seq_len - 1)
        if self.use_frequency_nce:
            return output, aug1, aug2
        else:
            return output  
        
        
    # def frequency_masking(self, item_seq, item_seq_len, rand=True):
    #     position_ids = torch.arange(
    #         item_seq.size(1), dtype=torch.long, device=item_seq.device
    #     )
    #     position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
    #     position_embedding = self.position_embedding(position_ids)

    #     item_emb = self.item_embedding(item_seq)
    #     input_emb = item_emb + position_embedding
    #     input_emb = self.LayerNorm(input_emb)
    #     input_emb = self.dropout(input_emb)

    #     batch, max_len, hidden = input_emb.shape
    #     x = torch.fft.rfft(input_emb, dim=1, norm='ortho')
    #     if rand=='low':
    #         x[:, self.c:, :] = 0
    #     elif rand=='high':
    #         x[:,:self.c,:] = 0
    #     elif rand==True:
    #         mask = torch.ones(batch, max_len//2+1, 1).to(input_emb.device)
    #         padding_ratio = self.pad_ratio
    #         num_elements = mask.numel()  
    #         num_padding = int(padding_ratio * num_elements)

    #         indices = random.sample(range(num_elements), num_padding)
    #         mask_flat = mask.view(-1) 

    #         mask_flat[indices] = 0
    #         mask = mask_flat.view(batch, max_len//2+1, 1)
    #         x = x * mask
    #     else:
    #         freqs = torch.arange(max_len//2 + 1).to(input_emb.device)
    #         seq_len_norm = torch.ones(batch) * self.c
    #         seq_len_norm_dim = seq_len_norm.unsqueeze(-1).to(x.device)

    #         # low mask를 통해 seq_len threshold 이하는 모두 useful로 학습되게함
    #         low_mask = (freqs <= seq_len_norm_dim).unsqueeze(-1)

    #         # high mask를 통해 seq_len threshold 바깥은 magnitude threshold에 따라 useful과 noise로 나뉨
    #         high_mask = (freqs > seq_len_norm_dim).unsqueeze(-1)
    #         padding_ratio = self.pad_ratio
    #         num_elements = high_mask[high_mask==True].numel()  
    #         num_padding = int(padding_ratio * num_elements)  

    #         indices = random.sample(range(num_elements), num_padding)

    #         flat = high_mask.view(-1) 
    #         true_indices = torch.nonzero(flat).squeeze()
    #         flat[true_indices[indices]] = 0 

    #         high_mask = flat.view(batch, max_len//2+1, 1)
    #         aug_mask = low_mask | high_mask
    #         x = x * aug_mask

    #     aug = torch.fft.irfft(x, n=max_len, dim=1, norm='ortho')
    #     #aug = self.dropout(aug)

    #     return aug # self.LayerNorm(input_emb + aug)
    

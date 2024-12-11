
import torch
import torch.nn as nn
import torch.nn.functional as F
# from recbole.model.layers import  LayerNorm
from recbole.model.loss import EmbLoss
from recbole.model.abstract_recommender import SequentialRecommender
from IPython import embed
import copy
import math

def gumbel_sigmoid(logits: torch.Tensor, tau: float = 1, hard: bool = False, threshold: float = 0.5) -> torch.Tensor:

    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0, 1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits, tau)
    y_soft = gumbels.sigmoid()

    indices = (y_soft > threshold).nonzero(as_tuple=True)
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
    y_hard[indices[0], indices[1]] = 1.0
    ret = y_hard - y_soft.detach() + y_soft

    return ret

class FMLPFilterLayer(nn.Module):
    def __init__(self, config, high):
        super(FMLPFilterLayer, self).__init__()
        self.high = high
        freq_n = config['MAX_ITEM_LIST_LENGTH']//2 + 1
        filter_ = torch.ones(freq_n, 2)
        if high:
            self.out_dropout = nn.Dropout(config['high_freq_dropout_prob'])
            filter_[:freq_n//2] *= 0
        else:
            self.out_dropout = nn.Dropout(config['freq_dropout_prob'])
            filter_[freq_n//2:] *= 0
        self.complex_weight1 = nn.Parameter(torch.randn(1, config['hidden_size'], config['MAX_ITEM_LIST_LENGTH']//2 + 1, 2, dtype=torch.float32) * 0.02)
        self.conv_layers = config['conv_layers']
        self.hidden_size = config['hidden_size']
        self.kernel_size = config['kernel_size']
        self.LayerNorm = nn.LayerNorm(config['hidden_size'], eps=1e-12)
        self.singate = config['use_singate']
        self.filter_ = torch.view_as_complex(filter_).reshape(1,-1,1)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=self.hidden_size,
                    out_channels=self.hidden_size,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size//2,
                    padding_mode= 'reflect'
                ),
                nn.BatchNorm1d(self.hidden_size),
            )
                for _ in range(self.conv_layers)
            ]) 
            
        self.linear = nn.Linear(freq_n, freq_n, bias=False)
        self.hidden_linear = nn.Linear(self.hidden_size, 1, bias=False)
        self.use_convfilter=config['use_convfilter']
        self.use_mlpfilter = config['use_mlpfilter']
        self.only_filter = config['only_filter']

    def forward(self, input_tensor, seq_mask):
        batch, seq_len, hidden = input_tensor.shape
        # input_signal = input_tensor * seq_mask
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight1)

        if self.use_convfilter:
            # conv filter
            x = x* self.filter_.cuda()
            x = x.permute(0,2,1)

            x_real = x.real
            for conv_layer in self.convs:
                x_real = conv_layer(x_real)

            x_real_maxpool = torch.max(x_real, dim=1, keepdim=True).values
            x_real_filter = torch.sigmoid(self.linear(x_real_maxpool))

            x_imag = x.imag
            for conv_layer in self.convs:
                x_imag = conv_layer(x_imag)
        
            x_imag_maxpool = torch.max(x_imag, dim=1, keepdim=True).values
            x_imag_filter = torch.sigmoid(self.linear(x_imag_maxpool))

            # filtered_frequency
            filtered_weight = torch.complex(x_real_filter * weight.real, x_imag_filter * weight.imag)
            x_ = (filtered_weight * x).permute(0,2,1)
        else:
            x_ = weight.permute(0,2,1) * x * self.filter_.cuda()

        sequence_emb_fft = torch.fft.irfft(x_, n=seq_len, dim=1, norm='ortho')
        # sequence_emb_fft *= seq_mask
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish, 'silu':F.silu}

class FMLPIntermediate(nn.Module):
    def __init__(self, config):
        super(FMLPIntermediate, self).__init__()
        self.dense_1 = nn.Linear(config['hidden_size'], config['inner_size'])
        if isinstance(config['hidden_act'], str):
            self.intermediate_act_fn = ACT2FN[config['hidden_act']]
        else:
            self.intermediate_act_fn = config['hidden_act']

        self.dense_2 = nn.Linear(config['inner_size'], config['hidden_size'])
        self.LayerNorm = nn.LayerNorm(config['hidden_size'], eps=1e-12)
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class FMLPLayer(nn.Module):
    def __init__(self, config, high):
        super(FMLPLayer, self).__init__()
        self.filterlayer = FMLPFilterLayer(config, high)
        self.intermediate = FMLPIntermediate(config)

    def forward(self, hidden_states, seq_mask):
        
        hidden_states = self.filterlayer(hidden_states, seq_mask)
        hidden_states = self.intermediate(hidden_states)
        # hidden_states = attention_mask * hidden_states
        return hidden_states

class FMLPEncoder(nn.Module):
    def __init__(self, config, high):
        super(FMLPEncoder, self).__init__()
        layer = FMLPLayer(config, high)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(config['n_layers'])])

    def forward(self, hidden_states, seq_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, seq_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers
    
class FMLPHP(SequentialRecommender):
    def __init__(self, config, dataset):
        super(FMLPHP, self).__init__(config, dataset)
        self.noise_mag = config['mag']
        self.gamma = config['gamma']
        self.mode = config['mode']
        self.th = config['th']
        self.alpha = config['alpha']
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.inner_size = config[
            "inner_size"
        ]  
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.aug_nce_fct = nn.CrossEntropyLoss()
        self.batch_size = config['train_batch_size']
        self.initializer_range = config["initializer_range"]
        self.mask_default = self.mask_correlated_samples(batch_size=self.batch_size)
        self.loss_type = config["loss_type"]
        self.config = config
        self.position_embeddings = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.hp_item_encoder = FMLPEncoder(config, high=True)
        self.lp_item_encoder = FMLPEncoder(config, high=False)
        self.alpha = nn.Parameter(torch.ones(1)*0.5)
        self.output_bias = nn.Parameter(torch.zeros(self.n_items))
        self.loss_fct = nn.CrossEntropyLoss()
        self.item_embeddings = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.concat_layer = nn.Linear(self.hidden_size*2, self.hidden_size, bias=False)

        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.emb_loss = EmbLoss()
        self.ssl = config['contrast']
        self.lmd_sem = config['lmd_sem']
        self.use_JSD= config['use_JSD']
        self.use_LHloss = config['use_LHloss']
        self.convex=0
        self.apply(self.init_weights)
        self.init_concat_layer()

    def init_concat_layer(self):
        with torch.no_grad():
            # Random 초기화된 가중치 생성
            concat_weights = torch.randn((self.hidden_size * 2, self.hidden_size))
            # 대각선 요소는 단위 행렬로 설정
            concat_weights[:self.hidden_size, :].fill_diagonal_(1)
            concat_weights[-self.hidden_size:, :].fill_diagonal_(1)
            # nn.Linear의 가중치를 초기화
            self.concat_layer.weight[:,:self.hidden_size].fill_diagonal_(1)
            self.concat_layer.weight[:,self.hidden_size:].fill_diagonal_(1)

    def add_position_embedding(self, sequence, seq_mask):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        # position_embeddings = self.position_embeddings(position_ids)
        item_embeddings = item_embeddings #+ position_embeddings
        item_embeddings *= seq_mask
        item_embeddings = self.LayerNorm(item_embeddings)
        item_embeddings = self.dropout(item_embeddings)
        return item_embeddings

    def forward(self, input_ids, item_seq_len, perturbed=False):
        seq_mask = self.sequence_mask(input_ids)
        sequence_emb = self.add_position_embedding(input_ids, seq_mask)
        
        item_encoded_layers = self.hp_item_encoder(sequence_emb,
                                                seq_mask, 
                                                output_all_encoded_layers=True,
                                                )
        hp_output = item_encoded_layers[-1]
        hp_output = self.gather_indexes(hp_output, item_seq_len - 1)
        item_encoded_layers = self.lp_item_encoder(sequence_emb,
                                                seq_mask, 
                                                output_all_encoded_layers=True,
                                                )
        lp_output = item_encoded_layers[-1]
        
        lp_output = self.gather_indexes(lp_output, item_seq_len - 1)
        if False:
            output = (self.alpha)*hp_output + (1-self.alpha)*lp_output
        if True:
            concate_output = torch.cat((hp_output, lp_output),dim=-1)
            # self.init_concat_layer()
            output = self.concat_layer(concate_output)
        if False:
            output = torch.cat((hp_output, lp_output),dim=-1)
            output = self.concat_layer(output) + (self.alpha)*hp_output + (1-self.alpha)*lp_output
        target_hidden_state = self.gather_indexes(sequence_emb, item_seq_len - 1)
        output = self.LayerNorm(output + target_hidden_state)
        output = self.dropout(output)
        return output, hp_output, lp_output, 0

    def sequence_mask(self, input_ids):
        mask = (input_ids != 0) * 1
        return mask.unsqueeze(-1) 
    
    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]

        seq_output, hp_output, lp_output,_ = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embeddings(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1) 
        return scores
    
    def JSD(self,h_logit, l_logit):
        h_probs =  F.softmax(h_logit, dim=1)
        l_probs=  F.softmax(l_logit, dim=1)

        h_probs, l_probs = h_probs.view(-1, h_probs.size(-1)), l_probs.view(-1, l_probs.size(-1))
        m = (0.5 * (h_probs + l_probs)).log()

        return 0.5 * (self.kl(m, h_probs.log()) + self.kl(m, l_probs.log()))
     
    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        
        seq_output, hp_output, lp_output, convex = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        # from IPython import embed; embed()
        test_item_emb = self.item_embeddings.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = self.loss_fct(logits, pos_items)
        # loss += self.gamma * ((hp_output - lp_output).pow(2).mul(-2).exp().mean()+1e-8).log()

        if self.use_LHloss:
            hp_logits = torch.matmul(hp_output, test_item_emb.transpose(0, 1)) # / 10
            hp_loss = self.loss_fct(hp_logits, pos_items)
            lp_logits = torch.matmul(lp_output, test_item_emb.transpose(0, 1)) # / 10
            lp_loss = self.loss_fct(lp_logits, pos_items)
            loss = loss + self.gamma*(lp_loss+hp_loss)

        if self.ssl == 'us_x':
            aug_seq_output,_,_,_ = self.forward(item_seq, item_seq_len)

            sem_aug, sem_aug_lengths = interaction['sem_aug'], interaction['sem_aug_lengths']
            sem_aug_seq_output,_,_,_ = self.forward(sem_aug, sem_aug_lengths)

            sem_nce_logits, sem_nce_labels = self.info_nce(aug_seq_output, sem_aug_seq_output, temp=1,
                                                        batch_size=item_seq_len.shape[0], sim='dot')

            loss = loss + self.lmd_sem * self.aug_nce_fct(sem_nce_logits, sem_nce_labels)
            # sem_hp_logits, sem_hp_labels = self.info_nce(aug_hp, sem_aug_hp, temp=1,
            #                                             batch_size=item_seq_len.shape[0], sim='dot')
            # sem_lp_logits, sem_lp_labels = self.info_nce(aug_lp, sem_aug_lp, temp=1,
            #                                             batch_size=item_seq_len.shape[0], sim='dot')
            # cl_loss =  self.lmd_sem * (self.aug_nce_fct(sem_nce_logits, sem_nce_labels)+self.aug_nce_fct(sem_hp_logits, sem_hp_labels) + self.aug_nce_fct(sem_lp_logits, sem_lp_labels))                
        
        if False:
            loss_f = nn.MSELoss()
            rand_seq_output = self.forward_noise(item_seq, item_seq_len)
            loss += loss_f(rand_seq_output, seq_output.detach()) * self.gamma
        return loss
    
    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        
        if self.mode == 'mag':
            seq_output = self.forward_noise(item_seq, item_seq_len)
        elif self.mode == 'rep':
            seq_output = self.forward_seq_noise(item_seq, item_seq_len)
        else:
            seq_output, hp_output, lp_output,_ = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embeddings.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1)) 
        return scores

    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        # if isinstance(module, nn.Linear) and module.bias is not None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

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
        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)
    
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels
    
    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask
    
    def forward2(self, input_ids, item_seq_len, perturbed=False):
        extended_attention_mask = self.sequence_mask(input_ids)
        sequence_emb = self.add_position_embedding(input_ids)

        item_encoded_layers = self.hp_item_encoder(sequence_emb,
                                                extended_attention_mask, 
                                                output_all_encoded_layers=True,
                                                )
        hp_total = item_encoded_layers[-1]
        hp_output = self.gather_indexes(hp_total, item_seq_len - 1)
        item_encoded_layers = self.lp_item_encoder(sequence_emb,
                                                extended_attention_mask, 
                                                output_all_encoded_layers=True,
                                                )
        lp_total = item_encoded_layers[-1]
        lp_output = self.gather_indexes(lp_total, item_seq_len - 1)
        if False:
            output = (self.alpha)*hp_output + (1-self.alpha)*lp_output
        if True:
            output = torch.cat((hp_output, lp_output),dim=-1)
            output = self.concat_layer(output)
        if False:
            output = torch.cat((hp_output, lp_output),dim=-1)
            output = self.concat_layer(output) + (self.alpha)*hp_output + (1-self.alpha)*lp_output
        output = self.LayerNorm(output)
        output = self.dropout(output)
        return output, hp_total, lp_total
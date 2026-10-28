from model import objectives

from .CrossEmbeddingLayer_tse import TexualEmbeddingLayer, VisualEmbeddingLayer
from .clip_model import build_CLIP_from_openai_pretrained, convert_weights
from .local_evidence import local_evidence_loss, validate_evidence_options
import torch
import torch.nn as nn 
import torch.nn.functional as F

def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

class RDE(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.evidence_mode = getattr(args, 'evidence_mode', 'none')
        if self.evidence_mode == 'off':
            self.evidence_mode = 'none'
        if self.evidence_mode not in ('none', 'a_only'):
            raise ValueError('evidence_mode must be none/off or a_only')
        if self.evidence_mode == 'a_only':
            self.evidence_options = dict(
                candidates=getattr(args, 'evidence_candidates', 4),
                weight=getattr(args, 'evidence_weight', 0.1),
                temperature=getattr(args, 'evidence_temperature', 0.07),
                margin=getattr(args, 'evidence_margin', 0.1))
            validate_evidence_options(**self.evidence_options)
            if getattr(args, 'noisy_rate', None) != 0.0:
                raise ValueError('mechanism A requires explicit noisy_rate=0.0')
            if 'TAL' not in self.current_task:
                raise ValueError('mechanism A pilot requires the original TAL task')

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 
 
        self.visul_emb_layer = VisualEmbeddingLayer(ratio=args.select_ratio)
        self.texual_emb_layer = TexualEmbeddingLayer(ratio=args.select_ratio)
 
        if 'TAL' in self.current_task:
            loss_type = 'TAL'
        elif 'TRL' in self.current_task:
            loss_type = 'TRL'
        elif 'InfoNCE' in self.current_task:
            loss_type = 'InfoNCE'
        elif 'SDM' in self.current_task:
            loss_type = 'SDM'
        else:
            exit()
        self.loss_type = loss_type
 
    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    def encode_image(self, image):
        x, _ = self.base_model.encode_image(image)
        return x[:, 0, :].float()
      
    def encode_text(self, text):
        x, _ = self.base_model.encode_text(text.long())
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def encode_image_tse(self, image):
        x,atten_i = self.base_model.encode_image(image)
        i_tse_f = self.visul_emb_layer(x, atten_i)   
        return i_tse_f.float()
 
    def encode_text_tse(self, text):
        x,atten_t = self.base_model.encode_text(text.long())
        t_tse_f = self.texual_emb_layer(x, text, atten_t)
        return t_tse_f.float()

    def compute_per_loss(self, batch):
        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        i_tse_f = self.visul_emb_layer(image_feats, atten_i)
        t_tse_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)

        lossA, simsA = objectives.compute_per_loss(i_feats, t_feats, batch['pids'], \
                                                    tau=self.args.tau, \
                                                    margin=self.args.margin, \
                                                    loss_type=self.loss_type, \
                                                    logit_scale=self.logit_scale)
        lossB, simsB = objectives.compute_per_loss(i_tse_f, t_tse_f, batch['pids'],\
                                                    tau=self.args.tau, \
                                                    margin=self.args.margin, \
                                                    loss_type=self.loss_type, \
                                                    logit_scale=self.logit_scale)
        
        return lossA.detach().cpu(), lossB.detach().cpu(), simsA, simsB

    def forward(self, batch, return_evidence=False):
        ret = dict()
        ret.update({'temperature': 1 / self.logit_scale})

        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        i_tse_f = self.visul_emb_layer(image_feats, atten_i)
        t_tse_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)
            
        label_hat = batch['label_hat'].to(i_feats.device) 
     
        loss1, loss2 = objectives.compute_rbs(i_feats, t_feats, i_tse_f, t_tse_f, batch['pids'], \
                                              label_hat=label_hat, margin=self.args.margin,tau=self.args.tau,\
                                                loss_type=self.loss_type,logit_scale=self.logit_scale)
        ret.update({'bge_loss':loss1})
        ret.update({'tse_loss':loss2})

        if self.training and self.evidence_mode == 'a_only':
            if torch.is_grad_enabled() and not (image_feats.requires_grad and text_feats.requires_grad):
                raise RuntimeError('mechanism A requires gradient-preserving visual/text encoder outputs')
            # The original projection preserves a gradient path for patches;
            # its output dimension alone does not establish local semantic alignment.
            evidence = local_evidence_loss(
                image_feats[:, 1:, :], t_feats, i_feats, batch['pids'],
                query_mask=label_hat, **self.evidence_options)
            ret['evidence_a_loss'] = evidence['loss']
            ret['evidence_valid_queries'] = evidence['valid_query_mask'].sum().detach()
            if return_evidence:
                ret['evidence_relations'] = evidence
  
        return ret


def build_model(args, num_classes=11003):
    model = RDE(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model

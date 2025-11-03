from yacs.config import CfgNode as CN
_CN = CN()

_CN.name = 'default'
_CN.suffix ='OSdata'
_CN.iters = 12
_CN.gamma = 0.85
_CN.max_flow = 400
_CN.batch_size = 2
_CN.sum_freq = 100
_CN.val_freq = 5000000
_CN.image_size = [400, 400]
_CN.add_noise = False
_CN.critical_params = []

_CN.transformer = 'featureformerwithlbl'
_CN.restore_ckpt = 'checkpoints/sintel.pth'

# featureformer
_CN.featureformer = CN()
_CN.featureformer.pe = 'linear'
_CN.featureformer.dropout = 0.0
_CN.featureformer.encoder_latent_dim = 256 # in twins, this is 256
_CN.featureformer.query_latent_dim = 64
_CN.featureformer.cost_latent_input_dim = 64
_CN.featureformer.cost_latent_token_num = 8
_CN.featureformer.cost_latent_dim = 128
_CN.featureformer.arc_type = 'transformer'
_CN.featureformer.cost_heads_num = 1
_CN.featureformer.num_transformer_layers = 6
_CN.featureformer.num_head = 1
_CN.featureformer.feature_channels = 256
_CN.featureformer.attention_type = 'swin'
_CN.featureformer.ffn_dim_expansion = 4
_CN.featureformer.attn_splits = 2

# encoder
_CN.featureformer.pretrain = True
_CN.featureformer.context_concat = False
_CN.featureformer.encoder_depth = 3
_CN.featureformer.feat_cross_attn = False
_CN.featureformer.patch_size = 8
_CN.featureformer.patch_embed = 'single'
_CN.featureformer.no_pe = False
_CN.featureformer.gma = "GMA"
_CN.featureformer.kernel_size = 9
_CN.featureformer.rm_res = True
_CN.featureformer.vert_c_dim = 64
_CN.featureformer.cost_encoder_res = True
_CN.featureformer.cnet = 'basicencoder'
_CN.featureformer.fnet = 'basicencoder'
_CN.featureformer.no_sc = False
_CN.featureformer.only_global = False
_CN.featureformer.add_flow_token = True
_CN.featureformer.use_mlp = False
_CN.featureformer.vertical_conv = False

# decoder
_CN.featureformer.decoder_depth = 12
_CN.featureformer.critical_params = ['cost_heads_num', 'vert_c_dim', 'cnet', 'pretrain' , 'add_flow_token', 'encoder_depth', 'gma', 'cost_encoder_res']


# featureformerwithlbl
_CN.featureformerwithlbl = CN()
_CN.featureformerwithlbl.pe = 'linear'
_CN.featureformerwithlbl.dropout = 0.0
_CN.featureformerwithlbl.encoder_latent_dim = 256 # in twins, this is 256
_CN.featureformerwithlbl.query_latent_dim = 64
_CN.featureformerwithlbl.cost_latent_input_dim = 64
_CN.featureformerwithlbl.cost_latent_token_num = 8
_CN.featureformerwithlbl.cost_latent_dim = 128
_CN.featureformerwithlbl.arc_type = 'transformer'
_CN.featureformerwithlbl.cost_heads_num = 1
_CN.featureformerwithlbl.num_transformer_layers = 1
_CN.featureformerwithlbl.num_head = 1
_CN.featureformerwithlbl.feature_channels = 256
_CN.featureformerwithlbl.attention_type = 'swin'
_CN.featureformerwithlbl.ffn_dim_expansion = 4
_CN.featureformerwithlbl.attn_splits = 2

# encoder
_CN.featureformerwithlbl.pretrain = True
_CN.featureformerwithlbl.context_concat = False
_CN.featureformerwithlbl.encoder_depth = 3
_CN.featureformerwithlbl.feat_cross_attn = False
_CN.featureformerwithlbl.patch_size = 8
_CN.featureformerwithlbl.patch_embed = 'single'
_CN.featureformerwithlbl.no_pe = False
_CN.featureformerwithlbl.gma = "GMA"
_CN.featureformerwithlbl.kernel_size = 9
_CN.featureformerwithlbl.rm_res = True
_CN.featureformerwithlbl.vert_c_dim = 64
_CN.featureformerwithlbl.cost_encoder_res = True
_CN.featureformerwithlbl.cnet = 'basicencoder'
_CN.featureformerwithlbl.fnet = 'basicencoder'
_CN.featureformerwithlbl.no_sc = False
_CN.featureformerwithlbl.only_global = False
_CN.featureformerwithlbl.add_flow_token = True
_CN.featureformerwithlbl.use_mlp = False
_CN.featureformerwithlbl.vertical_conv = False

# decoder
_CN.featureformerwithlbl.decoder_depth = 12
_CN.featureformerwithlbl.critical_params = ['cost_heads_num', 'vert_c_dim', 'cnet', 'pretrain' , 'add_flow_token', 'encoder_depth', 'gma', 'cost_encoder_res']


### TRAINER
_CN.trainer = CN()
_CN.trainer.scheduler = 'OneCycleLR'
_CN.trainer.optimizer = 'adamw'
_CN.trainer.canonical_lr = 12.5e-5
_CN.trainer.adamw_decay = 1e-5
_CN.trainer.clip = 1.0
_CN.trainer.num_steps = 120000
_CN.trainer.epsilon = 1e-8
_CN.trainer.anneal_strategy = 'linear'

def get_cfg():
    return _CN.clone()

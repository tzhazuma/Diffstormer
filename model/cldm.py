from typing import Mapping, Any
import copy
from collections import OrderedDict
import sys
import einops
import torch
import torch as th
import torch.nn as nn
import os
import numpy as np
from .cond_fn import MSEGuidance
from ..utils.metrics import calculate_psnr_pt, LPIPS
sys.path.append("/public_bme/data/lifeng/code/moco/DiffBIR-main")
curPath_ = os.path.dirname(__file__)
curPath = os.getcwd()

from ..ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)
from ..ldm.modules.attention import SpatialTransformer
from ..ldm.modules.diffusionmodules.openaimodel import TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock, UNetModel
from ..ldm.models.diffusion.ddpm import LatentDiffusion
from ..ldm.util import log_txt_as_img, exists, instantiate_from_config
from ..ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ..utils.common import frozen_module
from .spaced_sampler import *

class ControlledUnetModel(UNetModel):  #继承自SD的unetmodel
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:     #将x_noise经过每个encoder模块后的结果，都append到hs里面，后面通过pop跟control里面对应层输出的结果相加
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()     #中间层特征相加,[6, 1280, 8, 8]

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1) #[6, 2560, 8, 8]
            h = module(h, emb, context)     #把融合后的特征输入decoder模块

        h = h.type(x.dtype)    #[6, 320, 64, 64]
        return self.out(h)  # #[6, 4, 64, 64]


class ControlNet(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        hint_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims  #2
        self.image_size = image_size     #32
        self.in_channels = in_channels    #4
        self.model_channels = model_channels   #320
        if isinstance(num_res_blocks, int):  #2
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]  #4*[2] -> [2,2,2,2]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions  #[4,2,1]
        self.dropout = dropout
        self.channel_mult = channel_mult   #[1,2,4,4]
        self.conv_resample = conv_resample  #True
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads   #-1?
        self.num_head_channels = num_head_channels    #64
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4   # 320 * 4
        self.time_embed = nn.Sequential(      #320 -> 1280
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels + hint_channels, model_channels, 3, padding=1)  
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels  # resblock输出的通道数
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, **kwargs):  #x: x_noisy, hint: c_latent
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)  #[6, 320]
        emb = self.time_embed(t_emb)  #[6, 1280]
        x = torch.cat((x, hint), dim=1)  #[6, 8, 64, 64]
        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs


class ControlLDM(LatentDiffusion):

    def __init__(
        self,
        control_stage_config: Mapping[str, Any],
        control_key: str,
        sd_locked: bool,
        only_mid_control: bool,
        learning_rate: float,
        preprocess_config,
        *args,
        **kwargs
    ) -> "ControlLDM":
        super().__init__(*args, **kwargs)
        # instantiate control module
        self.control_model: ControlNet = instantiate_from_config(control_stage_config)    #新加的control model
        # with open('/public_bme/data/lifeng/code/moco/DiffBIR-main/control_model_arch.txt','a') as file0:
        #     print(self.control_model, file=file0)
        self.control_key = control_key
        self.sd_locked = sd_locked
        self.only_mid_control = only_mid_control
        self.learning_rate = learning_rate
        self.control_scales = [1.0] * 13
        
        # instantiate preprocess module (SwinIR)
        self.preprocess_model = instantiate_from_config(preprocess_config)              #新加的preprocess model (swinir)
        frozen_module(self.preprocess_model)
        
        # instantiate condition encoder, since our condition encoder has the same 
        # structure with AE encoder, we just make a copy of AE encoder. please
        # note that AE encoder's parameters has not been initialized here.
        self.cond_encoder = nn.Sequential(OrderedDict([
            ("encoder", copy.deepcopy(self.first_stage_model.encoder)), # cond_encoder.encoder
            ("quant_conv", copy.deepcopy(self.first_stage_model.quant_conv)) # cond_encoder.quant_conv
        ]))
        frozen_module(self.cond_encoder)
        frozen_module(self.cond_stage_model)
        #frozen_module(self.model)

    def apply_condition_encoder(self, control): #[1, 3, 1024, 1024]
        c_latent_meanvar = self.cond_encoder(control * 2 - 1)
        c_latent = DiagonalGaussianDistribution(c_latent_meanvar).mode() # only use mode
        c_latent = c_latent * self.scale_factor     #通过上面得到的高斯分布采样结果 * rescale系数，保证latent的标准差接近1（防止扩散过程的SNR较高，影响生成效果）
        return c_latent
    
    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)  #获取target_latent, key = 'jpg', 这里的c就是text features,用于后续做cross-attention
        control = batch[self.control_key]   #key = 'hint', 获取control的img, [12,6,512,512,3]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        #control = einops.rearrange(control, 'b h w c -> b c h w')
        control = einops.rearrange(control, 'b n h w c -> b n c h w')
        control = control.to(memory_format=torch.contiguous_format).float()   #内存连续，加快数据处理速度
        #lq = control
        lq = control[:,2,...]
        # apply preprocess model
        control = self.preprocess_model(control)      #跟SD不同的地方，加入了preprocess的model (SwinIR), lq图经过swinir得到I_reg
        
        """
        experiment 1: 做一个SD_naive对比模型效果:
        1) 不需要第一步恢复（注释掉上面一行，control = self.preprocess_model(control)）后的图像作为control,直接用latent(lq)作为control，看下结果指标如何
        2) sampling阶段不需要MSEGuidance, 将cond_fn变为None;
        3) 不需要log_images了，以免过多保存实验过程中的图片.
        """
        """
        experiment 2: 引入(hq, t, lq)作为输入的思路，
        """
        # apply condition encoder
        c_latent = self.apply_condition_encoder(control)   #I_reg经过AE，得到latent(I_reg)
        return x, dict(c_crossattn=[c], c_latent=[c_latent], lq=[lq], c_concat=[control])   #lq是含伪影图形，control是lq经过swinir恢复以后的I_reg, c_latent是AE后得到的latent(I_reg)

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model  #ControlledUnetModel

        cond_txt = torch.cat(cond['c_crossattn'], 1)  #outputs = torch.cat(inputs, dim=1), cond['c_crossattn']是一个list

        if cond['c_latent'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(
                x=x_noisy, hint=torch.cat(cond['c_latent'], 1),
                timesteps=t, context=cond_txt
            )
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)  #[6, 4, 64, 64]

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, sample_steps=30):
        log = dict()
        z, c = self.get_input(batch, self.first_stage_key)
        c_lq = c["lq"][0]
        c_latent = c["c_latent"][0]
        c_cat, c = c["c_concat"][0], c["c_crossattn"][0]

        log["hq"] = (self.decode_first_stage(z) + 1) / 2
        log["control"] = c_cat
        log["decoded_control"] = (self.decode_first_stage(c_latent) + 1) / 2
        log["lq"] = c_lq
        log["text"] = (log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16) + 1) / 2
        
        samples = self.sample_log(   #samples=self.sample_log(
            # TODO: remove c_concat from cond
            cond={"c_concat": [c_cat], "c_crossattn": [c], "c_latent": [c_latent]},
            steps=sample_steps
        )

        #x_samples = self.decode_first_stage(samples)
        #log["samples"] = (x_samples + 1) / 2
        log["samples"] = samples

        """Calculate quality score
        """
        # z = ((z+1) / 2).clamp(0,1)                #[0,1]
        # z_restored = (z_restored+1)/2   #[0,1]
        # qs_scores = self.cal_cossim(z,z_restored)   #[n,4,64,64]
        # log["qs"] = qs_scores

        return log
    
    @torch.no_grad()
    def calc_score(self, a: torch.tensor, b: torch.tensor, sim_f: torch.nn.Module, repeats: int):
        """ Calculates similarities between embeddings in a and b using similarity function sim_f.

        Args:
            a (torch.tensor): First tensor of embeddings.
            b (torch.tensor): Second tensor of embeddings.
            sim_f (torch.nn.Module): Similarity function.
            repeats (int): Number of repeats used.

        Returns:
            np.array: Numpy array of mean similarity values.
        """

        sims = sim_f(a, b)
        #sims_unpacked = einops.rearrange(sims, "(b n) -> b n", n=repeats)
        #sims_mean = torch.mean(sims_unpacked, dim=1).cpu().numpy()
        sims_mean = torch.mean(sims).cpu()#.numpy()

        return sims_mean

    @torch.no_grad()
    def cal_cossim(self, z_start, z_restored, t_steps=100):
        t = torch.tensor([t_steps]).long().cuda()
        noise = torch.randn_like(z_start)
        z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
        # Calculate the quality scores
        cossim = torch.nn.CosineSimilarity()
        repeats = 1
        qs_scores = self.calc_score(z_start, z_noisy, cossim, repeats) + self.calc_score(z_start, z_restored, cossim, repeats) 

        return qs_scores

    @torch.no_grad()
    def sample_log(self, cond, steps):
        sampler = SpacedSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        control = cond["c_concat"][0]
        shape = (b, self.channels, h // 8, w // 8)
        # samples = sampler.sample(
        #     steps, shape, cond, unconditional_guidance_scale=1.0,
        #     unconditional_conditioning=None
        #)
        cond_fn = MSEGuidance(
                    scale=200, t_start=200, t_stop=-1,
                    space="latent", repeat=5
        )
        if cond_fn is not None:
            cond_fn.load_target(2 * control - 1)
        samples = sampler.sample(          #samples
            steps, shape, cond_img=control, cfg_scale=1.0,
            positive_prompt="", negative_prompt="",
            cond_fn=None
        )
        return samples    #return samples

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):  #把validation_step按照training_step补上
        # TODO: 
        #pass
    
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

class ControlLDM_MS(LatentDiffusion):

    def __init__(
        self,
        control_stage_config: Mapping[str, Any],
        control_key: str,
        sd_locked: bool,
        only_mid_control: bool,
        learning_rate: float,
        preprocess_config,
        *args,
        **kwargs
    ) -> "ControlLDM":
        super().__init__(*args, **kwargs)
        # instantiate control module
        self.control_model: ControlNet = instantiate_from_config(control_stage_config)    #新加的control model
        # with open('/public_bme/data/lifeng/code/moco/DiffBIR-main/control_model_arch.txt','a') as file0:
        #     print(self.control_model, file=file0)
        self.control_key = control_key
        self.sd_locked = sd_locked
        self.only_mid_control = only_mid_control
        self.learning_rate = learning_rate
        self.control_scales = [1.0] * 13
        
        # instantiate preprocess module (SwinIR)
        self.preprocess_model = instantiate_from_config(preprocess_config)              #新加的preprocess model (swinir)
        frozen_module(self.preprocess_model)
        
        # instantiate condition encoder, since our condition encoder has the same 
        # structure with AE encoder, we just make a copy of AE encoder. please
        # note that AE encoder's parameters has not been initialized here.
        self.cond_encoder = nn.Sequential(OrderedDict([
            ("encoder", copy.deepcopy(self.first_stage_model.encoder)), # cond_encoder.encoder
            ("quant_conv", copy.deepcopy(self.first_stage_model.quant_conv)) # cond_encoder.quant_conv
        ]))
        frozen_module(self.cond_encoder)
        frozen_module(self.cond_stage_model)
        #frozen_module(self.model)

    def apply_condition_encoder(self, control): #[1, 3, 1024, 1024]
        c_latent_meanvar = self.cond_encoder(control * 2 - 1)
        c_latent = DiagonalGaussianDistribution(c_latent_meanvar).mode() # only use mode
        c_latent = c_latent * self.scale_factor     #通过上面得到的高斯分布采样结果 * rescale系数，保证latent的标准差接近1（防止扩散过程的SNR较高，影响生成效果）
        return c_latent
    
    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x = batch[self.first_stage_key]
        if bs is not None:
            x = x[:bs]

        b, t, c, h, w = x.shape
        x = x.view(-1, c, h, w).to(self.device)    #[bt,c,h,w]

        encoder_posterior = self.encode_first_stage(x)   #得到latent的均值和标准差
        x = self.get_first_stage_encoding(encoder_posterior).detach()  #得到target经过AE后得到的hq_latent, [5*12, 4, 64, 64] 
        """还要添加c
        """
        c = [""]*b
        #x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)  #获取target_latent, key = 'jpg', 这里的c是否为None??
        
        control = batch[self.control_key]   #key = 'hint', 获取control的img, [12,6,512,512,3] -> [12,10,h,w,3]
        control_res = []
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        #control = einops.rearrange(control, 'b h w c -> b c h w')
        control = einops.rearrange(control, 'b n h w c -> b n c h w')
        control = control.to(memory_format=torch.contiguous_format).float()   #内存连续，加快数据处理速度, [b,10,3,h,w]
        #lq = control
        lq = control[:,2:7,...]
        lq = lq.view(-1,c,h,w)  #[5*12, 3, 512, 512] 
        # apply preprocess model
        for i in range(control.shape[1]):
            control_pre = control[:,i:i+6,...]
            control_pre = self.preprocess_model(control_pre)      #跟SD不同的地方，加入了preprocess的model (SwinIR), lq图经过swinir得到I_reg
            control_res.append(control_pre)
        control = np.stack(control_res, axis=1)  #[12,5,3,512,512]
        b, t, c, h, w = control.shape
        control = control.view(-1, c, h, w).to(self.device)    #[bt,3,h,w]
        # apply condition encoder
        c_latent = self.apply_condition_encoder(control)   #I_reg经过AE，得到latent(I_reg), [bt,4,64,64]
        return x, dict(c_crossattn=[c], c_latent=[c_latent], lq=[lq], c_concat=[control])   #lq是含伪影图形，control是lq经过swinir恢复以后的I_reg, c_latent是AE后得到的latent(I_reg)

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model  #ControlledUnetModel

        cond_txt = torch.cat(cond['c_crossattn'], 1)  #outputs = torch.cat(inputs, dim=1), cond['c_crossattn']是一个list

        if cond['c_latent'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(
                x=x_noisy, hint=torch.cat(cond['c_latent'], 1),
                timesteps=t, context=cond_txt
            )
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)  #[6, 4, 64, 64]

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, sample_steps=30):
        log = dict()
        z, c = self.get_input(batch, self.first_stage_key)
        c_lq = c["lq"][0]
        c_latent = c["c_latent"][0]
        c_cat, c = c["c_concat"][0], c["c_crossattn"][0]

        log["hq"] = (self.decode_first_stage(z) + 1) / 2
        log["control"] = c_cat
        log["decoded_control"] = (self.decode_first_stage(c_latent) + 1) / 2
        log["lq"] = c_lq
        log["text"] = (log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16) + 1) / 2
        
        samples = self.sample_log(   #samples=self.sample_log(
            # TODO: remove c_concat from cond
            cond={"c_concat": [c_cat], "c_crossattn": [c], "c_latent": [c_latent]},
            steps=sample_steps
        )

        #x_samples = self.decode_first_stage(samples)
        #log["samples"] = (x_samples + 1) / 2
        log["samples"] = samples

        """Calculate quality score
        """
        # z = ((z+1) / 2).clamp(0,1)                #[0,1]
        # z_restored = (z_restored+1)/2   #[0,1]
        # qs_scores = self.cal_cossim(z,z_restored)   #[n,4,64,64]
        # log["qs"] = qs_scores

        return log
    
    @torch.no_grad()
    def calc_score(self, a: torch.tensor, b: torch.tensor, sim_f: torch.nn.Module, repeats: int):
        """ Calculates similarities between embeddings in a and b using similarity function sim_f.

        Args:
            a (torch.tensor): First tensor of embeddings.
            b (torch.tensor): Second tensor of embeddings.
            sim_f (torch.nn.Module): Similarity function.
            repeats (int): Number of repeats used.

        Returns:
            np.array: Numpy array of mean similarity values.
        """

        sims = sim_f(a, b)
        #sims_unpacked = einops.rearrange(sims, "(b n) -> b n", n=repeats)
        #sims_mean = torch.mean(sims_unpacked, dim=1).cpu().numpy()
        sims_mean = torch.mean(sims).cpu()#.numpy()

        return sims_mean

    @torch.no_grad()
    def cal_cossim(self, z_start, z_restored, t_steps=100):
        t = torch.tensor([t_steps]).long().cuda()
        noise = torch.randn_like(z_start)
        z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
        # Calculate the quality scores
        cossim = torch.nn.CosineSimilarity()
        repeats = 1
        qs_scores = self.calc_score(z_start, z_noisy, cossim, repeats) + self.calc_score(z_start, z_restored, cossim, repeats) 

        return qs_scores

    @torch.no_grad()
    def sample_log(self, cond, steps):
        sampler = SpacedSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        control = cond["c_concat"][0]
        shape = (b, self.channels, h // 8, w // 8)
        # samples = sampler.sample(
        #     steps, shape, cond, unconditional_guidance_scale=1.0,
        #     unconditional_conditioning=None
        #)
        cond_fn = MSEGuidance(
                    scale=200, t_start=200, t_stop=-1,
                    space="latent", repeat=5
        )
        if cond_fn is not None:
            cond_fn.load_target(2 * control - 1)
        samples = sampler.sample(          #samples
            steps, shape, cond_img=control, cfg_scale=1.0,
            positive_prompt="", negative_prompt="",
            cond_fn=None
        )
        return samples    #return samples

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):  #把validation_step按照training_step补上
        # TODO: 
        #pass
    
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

              
        
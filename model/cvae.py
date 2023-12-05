import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
import clip
from transformers import CLIPTokenizer, CLIPTextModel

from model.rotation2xyz import Rotation2xyz
from model.transformer_decoder import TransformerDecoderLayer, TransformerDecoder


class CVAE(nn.Module):
    def __init__(
        self,
        modeltype,
        njoints,
        nfeats,
        num_actions,
        translation,
        pose_rep,
        glob,
        glob_rot,
        latent_dim=256,
        ff_size=1024,
        num_layers=8,
        num_heads=4,
        dropout=0.1,
        ablation=None,
        activation="gelu",
        legacy=False,
        data_rep="rot6d",
        dataset="amass",
        clip_dim=512,
        arch="trans_enc",
        emb_trans_dec=False,
        clip_version=None,
        num_code: int = 512,
        use_transformers_clip: bool = False,
        use_word_emb: bool = False,
        **kargs
    ):
        super().__init__()

        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_actions = num_actions
        self.data_rep = data_rep
        self.dataset = dataset

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get("action_emb", None)
        self.use_transformers_clip = use_transformers_clip
        self.use_word_emb = use_word_emb
        if self.use_word_emb:
            self.word_emb_attn = nn.TransformerDecoderLayer(
                d_model=self.latent_dim,
                nhead=self.num_heads,
                dim_feedforward=self.ff_size,
                dropout=self.dropout,
                activation=activation,
            )

        self.input_feats = self.njoints * self.nfeats

        self.normalize_output = kargs.get("normalize_encoder_output", False)

        self.cond_mode = kargs.get("cond_mode", "no_cond")
        self.cond_mask_prob = kargs.get("cond_mask_prob", 0.0)
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == "gru" else 0
        self.input_process = InputProcess(
            self.data_rep, self.input_feats + self.gru_emb_dim, self.latent_dim
        )

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.emb_trans_dec = emb_trans_dec

        self.mu_layer = nn.Linear(clip_dim, self.latent_dim)
        self.sigma_layer = nn.Linear(clip_dim, self.latent_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.text_embedding = nn.Linear(clip_dim, self.latent_dim)

        decoder_layer = TransformerDecoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=activation,
        )
        self.decoder = TransformerDecoder(decoder_layer, num_layers=self.num_layers)

        self.num_code = num_code
        self.codebook = nn.parameter.Parameter(torch.randn(self.num_code, self.latent_dim))

        if self.cond_mode != "no_cond":
            if "text" in self.cond_mode:
                self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
                print("EMBED TEXT")
                print("Loading CLIP...")
                self.clip_version = clip_version
                if use_transformers_clip:
                    self.load_and_freeze_clip_transformers(clip_version)
                else:
                    self.load_and_freeze_clip(clip_version)
            if "action" in self.cond_mode:
                self.embed_action = EmbedAction(self.num_actions, self.latent_dim)
                print("EMBED ACTION")

        self.output_process = OutputProcess(
            self.data_rep, self.input_feats, self.latent_dim, self.njoints, self.nfeats
        )

        self.rot2xyz = Rotation2xyz(device="cpu", dataset=self.dataset)

    def parameters_wo_clip(self):
        return [
            p
            for name, p in self.named_parameters()
            if not name.startswith("clip_model.")
        ]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(
            clip_version, device="cpu", jit=False
        )  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model
        )  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        self.clip_model = clip_model

    def load_and_freeze_clip_transformers(self, clip_version):
        tokenizer = CLIPTokenizer.from_pretrained(clip_version)
        text_model = CLIPTextModel.from_pretrained(clip_version)

        self.clip_tokenizer = tokenizer
        self.clip_text_model = text_model

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.0:
            mask = torch.bernoulli(
                torch.ones(bs, device=cond.device) * self.cond_mask_prob
            ).view(
                bs, 1
            )  # 1-> use null_cond, 0-> use real cond
            return cond * (1.0 - mask)
        else:
            return cond

    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = (
            20 if self.dataset in ["humanml", "kit"] else None
        )  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2  # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(
                raw_text, context_length=context_length, truncate=True
            ).to(
                device
            )  # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros(
                [texts.shape[0], default_context_length - context_length],
                dtype=texts.dtype,
                device=texts.device,
            )
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(
                device
            )  # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()

    def encode_text_transformers(self, raw_text):
        token = self.clip_tokenizer(raw_text, return_tensors="pt", padding="longest", truncation=True)
        ids = token.input_ids.to(next(self.parameters()).device)
        mask = token.attention_mask.to(next(self.parameters()).device)
        word_embed = self.clip_text_model(ids, mask)[0].float()
        sentence_embed = self.clip_text_model(ids, mask)[1].float()
        return word_embed, sentence_embed, mask

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        y: dict with keys "mask", "lengths", "text", "tokens"
            mask: [batch_size, 1, 1, max_frames] (bool)
            lengths: [batch_size] (int)
            text: list of length batch_size (string)
            tokens: list of length batch_size (string)
        """
        bs, njoints, nfeats, nframes = x.shape

        force_mask = y.get("uncond", False)
        if "text" in self.cond_mode:
            if self.use_transformers_clip:
                enc_text = self.encode_text_transformers(y["text"])
                word_emb, sentence_emb, word_mask = enc_text
                sentence_emb = self.embed_text(sentence_emb)
            else:
                sentence_emb = self.encode_text(y["text"])
                sentence_emb = self.embed_text(sentence_emb)
        if "action" in self.cond_mode:
            action_emb = self.embed_action(y["action"])
            emb = self.mask_cond(action_emb, force_mask=force_mask)

        mu = self.mu_layer(sentence_emb)
        sigma = self.sigma_layer(sentence_emb)

        x = self.input_process(x)  # [seqlen, bs, d]
        x = torch.cat([mu[None], sigma[None], x], dim=0)  # [seqlen+2, bs, d]
        xseq = self.sequence_pos_encoder(x)  # [seqlen+2, bs, d]
        encoder_output = self.encoder(xseq)

        mu = encoder_output[0]
        logvar = encoder_output[1]

        z = self.reparameterize(mu, logvar)
        if self.use_word_emb:
            bias = self.text_embedding(word_emb)
            bias = rearrange(bias, "b t d -> t b d")
            z = repeat(z, "b d -> t b d", t=nframes)
            pos_emb = self.sequence_pos_encoder(torch.zeros_like(z))
            bias = self.word_emb_attn(pos_emb, bias, memory_key_padding_mask=~word_mask.bool())
            tgt = z + bias
        else:
            bias = self.text_embedding(sentence_emb)
            z = z + bias
            z = repeat(z, "b d -> t b d", t=nframes)
            tgt = self.sequence_pos_encoder(z)  # [seqlen, bs, d]

        memory = self.codebook.unsqueeze(1)
        memory = repeat(memory, "n b d -> n (repeat b) d", repeat=bs)

        output, att_list = self.decoder(tgt, memory)

        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        return {"output": output, "att": att_list, "mu": mu, "logvar": logvar, "codebook": self.codebook,}

    def _apply(self, fn):
        super()._apply(fn)
        self.rot2xyz.smpl_model._apply(fn)

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.rot2xyz.smpl_model.train(*args, **kwargs)

    def sample(self, y):
        """
        y: dict with keys "mask", "lengths", "text", "tokens"
            mask: [batch_size, 1, 1, max_frames] (bool)
            lengths: [batch_size] (int)
            text: list of length batch_size (string)
            tokens: list of length batch_size (string)
        """
        bs = len(y["lengths"])
        nframes = y["mask"].shape[-1]

        force_mask = y.get("uncond", False)
        if "text" in self.cond_mode:
            if self.use_transformers_clip:
                enc_text = self.encode_text_transformers(y["text"])
                word_emb, sentence_emb, word_mask = enc_text
                sentence_emb = self.embed_text(sentence_emb)
            else:
                sentence_emb = self.encode_text(y["text"])
                sentence_emb = self.embed_text(sentence_emb)
        if "action" in self.cond_mode:
            action_emb = self.embed_action(y["action"])
            emb = self.mask_cond(action_emb, force_mask=force_mask)
        
        z = torch.randn(bs, self.latent_dim).to(sentence_emb.device)
        if self.use_word_emb:
            bias = self.text_embedding(word_emb)
            bias = rearrange(bias, "b t d -> t b d")
            z = repeat(z, "b d -> t b d", t=nframes)
            pos_emb = self.sequence_pos_encoder(torch.zeros_like(z))
            bias = self.word_emb_attn(pos_emb, bias, memory_key_padding_mask=~word_mask.bool())
            tgt = z + bias
        else:
            bias = self.text_embedding(sentence_emb)
            z = z + bias
            z = repeat(z, "b d -> t b d", t=nframes)
            tgt = self.sequence_pos_encoder(z)  # [seqlen, bs, d]

        memory = self.codebook.unsqueeze(1)
        memory = repeat(memory, "n b d -> n (repeat b) d", repeat=bs)
        output, _ = self.decoder(tgt, memory)
        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == "rot_vel":
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints * nfeats)

        if self.data_rep in ["rot6d", "xyz", "hml_vec"]:
            x = self.poseEmbedding(x)  # [seqlen, bs, d]
            return x
        elif self.data_rep == "rot_vel":
            first_pose = x[[0]]  # [1, bs, 150]
            first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
            vel = x[1:]  # [seqlen-1, bs, 150]
            vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        else:
            raise ValueError


class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == "rot_vel":
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        if self.data_rep in ["rot6d", "xyz", "hml_vec"]:
            output = self.poseFinal(output)  # [seqlen, bs, 150]
        elif self.data_rep == "rot_vel":
            first_pose = output[[0]]  # [1, bs, d]
            first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
            vel = output[1:]  # [seqlen-1, bs, d]
            vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
            output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        else:
            raise ValueError
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output


class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output

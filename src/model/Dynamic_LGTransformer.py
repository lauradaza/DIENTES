import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.LocalGlobalUnetr import CrossAttnUnetr

from src.utils.utils import get_key


class Dynamic_LocalGlobal(nn.Module):
    def __init__(
        self,
        img_size,
        in_channels,
        out_channels,
        entities=False,
        training=True,
    ):
        # encoding: rand_embedding or word_embedding
        super().__init__()
        self.backbone = CrossAttnUnetr(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=48,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=False,
            entities=entities,
        )

        # CLIP embeddings have size 512
        self.register_buffer("organ_embedding", torch.randn(out_channels, 512))
        self.text_to_vision = nn.Linear(512, 256)
        self.class_num = out_channels

        if training:
            self.forward = self.forward_train
        else:
            self.forward = self.forward_test

    def get_embedding(self, task):
        # --- If I had more datasets ---
        # organ_embed = []
        # for t in task:
        #     idxs = self.organ_embed_idxs[get_key(t)]
        #     organ_embed.append(F.relu(self.text_to_vision(self.organ_embedding[idxs])))
        # ------------------------------
        idxs = self.organ_embed_idxs[get_key(task[0])]
        organ_embed = F.relu(self.text_to_vision(self.organ_embedding[idxs]))
        return organ_embed

    def get_test_kernels(self, task):
        task_embedding = self.get_embedding(task)
        weights, bias = self.backbone.get_dynamic_params(task_embedding)
        self.dyn_weights = weights
        self.dyn_bias = bias
        self.num_insts = len(task_embedding)

    def forward_train(self, x_in):
        x_in, metadata = x_in
        out_dec = self.backbone(x_in, metadata=metadata)

        task_embedding = self.get_embedding(metadata["task"])
        weights, bias = self.backbone.get_dynamic_params(task_embedding)
        out = self.backbone.forward_output(out_dec, weights, bias, len(task_embedding))
        return out, out_dec

    def forward_test(self, x_in, patch_coord, metadata=None):
        # start is already the upper corner
        coords = [[x.start, y.start, z.start] for (b, c, x, y, z) in patch_coord]
        coords = torch.tensor(coords, device=x_in.device) / metadata["shape"]
        metadata["corner"] = coords * 100

        out_dec = self.backbone(x_in, metadata=metadata)
        logits = self.backbone.forward_output(
            out_dec, self.dyn_weights, self.dyn_bias, self.num_insts
        )
        if isinstance(logits, list):
            logits = torch.stack(logits, 0)
        return logits, out_dec

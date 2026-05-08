import torch.nn.functional as F
from torch import nn


class PatchEmbed(nn.Module):
  def __init__(self, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, bias=True, dtype=None, prefix: str = ""):
    super().__init__()
    if isinstance(patch_size, list | tuple):
      if len(patch_size) == 1:
        patch_size = (1, patch_size[0], patch_size[0])
      elif len(patch_size) == 2:
        patch_size = (1, patch_size[0], patch_size[1])
    else:
      patch_size = (1, patch_size, patch_size)

    self.patch_size = patch_size
    self.flatten = flatten

    self.proj = nn.Conv3d(
      in_chans,
      embed_dim,
      kernel_size=patch_size,
      stride=patch_size,
      bias=bias,
      dtype=dtype,
    )
    self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

  def forward(self, x):
    if x.dim() == 5:
      B, C, T, H, W = x.shape
      pt, ph, pw = self.patch_size

      if T % pt == 0 and H % ph == 0 and W % pw == 0:
        T_ = T // pt
        H_ = H // ph
        W_ = W // pw

        x = x.reshape(B, C, T_, pt, H_, ph, W_, pw)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        x = x.reshape(B, T_ * H_ * W_, C * pt * ph * pw)

        w = self.proj.weight.reshape(self.proj.weight.shape[0], -1)
        x = F.linear(x, w, self.proj.bias)  # [B, T'*H'*W', embed_dim]

        if not self.flatten:
          x = x.reshape(B, T_, H_, W_, -1).permute(0, 4, 1, 2, 3).contiguous()

        x = self.norm(x)
        return x

    # Fallback to Conv3d for non-5D input or indivisible spatial dims.
    x = self.proj(x)
    if self.flatten:
      x = x.flatten(2).transpose(1, 2)
    x = self.norm(x)
    return x

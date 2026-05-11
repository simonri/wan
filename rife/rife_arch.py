"""
RIFE 4.7 architecture (covers checkpoints 4.7-4.9).
https://github.com/hzwer/Practical-RIFE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

_backwarp_grid_cache = {}


class ResConv(nn.Module):
  def __init__(self, c, dilation=1):
    super().__init__()
    self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
    self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
    self.relu = nn.LeakyReLU(0.2, True)

  def forward(self, x):
    return self.relu(self.conv(x) * self.beta + x)


def warp(x, flow):
  key = (str(flow.device), tuple(flow.shape))
  if key not in _backwarp_grid_cache:
    horizontal = (
      torch.linspace(-1.0, 1.0, flow.shape[3], device=flow.device)
      .view(1, 1, 1, flow.shape[3])
      .expand(flow.shape[0], -1, flow.shape[2], -1)
    )
    vertical = (
      torch.linspace(-1.0, 1.0, flow.shape[2], device=flow.device)
      .view(1, 1, flow.shape[2], 1)
      .expand(flow.shape[0], -1, -1, flow.shape[3])
    )
    _backwarp_grid_cache[key] = torch.cat([horizontal, vertical], 1)

  flow = torch.cat(
    [
      flow[:, 0:1, :, :] / ((x.shape[3] - 1.0) / 2.0),
      flow[:, 1:2, :, :] / ((x.shape[2] - 1.0) / 2.0),
    ],
    1,
  )
  grid = (_backwarp_grid_cache[key] + flow).permute(0, 2, 3, 1)
  return F.grid_sample(input=x, grid=grid, mode="bilinear", padding_mode="border", align_corners=True)


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
  return nn.Sequential(
    nn.Conv2d(
      in_planes,
      out_planes,
      kernel_size=kernel_size,
      stride=stride,
      padding=padding,
      dilation=dilation,
      bias=True,
    ),
    nn.LeakyReLU(0.2, True),
  )


class Conv2(nn.Module):
  def __init__(self, in_planes, out_planes, stride=2):
    super().__init__()
    self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
    self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

  def forward(self, x):
    return self.conv2(self.conv1(x))


class IFBlock(nn.Module):
  def __init__(self, in_planes, c=64):
    super().__init__()
    self.conv0 = nn.Sequential(
      conv(in_planes, c // 2, 3, 2, 1),
      conv(c // 2, c, 3, 2, 1),
    )
    self.convblock = nn.Sequential(*[ResConv(c) for _ in range(8)])
    self.lastconv = nn.Sequential(nn.ConvTranspose2d(c, 4 * 6, 4, 2, 1), nn.PixelShuffle(2))

  def forward(self, x, flow=None, scale=1):
    x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
    if flow is not None:
      flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False) / scale
      x = torch.cat((x, flow), 1)
    feat = self.conv0(x)
    feat = self.convblock(feat)
    tmp = self.lastconv(feat)
    tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
    flow = tmp[:, :4] * scale
    mask = tmp[:, 4:5]
    return flow, mask


class IFNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.block0 = IFBlock(7 + 8, c=192)
    self.block1 = IFBlock(8 + 4 + 8, c=128)
    self.block2 = IFBlock(8 + 4 + 8, c=96)
    self.block3 = IFBlock(8 + 4 + 8, c=64)
    self.encode = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.ConvTranspose2d(16, 4, 4, 2, 1))

  def forward(self, img0, img1, timestep=0.5, scale_list=(8, 4, 2, 1), ensemble=False):
    img0 = torch.clamp(img0, 0, 1)
    img1 = torch.clamp(img1, 0, 1)

    _, _, h, w = img0.shape
    ph = ((h - 1) // 64 + 1) * 64
    pw = ((w - 1) // 64 + 1) * 64
    padding = (0, pw - w, 0, ph - h)
    img0 = F.pad(img0, padding)
    img1 = F.pad(img1, padding)

    if not torch.is_tensor(timestep):
      timestep = torch.full_like(img0[:, :1], timestep)
    else:
      timestep = timestep.repeat(1, 1, img0.shape[2], img0.shape[3])

    f0 = self.encode(img0)
    f1 = self.encode(img1)

    flow = None
    mask = None
    warped_img0 = img0
    warped_img1 = img1
    blocks = [self.block0, self.block1, self.block2, self.block3]

    for i, block in enumerate(blocks):
      if flow is None:
        flow, mask = block(
          torch.cat((img0, img1, f0, f1, timestep), 1),
          None,
          scale=scale_list[i],
        )
        if ensemble:
          f_, m_ = block(
            torch.cat((img1, img0, f1, f0, 1 - timestep), 1),
            None,
            scale=scale_list[i],
          )
          flow = (flow + torch.cat((f_[:, 2:4], f_[:, :2]), 1)) / 2
          mask = (mask - m_) / 2
      else:
        fd, m0 = block(
          torch.cat(
            (warped_img0, warped_img1, warp(f0, flow[:, :2]), warp(f1, flow[:, 2:4]), timestep, mask),
            1,
          ),
          flow,
          scale=scale_list[i],
        )
        flow = flow + fd
        if ensemble:
          _, m_ = block(
            torch.cat(
              (
                warped_img1,
                warped_img0,
                warp(f1, flow[:, 2:4]),
                warp(f0, flow[:, :2]),
                1 - timestep,
                -mask,
              ),
              1,
            ),
            torch.cat((flow[:, 2:4], flow[:, :2]), 1),
            scale=scale_list[i],
          )
          mask = (m0 - m_) / 2
        else:
          mask = m0

      warped_img0 = warp(img0, flow[:, :2])
      warped_img1 = warp(img1, flow[:, 2:4])

    mask = torch.sigmoid(mask)
    merged = warped_img0 * mask + warped_img1 * (1 - mask)
    return merged[:, :, :h, :w]

"""The 2026-11-11 volumetric study's 3D network, verbatim (its `vol3d.py`).

Kept as a reference: the study's checkpoints must load into the packaged
`sleap_nn.architectures.unet3d.UNet3D` unchanged and give the same output (same module
names, same `[upsampled, skip]` concatenation order).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StudyBlock3d(nn.Sequential):
    def __init__(self, cin, cout, norm="group", convs=2):
        layers = []
        for i in range(convs):
            layers += [
                nn.Conv3d(cin if i == 0 else cout, cout, 3, padding=1),
                nn.GroupNorm(min(8, cout), cout),
                nn.ReLU(inplace=True),
            ]
        super().__init__(*layers)


class StudyUNet3D(nn.Module):
    def __init__(self, in_ch, out_ch, filters=32, rate=2.0, down=3, norm="group"):
        super().__init__()
        ch = [int(round(filters * rate**i)) for i in range(down + 1)]
        self.enc = nn.ModuleList()
        c = in_ch
        for i in range(down):
            self.enc.append(StudyBlock3d(c, ch[i], norm))
            c = ch[i]
        self.mid = StudyBlock3d(c, ch[down], norm)
        self.dec = nn.ModuleList()
        c = ch[down]
        for i in reversed(range(down)):
            self.dec.append(StudyBlock3d(c + ch[i], ch[i], norm))
            c = ch[i]
        self.head = nn.Conv3d(c, out_ch, 1)

    def forward(self, x):
        skips = []
        for blk in self.enc:
            x = blk(x)
            skips.append(x)
            x = F.max_pool3d(x, 2)
        x = self.mid(x)
        for blk, s in zip(self.dec, reversed(skips)):
            x = F.interpolate(
                x, size=s.shape[-3:], mode="trilinear", align_corners=False
            )
            x = blk(torch.cat([x, s], 1))
        return self.head(x)

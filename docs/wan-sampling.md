# Wan2.2 I2V-A14B sampling config

Verified values for the sampling/scheduler knobs, so they don't get second-guessed.
Set in `wan/configs/pipeline/wan.py` (`flow_shift`, `boundary_ratio`).

## flow_shift = 5.0 — for **both** 720p and 480p (not 8)

`flow_shift=5.0` is the official Wan2.2 I2V-A14B value and what the lightx2v distill
was trained with. Confirmed identical across every authoritative source:

| source | sample_shift |
|---|---|
| Wan2.2 official `wan_i2v_A14B.py` | 5.0 |
| lightx2v `wan_moe_i2v.json` (Wan2.2) | 5.0 |
| lightx2v `wan_moe_t2v_distill.json` (4-step) | 5.0 |

- **Not resolution-dependent.** Wan2.2 uses 5.0 at both 480p and 720p. (The 720p=5 / 480p=3 split was a *Wan2.1* thing and was dropped.)
- **Not 8.** Shift 8 is a community/ComfyUI value, off-recipe here — it would move the high/low MoE split away from what the distill expects.
- lightx2v docs say `sample_shift` is **pinned to the training value, don't modify** (and `enable_cfg=false`, which matches our CFG-off setup).

## boundary_ratio = 0.9 → high/low expert split

The A14B MoE runs a **high-noise** expert (`transformer`) for `t ≥ boundary_timestep`
and a **low-noise** expert (`transformer_2`) for `t < boundary_timestep`, where
`boundary_timestep = boundary_ratio · 1000 = 900`.

With `flow_shift=5.0` and **8 inference steps**, the schedule and split are:

```
timesteps: [1000.0, 967.9, 926.4 | 870.6, 791.4, 670.5, 463.1, 24.4]
HIGH (t≥900): steps 0,1,2   (3 steps)
LOW  (t<900): steps 3,4,5,6,7 (5 steps)
```

So it's **3 high / 5 low** at 8 steps — not 4/4. (Recompute if you change steps or shift.)

## Distill recipe (lightx2v 4-step + lightning)

- `enable_cfg = false` — CFG must be off for the distill; otherwise output blurs.
- Official base recipe is 40 steps @ CFG (3.5, 3.5); the distill runs few-step
  (4-step list `[1000, 750, 500, 250]`) with CFG off.
- High-noise checkpoint is the lightx2v 4-step distill; low-noise gets the lightning LoRA.

## Sources

- [Wan-Video/Wan2.2 — `wan_i2v_A14B` config](https://github.com/Wan-Video/Wan2.2)
- [ModelTC/LightX2V — `configs/wan22`](https://github.com/ModelTC/LightX2V/tree/main/configs/wan22)
- [lightx2v step-distillation docs](https://lightx2v-en.readthedocs.io/en/latest/method_tutorials/step_distill.html)
- [lightx2v/Wan2.2-Distill-Loras](https://huggingface.co/lightx2v/Wan2.2-Distill-Loras)

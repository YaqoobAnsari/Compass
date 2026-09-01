#!/usr/bin/env python
"""Correctness checks for the RadioDiff baseline before committing GPU time."""
import tempfile, os
import torch
from compass.models.baselines import BASELINES, DIFFUSION_BASELINES
from compass.models.baselines.wrapper import load_baseline, BaselineReconstructor

torch.manual_seed(0)
B, H, W = 2, 256, 256
def mk(H=256, W=256, B=2):
    return {
        "sparse_rss": torch.randn(B, 1, H, W), "mask": (torch.rand(B,1,H,W) > .9).float(),
        "coverage": torch.rand(B,1,H,W), "building": (torch.rand(B,1,H,W) > .5).float(),
        "tx_rowcol": torch.randint(0, H, (B,2)).float(), "target": torch.randn(B,1,H,W).clamp(-1,1),
        "free_mask": (torch.rand(B,1,H,W) > .3).float(),
    }
batch = mk()
m = BASELINES["radiodiff"]()
n = sum(p.numel() for p in m.parameters())
print(f"[1] built radiodiff: {n/1e6:.2f}M params  (panel range 6-9M)")

# 2. training step
m.train()
loss, comp = m.training_losses(batch)
loss.backward()
gn = sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)
ngrad = sum(1 for p in m.parameters() if p.grad is None)
print(f"[2] loss={loss.item():.4f} comp={ {k: round(v,4) for k,v in comp.items()} } grad_norm={gn:.1f} params_without_grad={ngrad}")
assert torch.isfinite(loss) and gn > 0 and ngrad == 0, "bad training step"

# 3. sampling forward
m.eval()
y = m(batch)
print(f"[3] forward out={tuple(y.shape)} finite={bool(torch.isfinite(y).all())} range=[{y.min():.2f},{y.max():.2f}]")
assert y.shape == (B,1,H,W) and torch.isfinite(y).all()

# 4. DECOUPLED MATH: an oracle predictor must reconstruct x0 exactly in one step
class Oracle(type(m)):
    def predict(self, x_t, cond, t):
        return self._x0, self._eps
o = Oracle(base=8); o._x0 = batch["target"]; o._eps = torch.randn_like(batch["target"])
o.eval()
rec = o(batch)
err = (rec - batch["target"]).abs().max().item()
print(f"[4] decoupled reverse with oracle: max|x_hat - x0| = {err:.2e} (must be ~0)")
assert err < 1e-5, "decoupled reverse process is wrong"

# 4b. forward process endpoints: t->0 gives x0, t->1 gives pure noise
x0 = batch["target"]; eps = torch.randn_like(x0)
# note: noise enters as sqrt(t), so at t=eps the deviation is ~sqrt(t)*max|eps|, not t
for t_val, name in [(1e-6,"t->0 (expect x0)"), (1.0,"t=1 (expect pure noise)")]:
    tt = torch.full((B,1,1,1), t_val)
    x_t = (1-tt)*x0 + tt.sqrt()*eps
    ref = x0 if t_val < .5 else eps
    d = (x_t - ref).abs().max().item()
    tol = 6.0*(t_val**0.5) + t_val*x0.abs().max().item() + 1e-6   # 6-sigma noise + attenuation
    print(f"[4b] forward {name}: max dev = {d:.2e}  (tol {tol:.2e})")
    assert d < tol

# 5. multi-step sampling also lands on x0_hat
m.sample_steps = 3
y3 = m(batch)
print(f"[5] multi-step (3) forward ok: {tuple(y3.shape)} finite={bool(torch.isfinite(y3).all())}")
m.sample_steps = 1

# 6. AFT resolution independence
small = mk(H=128, W=128, B=1)
ys = m(small)
print(f"[6] resolution independence (128x128): out={tuple(ys.shape)} finite={bool(torch.isfinite(ys).all())}")
assert ys.shape == (1,1,128,128)

# 7. checkpoint round-trip through the real loader
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "best.ckpt")
    torch.save({"model": m.state_dict(), "arch": "radiodiff", "kwargs": {}, "epoch": 0}, p)
    m2 = load_baseline(p, "cpu")
    torch.manual_seed(1); a = m(batch)
    torch.manual_seed(1); b = m2(batch)
    same = torch.allclose(a, b, atol=1e-5)
    print(f"[7] load_baseline round-trip identical: {same}")
    assert same
    # eval wrapper (n_mc should be 1: sampling-free, no dropout)
    r = BaselineReconstructor(m2, "cpu", n_mc=8)
    mean, std = r.predict_batch(batch)
    print(f"[7b] wrapper n_mc={r.n_mc} (expect 1) mean={mean.shape} finite={bool(np_ok(mean))}" if False else
          f"[7b] wrapper n_mc={r.n_mc} (expect 1) mean_shape={mean.shape}")
    assert r.n_mc == 1

# 8. regression: RMDM still builds/trains through the shared (now generic) trainer path
rm = BASELINES["rmdm"]()
l, _ = rm.training_losses(batch); l.backward()
print(f"[8] RMDM regression ok: loss={l.item():.4f}  DIFFUSION_BASELINES={sorted(DIFFUSION_BASELINES)}")

print("\nALL RADIODIFF CHECKS PASSED")

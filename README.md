# FCM + Graph Cut Image Segmentation

Binary image segmentation combining **Fuzzy C-Means** (for the data term)
with a **from-scratch Graph Cut / Min-Cut-Max-Flow** solver (for the
smoothness term).

- FCM uses `scikit-fuzzy` (library).
- Max-flow is implemented from scratch (`maxflow.py`) using Edmonds–Karp
  on an adjacency-list residual graph, with a min-cut extraction step.

## Project layout
```
fcm_graphcut/
├── fcm_graphcut/
│   ├── __init__.py
│   ├── fcm.py              # FCM wrapper (skfuzzy)
│   ├── maxflow.py          # from-scratch Graph + Edmonds-Karp + min-cut
│   ├── graph.py            # build flow graph from image + FCM memberships
│   ├── segment.py          # pipeline + Dice/IoU metrics
│   ├── viz.py              # overlay & panel plots
│   ├── cli.py              # command-line entry point
│   └── make_synthetic.py   # generates toy test images + GT
├── requirements.txt
└── README.md
```

## Install
```bash
pip install -r requirements.txt
```

## Quick demo (synthetic data)
```bash
python -m fcm_graphcut.make_synthetic --out samples
python -m fcm_graphcut.cli --image samples/circle.png \
    --gt samples/circle_gt.png --out outputs/circle
```

## CLI
```bash
python -m fcm_graphcut.cli \
    --image  path/to/input.png \
    --gt     path/to/gt_mask.png      # optional, for Dice/IoU
    --out    output_dir \
    --clusters 2 \
    --m 2.0 \
    --sigma 30 \
    --lambda-data 1.0 \
    --lambda-smooth 20.0 \
    --resize 256                       # optional: speeds up large images
```

### Arguments
| flag | default | meaning |
|---|---|---|
| `--image` | required | grayscale input image (any PIL-readable format) |
| `--gt` | None | binary ground-truth mask for Dice/IoU |
| `--out` | `output` | directory for `mask.png`, `overlay.png`, `panel.png` |
| `--clusters` | 2 | FCM clusters (keep 2 for binary) |
| `--m` | 2.0 | FCM fuzzifier |
| `--sigma` | 30 | Gaussian σ for n-link intensity similarity |
| `--lambda-data` | 1.0 | weight of data term (FCM) |
| `--lambda-smooth` | 20.0 | weight of smoothness term (n-links) |
| `--resize` | None | shrink longest side to N pixels |
| `--invert-gt` | off | invert GT if foreground is encoded as 0 |

### Outputs
- `mask.png` — binary segmentation (255=object, 0=background)
- `overlay.png` — input with object pixels tinted red
- `panel.png` — 4- or 5-panel figure: input, FCM membership, mask, overlay, (GT)
- stdout — max-flow value, foreground count, **Dice** and **IoU** if GT given

## Pipeline
```
Input image
   │
   ▼
FCM (c=2, m=2)        ──► cluster centers & membership matrix U
   │
   ▼
Data costs
   D_bg = −ln(u_bg)   ──► capacity(S → pᵢ)
   D_obj = −ln(u_obj) ──► capacity(pᵢ → T)
   │
   ▼
n-links (4-neighbors)
   w_ij = exp(−(I_i−I_j)² / 2σ²)
   │
   ▼
Max-flow / Min-cut (Edmonds–Karp, from scratch)
   │
   ▼
Source-side nodes → label = object → binary mask
```

## Convention reminder
With source S = object, sink T = background:
- `cap(S → pᵢ) = D_i(bg)` — cost paid if pᵢ ends on the **T side**
- `cap(pᵢ → T) = D_i(obj)` — cost paid if pᵢ ends on the **S side**

After `max_flow()`, nodes still reachable from S in the residual graph
are the **object** pixels.

## Tuning
- Bumping `--lambda-smooth` cleans up speckle but erodes thin structures.
- Lowering `--sigma` makes n-links more sensitive to small intensity changes
  (sharper boundaries, but jaggier).
- If the mask comes out inverted (foreground/background swapped), the CLI
  auto-flips it when a GT is provided; otherwise set `--invert-gt` or flip
  manually.

## Complexity
Edmonds–Karp is O(V·E²) worst-case; for a grid with V ≈ H·W and
E ≈ 5·H·W this handles 128×128 in a few seconds. For larger images
use `--resize 256` (or swap in Boykov–Kolmogorov).

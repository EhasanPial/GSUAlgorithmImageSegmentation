"""Interactive tool to draw FG/BG scribbles on an image and save them as masks.

Usage:
    python -m fcm_graphcut.draw_seeds --image my_image.png --out seeds_dir

Controls:
    Left-click + drag  : draw FG (red)
    Right-click + drag : draw BG (blue)
    Key 'f'            : switch to FG mode (left-click)
    Key 'b'            : switch to BG mode (left-click)
    Key 'e'            : eraser mode
    Key '+' / '-'      : brush size up/down
    Key 'c'            : clear all scribbles
    Key 's'            : save fg_mask.png and bg_mask.png to --out and exit
    Key 'q'            : quit without saving
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


class SeedDrawer:
    def __init__(self, image: np.ndarray, out_dir: Path, brush: int = 5):
        self.image = image
        self.out_dir = out_dir
        self.fg = np.zeros(image.shape, dtype=np.uint8)
        self.bg = np.zeros(image.shape, dtype=np.uint8)
        self.brush = brush
        self.mode = "fg"   # "fg", "bg", "erase"
        self.button_held = None

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.imshow(self.image, cmap="gray")
        self.overlay = self.ax.imshow(self._make_overlay(), alpha=0.55)
        self.title = self.ax.set_title(self._title_text())
        self.ax.axis("off")

        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def _title_text(self):
        return (f"mode={self.mode}  brush={self.brush}    "
                f"[L=draw  R=BG  f/b/e=mode  +/-=size  c=clear  s=save  q=quit]")

    def _make_overlay(self) -> np.ndarray:
        H, W = self.image.shape
        rgba = np.zeros((H, W, 4), dtype=np.float32)
        # FG = red, BG = blue
        rgba[self.fg > 0] = [1.0, 0.1, 0.1, 1.0]
        rgba[self.bg > 0] = [0.1, 0.3, 1.0, 1.0]
        return rgba

    def _refresh(self):
        self.overlay.set_data(self._make_overlay())
        self.title.set_text(self._title_text())
        self.fig.canvas.draw_idle()

    def _paint(self, x: int, y: int):
        H, W = self.image.shape
        r = self.brush
        yy, xx = np.ogrid[:H, :W]
        disk = (yy - y) ** 2 + (xx - x) ** 2 <= r * r
        if self.mode == "fg":
            self.fg[disk] = 1
            self.bg[disk] = 0
        elif self.mode == "bg":
            self.bg[disk] = 1
            self.fg[disk] = 0
        elif self.mode == "erase":
            self.fg[disk] = 0
            self.bg[disk] = 0

    def on_press(self, evt):
        if evt.inaxes != self.ax or evt.xdata is None:
            return
        self.button_held = evt.button
        prev_mode = self.mode
        if evt.button == 3:           # right click forces BG
            self.mode = "bg"
        self._paint(int(evt.xdata), int(evt.ydata))
        self._refresh()
        self.mode = prev_mode

    def on_release(self, evt):
        self.button_held = None

    def on_motion(self, evt):
        if self.button_held is None or evt.inaxes != self.ax or evt.xdata is None:
            return
        prev_mode = self.mode
        if self.button_held == 3:
            self.mode = "bg"
        self._paint(int(evt.xdata), int(evt.ydata))
        self._refresh()
        self.mode = prev_mode

    def on_key(self, evt):
        if evt.key == "f":
            self.mode = "fg"
        elif evt.key == "b":
            self.mode = "bg"
        elif evt.key == "e":
            self.mode = "erase"
        elif evt.key in ("+", "="):
            self.brush = min(self.brush + 1, 80)
        elif evt.key == "-":
            self.brush = max(self.brush - 1, 1)
        elif evt.key == "c":
            self.fg[:] = 0
            self.bg[:] = 0
        elif evt.key == "s":
            self.save()
            plt.close(self.fig)
            return
        elif evt.key == "q":
            print("[draw_seeds] quit without saving.")
            plt.close(self.fig)
            return
        self._refresh()

    def save(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        fg_path = self.out_dir / "fg_mask.png"
        bg_path = self.out_dir / "bg_mask.png"
        Image.fromarray(self.fg * 255).save(fg_path)
        Image.fromarray(self.bg * 255).save(bg_path)
        print(f"[draw_seeds] wrote {fg_path}  ({int(self.fg.sum())} px)")
        print(f"[draw_seeds] wrote {bg_path}  ({int(self.bg.sum())} px)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--out", default="seeds")
    p.add_argument("--brush", type=int, default=5)
    p.add_argument("--resize", type=int, default=None,
                   help="Resize image's longest side before drawing (matches CLI --resize).")
    args = p.parse_args()

    img = np.array(Image.open(args.image).convert("L"))
    if args.resize is not None:
        h, w = img.shape
        m = max(h, w)
        if m > args.resize:
            scale = args.resize / m
            new_h, new_w = int(round(h * scale)), int(round(w * scale))
            img = np.array(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))

    drawer = SeedDrawer(img, Path(args.out), brush=args.brush)
    plt.show()


if __name__ == "__main__":
    main()
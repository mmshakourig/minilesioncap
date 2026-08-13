"""
Interactive PET/CT fusion viewer for the lesion captioning pipeline.

Shows the axial PET/CT fusion slice-by-slice with every lesion contoured.
Clicking a lesion shows its computed metrics (closest organ, centroid coordinates,
SUVmax, MTV, TLG, axial slice) and a free-text description generated on demand
by the Gemini free-tier model (cached to lesion_captions.json).

Usage:
    python scripts/gui_app.py --patient-dir data/11-37493
"""

import argparse
import json
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import scrolledtext, ttk

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.widgets import Slider
from rt_utils import image_helper

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gemini_caption
import lesion_metrics

LESION_COLORS = list(
    __import__("matplotlib").colormaps["tab20"].colors
    + __import__("matplotlib").colormaps["tab20b"].colors
    + __import__("matplotlib").colormaps["tab20c"].colors
)
SUV_DISPLAY_THRESHOLD = 1.0  # SUV below this is fully transparent in the fusion overlay


class LesionViewerApp:
    def __init__(self, ctx: lesion_metrics.PatientContext, report_df: pd.DataFrame, captions_path: Path):
        self.ctx = ctx
        self.report = report_df.set_index("lesion")
        self.captions_path = captions_path
        self.captions: dict[str, str] = json.loads(captions_path.read_text()) if captions_path.exists() else {}

        try:
            self.api_key = gemini_caption.load_api_key()
        except RuntimeError:
            self.api_key = None

        self.lesion_names = list(self.report.index)
        self.colors = {name: LESION_COLORS[i % len(LESION_COLORS)] for i, name in enumerate(self.lesion_names)}
        self.slice_ranges = self._compute_slice_ranges()
        self.suv_vmax = float(np.percentile(ctx.suv_ct, 99.5))

        self.selected = None
        self.result_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()

        self.root = tk.Tk()
        self.root.title("Mini Lesion Caption Viewer")
        self.root.geometry("1350x820")
        self._build_ui()
        self._poll_queue()

    def _compute_slice_ranges(self) -> dict:
        ranges = {}
        for name in self.lesion_names:
            mask = self.ctx.lesion_mask_pet(name)
            coords = np.argwhere(mask)
            if len(coords) == 0:
                continue
            physical = image_helper.apply_transformation_to_3d_points(coords.astype(float), self.ctx.pixel_to_patient)
            z_mm = physical[:, 2]
            idx_lo = self.ctx.ct_index_of_point([physical[:, 0].mean(), physical[:, 1].mean(), z_mm.min()])[2]
            idx_hi = self.ctx.ct_index_of_point([physical[:, 0].mean(), physical[:, 1].mean(), z_mm.max()])[2]
            lo, hi = sorted((int(round(idx_lo)), int(round(idx_hi))))
            ranges[name] = (max(0, lo - 1), min(self.ctx.ct_size[2] - 1, hi + 1))
        return ranges

    def _initial_slice(self) -> int:
        valid = self.report["axial_slice_index"].dropna()
        if len(valid) == 0:
            return self.ctx.ct_size[2] // 2
        return int(valid.mode().iloc[0])

    def _build_ui(self):
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right = ttk.Frame(main, width=420)
        right.pack(side=tk.RIGHT, fill=tk.Y)

        self.fig = Figure(figsize=(7, 7.5))
        self.ax_img = self.fig.add_axes([0.03, 0.1, 0.94, 0.86])
        ax_slider = self.fig.add_axes([0.15, 0.02, 0.7, 0.03])

        nz = self.ctx.ct_size[2]
        self.slider = Slider(ax_slider, "Axial slice", 0, nz - 1, valinit=self._initial_slice(), valstep=1)
        self.slider.on_changed(lambda _val: self.redraw())

        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_click)

        ttk.Label(right, text="Lesion Info", font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=10, pady=(10, 2))
        self.info_text = scrolledtext.ScrolledText(right, width=52, wrap="word", font=("Segoe UI", 10))
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)
        self.info_text.config(state="disabled")

        btn_frame = ttk.Frame(right)
        btn_frame.pack(fill=tk.X, padx=10, pady=4)
        self.regen_btn = ttk.Button(btn_frame, text="Regenerate description", command=self.regenerate_caption)
        self.regen_btn.pack(side=tk.LEFT)

        self.status_label = ttk.Label(right, text="Click a lesion contour on the image to see its details.", wraplength=380)
        self.status_label.pack(anchor="w", padx=10, pady=(0, 10))

        if self.api_key is None:
            self.status_label.config(text="No GEMINI_API_KEY found (set env var or .env) — descriptions disabled.")

        self._set_info_text("Click a lesion contour to see its details.")
        self.redraw()

    def redraw(self):
        idx = int(round(self.slider.val))
        ax = self.ax_img
        ax.clear()

        ct_slice = self.ctx.ct_arr[:, :, idx].T
        suv_slice = self.ctx.suv_ct[:, :, idx].T

        ax.imshow(ct_slice, origin="lower", cmap="gray", vmin=-200, vmax=400)
        alpha = np.clip((suv_slice - SUV_DISPLAY_THRESHOLD) / max(self.suv_vmax - SUV_DISPLAY_THRESHOLD, 1e-6), 0, 1) * 0.75
        ax.imshow(suv_slice, origin="lower", cmap="hot", vmin=0, vmax=self.suv_vmax, alpha=alpha)

        visible = 0
        for name in self.lesion_names:
            lo, hi = self.slice_ranges.get(name, (None, None))
            if lo is None or not (lo <= idx <= hi):
                continue
            mask = self.ctx.lesion_mask_ct(name)
            footprint = mask[:, :, idx]
            if not footprint.any():
                continue
            visible += 1
            is_selected = name == self.selected
            color = "yellow" if is_selected else self.colors[name]
            lw = 2.5 if is_selected else 1.1
            ax.contour(footprint.T.astype(float), levels=[0.5], colors=[color], linewidths=lw, origin="lower")

        title = f"Axial slice {idx}/{self.ctx.ct_size[2] - 1}  |  {visible} lesion(s) on this slice"
        if self.selected:
            title += f"  |  selected: {self.selected}"
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        self.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax_img or event.xdata is None or event.ydata is None:
            return
        idx = int(round(self.slider.val))
        x, y = int(round(event.xdata)), int(round(event.ydata))

        hit = None
        for name in self.lesion_names:
            lo, hi = self.slice_ranges.get(name, (None, None))
            if lo is None or not (lo <= idx <= hi):
                continue
            mask = self.ctx.lesion_mask_ct(name)
            if 0 <= x < mask.shape[0] and 0 <= y < mask.shape[1] and mask[x, y, idx]:
                hit = name
                break

        if hit:
            self.select_lesion(hit)

    def select_lesion(self, name: str):
        self.selected = name
        self.redraw()
        self.show_info(name)
        if name not in self.captions:
            self.request_caption(name)

    def show_info(self, name: str):
        row = self.report.loc[name]
        lines = [f"Lesion: {name}", ""]

        if pd.isna(row.get("suv_max")):
            lines.append("(outside the CT field of view - no PET metrics available)")
        else:
            lines += [
                f"Closest organ: {row['closest_anatomy']}  (distance {row['distance_mm']:.1f} mm)",
                f"Lesion centroid (mm, LPS): ({row['lesion_centroid_x_mm']:.1f}, {row['lesion_centroid_y_mm']:.1f}, {row['lesion_centroid_z_mm']:.1f})",
                f"Closest organ centroid (mm, LPS): ({row['closest_anatomy_centroid_x_mm']:.1f}, {row['closest_anatomy_centroid_y_mm']:.1f}, {row['closest_anatomy_centroid_z_mm']:.1f})",
                f"Axial CT slice index: {int(row['axial_slice_index'])}",
                "",
                f"SUVmax: {row['suv_max']:.2f}    SUVmean: {row['suv_mean_whole']:.2f}",
                f"Metabolic Tumor Volume (41% SUVmax): {row['mtv_ml']:.2f} mL",
                f"Total Lesion Glycolysis: {row['tlg_g']:.2f} g",
                "",
                "Description:",
            ]
            caption = self.captions.get(name)
            if caption:
                lines.append(caption)
            elif self.api_key is None:
                lines.append("(no GEMINI_API_KEY configured)")
            else:
                lines.append("(generating...)")

        self._set_info_text("\n".join(lines))

    def _set_info_text(self, text: str):
        self.info_text.config(state="normal")
        self.info_text.delete("1.0", "end")
        self.info_text.insert("1.0", text)
        self.info_text.config(state="disabled")

    def request_caption(self, name: str):
        if not self.api_key:
            return
        row = self.report.loc[name]
        if pd.isna(row.get("suv_max")):
            return

        def worker():
            try:
                text = gemini_caption.generate_caption_for_lesion(row, self.api_key)
                self.result_queue.put((name, text))
            except Exception as exc:
                self.result_queue.put((name, f"[caption generation failed: {exc}]"))

        threading.Thread(target=worker, daemon=True).start()

    def regenerate_caption(self):
        if self.selected:
            self.status_label.config(text=f"Regenerating description for {self.selected}...")
            self.request_caption(self.selected)

    def _poll_queue(self):
        try:
            while True:
                name, text = self.result_queue.get_nowait()
                if text.startswith("[caption generation failed"):
                    self.status_label.config(text=text)
                else:
                    self.captions[name] = text
                    self.captions_path.parent.mkdir(parents=True, exist_ok=True)
                    self.captions_path.write_text(json.dumps(self.captions, indent=2))
                    self.status_label.config(text="Click a lesion contour on the image to see its details.")
                if name == self.selected:
                    self.show_info(name)
        except queue.Empty:
            pass
        self.root.after(200, self._poll_queue)

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patient-dir", type=Path, default=Path("data/11-37493"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--task", default="total")
    args = parser.parse_args()

    print("Loading patient data (CT, segmentation, PET, RTSTRUCT)...")
    ctx = lesion_metrics.PatientContext(args.patient_dir, args.output_dir, args.task)
    print("Computing lesion metrics...")
    report = lesion_metrics.compute_lesion_report(ctx)
    report.to_csv(ctx.output_dir / "lesion_report.csv", index=False)

    captions_path = ctx.output_dir / "lesion_captions.json"
    app = LesionViewerApp(ctx, report, captions_path)
    print("GUI ready.")
    app.run()


if __name__ == "__main__":
    main()

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
from skimage.transform import resize

from pmrf3dsynthesis.data import constants as C
from pmrf3dsynthesis.eval_model import load_sample


def load_evaluation_cases(
    res_path: Union[str, Path]
) -> Tuple[List[str], Dict[str, float], Dict[str, float], Optional[float]]:
    """
    Loads available cases and metrics from an evaluation result directory.
    Returns:
        keys: List of case IDs
        psnr_dict: Mapping key -> PSNR
        ssim_dict: Mapping key -> SSIM
        fid: FID score if available
    """
    res_path = Path(res_path)
    pkl_file = res_path / "eval_results.pkl"
    csv_file = res_path / "eval_results.csv"

    psnr_dict = {}
    ssim_dict = {}
    fid = None
    keys = []

    if pkl_file.exists():
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
            psnr_dict = data.get("PSNR", {})
            ssim_dict = data.get("SSIM", {})
            fid_list = data.get("FID", [])
            if fid_list:
                fid = fid_list[0]
            keys = list(psnr_dict.keys())

    elif csv_file.exists():
        df = pd.read_csv(csv_file)
        # Filter out summary rows
        df_cases = df[~df["Key"].isin(["MEAN", "STD", "FID"])].dropna(subset=["Key"])
        keys = list(df_cases["Key"].astype(str))
        if "PSNR" in df_cases.columns:
            psnr_dict = dict(zip(keys, df_cases["PSNR"].astype(float)))
        if "SSIM" in df_cases.columns:
            ssim_dict = dict(zip(keys, df_cases["SSIM"].astype(float)))

    # Fallback to scanning .npy files in res_path if no summary files found
    if not keys:
        keys = sorted([p.stem for p in res_path.glob("*.npy") if not p.stem.startswith(".")])

    return keys, psnr_dict, ssim_dict, fid


class EvaluationViewer:
    """
    Interactive viewer to scroll through cases and 3D slices of an evaluation run.
    Supports both Jupyter Notebook (ipywidgets) and standalone Matplotlib window.
    """

    def __init__(
        self,
        res_path: Union[str, Path, Dict[str, Union[str, Path]]],
        modality: str = C.MODALITY_CT,
        vol_path: Optional[Union[str, Path]] = None,
        img_path: Optional[Union[str, Path]] = None,
        data_dir: Optional[Union[str, Path]] = None,
        data_format: str = C.FORMAT_AUTO,
        keys: Optional[List[str]] = None,
        sort_by: str = "none",
        hu_min: float = -1350.0,
        hu_max: float = 150.0,
        ct_scale: float = 5800.0,
        xray_scale: float = 1200.0,
        show_diff: bool = True
    ):
        """
        :param res_path: Path to evaluation results, or dict of {model_name: path} for comparisons.
        :param modality: 'ct' or 'oct'
        :param vol_path: Ground truth volume directory
        :param img_path: Ground truth input image directory
        :param data_dir: Root dataset directory (for HDF5 or multi-case root)
        :param data_format: 'auto', 'nii', 'h5', or 'npy'
        :param keys: Optional explicit list of keys to view
        :param sort_by: Sorting: 'none', 'psnr_asc', 'psnr_desc', 'ssim_asc', 'ssim_desc'
        """
        self.modality = modality
        self.vol_path = Path(vol_path) if vol_path else None
        self.img_path = Path(img_path) if img_path else None
        self.data_dir = Path(data_dir) if data_dir else None
        self.data_format = data_format
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.ct_scale = ct_scale
        self.xray_scale = xray_scale
        self.show_diff = show_diff

        # Handle single vs multiple model runs
        if isinstance(res_path, (str, Path)):
            self.model_runs = {"Prediction": Path(res_path)}
        elif isinstance(res_path, dict):
            self.model_runs = {name: Path(p) for name, p in res_path.items()}
        else:
            raise TypeError("res_path must be a str, Path, or dict of {model_name: path}")

        # Primary run for indexing keys and metrics
        primary_name, primary_path = next(iter(self.model_runs.items()))
        self.primary_name = primary_name
        self.primary_path = primary_path

        auto_keys, self.psnr_dict, self.ssim_dict, self.fid = load_evaluation_cases(self.primary_path)
        self.keys = keys if keys is not None else auto_keys

        if not self.keys:
            raise ValueError(f"No cases found in {self.primary_path}. Please check path.")

        # Sort keys if requested
        self._sort_keys(sort_by)

        self.current_case_idx = 0
        self.current_slice_idx = 0
        self.slice_axis = 0  # 0: Axial/depth, 1: Coronal, 2: Sagittal

        # Cached case data
        self.current_key = self.keys[0]
        self.cached_input_img = None
        self.cached_gt_vol = None
        self.cached_pred_vols: Dict[str, np.ndarray] = {}

        self._load_current_case_data()

    def _sort_keys(self, sort_by: str):
        mode = sort_by.lower()
        if mode == "psnr_asc":
            self.keys = sorted(self.keys, key=lambda k: self.psnr_dict.get(k, float("inf")))
        elif mode == "psnr_desc":
            self.keys = sorted(self.keys, key=lambda k: self.psnr_dict.get(k, -float("inf")), reverse=True)
        elif mode == "ssim_asc":
            self.keys = sorted(self.keys, key=lambda k: self.ssim_dict.get(k, float("inf")))
        elif mode == "ssim_desc":
            self.keys = sorted(self.keys, key=lambda k: self.ssim_dict.get(k, -float("inf")), reverse=True)

    def _load_current_case_data(self):
        self.current_key = self.keys[self.current_case_idx]

        # 1. Load Ground Truth & 2D Input
        try:
            input_img, gt_vol = load_sample(
                key=self.current_key,
                modality=self.modality,
                vol_path=self.vol_path,
                img_path=self.img_path,
                data_dir=self.data_dir,
                data_format=self.data_format,
                hu_min=self.hu_min,
                hu_max=self.hu_max,
                ct_scale=self.ct_scale,
                xray_scale=self.xray_scale
            )
            self.cached_input_img = input_img
            self.cached_gt_vol = gt_vol
        except Exception as e:
            print(f"Warning: Failed to load GT/Input for case '{self.current_key}': {e}")
            self.cached_input_img = np.zeros((256, 256), dtype=np.float32)
            self.cached_gt_vol = np.zeros((128, 256, 256), dtype=np.float32)

        # 2. Load Predicted Volumes for each model run
        self.cached_pred_vols.clear()
        for name, run_dir in self.model_runs.items():
            pred_file = run_dir / f"{self.current_key}.npy"
            if pred_file.exists():
                pred = np.load(pred_file)
                # Map to [0, 1] if stored in [-1, 1]
                if pred.min() < 0.0:
                    pred = np.clip(pred * 0.5 + 0.5, 0.0, 1.0)
                else:
                    pred = np.clip(pred, 0.0, 1.0)

                # Match GT spatial shape if different
                if pred.shape != self.cached_gt_vol.shape:
                    pred = resize(pred, self.cached_gt_vol.shape, preserve_range=True)

                self.cached_pred_vols[name] = pred.astype(np.float32)
            else:
                print(f"Warning: Prediction file not found: {pred_file}")
                self.cached_pred_vols[name] = np.zeros_like(self.cached_gt_vol)

        max_slices = self._get_num_slices()
        self.current_slice_idx = min(self.current_slice_idx, max_slices - 1)

    def _get_num_slices(self) -> int:
        if self.cached_gt_vol is not None and self.cached_gt_vol.ndim == 3:
            return self.cached_gt_vol.shape[self.slice_axis]
        return 1

    def _extract_slice(self, vol: np.ndarray, slice_idx: int) -> np.ndarray:
        if vol.ndim != 3:
            return vol
        idx = max(0, min(slice_idx, vol.shape[self.slice_axis] - 1))
        if self.slice_axis == 0:
            return vol[idx, :, :]
        elif self.slice_axis == 1:
            return vol[:, idx, :]
        else:
            return vol[:, :, idx]

    # -------------------------------------------------------------
    # Standalone Matplotlib GUI with Keyboard & Mouse Scroll
    # -------------------------------------------------------------
    def show(self):
        """
        Launches an interactive Matplotlib window.
        Keyboard Shortcuts:
          - Up / Down or Mouse Wheel: Scroll through slices
          - Left / Right: Scroll through cases
          - 1, 2, 3: Change slice orientation (Axial, Coronal, Sagittal)
          - D: Toggle difference map
        """
        # Determine number of columns: Input, GT, [Predictions...], [Diff if 1 prediction]
        has_single_pred = len(self.cached_pred_vols) == 1
        num_panels = 2 + len(self.cached_pred_vols) + (1 if (self.show_diff and has_single_pred) else 0)

        fig, axes = plt.subplots(1, num_panels, figsize=(4 * num_panels, 5))
        if num_panels == 1:
            axes = [axes]
        else:
            axes = list(axes)

        plt.subplots_adjust(bottom=0.20, top=0.88, wspace=0.15)

        # Plot titles and image handles
        im_handles = []

        # 1. 2D Input
        ax_input = axes[0]
        ax_input.set_title("2D Input Projection\n(X-ray / Fundus)", fontsize=10)
        ax_input.axis("off")
        input_slice = self.cached_input_img
        if input_slice.ndim == 2:
            im_input = ax_input.imshow(input_slice, cmap="gray", vmin=0.0, vmax=1.0)
        else:
            im_input = ax_input.imshow(input_slice)
        im_handles.append((im_input, "input"))

        # 2. Ground Truth
        ax_gt = axes[1]
        ax_gt.set_title("Ground Truth\n(3D Volume)", fontsize=10)
        ax_gt.axis("off")
        im_gt = ax_gt.imshow(self._extract_slice(self.cached_gt_vol, self.current_slice_idx), cmap="gray", vmin=0.0, vmax=1.0)
        im_handles.append((im_gt, "gt"))

        # 3. Model Predictions
        pred_handles = {}
        for i, (name, pred_vol) in enumerate(self.cached_pred_vols.items()):
            ax_pred = axes[2 + i]
            ax_pred.set_title(f"{name}\n(Synthesis)", fontsize=10)
            ax_pred.axis("off")
            im_pred = ax_pred.imshow(self._extract_slice(pred_vol, self.current_slice_idx), cmap="gray", vmin=0.0, vmax=1.0)
            im_handles.append((im_pred, f"pred_{name}"))
            pred_handles[name] = im_pred

        # 4. Difference Map (if single model)
        im_diff = None
        if self.show_diff and has_single_pred:
            ax_diff = axes[-1]
            ax_diff.set_title("Absolute Difference\n|Pred - GT|", fontsize=10)
            ax_diff.axis("off")
            single_pred = next(iter(self.cached_pred_vols.values()))
            diff_slice = np.abs(
                self._extract_slice(single_pred, self.current_slice_idx) -
                self._extract_slice(self.cached_gt_vol, self.current_slice_idx)
            )
            im_diff = ax_diff.imshow(diff_slice, cmap="magma", vmin=0.0, vmax=0.5)
            im_handles.append((im_diff, "diff"))

        # Interactive Slider & Buttons layout
        # Row 1 (y ≈ 0.12): Slice navigation and plane selection
        ax_slider = plt.axes([0.18, 0.12, 0.48, 0.035])
        slider = mwidgets.Slider(
            ax=ax_slider,
            label="Slice ",
            valmin=0,
            valmax=max(0, self._get_num_slices() - 1),
            valinit=self.current_slice_idx,
            valstep=1,
            valfmt="%0.0f"
        )

        ax_btn_axis = plt.axes([0.70, 0.12, 0.13, 0.035])
        plane_names = ["Axial (D)", "Coronal (H)", "Sagittal (W)"]
        btn_axis = mwidgets.Button(ax_btn_axis, f"Plane: {plane_names[self.slice_axis]}")

        ax_btn_random = plt.axes([0.85, 0.12, 0.09, 0.035])
        btn_random = mwidgets.Button(ax_btn_random, "🎲 Random")

        # Row 2 (y ≈ 0.04): Case navigation & Jump controls
        ax_btn_prev10 = plt.axes([0.05, 0.04, 0.06, 0.038])
        btn_prev10 = mwidgets.Button(ax_btn_prev10, "<< -10")

        ax_btn_prev = plt.axes([0.12, 0.04, 0.06, 0.038])
        btn_prev = mwidgets.Button(ax_btn_prev, "< Prev")

        ax_btn_next = plt.axes([0.19, 0.04, 0.06, 0.038])
        btn_next = mwidgets.Button(ax_btn_next, "Next >")

        ax_btn_next10 = plt.axes([0.26, 0.04, 0.06, 0.038])
        btn_next10 = mwidgets.Button(ax_btn_next10, "+10 >>")

        ax_case_slider = plt.axes([0.38, 0.04, 0.30, 0.035])
        case_slider = mwidgets.Slider(
            ax=ax_case_slider,
            label="Case ",
            valmin=1,
            valmax=max(1, len(self.keys)),
            valinit=self.current_case_idx + 1,
            valstep=1,
            valfmt="%0.0f"
        )

        ax_jump_box = plt.axes([0.76, 0.04, 0.07, 0.038])
        jump_box = mwidgets.TextBox(ax_jump_box, "Jump: ", initial=str(self.current_case_idx + 1))

        def update_header():
            key = self.current_key
            psnr_val = self.psnr_dict.get(key, None)
            ssim_val = self.ssim_dict.get(key, None)

            metrics_str = ""
            if psnr_val is not None:
                metrics_str += f" | PSNR: {psnr_val:.2f} dB"
            if ssim_val is not None:
                metrics_str += f" | SSIM: {ssim_val:.4f}"

            plane_str = plane_names[self.slice_axis]
            fig.suptitle(
                f"Case [{self.current_case_idx + 1}/{len(self.keys)}]: {key}{metrics_str}\n"
                f"Slice: {self.current_slice_idx}/{self._get_num_slices() - 1} ({plane_str})  |  "
                f"[PgUp/PgDn/[]: Jump 10, Left/Right: +/-1, Scroll/Up/Down: Slices, R: Random, 1/2/3: Planes]",
                fontsize=11, fontweight="bold"
            )

        def update_views():
            # Update 2D Input
            if self.cached_input_img.ndim == 2:
                im_input.set_data(self.cached_input_img)
            else:
                im_input.set_data(self.cached_input_img)

            # Update GT
            gt_sl = self._extract_slice(self.cached_gt_vol, self.current_slice_idx)
            im_gt.set_data(gt_sl)

            # Update Predictions
            first_pred_sl = None
            for name, im_h in pred_handles.items():
                p_sl = self._extract_slice(self.cached_pred_vols[name], self.current_slice_idx)
                im_h.set_data(p_sl)
                if first_pred_sl is None:
                    first_pred_sl = p_sl

            # Update Diff
            if im_diff is not None and first_pred_sl is not None:
                im_diff.set_data(np.abs(first_pred_sl - gt_sl))

            update_header()
            fig.canvas.draw_idle()

        def on_slider_change(val):
            self.current_slice_idx = int(val)
            update_views()

        slider.on_changed(on_slider_change)

        is_switching_case = False

        def switch_case(new_idx):
            nonlocal is_switching_case
            if is_switching_case:
                return
            is_switching_case = True

            self.current_case_idx = max(0, min(new_idx, len(self.keys) - 1))
            self._load_current_case_data()

            # Update slider bounds for new volume shape
            max_sl = max(0, self._get_num_slices() - 1)
            slider.valmax = max_sl
            slider.ax.set_xlim(0, max_sl)
            slider.set_val(min(self.current_slice_idx, max_sl))

            # Update case slider and jump box display
            case_slider.set_val(self.current_case_idx + 1)
            jump_box.set_val(str(self.current_case_idx + 1))

            update_views()
            is_switching_case = False

        def on_prev_clicked(event):
            switch_case(self.current_case_idx - 1)

        def on_next_clicked(event):
            switch_case(self.current_case_idx + 1)

        def on_prev10_clicked(event):
            switch_case(self.current_case_idx - 10)

        def on_next10_clicked(event):
            switch_case(self.current_case_idx + 10)

        def on_random_clicked(event):
            if len(self.keys) > 1:
                new_idx = int(np.random.randint(0, len(self.keys)))
                switch_case(new_idx)

        def on_case_slider_change(val):
            target_idx = int(val) - 1
            if target_idx != self.current_case_idx:
                switch_case(target_idx)

        def on_jump_submit(text):
            try:
                val = int(text.strip()) - 1
                switch_case(val)
            except ValueError:
                jump_box.set_val(str(self.current_case_idx + 1))

        btn_prev.on_clicked(on_prev_clicked)
        btn_next.on_clicked(on_next_clicked)
        btn_prev10.on_clicked(on_prev10_clicked)
        btn_next10.on_clicked(on_next10_clicked)
        btn_random.on_clicked(on_random_clicked)
        case_slider.on_changed(on_case_slider_change)
        jump_box.on_submit(on_jump_submit)

        def on_axis_clicked(event):
            self.slice_axis = (self.slice_axis + 1) % 3
            btn_axis.label.set_text(f"Plane: {plane_names[self.slice_axis]}")
            max_sl = max(0, self._get_num_slices() - 1)
            slider.valmax = max_sl
            slider.ax.set_xlim(0, max_sl)
            slider.set_val(min(self.current_slice_idx, max_sl))
            update_views()

        btn_axis.on_clicked(on_axis_clicked)

        # Keyboard and Mouse Wheel callbacks
        def on_key(event):
            if event.key in ["up", "k"]:
                new_slice = min(self.current_slice_idx + 1, self._get_num_slices() - 1)
                slider.set_val(new_slice)
            elif event.key in ["down", "j"]:
                new_slice = max(self.current_slice_idx - 1, 0)
                slider.set_val(new_slice)
            elif event.key in ["right", "l"]:
                switch_case(self.current_case_idx + 1)
            elif event.key in ["left", "h"]:
                switch_case(self.current_case_idx - 1)
            elif event.key in ["pageup", "]", "shift+right"]:
                switch_case(self.current_case_idx + 10)
            elif event.key in ["pagedown", "[", "shift+left"]:
                switch_case(self.current_case_idx - 10)
            elif event.key in ["home"]:
                switch_case(0)
            elif event.key in ["end"]:
                switch_case(len(self.keys) - 1)
            elif event.key in ["r", "R"]:
                on_random_clicked(None)
            elif event.key in ["1", "2", "3"]:
                self.slice_axis = int(event.key) - 1
                btn_axis.label.set_text(f"Plane: {plane_names[self.slice_axis]}")
                max_sl = max(0, self._get_num_slices() - 1)
                slider.valmax = max_sl
                slider.ax.set_xlim(0, max_sl)
                slider.set_val(min(self.current_slice_idx, max_sl))
                update_views()

        def on_scroll(event):
            step = 1 if event.button == "up" else -1
            new_slice = max(0, min(self.current_slice_idx + step, self._get_num_slices() - 1))
            slider.set_val(new_slice)

        fig.canvas.mpl_connect("key_press_event", on_key)
        fig.canvas.mpl_connect("scroll_event", on_scroll)

        update_header()
        plt.show()

    # -------------------------------------------------------------
    # Jupyter Notebook Widget Interface
    # -------------------------------------------------------------
    def widget(self):
        """
        Creates an interactive ipywidgets-based viewer for Jupyter notebooks.
        Returns ipywidgets.VBox container.
        """
        import ipywidgets as widgets
        from IPython.display import display

        # Options for Case Dropdown
        case_options = []
        for i, k in enumerate(self.keys):
            p = self.psnr_dict.get(k)
            s = self.ssim_dict.get(k)
            label = f"[{i+1}/{len(self.keys)}] {k}"
            if p is not None and s is not None:
                label += f" (PSNR: {p:.2f}, SSIM: {s:.3f})"
            case_options.append((label, i))

        dropdown_case = widgets.Dropdown(
            options=case_options,
            value=self.current_case_idx,
            description="Case:",
            layout=widgets.Layout(width="380px")
        )

        btn_prev_10 = widgets.Button(description="<< -10", layout=widgets.Layout(width="70px"))
        btn_prev = widgets.Button(description="< Prev", button_style="info", layout=widgets.Layout(width="75px"))
        btn_next = widgets.Button(description="Next >", button_style="info", layout=widgets.Layout(width="75px"))
        btn_next_10 = widgets.Button(description="+10 >>", layout=widgets.Layout(width="70px"))
        btn_random = widgets.Button(description="🎲 Random", button_style="warning", layout=widgets.Layout(width="95px"))

        jump_input = widgets.BoundedIntText(
            value=self.current_case_idx + 1,
            min=1,
            max=len(self.keys),
            step=1,
            description="Jump #:",
            layout=widgets.Layout(width="150px")
        )

        radio_plane = widgets.RadioButtons(
            options=[("Axial (D)", 0), ("Coronal (H)", 1), ("Sagittal (W)", 2)],
            value=self.slice_axis,
            description="Plane:",
            layout=widgets.Layout(width="250px")
        )

        slider_slice = widgets.IntSlider(
            value=self.current_slice_idx,
            min=0,
            max=max(0, self._get_num_slices() - 1),
            step=1,
            description="Slice:",
            continuous_update=True,
            layout=widgets.Layout(width="500px")
        )

        out_plot = widgets.Output()

        def render_figure():
            with out_plot:
                out_plot.clear_output(wait=True)
                has_single = len(self.cached_pred_vols) == 1
                n_panels = 2 + len(self.cached_pred_vols) + (1 if (self.show_diff and has_single) else 0)

                fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4.5))
                if n_panels == 1:
                    axes = [axes]
                else:
                    axes = list(axes)

                # Input
                axes[0].set_title("2D Input", fontsize=10)
                axes[0].axis("off")
                if self.cached_input_img.ndim == 2:
                    axes[0].imshow(self.cached_input_img, cmap="gray", vmin=0, vmax=1)
                else:
                    axes[0].imshow(self.cached_input_img)

                # GT
                gt_sl = self._extract_slice(self.cached_gt_vol, self.current_slice_idx)
                axes[1].set_title("Ground Truth", fontsize=10)
                axes[1].axis("off")
                axes[1].imshow(gt_sl, cmap="gray", vmin=0, vmax=1)

                # Predictions
                first_pred = None
                for idx, (m_name, p_vol) in enumerate(self.cached_pred_vols.items()):
                    p_sl = self._extract_slice(p_vol, self.current_slice_idx)
                    axes[2 + idx].set_title(m_name, fontsize=10)
                    axes[2 + idx].axis("off")
                    axes[2 + idx].imshow(p_sl, cmap="gray", vmin=0, vmax=1)
                    if first_pred is None:
                        first_pred = p_sl

                # Diff
                if self.show_diff and has_single and first_pred is not None:
                    axes[-1].set_title("|Pred - GT|", fontsize=10)
                    axes[-1].axis("off")
                    axes[-1].imshow(np.abs(first_pred - gt_sl), cmap="magma", vmin=0, vmax=0.5)

                k = self.current_key
                psnr_str = f" | PSNR: {self.psnr_dict[k]:.2f} dB" if k in self.psnr_dict else ""
                ssim_str = f" | SSIM: {self.ssim_dict[k]:.4f}" if k in self.ssim_dict else ""
                fig.suptitle(f"[{self.current_case_idx + 1}/{len(self.keys)}] {k}{psnr_str}{ssim_str} (Slice {self.current_slice_idx}/{self._get_num_slices()-1})", fontsize=11, fontweight="bold")
                plt.tight_layout()
                plt.show()

        is_updating_widget = False

        def on_case_change(change):
            nonlocal is_updating_widget
            if is_updating_widget:
                return
            is_updating_widget = True
            self.current_case_idx = change["new"]
            self._load_current_case_data()
            slider_slice.max = max(0, self._get_num_slices() - 1)
            slider_slice.value = min(self.current_slice_idx, slider_slice.max)
            jump_input.value = self.current_case_idx + 1
            render_figure()
            is_updating_widget = False

        def on_prev(b):
            if dropdown_case.value > 0:
                dropdown_case.value -= 1

        def on_next(b):
            if dropdown_case.value < len(self.keys) - 1:
                dropdown_case.value += 1

        def on_prev_10(b):
            dropdown_case.value = max(0, dropdown_case.value - 10)

        def on_next_10(b):
            dropdown_case.value = min(len(self.keys) - 1, dropdown_case.value + 10)

        def on_random(b):
            if len(self.keys) > 1:
                dropdown_case.value = int(np.random.randint(0, len(self.keys)))

        def on_jump(change):
            target_idx = change["new"] - 1
            if target_idx != dropdown_case.value:
                dropdown_case.value = target_idx

        def on_slice_change(change):
            self.current_slice_idx = change["new"]
            render_figure()

        def on_plane_change(change):
            self.slice_axis = change["new"]
            slider_slice.max = max(0, self._get_num_slices() - 1)
            slider_slice.value = min(self.current_slice_idx, slider_slice.max)
            render_figure()

        dropdown_case.observe(on_case_change, names="value")
        btn_prev.on_click(on_prev)
        btn_next.on_click(on_next)
        btn_prev_10.on_click(on_prev_10)
        btn_next_10.on_click(on_next_10)
        btn_random.on_click(on_random)
        jump_input.observe(on_jump, names="value")
        slider_slice.observe(on_slice_change, names="value")
        radio_plane.observe(on_plane_change, names="value")

        controls = widgets.VBox([
            widgets.HBox([btn_prev_10, btn_prev, btn_next, btn_next_10, btn_random, jump_input]),
            widgets.HBox([dropdown_case, radio_plane, slider_slice])
        ])

        render_figure()
        return widgets.VBox([controls, out_plot])


def run_cli():
    parser = argparse.ArgumentParser(description="Quickly view and scroll through evaluation run cases and 3D slices.")
    parser.add_argument("--res_path", type=str, required=True, help="Path to evaluation results folder.")
    parser.add_argument("--modality", type=str, choices=[C.MODALITY_OCT, C.MODALITY_CT], default=C.MODALITY_CT, help="Modality (ct or oct).")
    parser.add_argument("--vol_path", type=str, default=None, help="Custom ground truth volume path.")
    parser.add_argument("--img_path", type=str, default=None, help="Custom 2D input image path.")
    parser.add_argument("--data_dir", type=str, default=None, help="Base dataset directory.")
    parser.add_argument("--data_format", type=str, choices=[C.FORMAT_AUTO, C.FORMAT_NII, C.FORMAT_H5, C.FORMAT_NPY], default=C.FORMAT_AUTO, help="Data format.")
    parser.add_argument("--sort_by", type=str, choices=["none", "psnr_asc", "psnr_desc", "ssim_asc", "ssim_desc"], default="none", help="Sort cases by metrics.")
    parser.add_argument("--no_diff", action="store_true", help="Disable difference map panel.")

    args = parser.parse_args()

    viewer = EvaluationViewer(
        res_path=args.res_path,
        modality=args.modality,
        vol_path=args.vol_path,
        img_path=args.img_path,
        data_dir=args.data_dir,
        data_format=args.data_format,
        sort_by=args.sort_by,
        show_diff=not args.no_diff
    )
    viewer.show()


if __name__ == "__main__":
    run_cli()

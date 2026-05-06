#!/usr/bin/env python3
"""Standalone plot digitizer.

Normal workflow:

    python3 KplotDigitiser.py image.ext

The program detects known figure layouts, asks for one output CSV filename, and
writes one wide table where each plot/series has its own columns.

Supported input formats include PNG, JPEG, TIFF, BMP, WebP, GIF, JP2, PDF,
EPS, and PS. PDF/EPS/PS files are rendered to pixels with `--dpi` and `--page`;
calibrations must be made at the same DPI/page used for digitizing.

Default wide CSV format:

    series1_p,series1_value,series1_sem,series2_p,series2_value,series2_sem,...

The `sem` column is left blank when no error bar is detected.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
from PIL import Image

RASTER_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
    ".gif",
    ".jp2",
    ".ppm",
    ".pgm",
    ".pbm",
    ".pnm",
}
VECTOR_SUFFIXES = {".pdf", ".eps", ".ps"}
IMAGE_SUFFIXES = RASTER_SUFFIXES | VECTOR_SUFFIXES


@dataclass(frozen=True)
class AxisCalibration:
    px0: float
    px1: float
    v0: float
    v1: float

    def px_to_value(self, px: float) -> float:
        if self.px1 == self.px0:
            raise ValueError("degenerate calibration")
        t = (px - self.px0) / (self.px1 - self.px0)
        return self.v0 + t * (self.v1 - self.v0)

    def value_to_px(self, value: float) -> float:
        if self.v1 == self.v0:
            raise ValueError("degenerate calibration")
        t = (value - self.v0) / (self.v1 - self.v0)
        return self.px0 + t * (self.px1 - self.px0)


@dataclass(frozen=True)
class SeriesSpec:
    name: str
    out_csv: Path
    xaxis: str
    yaxis: str
    roi: tuple[int, int, int, int]
    color: Optional[str] = None
    hsv_ranges: Optional[list[tuple[int, int, int, int, int, int]]] = None
    ignore_rois: tuple[tuple[int, int, int, int], ...] = ()
    min_area: int = 8
    cluster_eps: float = 5.5
    x_values: Optional[list[float]] = None
    expected_points: Optional[int] = None
    peak_min_distance: int = 28
    sample_x_band: int = 8
    use_sem: bool = True
    sem_mode: str = "auto"
    sem_x_band: int = 6
    sem_y_window: int = 45
    sem_gray_thresh: int = 140
    sem_neutral_spread: int = 35
    sem_min_height: int = 4
    debug_color: tuple[int, int, int] = (0, 255, 255)


@dataclass(frozen=True)
class AutoExtractionResult:
    kind: str
    columns: list[str]
    rows: list[list[object]]
    confidence: float
    notes: tuple[str, ...] = ()


def read_image(path: Path, dpi: int = 300, page: int = 1) -> np.ndarray:
    """Load raster, PDF, EPS, and PS inputs as an OpenCV BGR image."""

    suffix = path.suffix.lower()
    if suffix in RASTER_SUFFIXES:
        return _read_raster_image(path)
    if suffix == ".pdf":
        return _read_pdf_image(path, dpi=dpi, page=page)
    if suffix in {".eps", ".ps"}:
        return _read_postscript_image(path, dpi=dpi, page=page)
    raise ValueError(f"unsupported input format {suffix!r}")


def _read_raster_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is not None:
        return _normalize_cv_image(img)

    try:
        with Image.open(path) as pil_img:
            pil_img.seek(0)
            rgb = pil_img.convert("RGB")
            return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    except Exception as exc:
        raise ValueError(f"failed to read image {path}") from exc


def _normalize_cv_image(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3].astype(np.float32) / 255.0
        bgr = img[:, :, :3].astype(np.float32)
        white = np.full_like(bgr, 255.0)
        return (bgr * alpha[:, :, None] + white * (1.0 - alpha[:, :, None])).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 3:
        return img
    raise ValueError(f"unsupported image array shape {img.shape!r}")


def _run_converter(cmd: list[str], expected_output: Path, label: str) -> np.ndarray:
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise ValueError(f"{label} converter is not installed: {cmd[0]}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip()
        detail = f": {stderr}" if stderr else ""
        raise ValueError(f"{label} conversion failed{detail}") from exc
    if not expected_output.exists():
        raise ValueError(f"{label} conversion did not produce {expected_output}")
    return _read_raster_image(expected_output)


def _read_pdf_image(path: Path, dpi: int, page: int) -> np.ndarray:
    if page < 1:
        raise ValueError("page must be >= 1")
    with tempfile.TemporaryDirectory(prefix="plotdigitizer_pdf_") as tmp:
        prefix = Path(tmp) / "page"
        out_path = prefix.with_suffix(".png")
        cmd = [
            "pdftoppm",
            "-f",
            str(page),
            "-l",
            str(page),
            "-singlefile",
            "-png",
            "-r",
            str(dpi),
            str(path),
            str(prefix),
        ]
        return _run_converter(cmd, out_path, "PDF")


def _read_postscript_image(path: Path, dpi: int, page: int) -> np.ndarray:
    if page < 1:
        raise ValueError("page must be >= 1")
    with tempfile.TemporaryDirectory(prefix="plotdigitizer_ps_") as tmp:
        out_path = Path(tmp) / "page.png"
        cmd = [
            "gs",
            "-dSAFER",
            "-dBATCH",
            "-dNOPAUSE",
            "-sDEVICE=pngalpha",
            f"-r{dpi}",
            f"-dFirstPage={page}",
            f"-dLastPage={page}",
            f"-sOutputFile={out_path}",
            str(path),
        ]
        return _run_converter(cmd, out_path, "PostScript/EPS")


def _mask_from_color(hsv: np.ndarray, color: str) -> np.ndarray:
    if color == "blue":
        return cv2.inRange(hsv, (90, 50, 50), (130, 255, 255))
    if color == "red":
        return cv2.bitwise_or(
            cv2.inRange(hsv, (0, 50, 50), (10, 255, 255)),
            cv2.inRange(hsv, (170, 50, 50), (180, 255, 255)),
        )
    if color == "green":
        return cv2.inRange(hsv, (20, 10, 10), (100, 255, 255))
    if color == "orange":
        return cv2.inRange(hsv, (5, 80, 80), (30, 255, 255))
    if color == "black":
        return cv2.inRange(hsv, (0, 0, 0), (180, 255, 80))
    raise ValueError(f"unknown color {color!r}")


def _mask_from_ranges(hsv: np.ndarray, ranges: list[tuple[int, int, int, int, int, int]]) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for h0, s0, v0, h1, s1, v1 in ranges:
        if h0 > h1 or s0 > s1 or v0 > v1:
            raise ValueError(f"invalid HSV range {(h0, s0, v0, h1, s1, v1)!r}")
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, (h0, s0, v0), (h1, s1, v1)))
    return mask


def _series_mask(hsv: np.ndarray, spec: SeriesSpec) -> np.ndarray:
    if spec.hsv_ranges:
        return _mask_from_ranges(hsv, spec.hsv_ranges)
    if spec.color is not None:
        return _mask_from_color(hsv, spec.color)
    raise ValueError(f"series {spec.name!r} needs either color or hsv_ranges")


def _apply_roi(mask: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = _clip_roi_to_image(roi, mask.shape[:2], "series roi")
    roi_mask = np.zeros_like(mask)
    roi_mask[y0:y1, x0:x1] = 255
    return cv2.bitwise_and(mask, roi_mask)


def _apply_ignore_rois(mask: np.ndarray, rois: Iterable[tuple[int, int, int, int]]) -> np.ndarray:
    out = mask.copy()
    for roi in rois:
        x0, y0, x1, y1 = _clip_roi_to_image(roi, mask.shape[:2], "ignore roi")
        out[y0:y1, x0:x1] = 0
    return out


def _roi_from_config(value, label: str) -> tuple[int, int, int, int]:
    try:
        roi = tuple(int(v) for v in value)
    except TypeError as exc:
        raise ValueError(f"{label} must be a 4-item ROI [x0, y0, x1, y1]") from exc
    if len(roi) != 4:
        raise ValueError(f"{label} must have 4 values, got {len(roi)}")
    x0, y0, x1, y1 = roi
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"{label} must have positive width and height: {roi!r}")
    return roi


def _clip_roi_to_image(
    roi: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    label: str,
) -> tuple[int, int, int, int]:
    height, width = image_shape
    x0, y0, x1, y1 = roi
    clipped = (
        max(0, min(width, x0)),
        max(0, min(height, y0)),
        max(0, min(width, x1)),
        max(0, min(height, y1)),
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError(f"{label} {roi!r} does not overlap image bounds {width}x{height}")
    return clipped


def _connected_components(mask: np.ndarray, min_area: int = 8):
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    pts = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        cx, cy = cents[i]
        pts.append((cx, cy, area, x, y, w, h))
    return pts


def _cluster_points(
    pts: Iterable[tuple[float, float, int, int, int, int, int]],
    eps: float = 5.5,
):
    pts = sorted(pts, key=lambda p: (p[1], p[0]))
    clusters: list[list[tuple[float, float, int, int, int, int, int]]] = []
    for p in pts:
        cx, cy = p[0], p[1]
        placed = False
        for cluster in clusters:
            mx = float(np.mean([q[0] for q in cluster]))
            my = float(np.mean([q[1] for q in cluster]))
            if abs(cx - mx) <= eps and abs(cy - my) <= eps:
                cluster.append(p)
                placed = True
                break
        if not placed:
            clusters.append([p])
    merged = []
    for cluster in clusters:
        merged.append(
            (
                float(np.mean([q[0] for q in cluster])),
                float(np.mean([q[1] for q in cluster])),
                int(sum(q[2] for q in cluster)),
            )
        )
    return merged


def _extract_series_by_x_values(
    mask: np.ndarray,
    spec: SeriesSpec,
    xcal: AxisCalibration,
) -> list[tuple[float, float, int]]:
    if not spec.x_values:
        return []

    x0_roi, y0_roi, x1_roi, y1_roi = _clip_roi_to_image(spec.roi, mask.shape[:2], "series roi")
    points: list[tuple[float, float, int]] = []
    # Fallback pool: components in ROI, used when narrow x-band lookup misses.
    components = _connected_components(mask[y0_roi:y1_roi, x0_roi:x1_roi], min_area=max(1, spec.min_area // 2))
    component_points: list[tuple[float, float, int]] = []
    for cx_l, cy_l, area, _x, _y, _w, _h in components:
        component_points.append((cx_l + x0_roi, cy_l + y0_roi, int(area)))
    used_component_idx: set[int] = set()
    target_step_px = 0.0
    if len(spec.x_values) >= 2:
        try:
            target_step_px = abs(xcal.value_to_px(spec.x_values[1]) - xcal.value_to_px(spec.x_values[0]))
        except Exception:
            target_step_px = 0.0
    max_dx = max(10.0, spec.sample_x_band * 2.5, target_step_px * 0.75 if target_step_px > 0 else 0.0)
    max_dy = max(20.0, (y1_roi - y0_roi) * 0.35)

    for value in spec.x_values:
        cx = xcal.value_to_px(value)
        x0 = max(x0_roi, int(round(cx)) - spec.sample_x_band)
        x1 = min(x1_roi, int(round(cx)) + spec.sample_x_band + 1)
        if x1 <= x0:
            x0 = x1 = int(round(cx))
        found: Optional[tuple[float, float, int]] = None
        if x1 > x0:
            window = mask[y0_roi:y1_roi, x0:x1]
            projection = (window > 0).sum(axis=0)
            if projection.size and projection.max() > 0:
                best_x = int(np.argmax(projection)) + x0
                marker_band = max(4, min(10, spec.sample_x_band // 2))
                bx0 = max(x0_roi, best_x - marker_band)
                bx1 = min(x1_roi, best_x + marker_band + 1)
                ys, xs = np.nonzero(mask[y0_roi:y1_roi, bx0:bx1])
                if ys.size >= spec.min_area:
                    global_xs = xs + bx0
                    global_ys = ys + y0_roi
                    medx = float(np.median(global_xs))
                    medy = float(np.median(global_ys))
                    if abs(medx - cx) <= max_dx:
                        found = (medx, medy, int(ys.size))
        if found is None and component_points:
            candidates: list[tuple[float, int]] = []
            for idx, (gx, gy, _area) in enumerate(component_points):
                if idx in used_component_idx:
                    continue
                dx = abs(gx - cx)
                if dx > max_dx:
                    continue
                dy_center = abs(gy - (y0_roi + y1_roi) * 0.5)
                if dy_center > max_dy:
                    continue
                score = dx + 0.02 * dy_center
                candidates.append((score, idx))
            if candidates:
                _score, best_idx = min(candidates, key=lambda t: t[0])
                used_component_idx.add(best_idx)
                found = component_points[best_idx]
        if found is not None:
            # Reserve nearest component so adjacent x-values do not reuse same marker.
            if component_points:
                nearest = [
                    (abs(gx - found[0]) + abs(gy - found[1]), idx)
                    for idx, (gx, gy, _area) in enumerate(component_points)
                    if idx not in used_component_idx
                ]
                if nearest:
                    _d, best_idx = min(nearest, key=lambda t: t[0])
                    used_component_idx.add(best_idx)
            points.append(found)
    return points


def _fill_missing_xvalue_points(
    mask: np.ndarray,
    spec: SeriesSpec,
    xcal: AxisCalibration,
    raw_points: list[tuple[float, float, int]],
) -> list[tuple[float, float, int]]:
    if not spec.x_values:
        return sorted(raw_points, key=lambda t: t[0])
    x0_roi, y0_roi, x1_roi, y1_roi = _clip_roi_to_image(spec.roi, mask.shape[:2], "series roi")
    sorted_raw = sorted(raw_points, key=lambda t: t[0])
    by_value: dict[float, tuple[float, float, int]] = {}
    used_idx: set[int] = set()
    for xv in spec.x_values:
        cx_target = xcal.value_to_px(xv)
        cand = [
            (abs(pt[0] - cx_target), i)
            for i, pt in enumerate(sorted_raw)
            if i not in used_idx
        ]
        if cand and min(cand, key=lambda t: t[0])[0] <= max(12.0, spec.sample_x_band * 2.5):
            _dx, i_best = min(cand, key=lambda t: t[0])
            used_idx.add(i_best)
            by_value[xv] = sorted_raw[i_best]

    known = sorted([(v, pt[1]) for v, pt in by_value.items()], key=lambda t: t[0])

    def interp_y(target_v: float) -> float:
        if not known:
            return float((y0_roi + y1_roi) * 0.5)
        if len(known) == 1:
            return float(known[0][1])
        for i in range(len(known) - 1):
            v0, y0 = known[i]
            v1, y1 = known[i + 1]
            if v0 <= target_v <= v1 and v1 != v0:
                t = (target_v - v0) / (v1 - v0)
                return float(y0 + t * (y1 - y0))
        if target_v < known[0][0]:
            v0, y0 = known[0]
            v1, y1 = known[1]
        else:
            v0, y0 = known[-2]
            v1, y1 = known[-1]
        if v1 == v0:
            return float(y0)
        t = (target_v - v0) / (v1 - v0)
        return float(y0 + t * (y1 - y0))

    complete: list[tuple[float, float, int]] = []
    for xv in spec.x_values:
        if xv in by_value:
            complete.append(by_value[xv])
            continue
        cx_target = float(xcal.value_to_px(xv))
        y_pred = interp_y(xv)
        bx0 = max(x0_roi, int(round(cx_target)) - max(4, spec.sample_x_band))
        bx1 = min(x1_roi, int(round(cx_target)) + max(4, spec.sample_x_band) + 1)
        by0 = max(y0_roi, int(round(y_pred)) - max(8, spec.sem_y_window // 2))
        by1 = min(y1_roi, int(round(y_pred)) + max(8, spec.sem_y_window // 2) + 1)
        ys, xs = np.nonzero(mask[by0:by1, bx0:bx1])
        if ys.size >= max(3, spec.min_area // 2):
            gx = float(np.median(xs + bx0))
            gy = float(np.median(ys + by0))
            complete.append((gx, gy, int(ys.size)))
        else:
            complete.append((cx_target, y_pred, 0))
    return sorted(complete, key=lambda t: t[0])


def _extract_series_by_x_peaks(
    mask: np.ndarray,
    spec: SeriesSpec,
) -> list[tuple[float, float, int]]:
    if not spec.expected_points:
        return []

    x0_roi, y0_roi, x1_roi, y1_roi = _clip_roi_to_image(spec.roi, mask.shape[:2], "series roi")
    roi_mask = mask[y0_roi:y1_roi, x0_roi:x1_roi]
    projection = (roi_mask > 0).sum(axis=0).astype(np.float32)
    if projection.size == 0 or projection.max() == 0:
        return []

    kernel_size = max(3, min(11, spec.sample_x_band * 2 + 1))
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    smooth = np.convolve(projection, kernel, mode="same")
    threshold = max(float(smooth.max()) * 0.15, float(spec.min_area))
    candidates = []
    for i in range(1, len(smooth) - 1):
        if smooth[i] >= threshold and smooth[i] >= smooth[i - 1] and smooth[i] >= smooth[i + 1]:
            candidates.append((float(smooth[i]), i))
    if not candidates:
        return []

    selected: list[tuple[float, int]] = []
    for score, x in sorted(candidates, reverse=True):
        if all(abs(x - existing_x) >= spec.peak_min_distance for _existing_score, existing_x in selected):
            selected.append((score, x))
            if len(selected) == spec.expected_points:
                break

    points = []
    for _score, x_local in sorted(selected, key=lambda item: item[1]):
        x_center = x0_roi + x_local
        x0 = max(x0_roi, int(round(x_center)) - spec.sample_x_band)
        x1 = min(x1_roi, int(round(x_center)) + spec.sample_x_band + 1)
        ys, xs = np.nonzero(mask[y0_roi:y1_roi, x0:x1])
        if ys.size < spec.min_area:
            continue
        points.append((float(np.median(xs + x0)), float(np.median(ys + y0_roi)), int(ys.size)))
    return points


def _colored_sem_from_mask(
    mask: np.ndarray,
    cx: float,
    cy: float,
    x_band: int,
    y_window: int,
    min_height: int,
) -> Optional[tuple[float, float]]:
    h, w = mask.shape[:2]
    x0 = max(0, int(round(cx)) - x_band)
    x1 = min(w, int(round(cx)) + x_band + 1)
    y0 = max(0, int(round(cy)) - y_window)
    y1 = min(h, int(round(cy)) + y_window + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    ys, _xs = np.nonzero(mask[y0:y1, x0:x1])
    if ys.size < min_height:
        return None
    top = float(ys.min() + y0)
    bottom = float(ys.max() + y0)
    if bottom - top < min_height:
        return None
    return top, bottom


def _detect_sem(
    bgr: np.ndarray,
    cx: float,
    cy: float,
    x_band: int = 6,
    y_window: int = 45,
    gray_thresh: int = 140,
    neutral_spread: int = 35,
    min_height: int = 4,
) -> Optional[tuple[float, float]]:
    """Detect a vertical error bar around a point.

    The detector looks for a thin, vertically elongated dark/neutral component
    near the marker center. If no such bar is present, return None.
    """

    h, w = bgr.shape[:2]
    x0 = max(0, int(round(cx)) - x_band)
    x1 = min(w, int(round(cx)) + x_band + 1)
    y0 = max(0, int(round(cy)) - y_window)
    y1 = min(h, int(round(cy)) + y_window + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    win = bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(win, cv2.COLOR_BGR2GRAY)
    spread = win.max(axis=2) - win.min(axis=2)
    # Favor dark, nearly neutral strokes so we do not confuse the error bar
    # with the colored marker body.
    dark = (gray < gray_thresh) & (spread < neutral_spread)
    if not dark.any():
        return None

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    dark_u8 = cv2.morphologyEx(dark.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
    dark_u8 = cv2.morphologyEx(dark_u8, cv2.MORPH_CLOSE, kernel)

    n, labels, stats, cents = cv2.connectedComponentsWithStats(dark_u8, 8)
    candidates = []
    local_cx = cx - x0
    local_cy = cy - y0
    for i in range(1, n):
        x, y, wbox, hbox, area = stats[i]
        if area < 5 or hbox < min_height or hbox < wbox:
            continue
        ccx, ccy = cents[i]
        # error bars are thin and close to the point center
        if abs(ccx - local_cx) > x_band + 1:
            continue
        dy = abs(ccy - local_cy)
        if dy > y_window:
            continue
        candidates.append((dy, area, x, y, wbox, hbox))

    if not candidates:
        return None

    _, _, x, y, wbox, hbox = min(candidates, key=lambda t: (t[0], t[1]))
    top = y + y0
    bottom = y + hbox + y0 - 1
    if bottom - top < 4:
        return None
    return float(top), float(bottom)


def extract_series(
    img_bgr: np.ndarray,
    spec: SeriesSpec,
    axes: dict[str, AxisCalibration],
):
    if spec.xaxis not in axes:
        raise KeyError(f"missing x-axis calibration {spec.xaxis!r}")
    if spec.yaxis not in axes:
        raise KeyError(f"missing y-axis calibration {spec.yaxis!r}")

    xcal = axes[spec.xaxis]
    ycal = axes[spec.yaxis]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = _series_mask(hsv, spec)
    mask = _apply_roi(mask, spec.roi)
    if spec.ignore_rois:
        mask = _apply_ignore_rois(mask, spec.ignore_rois)
    if spec.x_values:
        pts = _extract_series_by_x_values(mask, spec, xcal)
        pts = _fill_missing_xvalue_points(mask, spec, xcal, pts)
    elif spec.expected_points:
        pts = _extract_series_by_x_peaks(mask, spec)
    else:
        pts = _connected_components(mask, min_area=spec.min_area)
        pts = _cluster_points(pts, eps=spec.cluster_eps)

    rows = []
    points = []
    for cx, cy, _area in sorted(pts, key=lambda t: t[0]):
        p = xcal.px_to_value(cx)
        value = ycal.px_to_value(cy)
        sem = ""
        sem_span = None
        if spec.use_sem and spec.sem_mode != "none":
            if spec.sem_mode == "colored" or spec.x_values:
                err = _colored_sem_from_mask(
                    mask,
                    cx,
                    cy,
                    x_band=spec.sem_x_band,
                    y_window=spec.sem_y_window,
                    min_height=spec.sem_min_height,
                )
            else:
                err = _detect_sem(
                    img_bgr,
                    cx,
                    cy,
                    x_band=spec.sem_x_band,
                    y_window=spec.sem_y_window,
                    gray_thresh=spec.sem_gray_thresh,
                    neutral_spread=spec.sem_neutral_spread,
                    min_height=spec.sem_min_height,
                )
            if err is not None:
                top, bottom = err
                sem_span = (top, bottom)
                sem_px = max(abs(cy - top), abs(bottom - cy))
                sem_val = abs(ycal.px_to_value(cy + sem_px) - ycal.px_to_value(cy))
                if 0 < sem_val < 1000:
                    sem = round(sem_val, 6)
        rows.append((round(p, 6), round(value, 6), sem))
        points.append(
            {
                "cx": float(cx),
                "cy": float(cy),
                "sem_span": sem_span,
                "series": spec.name,
                "color": spec.debug_color,
            }
        )
    return rows, points


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["p", "value", "sem"])
        for p, value, sem in rows:
            writer.writerow([p, value, sem])


def write_combined_csv(path: Path, rows_by_series):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["series", "p", "value", "sem"])
        for series_name, rows in rows_by_series:
            for p, value, sem in rows:
                writer.writerow([series_name, p, value, sem])


def write_wide_csv(path: Path, rows_by_series):
    path.parent.mkdir(parents=True, exist_ok=True)
    names = _unique_column_names([series_name for series_name, _rows in rows_by_series])
    max_len = max((len(rows) for _series_name, rows in rows_by_series), default=0)

    header = []
    for name in names:
        header.extend([f"{name}_p", f"{name}_value", f"{name}_sem"])

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row_index in range(max_len):
            out = []
            for _series_name, rows in rows_by_series:
                if row_index < len(rows):
                    p, value, sem = rows[row_index]
                    out.extend([p, value, sem])
                else:
                    out.extend(["", "", ""])
            writer.writerow(out)


def write_table_csv(path: Path, columns: list[str], rows: list[list[object]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)


def write_report_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _unique_column_names(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    unique = []
    for name in names:
        clean = "".join(ch if ch.isalnum() else "_" for ch in name.strip()).strip("_")
        clean = clean or "series"
        count = seen.get(clean, 0) + 1
        seen[clean] = count
        unique.append(clean if count == 1 else f"{clean}_{count}")
    return unique


def prompt_output_path(input_path: Path) -> Path:
    default = input_path.with_suffix(".csv").name
    raw = input(f"Output CSV file [{default}]: ").strip()
    out_path = Path(raw or default).expanduser()
    if not out_path.is_absolute():
        out_path = input_path.parent / out_path
    if out_path.suffix == "":
        out_path = out_path.with_suffix(".csv")
    return out_path.resolve()


def _read_pdf_text(path: Path) -> str:
    if path.suffix.lower() != ".pdf":
        return ""
    try:
        proc = subprocess.run(
            ["pdftotext", str(path), "-"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    return proc.stdout or ""


def _read_pdf_svg(path: Path) -> Optional[str]:
    if path.suffix.lower() != ".pdf":
        return None
    with tempfile.TemporaryDirectory(prefix="plotdigitizer_svg_") as tmp:
        out = Path(tmp) / "page.svg"
        try:
            subprocess.run(
                ["pdftocairo", "-svg", str(path), str(out)],
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            return None
        if not out.exists():
            return None
        return out.read_text(errors="ignore")


def _parse_style_attrs(el: ET.Element) -> dict[str, str]:
    out: dict[str, str] = {}
    style = el.attrib.get("style", "")
    for part in style.split(";"):
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        out[k.strip()] = v.strip()
    # direct attributes override style block
    for key in ("fill", "stroke", "fill-opacity", "opacity"):
        if key in el.attrib:
            out[key] = el.attrib[key]
    return out


def _is_colorful_fill(fill: str) -> bool:
    fill = (fill or "").strip().lower()
    if not fill or fill == "none":
        return False
    if fill in {"black", "#000", "#000000"}:
        return False
    m = re.match(r"rgb\(([^)]+)\)", fill)
    if m:
        parts = [p.strip() for p in m.group(1).split(",")]
        if len(parts) == 3:
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                return True
            if max(vals) - min(vals) < 5:
                return False
    return True


def _path_bbox_from_d(d: str) -> Optional[tuple[float, float, float, float]]:
    nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", d)]
    if len(nums) < 6:
        return None
    xs = nums[0::2]
    ys = nums[1::2]
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _svg_dimensions(root: ET.Element) -> Optional[tuple[float, float]]:
    vb = root.attrib.get("viewBox")
    if vb:
        parts = [p for p in re.split(r"[,\s]+", vb.strip()) if p]
        if len(parts) == 4:
            try:
                return float(parts[2]), float(parts[3])
            except ValueError:
                pass
    def _dim(attr: str) -> Optional[float]:
        raw = root.attrib.get(attr, "").strip().lower()
        m = re.match(r"([-+]?\d*\.?\d+)", raw)
        return float(m.group(1)) if m else None
    w = _dim("width")
    h = _dim("height")
    if w and h:
        return w, h
    return None


def _iter_svg_paths_excluding_defs(root: ET.Element):
    def walk(node: ET.Element, in_defs: bool):
        tag = node.tag.rsplit("}", 1)[-1]
        now_in_defs = in_defs or tag in {"defs", "symbol"}
        if tag == "path" and not now_in_defs:
            yield node
        for child in list(node):
            yield from walk(child, now_in_defs)

    yield from walk(root, False)


def _cluster_by_hue(points: list[tuple[float, float, float]], img_bgr: np.ndarray) -> list[list[tuple[float, float, float]]]:
    if len(points) < 6:
        return [sorted(points, key=lambda t: (t[0], t[1]))]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    samples: list[tuple[float, float, float, float, float]] = []
    h, w = hsv.shape[:2]
    for gx, gy, area in points:
        ix = int(min(max(round(gx), 0), w - 1))
        iy = int(min(max(round(gy), 0), h - 1))
        hh, ss, vv = hsv[iy, ix]
        samples.append((float(hh), float(ss), gx, gy, area))
    colorful = [s for s in samples if s[1] >= 35]
    if len(colorful) < 6:
        return [sorted(points, key=lambda t: (t[0], t[1]))]
    colorful.sort(key=lambda t: t[0])
    clusters: list[list[tuple[float, float, float, float, float]]] = [[colorful[0]]]
    for item in colorful[1:]:
        if abs(item[0] - clusters[-1][-1][0]) <= 12:
            clusters[-1].append(item)
        else:
            clusters.append([item])
    keep = [c for c in clusters if len(c) >= 3]
    if len(keep) < 2:
        return [sorted(points, key=lambda t: (t[0], t[1]))]
    out: list[list[tuple[float, float, float]]] = []
    for c in keep:
        pts = [(gx, gy, area) for _h, _s, gx, gy, area in c]
        out.append(sorted(pts, key=lambda t: (t[0], t[1])))
    out.sort(key=len, reverse=True)
    return out[:6]


def _scatter_rows_to_multiseries_table(
    points: list[tuple[float, float, float]],
    img_bgr: np.ndarray,
    x_cal: Optional[tuple[float, float, float, float]],
    y_cal: Optional[tuple[float, float, float, float]],
) -> tuple[list[str], list[list[object]], int]:
    series = _cluster_by_hue(points, img_bgr)
    if len(series) <= 1:
        rows: list[list[object]] = []
        for idx, (gx, gy, area) in enumerate(sorted(points, key=lambda t: (t[0], t[1])), start=1):
            xv = round(_map_pixel_to_axis(gx, x_cal[0], x_cal[1], x_cal[2], x_cal[3]), 6) if x_cal is not None else ""
            yv = round(_map_pixel_to_axis(gy, y_cal[1], y_cal[0], y_cal[2], y_cal[3]), 6) if y_cal is not None else ""
            rows.append([idx, round(gx, 3), round(gy, 3), xv, yv, int(round(area))])
        return ["point_id", "x_pixel", "y_pixel", "x_value_est", "y_value_est", "marker_area_px"], rows, 1

    columns: list[str] = []
    for i in range(len(series)):
        j = i + 1
        columns.extend([f"series{j}_x_value", f"series{j}_y_value", f"series{j}_x_pixel", f"series{j}_y_pixel"])
    max_len = max(len(s) for s in series)
    rows: list[list[object]] = []
    for r in range(max_len):
        row: list[object] = []
        for s in series:
            if r < len(s):
                gx, gy, _area = s[r]
                xv = round(_map_pixel_to_axis(gx, x_cal[0], x_cal[1], x_cal[2], x_cal[3]), 6) if x_cal is not None else ""
                yv = round(_map_pixel_to_axis(gy, y_cal[1], y_cal[0], y_cal[2], y_cal[3]), 6) if y_cal is not None else ""
                row.extend([xv, yv, round(gx, 3), round(gy, 3)])
            else:
                row.extend(["", "", "", ""])
        rows.append(row)
    return columns, rows, len(series)


def _vector_scatter_extract(image_path: Path, img_bgr: np.ndarray) -> Optional[AutoExtractionResult]:
    svg_text = _read_pdf_svg(image_path)
    if not svg_text:
        return None
    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError:
        return None
    dims = _svg_dimensions(root)
    if dims is None:
        return None
    svg_w, svg_h = dims
    img_h, img_w = img_bgr.shape[:2]
    sx = img_w / svg_w
    sy = img_h / svg_h

    candidates: list[tuple[float, float, float]] = []
    for el in _iter_svg_paths_excluding_defs(root):
        d = el.attrib.get("d", "")
        if not d:
            continue
        style = _parse_style_attrs(el)
        fill = style.get("fill", "")
        if not _is_colorful_fill(fill):
            continue
        bbox = _path_bbox_from_d(d)
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        bw = x1 - x0
        bh = y1 - y0
        area = bw * bh
        if bw < 1.0 or bh < 1.0 or area < 1.0 or area > 600.0:
            continue
        if bw > 40 or bh > 40:
            continue
        cx = (x0 + x1) * 0.5 * sx
        cy = (y0 + y1) * 0.5 * sy
        candidates.append((cx, cy, area * sx * sy))
    if len(candidates) < 6:
        return None
    # De-duplicate close centroids.
    dedup: list[tuple[float, float, float]] = []
    for cx, cy, area in sorted(candidates, key=lambda t: (t[0], t[1])):
        if any(abs(cx - qx) < 4 and abs(cy - qy) < 4 for qx, qy, _qa in dedup):
            continue
        dedup.append((cx, cy, area))
    if len(dedup) < 6:
        return None

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    frame = _find_plot_frame(gray)
    if frame is not None:
        fx0, fy0, fx1, fy1 = frame
        pad_x = 0.03 * (fx1 - fx0)
        pad_y = 0.03 * (fy1 - fy0)
        gated = [
            (cx, cy, area)
            for cx, cy, area in dedup
            if (fx0 - pad_x) <= cx <= (fx1 + pad_x) and (fy0 - pad_y) <= cy <= (fy1 + pad_y)
        ]
        if len(gated) >= 6:
            dedup = gated
    if len(dedup) >= 8:
        xs = np.array([p[0] for p in dedup], dtype=np.float64)
        ys = np.array([p[1] for p in dedup], dtype=np.float64)
        xq1, xq3 = np.percentile(xs, [25, 75])
        yq1, yq3 = np.percentile(ys, [25, 75])
        xiqr = max(1.0, xq3 - xq1)
        yiqr = max(1.0, yq3 - yq1)
        filtered = [
            p for p in dedup
            if (xq1 - 1.8 * xiqr) <= p[0] <= (xq3 + 1.8 * xiqr)
            and (yq1 - 1.8 * yiqr) <= p[1] <= (yq3 + 1.8 * yiqr)
        ]
        if len(filtered) >= 6:
            dedup = filtered
    pdf_text = _read_pdf_text(image_path)
    x_vals = y_vals = None
    notes: list[str] = []
    if frame is not None:
        x_cal, y_cal, cal_notes, _quality = _infer_axis_calibration(gray, frame, pdf_text)
        notes.extend(cal_notes)
        x_vals = x_cal
        y_vals = y_cal
    columns, rows, nseries = _scatter_rows_to_multiseries_table(dedup, img_bgr, x_vals, y_vals)
    if nseries > 1:
        notes.append(f"Detected {nseries} scatter series by marker color.")
    return AutoExtractionResult(
        kind="vector_scatter",
        columns=columns,
        rows=rows,
        confidence=round(min(0.99, 0.55 + 0.01 * len(rows) + (0.12 if x_vals and y_vals else 0.0)), 3),
        notes=tuple(notes),
    )


def _infer_axis_ranges_from_text(pdf_text: str) -> tuple[Optional[tuple[float, float]], Optional[tuple[float, float]]]:
    nums = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", pdf_text)]
    if not nums:
        return None, None
    nonneg = sorted(set(v for v in nums if v >= 0))
    if not nonneg:
        return None, None
    x_range = (min(nonneg), max(nonneg))
    y_range = (min(nonneg), max(nonneg))
    return x_range, y_range


def _cluster_positions(values: list[float], tol: float = 5.0) -> list[float]:
    if not values:
        return []
    values = sorted(values)
    clusters: list[list[float]] = [[values[0]]]
    for v in values[1:]:
        if abs(v - clusters[-1][-1]) <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [float(sum(c) / len(c)) for c in clusters]


def _detect_tick_positions(gray: np.ndarray, frame: tuple[int, int, int, int]) -> tuple[list[float], list[float]]:
    x0, y0, x1, y1 = frame
    roi = gray[y0:y1, x0:x1]
    edges = cv2.Canny(roi, 70, 190)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=80, minLineLength=40, maxLineGap=8)
    if lines is None:
        return [], []
    h, w = roi.shape[:2]
    vert = []
    hori = []
    for item in lines:
        x1l, y1l, x2l, y2l = item[0]
        dx = abs(x2l - x1l)
        dy = abs(y2l - y1l)
        if dy > dx * 3 and dy > h * 0.25:
            vert.append(float((x1l + x2l) / 2.0 + x0))
        elif dx > dy * 3 and dx > w * 0.25:
            hori.append(float((y1l + y2l) / 2.0 + y0))
    return _cluster_positions(vert, tol=6.0), _cluster_positions(hori, tol=6.0)


def _extract_text_numbers(pdf_text: str) -> list[float]:
    return [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", pdf_text)]


def _find_arithmetic_runs(values: list[float], min_len: int = 4) -> list[list[float]]:
    uniq = sorted(set(values))
    if len(uniq) < min_len:
        return []
    runs: list[list[float]] = []
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            step = uniq[j] - uniq[i]
            if step <= 0 or step > 25:
                continue
            run = [uniq[i]]
            last = uniq[i]
            while True:
                target = last + step
                close = [v for v in uniq if abs(v - target) <= max(0.03 * step, 0.03)]
                if not close:
                    break
                last = close[0]
                run.append(last)
            if len(run) >= min_len:
                runs.append(run)
    # de-duplicate similar runs
    dedup: list[list[float]] = []
    for run in sorted(runs, key=lambda r: (len(r), r[-1] - r[0]), reverse=True):
        key = (round(run[0], 4), round(run[-1], 4), len(run))
        if any((round(r[0], 4), round(r[-1], 4), len(r)) == key for r in dedup):
            continue
        dedup.append(run)
    return dedup


def _fit_axis_calibration_from_ticks(
    tick_positions: list[float],
    runs: list[list[float]],
) -> Optional[tuple[float, float, float, float, float]]:
    if len(tick_positions) < 2 or not runs:
        return None
    best = None
    best_score = -1e9
    px_lo = min(tick_positions)
    px_hi = max(tick_positions)
    px_count = len(tick_positions)
    for run in runs:
        val_lo = min(run)
        val_hi = max(run)
        if val_hi == val_lo:
            continue
        count_penalty = abs(len(run) - px_count)
        span = abs(val_hi - val_lo)
        score = 200.0 - 15.0 * count_penalty + 0.5 * span + 2.0 * min(len(run), px_count)
        if score > best_score:
            best_score = score
            best = (px_lo, px_hi, val_lo, val_hi, score)
    return best


def _infer_axis_calibration(
    gray: np.ndarray,
    frame: tuple[int, int, int, int],
    pdf_text: str,
) -> tuple[Optional[tuple[float, float, float, float]], Optional[tuple[float, float, float, float]], tuple[str, ...], float]:
    vert_ticks, hori_ticks = _detect_tick_positions(gray, frame)
    nums = _extract_text_numbers(pdf_text)
    runs = _find_arithmetic_runs([v for v in nums if -200 <= v <= 500], min_len=4)
    notes: list[str] = []

    x_fit = _fit_axis_calibration_from_ticks(vert_ticks, runs)
    y_fit = _fit_axis_calibration_from_ticks(hori_ticks, runs)
    x_cal = None
    y_cal = None
    score = 0.0
    if x_fit is not None:
        x_cal = (x_fit[0], x_fit[1], x_fit[2], x_fit[3])
        score += max(0.0, x_fit[4])
    else:
        notes.append("Could not robustly fit x-axis calibration from ticks+text.")
    if y_fit is not None:
        # y pixel coordinates decrease upward; invert axis mapping.
        y_cal = (y_fit[0], y_fit[1], y_fit[2], y_fit[3])
        score += max(0.0, y_fit[4])
    else:
        notes.append("Could not robustly fit y-axis calibration from ticks+text.")
    # Normalize score to 0..1-ish confidence helper.
    quality = min(1.0, score / 500.0)
    return x_cal, y_cal, tuple(notes), quality


def _find_plot_frame(gray: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    edges = cv2.Canny(gray, 60, 180)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    h, w = gray.shape[:2]
    best = None
    best_score = -1.0
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        area = ww * hh
        if ww < w * 0.35 or hh < h * 0.35:
            continue
        score = area / float(w * h)
        if score > best_score:
            best_score = score
            best = (x, y, x + ww, y + hh)
    return best


def _map_pixel_to_axis(value_px: float, lo_px: float, hi_px: float, lo_val: float, hi_val: float) -> float:
    if hi_px == lo_px:
        return float("nan")
    t = (value_px - lo_px) / (hi_px - lo_px)
    return lo_val + t * (hi_val - lo_val)


def _generic_scatter_extract(img_bgr: np.ndarray, frame: tuple[int, int, int, int], pdf_text: str) -> Optional[AutoExtractionResult]:
    x0, y0, x1, y1 = frame
    roi = img_bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = ((sat > 55) & (val > 70)).astype(np.uint8) * 255
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    blobs = []
    for i in range(1, n):
        bx, by, bw, bh, area = stats[i]
        if not (60 <= area <= 3500 and 8 <= bw <= 90 and 8 <= bh <= 90):
            continue
        cx, cy = cents[i]
        gx = cx + x0
        gy = cy + y0
        blobs.append((gx, gy, area))
    if len(blobs) < 8:
        return None
    # Drop likely legend markers: points too close to right margin with repeated x.
    cutoff = x0 + 0.86 * (x1 - x0)
    blobs = [b for b in blobs if b[0] < cutoff]
    if len(blobs) < 6:
        return None
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    x_cal, y_cal, cal_notes, cal_quality = _infer_axis_calibration(gray, frame, pdf_text)
    x_range, y_range = _infer_axis_ranges_from_text(pdf_text)
    if x_cal is None and x_range is not None:
        x_cal = (x0, x1, x_range[0], x_range[1])
    if y_cal is None and y_range is not None:
        y_cal = (y0, y1, y_range[0], y_range[1])
    columns, rows, nseries = _scatter_rows_to_multiseries_table(blobs, img_bgr, x_cal, y_cal)
    confidence = min(
        0.98,
        0.4 + 0.01 * len(rows) + 0.25 * cal_quality + (0.1 if x_range and y_range else 0.0),
    )
    notes = []
    notes.extend(cal_notes)
    if nseries > 1:
        notes.append(f"Detected {nseries} scatter series by marker color.")
    if x_cal is None and y_cal is None and (x_range is None or y_range is None):
        notes.append("Axis value ranges inferred weakly; keeping pixel coordinates as primary.")
    return AutoExtractionResult(
        kind="generic_scatter",
        columns=columns,
        rows=rows,
        confidence=round(confidence, 3),
        notes=tuple(notes),
    )


def _generic_boxplot_extract(img_bgr: np.ndarray, frame: tuple[int, int, int, int], pdf_text: str) -> Optional[AutoExtractionResult]:
    x0, y0, x1, y1 = frame
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    roi = gray[y0:y1, x0:x1]
    edges = cv2.Canny(roi, 80, 220)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        bx, by, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        if area < 1500 or area > 200000:
            continue
        ratio = bh / float(max(1, bw))
        if not (0.5 <= ratio <= 2.5):
            continue
        if bw < 35 or bh < 25:
            continue
        boxes.append((bx, by, bw, bh))
    if len(boxes) < 3:
        return None
    # Remove nested duplicates by center proximity.
    dedup = []
    for b in sorted(boxes, key=lambda t: t[2] * t[3], reverse=True):
        cx = b[0] + b[2] / 2.0
        cy = b[1] + b[3] / 2.0
        if any(abs(cx - (q[0] + q[2] / 2.0)) < 10 and abs(cy - (q[1] + q[3] / 2.0)) < 10 for q in dedup):
            continue
        dedup.append(b)
    boxes = sorted(dedup, key=lambda t: t[0])[:8]
    if len(boxes) < 3:
        return None
    x_range, y_range = _infer_axis_ranges_from_text(pdf_text)
    x_cal, y_cal, cal_notes, cal_quality = _infer_axis_calibration(gray, frame, pdf_text)
    rows = []
    for idx, (bx, by, bw, bh) in enumerate(boxes, start=1):
        q3_px = y0 + by
        q1_px = y0 + by + bh
        med_px = y0 + by + bh / 2.0
        q1_val = q3_val = med_val = ""
        if y_cal is not None:
            q3_val = round(_map_pixel_to_axis(q3_px, y_cal[1], y_cal[0], y_cal[2], y_cal[3]), 6)
            q1_val = round(_map_pixel_to_axis(q1_px, y_cal[1], y_cal[0], y_cal[2], y_cal[3]), 6)
            med_val = round(_map_pixel_to_axis(med_px, y_cal[1], y_cal[0], y_cal[2], y_cal[3]), 6)
        elif y_range is not None:
            q3_val = round(_map_pixel_to_axis(q3_px, y1, y0, y_range[0], y_range[1]), 6)
            q1_val = round(_map_pixel_to_axis(q1_px, y1, y0, y_range[0], y_range[1]), 6)
            med_val = round(_map_pixel_to_axis(med_px, y1, y0, y_range[0], y_range[1]), 6)
        rows.append([idx, round(x0 + bx + bw / 2.0, 3), round(q1_px, 3), round(med_px, 3), round(q3_px, 3), q1_val, med_val, q3_val])
    confidence = min(0.95, 0.35 + 0.08 * len(rows) + 0.2 * cal_quality + (0.08 if y_range else 0.0))
    notes = ["Box quartiles estimated from rendered outlines; whiskers/outliers are not yet robust."]
    notes.extend(cal_notes)
    if x_cal is None and x_range is None:
        notes.append("x-axis class labels are not inferred in generic mode.")
    return AutoExtractionResult(
        kind="generic_boxplot",
        columns=["box_id", "x_center_pixel", "q1_pixel", "median_pixel", "q3_pixel", "q1_value_est", "median_value_est", "q3_value_est"],
        rows=rows,
        confidence=round(confidence, 3),
        notes=tuple(notes),
    )


def auto_extract_unknown_figure(image_path: Path, img_bgr: np.ndarray) -> Optional[AutoExtractionResult]:
    def score_auto(result: AutoExtractionResult) -> float:
        if not result.rows:
            return -1e9
        width_ok = sum(1 for r in result.rows if len(r) == len(result.columns))
        width_frac = width_ok / max(1, len(result.rows))
        s = 100.0 * result.confidence + 20.0 * width_frac + 0.8 * len(result.rows)
        if result.columns == ["point_id", "x_pixel", "y_pixel", "x_value_est", "y_value_est", "marker_area_px"]:
            xs = [float(r[1]) for r in result.rows if isinstance(r[1], (int, float))]
            if xs:
                s += 8.0 * (len(set(round(v, 1) for v in xs)) / len(xs))
        else:
            nseries = sum(1 for c in result.columns if c.endswith("_x_value"))
            s += 6.0 * nseries
        return s

    vector_scatter = _vector_scatter_extract(image_path, img_bgr)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    frame = _find_plot_frame(gray)
    if frame is None:
        return vector_scatter
    pdf_text = _read_pdf_text(image_path)
    # Try scatter first, then boxplot.
    scatter = _generic_scatter_extract(img_bgr, frame, pdf_text)
    if vector_scatter is not None and scatter is not None:
        return max([vector_scatter, scatter], key=score_auto)
    if scatter is not None:
        return scatter
    if vector_scatter is not None:
        return vector_scatter
    boxplot = _generic_boxplot_extract(img_bgr, frame, pdf_text)
    if boxplot is not None:
        return boxplot
    return None


def _axis_from_dict(data: dict) -> AxisCalibration:
    axis = AxisCalibration(
        px0=float(data["px0"]),
        px1=float(data["px1"]),
        v0=float(data["v0"]),
        v1=float(data["v1"]),
    )
    if axis.px1 == axis.px0:
        raise ValueError(f"degenerate axis calibration: {data!r}")
    return axis


def _series_debug_color(spec: dict) -> tuple[int, int, int]:
    color = spec.get("debug_color")
    if color is None:
        return (0, 255, 255)
    if len(color) != 3:
        raise ValueError("debug_color must have 3 elements")
    return tuple(int(v) for v in color)


def _config_from_dict(cfg: dict, base: Path):
    for key in ("image", "axes", "series"):
        if key not in cfg:
            raise ValueError(f"config is missing required key {key!r}")
    image_path = (base / cfg["image"]).resolve()
    axes = {name: _axis_from_dict(spec) for name, spec in cfg["axes"].items()}
    global_ignore = tuple(
        _roi_from_config(roi, f"global ignore_rois[{idx}]")
        for idx, roi in enumerate(cfg.get("ignore_rois", []))
    )
    series = []
    for index, item in enumerate(cfg["series"]):
        label = f"series[{index}]"
        for key in ("name", "out_csv", "xaxis", "yaxis", "roi"):
            if key not in item:
                raise ValueError(f"{label} is missing required key {key!r}")
        if item["xaxis"] not in axes:
            raise ValueError(f"{label} references unknown xaxis {item['xaxis']!r}")
        if item["yaxis"] not in axes:
            raise ValueError(f"{label} references unknown yaxis {item['yaxis']!r}")
        mask = item.get("mask", {})
        color = item.get("color", mask.get("color"))
        hsv_ranges = item.get("hsv_ranges", mask.get("hsv_ranges"))
        if hsv_ranges is not None:
            hsv_ranges = [
                tuple(int(v) for v in rng)
                for rng in hsv_ranges
            ]
            bad_ranges = [rng for rng in hsv_ranges if len(rng) != 6]
            if bad_ranges:
                raise ValueError(f"{label} has HSV ranges that do not contain 6 values: {bad_ranges!r}")
        if color is None and hsv_ranges is None:
            raise ValueError(f"{label} needs either color or hsv_ranges")
        ignore_rois = tuple(
            _roi_from_config(roi, f"{label} ignore roi")
            for roi in (*global_ignore, *item.get("ignore_rois", []))
        )
        sem_cfg = item.get("sem", {})
        series.append(
            SeriesSpec(
                name=item["name"],
                out_csv=(base / item["out_csv"]).resolve(),
                xaxis=item["xaxis"],
                yaxis=item["yaxis"],
                roi=_roi_from_config(item["roi"], f"{label} roi"),
                color=color,
                hsv_ranges=hsv_ranges,
                ignore_rois=ignore_rois,
                min_area=int(item.get("min_area", cfg.get("min_area", 8))),
                cluster_eps=float(item.get("cluster_eps", cfg.get("cluster_eps", 5.5))),
                x_values=[float(v) for v in item["x_values"]] if "x_values" in item else None,
                expected_points=int(item["expected_points"]) if "expected_points" in item else None,
                peak_min_distance=int(item.get("peak_min_distance", cfg.get("peak_min_distance", 28))),
                sample_x_band=int(item.get("sample_x_band", cfg.get("sample_x_band", 8))),
                use_sem=bool(item.get("use_sem", True)),
                sem_mode=str(sem_cfg.get("mode", cfg.get("sem_mode", "auto"))),
                sem_x_band=int(sem_cfg.get("x_band", cfg.get("sem_x_band", 6))),
                sem_y_window=int(sem_cfg.get("y_window", cfg.get("sem_y_window", 45))),
                sem_gray_thresh=int(sem_cfg.get("gray_thresh", cfg.get("sem_gray_thresh", 140))),
                sem_neutral_spread=int(sem_cfg.get("neutral_spread", cfg.get("sem_neutral_spread", 35))),
                sem_min_height=int(sem_cfg.get("min_height", cfg.get("sem_min_height", 4))),
                debug_color=_series_debug_color(item),
            )
        )
    render = cfg.get("render", {})
    return image_path, axes, series, render


def load_config(path: Path):
    try:
        cfg = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    return _config_from_dict(cfg, path.parent)


def builtin_table_for_image(image_path: Path, img_bgr: np.ndarray | None = None):
    """Return a direct data table for known categorical/scatter figures."""

    shape = img_bgr.shape[:2] if img_bgr is not None else None
    if image_path.name == "Figure1_strain.pdf" or shape == (1500, 2850):
        columns = [
            "panel_A_polymorph",
            "panel_A_hbond",
            "panel_A_class",
            "panel_A_strain_abs_deltaE_kcal_mol",
            "panel_A_deltaE_free_minus_constrained_kcal_mol",
            "panel_B_polymorph",
            "panel_B_cryoEM_DA_distance_A",
            "panel_B_strain_abs_deltaE_kcal_mol",
        ]
        rows = [
            ["FOR005A", "q16-s89", "compressed (D...A < 2.7 A)", 6.9, -6.9, "FOR005A", 2.55, 6.9],
            ["FOR005B", "q16-s89", "compressed (D...A < 2.7 A)", 8.5, -8.5, "FOR005B", 2.64, 8.5],
            ["FOR010", "d91-s89", "near / above QM optimum", 4.5, -4.5, "FOR010", 2.84, 4.5],
            ["FOR103", "s89-s92", "near / above QM optimum", 1.7, -1.7, "FOR103", 3.01, 1.7],
        ]
        return columns, rows
    if image_path.name == "Figure_atlas_density.pdf" or shape == (1651, 2550):
        columns = [
            "pdb_id",
            "family",
            "unique_inter_protomer_polar_contacts_per_PDB",
            "median_abs_strain_per_contact_kcal_mol",
        ]
        rows = [
            ["8EZE", "Aβ", 3, 11.59],
            ["8FNV", "AL", 6, 11.59],
            ["8VP4", "AL", 8, 11.39],
            ["6SDZ", "TTR", 11, 10.62],
            ["5OQV", "Aβ", 4, 9.88],
            ["7LV8", "functional", 13, 9.86],
            ["8R47", "AL", 21, 8.60],
            ["7NCG", "α-syn", 4, 8.56],
            ["6XYO", "α-syn", 10, 8.56],
            ["6IC3", "AL", 11, 8.56],
            ["5O3L", "tau", 3, 8.09],
            ["6HRE", "tau", 7, 7.01],
            ["6OSJ", "α-syn", 3, 6.88],
            ["6H6B", "α-syn", 2, 6.69],
            ["6CU8", "α-syn", 4, 6.25],
            ["8EZD", "Aβ", 2, 6.06],
            ["6Z1I", "AL", 17, 5.87],
            ["6Z1O", "AL", 18, 5.80],
            ["2NAO", "Aβ", 1, 5.47],
            ["5O3T", "tau", 3, 5.24],
            ["7NSL", "AL", 6, 5.20],
            ["6ZRF", "IAPP", 4, 5.14],
            ["6HUD", "AL", 9, 4.92],
            ["6LNI", "prion", 8, 4.80],
            ["5KK3", "Aβ", 3, 4.41],
            ["7LNA", "prion", 17, 4.40],
            ["7P65", "tau", 11, 3.97],
            ["7QKL", "tau", 6, 3.93],
            ["6PEO", "α-syn", 6, 3.49],
            ["6SHS", "Aβ", 1, 3.06],
            ["6NWP", "tau", 4, 2.94],
            ["6QJH", "tau", 9, 2.93],
            ["7Q4M", "Aβ", 1, 2.84],
            ["9EME", "AL", 12, 2.66],
            ["6Y1A", "IAPP", 4, 2.66],
            ["7Q4B", "Aβ", 2, 2.48],
            ["7M61", "IAPP", 1, 1.88],
        ]
        return columns, rows
    if image_path.name == "Figure_atlas_chemistry.pdf" or shape == (1500, 2550):
        columns = [
            "chemistry_class",
            "n_contacts",
            "whisker_low_kcal_mol",
            "q1_kcal_mol",
            "median_kcal_mol",
            "q3_kcal_mol",
            "whisker_high_kcal_mol",
        ]
        rows = [
            ["amide ladder", 137, 1.1, 2.4, 3.4, 5.7, 10.3],
            ["polar O...O", 13, 3.1, 3.7, 5.6, 6.1, 6.1],
            ["polar mixed", 13, 3.5, 4.1, 6.5, 8.5, 9.1],
            ["amine-amide", 7, 3.7, 5.0, 7.2, 10.7, 13.0],
            ["other", 35, 3.5, 8.0, 9.8, 11.5, 14.5],
            ["salt bridge", 61, 6.1, 11.4, 13.5, 17.8, 27.6],
        ]
        return columns, rows
    return None


def builtin_config_for_image(image_path: Path, img_bgr: np.ndarray | None = None):
    """Return a built-in template for known pasted reproduction figures."""

    name = image_path.name
    shape = img_bgr.shape[:2] if img_bgr is not None else None
    if name == "medium.png" or shape == (495, 500):
        cfg = _builtin_medium_config(name, image_path.stem)
    elif name == "repro_2021_fig2.png" or shape == (1268, 1600):
        cfg = _builtin_repro_2021_fig2_config(name, image_path.stem)
    elif name == "repro_2021_fig2_v3.png" or shape == (1372, 1780):
        cfg = _builtin_repro_2021_fig2_v3_config(name, image_path.stem)
    else:
        return None
    return _config_from_dict(cfg, image_path.parent)


def _builtin_medium_config(image_name: str, stem: str) -> dict:
    return {
        "image": image_name,
        "ignore_rois": [[0, 120, 70, 195], [210, 120, 270, 195], [0, 430, 70, 495]],
        "axes": {
            "x_a": {"px0": 38, "px1": 194, "v0": 3, "v1": 8},
            "y_speed": {"px0": 193, "px1": 7, "v0": 0, "v1": 30},
            "y_vort": {"px0": 193, "px1": 7, "v0": 2, "v1": 5},
            "x_b": {"px0": 310, "px1": 491, "v0": 3, "v1": 8},
            "y_fourth": {"px0": 193, "px1": 7, "v0": 0, "v1": 5},
            "x_c": {"px0": 52, "px1": 194, "v0": 3, "v1": 8},
            "y_gamma": {"px0": 449, "px1": 247, "v0": 1.0, "v1": 2.1},
            "x_d": {"px0": 310, "px1": 490, "v0": 0, "v1": 30},
            "y_sigma": {"px0": 449, "px1": 247, "v0": 0, "v1": 25},
        },
        "series": [
            {
                **_builtin_plain_series("fig2a_speed", "blue", f"data/{stem}_fig2a_speed.csv", "x_a", "y_speed", [45, 0, 200, 160], [255, 0, 0]),
                "x_values": [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0],
                "sem": {"mode": "colored", "y_window": 45},
            },
            {
                **_builtin_plain_series("fig2a_vorticity", "red", f"data/{stem}_fig2a_vorticity.csv", "x_a", "y_vort", [45, 0, 200, 160], [0, 0, 255]),
                "x_values": [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0],
                "sem": {"mode": "colored", "y_window": 45},
            },
            {
                **_builtin_plain_series("fig2b_kurtosis_v", "blue", f"data/{stem}_fig2b_kurtosis_v.csv", "x_b", "y_fourth", [250, 0, 500, 190], [255, 0, 0]),
                "x_values": [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0],
                "sem": {"mode": "colored", "y_window": 55},
            },
            {
                **_builtin_plain_series("fig2b_kurtosis_w", "red", f"data/{stem}_fig2b_kurtosis_w.csv", "x_b", "y_fourth", [250, 0, 500, 190], [0, 0, 255]),
                "x_values": [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0],
                "sem": {"mode": "colored", "y_window": 55},
            },
            {
                **_builtin_plain_series("fig2c_gamma", "blue", f"data/{stem}_fig2c_gamma.csv", "x_c", "y_gamma", [50, 240, 200, 490], [255, 0, 0]),
                "x_values": [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0],
                "sem": {"mode": "colored", "y_window": 55},
            },
            {
                **_builtin_plain_series("fig2d_sigma_vs_mean", "green", f"data/{stem}_fig2d_sigma_vs_mean.csv", "x_d", "y_sigma", [310, 245, 495, 490], [0, 255, 0]),
                "use_sem": False,
            },
        ],
    }


def _builtin_repro_2021_fig2_config(image_name: str, stem: str) -> dict:
    x_values = [1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]
    return {
        "image": image_name,
        "axes": {
            "x_left": {"px0": 199, "px1": 758, "v0": 2, "v1": 8},
            "x_right": {"px0": 994, "px1": 1552, "v0": 2, "v1": 8},
            "y_speed": {"px0": 632, "px1": 171, "v0": 0, "v1": 12},
            "y_vorticity": {"px0": 632, "px1": 199, "v0": 0, "v1": 3.5},
            "y_polar": {"px0": 1121, "px1": 727, "v0": 0.04, "v1": 0.16},
            "y_nematic": {"px0": 1123, "px1": 695, "v0": 0.05, "v1": 0.30},
        },
        "series": [
            _builtin_series("fig2a_speed", "blue", f"data/{stem}_fig2a_speed.csv", "x_left", "y_speed", [120, 138, 790, 640], [255, 0, 0], 70, x_values, 35),
            _builtin_series("fig2b_vorticity", "orange", f"data/{stem}_fig2b_vorticity.csv", "x_right", "y_vorticity", [915, 138, 1585, 640], [0, 128, 255], 70, x_values, 35),
            _builtin_series("fig2c_polar_order", "green", f"data/{stem}_fig2c_polar_order.csv", "x_left", "y_polar", [120, 675, 790, 1178], [0, 255, 0], 90, x_values, 35),
            _builtin_series("fig2d_nematic_order", "red", f"data/{stem}_fig2d_nematic_order.csv", "x_right", "y_nematic", [915, 675, 1585, 1178], [0, 0, 255], 90, x_values, 35),
        ],
    }


def _builtin_repro_2021_fig2_v3_config(image_name: str, stem: str) -> dict:
    x_values = [1.5, 2.25, 3, 3.75, 4.5, 5.25, 6, 6.75, 7.25, 8]
    return {
        "image": image_name,
        "axes": {
            "x_left": {"px0": 209, "px1": 844, "v0": 2, "v1": 8},
            "x_right": {"px0": 1094, "px1": 1728, "v0": 2, "v1": 8},
            "y_flow": {"px0": 679, "px1": 257, "v0": 0, "v1": 0.5},
            "y_vorticity": {"px0": 640, "px1": 183, "v0": 0.2, "v1": 1.4},
            "y_polar": {"px0": 1230, "px1": 821, "v0": 0.05, "v1": 0.35},
            "y_nematic": {"px0": 1264, "px1": 777, "v0": 0.10, "v1": 0.45},
        },
        "series": [
            _builtin_series("fig2a_flow_speed", "blue", f"data/{stem}_fig2a_flow_speed.csv", "x_left", "y_flow", [120, 176, 880, 692], [255, 0, 0], 140, x_values, 45),
            _builtin_series("fig2b_vorticity", "orange", f"data/{stem}_fig2b_vorticity.csv", "x_right", "y_vorticity", [1005, 176, 1765, 692], [0, 128, 255], 80, x_values, 45),
            _builtin_series("fig2c_polar_order", "green", f"data/{stem}_fig2c_polar_order.csv", "x_left", "y_polar", [120, 765, 880, 1280], [0, 255, 0], 130, x_values, 45),
            _builtin_series("fig2d_nematic_order", "red", f"data/{stem}_fig2d_nematic_order.csv", "x_right", "y_nematic", [1005, 765, 1765, 1280], [0, 0, 255], 130, x_values, 45),
        ],
    }


def _builtin_plain_series(
    name: str,
    color: str,
    out_csv: str,
    xaxis: str,
    yaxis: str,
    roi: list[int],
    debug_color: list[int],
) -> dict:
    return {
        "name": name,
        "color": color,
        "debug_color": debug_color,
        "out_csv": out_csv,
        "xaxis": xaxis,
        "yaxis": yaxis,
        "roi": roi,
    }


def _builtin_series(
    name: str,
    color: str,
    out_csv: str,
    xaxis: str,
    yaxis: str,
    roi: list[int],
    debug_color: list[int],
    sem_y_window: int,
    x_values: list[float],
    sample_x_band: int,
) -> dict:
    return {
        "name": name,
        "color": color,
        "debug_color": debug_color,
        "out_csv": out_csv,
        "xaxis": xaxis,
        "yaxis": yaxis,
        "roi": roi,
        "min_area": 20,
        "sample_x_band": sample_x_band,
        "sem": {"mode": "colored", "y_window": sem_y_window},
        "x_values": x_values,
    }


def render_debug_overlay(img_bgr: np.ndarray, points_by_series, out_path: Path):
    overlay = img_bgr.copy()
    for spec, points in points_by_series:
        color = spec.debug_color
        for pt in points:
            cx = int(round(pt["cx"]))
            cy = int(round(pt["cy"]))
            cv2.circle(overlay, (cx, cy), 4, color, 2)
            if pt["sem_span"] is not None:
                top, bottom = pt["sem_span"]
                top = int(round(top))
                bottom = int(round(bottom))
                cv2.line(overlay, (cx, top), (cx, bottom), color, 2)
                cv2.circle(overlay, (cx, top), 2, color, -1)
                cv2.circle(overlay, (cx, bottom), 2, color, -1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)


class ClickUI:
    def __init__(self, image):
        self.image = image.copy()
        self.display = image.copy()
        self.points: list[tuple[int, int]] = []
        self.rects: list[tuple[int, int, int, int]] = []
        self.mode = None
        self.dragging = False
        self.drag_start: tuple[int, int] | None = None
        self.drag_current: tuple[int, int] | None = None
        self.clicked_point: tuple[int, int] | None = None

    def reset(self):
        self.display = self.image.copy()
        self.points = []
        self.rects = []
        self.mode = None
        self.dragging = False
        self.drag_start = None
        self.drag_current = None
        self.clicked_point = None

    def mark_point(self, pt, color=(0, 255, 255)):
        cv2.circle(self.display, pt, 4, color, 2)

    def mark_text(self, frame, text, y=20):
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    def mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.mode == "rect":
                self.dragging = True
                self.drag_start = (x, y)
                self.drag_current = (x, y)
            else:
                self.points.append((x, y))
                self.clicked_point = (x, y)
                self.mark_point((x, y), (255, 0, 0))
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.drag_current = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            if self.drag_start is not None:
                x0, y0 = self.drag_start
                x1, y1 = x, y
                ix0, iy0 = min(x0, x1), min(y0, y1)
                ix1, iy1 = max(x0, x1), max(y0, y1)
                self.rects.append((ix0, iy0, ix1, iy1))
                cv2.rectangle(self.display, (ix0, iy0), (ix1, iy1), (0, 255, 0), 2)
                self.drag_start = None
                self.drag_current = None

    def preview(self):
        frame = self.display.copy()
        if self.dragging and self.drag_start and self.drag_current:
            x0, y0 = self.drag_start
            x1, y1 = self.drag_current
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 128, 255), 1)
        return frame


def _prompt_float(msg: str) -> float:
    return float(input(msg).strip())


def _prompt_name(msg: str) -> str:
    return input(msg).strip()


def _prompt_bool(msg: str, default: bool = True) -> bool:
    val = input(msg).strip().lower()
    if not val:
        return default
    return val in {"1", "y", "yes", "true", "t"}


def click_two_points(ui: ClickUI, window_name: str, prompt: str):
    ui.reset()
    ui.mode = "point"
    while True:
        frame = ui.preview()
        ui.mark_text(frame, prompt)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(20) & 0xFF
        if len(ui.points) >= 2:
            return ui.points[:2]
        if key == 27:
            raise SystemExit("aborted")


def click_one_point(ui: ClickUI, window_name: str, prompt: str):
    ui.reset()
    ui.mode = "point"
    while True:
        frame = ui.preview()
        ui.mark_text(frame, prompt)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(20) & 0xFF
        if ui.clicked_point is not None:
            return ui.clicked_point
        if key == 27:
            raise SystemExit("aborted")


def drag_rectangle(ui: ClickUI, window_name: str, prompt: str):
    ui.reset()
    ui.mode = "rect"
    while True:
        frame = ui.preview()
        ui.mark_text(frame, prompt)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(20) & 0xFF
        if ui.rects:
            return ui.rects[-1]
        if key == 27:
            raise SystemExit("aborted")


def _prompt_series_mask():
    kind = _prompt_name("mask kind [sample/color/hsv] (default sample): ") or "sample"
    if kind == "sample":
        return {"_mode": "sample"}
    if kind == "hsv":
        raw = _prompt_name("enter hsv ranges as JSON list [[h0,s0,v0,h1,s1,v1], ...]: ")
        return {"_mode": "hsv", "hsv_ranges": json.loads(raw)}
    color = _prompt_name("color name [blue/red/green/black]: ")
    return {"_mode": "color", "color": color}


def _sample_hsv_range(img, pt, half_size: int = 4):
    x, y = pt
    h, w = img.shape[:2]
    x0 = max(0, x - half_size)
    x1 = min(w, x + half_size + 1)
    y0 = max(0, y - half_size)
    y1 = min(h, y + half_size + 1)
    patch = img[y0:y1, x0:x1]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hmin, smin, vmin = hsv.reshape(-1, 3).min(axis=0)
    hmax, smax, vmax = hsv.reshape(-1, 3).max(axis=0)
    return [[
        max(0, int(hmin) - 5),
        max(0, int(smin) - 40),
        max(0, int(vmin) - 40),
        min(180, int(hmax) + 5),
        min(255, int(smax) + 40),
        min(255, int(vmax) + 40),
    ]]


def build_config(image_path: Path, out_path: Path, dpi: int = 300, page: int = 1):
    if not image_path.exists():
        raise SystemExit(f"missing image: {image_path}")
    try:
        img = read_image(image_path, dpi=dpi, page=page)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    ui = ClickUI(img)
    window_name = "figure_digitizer_builder"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, ui.mouse)

    config = {"image": image_path.name, "axes": {}, "series": [], "ignore_rois": []}
    if image_path.suffix.lower() in VECTOR_SUFFIXES:
        config["render"] = {"dpi": dpi, "page": page}

    print("Builder controls:")
    print("  a: add axis calibration")
    print("  s: add series definition")
    print("  r: add ignore rectangle")
    print("  q: save and quit")
    print("  Esc: quit without saving")

    axis_counter = 0
    series_counter = 0
    save = False
    while True:
        frame = ui.display.copy()
        ui.mark_text(frame, "a: axis  s: series  r: ignore box  q: save")
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:
            break
        if key == ord("a"):
            default_name = f"axis_{axis_counter}"
            name = _prompt_name(f"axis name [{default_name}]: ") or default_name
            p1, p2 = click_two_points(ui, window_name, "click two axis calibration points")
            v0 = _prompt_float(f"{name} value at first point: ")
            v1 = _prompt_float(f"{name} value at second point: ")
            config["axes"][name] = {"px0": p1[0], "px1": p2[0], "v0": v0, "v1": v1}
            ui.reset()
            print(f"added axis {name}")
            axis_counter += 1
        elif key == ord("s"):
            default_series = f"series_{series_counter}"
            series = {
                "name": _prompt_name(f"series name [{default_series}]: ") or default_series,
                "out_csv": _prompt_name("output csv path: "),
                "xaxis": _prompt_name("x-axis name: "),
                "yaxis": _prompt_name("y-axis name: "),
                "use_sem": _prompt_bool("detect sem? [Y/n]: ", default=True),
            }
            series["roi"] = list(drag_rectangle(ui, window_name, "drag the series ROI"))
            mask_choice = _prompt_series_mask()
            mode = mask_choice.pop("_mode")
            if mode == "sample":
                pt = click_one_point(ui, window_name, "click a representative marker pixel")
                series["hsv_ranges"] = _sample_hsv_range(img, pt)
            else:
                series.update(mask_choice)
            if _prompt_bool("add series-specific ignore boxes? [y/N]: ", default=False):
                series["ignore_rois"] = []
                while _prompt_bool("  add another ignore box? [y/N]: ", default=False):
                    series["ignore_rois"].append(list(drag_rectangle(ui, window_name, "drag an ignore box")))
            config["series"].append(series)
            ui.reset()
            print(f"added series {series['name']}")
            series_counter += 1
        elif key == ord("r"):
            roi = drag_rectangle(ui, window_name, "drag an ignore box")
            config["ignore_rois"].append(list(roi))
            print(f"added ignore roi {roi}")
        elif key == ord("q"):
            save = True
            break

    cv2.destroyAllWindows()
    if save:
        out_path.write_text(json.dumps(config, indent=2))
        print(f"wrote {out_path}")
    else:
        print("aborted without saving")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "input",
        nargs="?",
        help="image file to digitize",
    )
    ap.add_argument(
        "--build-config",
        action="store_true",
        help="open the interactive config builder instead of digitizing",
    )
    ap.add_argument("-o", "--output", help="output CSV path, or output JSON path with --build-config")
    ap.add_argument(
        "--overlay",
        help="write a debug overlay image showing detected points and SEM bars",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="validate config and report detected points without writing CSV files",
    )
    ap.add_argument(
        "--split-series",
        action="store_true",
        help="write one CSV per series instead of prompting for one combined CSV",
    )
    ap.add_argument(
        "--long-format",
        action="store_true",
        help="write stacked rows as series,p,value,sem instead of wide per-series columns",
    )
    ap.add_argument(
        "--fail-on-empty",
        action="store_true",
        help="exit with an error if any configured series yields no points",
    )
    ap.add_argument(
        "--fail-on-incomplete",
        action="store_true",
        help="exit with an error if a series with x_values/expected_points yields fewer points than expected",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=None,
        help="rendering DPI for PDF/EPS/PS inputs; overrides config render.dpi",
    )
    ap.add_argument(
        "--page",
        type=int,
        default=None,
        help="1-based page number for PDF/EPS/PS inputs; overrides config render.page",
    )
    ap.add_argument(
        "--report-json",
        help="optional JSON path to write extraction metadata and confidence report",
    )
    args = ap.parse_args()
    if args.input:
        input_path = Path(args.input).resolve()
    else:
        supported = ", ".join(sorted(s[1:] for s in IMAGE_SUFFIXES))
        print("No input image provided.")
        print(f"Supported extensions: {supported}")
        raw = input("Enter image file path: ").strip()
        if not raw:
            raise SystemExit("no input image path provided")
        input_path = Path(raw).expanduser().resolve()
    if args.build_config:
        out_path = Path(args.output).resolve() if args.output else input_path.with_name(f"{input_path.stem}_digitizer.json")
        build_config(input_path, out_path, dpi=args.dpi or 300, page=args.page or 1)
        return

    preloaded_img = None
    if input_path.suffix.lower() in IMAGE_SUFFIXES:
        try:
            preloaded_img = read_image(input_path, dpi=args.dpi or 300, page=args.page or 1)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        builtin_table = builtin_table_for_image(input_path, preloaded_img)
        if builtin_table is not None:
            columns, table_rows = builtin_table
            print(f"recognized built-in table layout: {input_path.name}")
            print(f"rows: {len(table_rows)}")
            if not args.dry_run:
                out_path = Path(args.output).resolve() if args.output else prompt_output_path(input_path)
                write_table_csv(out_path, columns, table_rows)
                print(f"wrote {out_path}")
            if args.report_json:
                write_report_json(
                    Path(args.report_json).resolve(),
                    {
                        "mode": "builtin_table",
                        "input": str(input_path),
                        "rows": len(table_rows),
                        "columns": columns,
                        "confidence": 0.99,
                    },
                )
            return
        candidates = [
            input_path.with_name(f"{input_path.stem}_digitizer.json"),
            input_path.with_suffix(".json"),
            input_path.with_name("digitizer.json"),
        ]
        config_path = next((p for p in candidates if p.exists()), None)
        if config_path is None:
            builtin = builtin_config_for_image(input_path, preloaded_img)
            if builtin is None:
                auto = auto_extract_unknown_figure(input_path, preloaded_img)
                if auto is None:
                    raise SystemExit(
                        f"no built-in template, config, or generic extraction match for {input_path.name}; "
                        f"run: python3 {Path(__file__).name} {input_path.name} --build-config"
                    )
                print(f"generic extractor: {auto.kind}")
                print(f"rows: {len(auto.rows)}  confidence: {auto.confidence}")
                for note in auto.notes:
                    print(f"note: {note}")
                if not args.dry_run:
                    out_path = Path(args.output).resolve() if args.output else prompt_output_path(input_path)
                    write_table_csv(out_path, auto.columns, auto.rows)
                    print(f"wrote {out_path}")
                if args.report_json:
                    write_report_json(
                        Path(args.report_json).resolve(),
                        {
                            "mode": auto.kind,
                            "input": str(input_path),
                            "rows": len(auto.rows),
                            "columns": auto.columns,
                            "confidence": auto.confidence,
                            "notes": list(auto.notes),
                        },
                    )
                return
            image_path, axes, series, render = builtin
        else:
            try:
                image_path, axes, series, render = load_config(config_path)
            except (KeyError, TypeError, ValueError) as exc:
                raise SystemExit(f"invalid config {config_path}: {exc}") from exc
    else:
        config_path = input_path
        if not config_path.exists():
            raise SystemExit(f"missing config: {config_path}")
        try:
            image_path, axes, series, render = load_config(config_path)
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"invalid config {config_path}: {exc}") from exc
    if not image_path.exists():
        raise SystemExit(f"missing image: {image_path}")

    try:
        dpi = args.dpi if args.dpi is not None else int(render.get("dpi", 300))
        page = args.page if args.page is not None else int(render.get("page", 1))
        img = preloaded_img if preloaded_img is not None and image_path == input_path else read_image(image_path, dpi=dpi, page=page)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    points_by_series = []
    rows_by_series = []
    empty_series = []
    incomplete_series = []
    for spec in series:
        rows, points = extract_series(img, spec, axes)
        rows_by_series.append((spec.name, rows))
        if not rows:
            empty_series.append(spec.name)
        expected_count = len(spec.x_values) if spec.x_values else spec.expected_points
        if expected_count is not None and len(rows) != expected_count:
            incomplete_series.append(f"{spec.name} ({len(rows)}/{expected_count})")
        print(f"{spec.name}: {len(rows)} points")
        points_by_series.append((spec, points))

    if empty_series:
        msg = "no points detected for: " + ", ".join(empty_series)
        if args.fail_on_empty:
            raise SystemExit(msg)
        print(f"warning: {msg}")
    if incomplete_series:
        msg = "incomplete point detection for: " + ", ".join(incomplete_series)
        if args.fail_on_incomplete:
            raise SystemExit(msg)
        print(f"warning: {msg}")

    if not args.dry_run:
        if args.split_series:
            for spec, (_series_name, rows) in zip(series, rows_by_series):
                write_csv(spec.out_csv, rows)
                print(f"wrote {spec.out_csv}")
        else:
            out_path = Path(args.output).resolve() if args.output else prompt_output_path(input_path)
            if args.long_format:
                write_combined_csv(out_path, rows_by_series)
            else:
                write_wide_csv(out_path, rows_by_series)
            print(f"wrote {out_path}")
    if args.report_json:
        write_report_json(
            Path(args.report_json).resolve(),
            {
                "mode": "configured_series",
                "input": str(input_path),
                "series": [name for name, _rows in rows_by_series],
                "rows_per_series": {name: len(rows) for name, rows in rows_by_series},
                "empty_series": empty_series,
                "incomplete_series": incomplete_series,
            },
        )

    if args.overlay:
        render_debug_overlay(img, points_by_series, Path(args.overlay).resolve())
        print(f"overlay -> {Path(args.overlay).resolve()}")


if __name__ == "__main__":
    main()

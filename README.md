# KplotDigitiser Usage

## Run

```bash
python3 KplotDigitiser.py image.ext
```

Example:

```bash
python3 KplotDigitiser.py Figure1_strain.pdf
```

The script will ask for output CSV file name (unless `-o/--output` is provided).

## Run Without Input Argument

```bash
python3 KplotDigitiser.py
```

It will:
1. Show supported extensions.
2. Ask for image file path.
3. Continue normal extraction flow.

## Supported Input Formats

- Raster: `png, jpg, jpeg, tif, tiff, bmp, webp, gif, jp2, ppm, pgm, pbm, pnm`
- Vector: `pdf, eps, ps`

For vector inputs, image rendering is done internally (DPI/page configurable).

## Common Options

- `-o, --output PATH`  
  Output CSV path.

- `--dry-run`  
  Detect and report points without writing CSV.

- `--report-json PATH`  
  Save extraction metadata report as JSON.

- `--long-format`  
  Write combined rows as `series,p,value,sem`.

- `--split-series`  
  Write one CSV per configured series (template/config mode).

- `--dpi N`  
  Rendering DPI for `pdf/eps/ps` (default 300 if not overridden by config).

- `--page N`  
  1-based page for `pdf/eps/ps` (default 1).

## Build Config (Optional Advanced Mode)

```bash
python3 KplotDigitiser.py image.ext --build-config
```

This opens interactive calibration/ROI setup and writes a JSON config.

## Output Behavior

- Known built-in figures: uses built-in templates/tables directly.
- Unknown figures: uses generic auto-extraction.
- Multi-series scatter: outputs separate columns per detected series in generic mode.

## Quick Examples

```bash
python3 KplotDigitiser.py medium.png
python3 KplotDigitiser.py Figure_atlas_density.pdf --dry-run --report-json density_report.json
python3 KplotDigitiser.py Figure_dft_validation.pdf -o dft_points.csv --report-json dft_report.json
```

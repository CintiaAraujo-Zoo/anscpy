# anscpy — Animal Science Python Toolkit

A Python library for quantitative analysis in animal science and livestock research.
Designed for researchers with basic Python knowledge.

> **Current status:** Early development. Only `anscpy.gas` is currently available.
> Additional modules are planned but have no release date yet.

---

## Installation

```bash
pip install anscpy
```

*Requires Python 3.9+. NumPy and SciPy are installed automatically.*
*Pandas and Matplotlib are optional but recommended.*

---

## Available now — `anscpy.gas`

Fits mathematical models to cumulative in vitro gas production profiles.
Compatible with semi-automatic (pressure transducer) and automatic techniques.

### Minimal example (data in mL)

Fits mathematical models to cumulative in vitro gas production profiles.
Compatible with semi-automatic and automatic techniques.

> **Note on units:** `anscpy.gas` works exclusively with volumes in **mL**.
> If your equipment records data in PSI or other pressure units, convert to mL
> before using any function. Each laboratory should derive its own calibration
> equation experimentally.

### Minimal example

```python
from anscpy.gas import fit_gas_production

time   = [0, 2, 4, 6, 8, 12, 24, 48, 72, 96]
volume = [0, 10.3, 19.9, 28.9, 38.9, 61.9, 112.6, 140.1, 145.3, 149.1]

result = fit_gas_production(time, volume)
result.summary()
result.plot()
```

### With blank correction (PSI input, semi-automatic technique)

```python
# Convert your data to mL first, then:
result = fit_gas_production(
    time   = time,
    volume = volume_ml,
    blank  = blank_ml,
)
result.summary()
result.plot()
```

**Available models:** LE0 (default), LEL, MM, MIT, EXPL, GOM, LOG

---

## Planned modules

These modules are part of the project roadmap but are not yet implemented.
No estimated release dates.

| Module | Description |
|--------|-------------|
| `anscpy.deg` | Ruminal degradation kinetics (in situ / nylon bag) |
| `anscpy.growth` | Animal growth modeling and weight gain curves |
| `anscpy.feed` | Feed efficiency analysis (RFI, FCR) |
| `anscpy.forage` | Forage quality assessment |
| `anscpy.silage` | Silage fermentation kinetics |
| `anscpy.field` | Field decision support tools |
| `anscpy.ration` | Ration formulation algorithms |

---

## License

MIT — see `LICENSE` for details.

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
result = fit_gas_production(
    time       = time,
    volume     = sample_psi,
    input_unit = 'psi',
    blank      = blanks_psi,
    blank_unit = 'psi',
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

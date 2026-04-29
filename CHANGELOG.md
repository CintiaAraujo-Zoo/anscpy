# Changelog

## [Unreleased]

### Changed
- `anscpy.gas` now accepts only volumes in **mL**. PSI and other pressure
  units are no longer accepted by any function in this module.

### Removed
- `correct_blank()`: removed parameters `blank_unit`, `sample_unit`,
  `slope_psi`, `intercept_psi`.
- `fit_gas_production()`: removed parameters `input_unit`, `blank_unit`,
  `slope_psi`, `intercept_psi`.
- Internal constants `PSI_SLOPE` and `PSI_INTERCEPT`.
- Internal function `_to_ml()`.

### Migration guide
If you were passing PSI data to `fit_gas_production()` or `correct_blank()`,
convert your data to mL before calling these functions:

```python
# Before (no longer works)
result = fit_gas_production(time, readings, input_unit='psi')

# After — convert first using your lab's calibration equation
volume_ml = slope * readings_psi + intercept
result = fit_gas_production(time, volume_ml)
```

Each laboratory should derive its own PSI → mL calibration equation
experimentally. See: Maurício et al. (1999), Pereira et al. (2009).

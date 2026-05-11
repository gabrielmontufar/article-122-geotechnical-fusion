# Shared latent-field benchmark for article 122

This folder contains a fully reproducible synthetic benchmark for a manuscript on Bayesian estimation of geotechnical correlation lengths by geotechnical-geophysical fusion.

Run:

```powershell
python benchmark_latent_fusion.py
```

The script generates random fields with known correlation lengths, sparse direct CPT/SPT-like observations, dense geophysical observations and four baseline comparators plus the proposed shared latent-field fusion model. Figures and table images are generated from the CSV outputs.

Primary true effective correlation length: 3.987 m.
Fusion median estimate: 5.503 m.
Direct-only median estimate: 8.029 m.
90% interval width reduction versus direct-only: 52.8%.
Holdout RMSE reduction versus direct-only: 8.3%.

# Data directory

Large generated/downloaded artifacts are ignored by Git.

For the primary real-data version of ClaimsIQ, run:

```powershell
python scripts/bootstrap_real.py
```

This downloads the public French MTPL2 frequency/severity datasets through OpenML and creates:

- `raw/freMTPL2freq.csv`
- `raw/freMTPL2sev.csv`
- `mtpl2_policy.csv`
- `claims_real.db`

Source: CASdatasets / OpenML. The CASdatasets documentation identifies the underlying source as an unknown private insurer.

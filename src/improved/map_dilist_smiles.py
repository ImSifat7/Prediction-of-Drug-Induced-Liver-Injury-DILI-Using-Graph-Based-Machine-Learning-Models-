"""Map the official FDA DILIst supplementary table (drug names) -> SMILES.

Strategy: (1) reuse SMILES we already have (data/dili_with_smiles.csv, name match),
(2) resolve the remainder via PubChem PUG-REST name->CanonicalSMILES.
Writes data/external/dilist_official_1279.csv  (CompoundName, label, smiles).
"""
from __future__ import annotations
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "data" / "external" / "dilist_official_1279.csv"


def pubchem_smiles(name):
    url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
           + urllib.parse.quote(name)
           + "/property/CanonicalSMILES/TXT")
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            s = r.read().decode().strip().splitlines()
            return s[0].strip() if s else None
    except Exception:
        return None


def main():
    dilist = pd.read_excel(BASE / "DILIst-Supplementary-Table.xlsx", header=0)
    dilist.columns = [str(c).strip() for c in dilist.columns]
    lab_col = [c for c in dilist.columns if "Classification" in c][0]
    dilist["name_l"] = dilist["CompoundName"].astype(str).str.strip().str.lower()

    local = pd.read_csv(BASE / "data" / "dili_with_smiles.csv")
    local["name_l"] = local["drug_name"].astype(str).str.strip().str.lower()
    local_map = dict(zip(local["name_l"], local["smiles"]))

    rows, n_local, n_pub, n_fail = [], 0, 0, 0
    for i, r in dilist.iterrows():
        nm, nl, lab = r["CompoundName"], r["name_l"], int(r[lab_col])
        smi = local_map.get(nl)
        if smi:
            n_local += 1
        else:
            smi = pubchem_smiles(str(nm))
            time.sleep(0.2)
            if smi:
                n_pub += 1
            else:
                n_fail += 1
        if smi:
            rows.append({"CompoundName": nm, "label": lab, "smiles": smi})
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(dilist)}  local={n_local} pubchem={n_pub} fail={n_fail}", flush=True)

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nDONE: {len(out)}/{len(dilist)} resolved "
          f"(local={n_local}, pubchem={n_pub}, failed={n_fail})", flush=True)
    print(f"label balance: {out['label'].value_counts().to_dict()}", flush=True)
    print(f"saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()

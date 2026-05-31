"""
Compound featurization for perturbation-response modeling.
Portable, reusable. Morgan FP path matches DrugSynthAI S03/S06.
"""
from __future__ import annotations
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    _RDKIT = True
except Exception:
    _RDKIT = False


def morgan_fp(smiles: str, n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    if not _RDKIT or not smiles:
        return np.zeros(n_bits, dtype=np.float32)
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def featurize_ids(ids, smiles_map: dict, n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    return np.vstack([morgan_fp(smiles_map.get(i, ""), n_bits, radius) for i in ids])


def cosine_sim(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return Bn @ An.T


def coverage_report(ids, smiles_map: dict) -> dict:
    total = len(ids)
    missing = sum(1 for i in ids if not smiles_map.get(i))
    invalid = 0
    if _RDKIT:
        for i in ids:
            s = smiles_map.get(i, "")
            if s and Chem.MolFromSmiles(str(s)) is None:
                invalid += 1
    return {"total": total, "missing_smiles": missing,
            "invalid_smiles": invalid, "resolved": total - missing - invalid}

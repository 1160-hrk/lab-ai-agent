"""matplotlib でスペクトルを描くツール関数"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List

def plot_spectrum(x: List[float], y: List[float], out: Path = Path("spectrum.png")) -> str:
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Intensity (a.u.)")
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    return str(out)

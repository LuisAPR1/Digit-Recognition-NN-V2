#!/usr/bin/env python3
"""
Descarrega e extrai automaticamente o dataset MNIST em formato IDX.
Os ficheiros finais são colocados na pasta data/.
"""

from __future__ import annotations

import gzip
import shutil
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

MNIST_FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

MIRRORS = [
    "https://yann.lecun.com/exdb/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]

DATA_DIR = Path("data")


def download_file(name: str) -> Path:
    target = DATA_DIR / name
    if target.exists():
        print(f"[skip] {name} já existe.")
        return target

    last_error: Exception | None = None
    for base_url in MIRRORS:
        url = base_url + name
        print(f"[download] {name} a partir de {base_url} ...")
        tmp_target = target.with_suffix(target.suffix + ".tmp")
        try:
            urlretrieve(url, tmp_target)
            tmp_target.rename(target)
            return target
        except (URLError, OSError) as exc:
            last_error = exc
            print(f"  -> falhou: {exc}")
            if tmp_target.exists():
                tmp_target.unlink()

    raise RuntimeError(f"Não foi possível descarregar {name}. Último erro: {last_error}")


def extract_gzip(gz_path: Path) -> Path:
    output_path = gz_path.with_suffix("")  # remove .gz
    if output_path.exists():
        print(f"[skip] {output_path.name} já extraído.")
        return output_path

    print(f"[extract] {gz_path.name} -> {output_path.name}")
    with gzip.open(gz_path, "rb") as src, open(output_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return output_path


def main() -> int:
    DATA_DIR.mkdir(exist_ok=True)
    for filename in MNIST_FILES:
        gz_path = download_file(filename)
        extract_gzip(gz_path)
    print("\n✓ MNIST disponível em data/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

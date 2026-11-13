"""Загрузка датасета THULab/SMAP с Hugging Face."""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from pathlib import Path


REPOSITORY_URL = "https://huggingface.co/datasets/THULab/SMAP"
RESOLVE_URL = f"{REPOSITORY_URL}/resolve/main"
DATA_FILES = ("SMAP_train.tsfile", "SMAP_test.tsfile")


def _download(url: str, destination: Path) -> None:
    """Скачать файл атомарно и повторить запрос при сетевом сбое."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url, headers={"User-Agent": "smap-lstm-course-project/2.0"}
    )
    last_error: Exception | None = None

    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open(
                "wb"
            ) as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            # Первые шесть байт любого корректного Apache TsFile — `TsFile`.
            if temporary.read_bytes()[:6] != b"TsFile":
                raise ValueError(f"{destination.name} не является Apache TsFile")
            temporary.replace(destination)
            return
        except (OSError, urllib.error.URLError, ValueError) as error:
            last_error = error
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            if attempt < 3:
                print(f"Попытка {attempt} не удалась ({error}); повторяем...")
                time.sleep(2**attempt)
    raise RuntimeError(f"Не удалось скачать {url} после 3 попыток") from last_error


def download_dataset(data_dir: str | Path = "data/raw", force: bool = False) -> Path:
    """Получить два официальных split-файла THULab/SMAP.

    По умолчанию путь относительный, поэтому репозиторий можно перемещать без
    изменения исходного кода.
    """
    data_dir = Path(data_dir)
    for filename in DATA_FILES:
        destination = data_dir / filename
        if force or not destination.exists():
            print(f"Загрузка {filename}...")
            _download(f"{RESOLVE_URL}/{filename}", destination)
        elif destination.read_bytes()[:6] != b"TsFile":
            raise ValueError(f"Повреждён локальный файл: {destination}")

    print(f"Датасет THULab/SMAP готов: {data_dir.resolve()}")
    return data_dir


def dataset_paths(data_dir: str | Path = "data/raw") -> tuple[Path, Path]:
    """Вернуть пути train/test или понятную ошибку до чтения данных."""
    root = Path(data_dir)
    train_path = root / "SMAP_train.tsfile"
    test_path = root / "SMAP_test.tsfile"
    missing = [str(path) for path in (train_path, test_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Не найдены файлы: {missing}. Запустите scripts/download_data.py."
        )
    return train_path, test_path


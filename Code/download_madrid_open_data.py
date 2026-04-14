from __future__ import annotations

import io
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import urlopen

from bs4 import BeautifulSoup


BASE = "https://datos.madrid.es"
ROOT = Path(r"d:\5_Article_Rapit_Manuel\Code\Datos\datos")

MONTHS = {
    "Enero": "ene",
    "Febrero": "feb",
    "Marzo": "mar",
    "Abril": "abr",
    "Mayo": "may",
    "Junio": "jun",
    "Julio": "jul",
    "Agosto": "ago",
    "Septiembre": "sep",
    "Octubre": "oct",
    "Noviembre": "nov",
    "Diciembre": "dic",
}


def normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def fetch_download_cards(page_url: str) -> list[tuple[str, str]]:
    html = urlopen(page_url, timeout=60).read().decode("utf-8", "ignore")
    soup = BeautifulSoup(html, "html.parser")
    cards: list[tuple[str, str]] = []
    for a in soup.select('a.resource-url-analytics[href*="/download/"]'):
        parent = a
        for _ in range(4):
            parent = parent.parent
        text = " ".join(parent.get_text(" ", strip=True).split())
        href = a.get("href")
        if href:
            cards.append((text, urljoin(BASE, href)))
    return cards


def download_bytes(url: str) -> bytes:
    with urlopen(url, timeout=120) as response:
        return response.read()


def save_bytes(target: Path, payload: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as fh:
        fh.write(payload)


def extract_single_csv_from_zip(payload: bytes, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise RuntimeError(f"Expected one CSV in zip for {target.name}, found: {csv_names}")
        with zf.open(csv_names[0]) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)


def extract_air_quality_zip(payload: bytes, year: int) -> None:
    short = str(year)[-2:]
    outdir = ROOT / "contaminacion" / f"anio{short}"
    outdir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        for name in zf.namelist():
            if name.lower().endswith(".csv"):
                target = outdir / Path(name).name
                with zf.open(name) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)


def build_index(cards: list[tuple[str, str]]) -> dict[str, str]:
    return {text: url for text, url in cards}


def find_first(cards: list[tuple[str, str]], predicate) -> tuple[str, str]:
    for text, url in cards:
        if predicate(text):
            return text, url
    raise KeyError("No download card matched the requested predicate")


def download_air_quality() -> None:
    cards = fetch_download_cards("https://datos.madrid.es/dataset/201200-0-calidad-aire-horario/downloads")
    for year in (2021, 2022, 2023):
        outdir = ROOT / "contaminacion" / f"anio{str(year)[-2:]}"
        if outdir.exists() and len(list(outdir.glob("*.csv"))) >= 12:
            print(f"[air] {year}: already present")
            continue
        text, url = find_first(
            cards,
            lambda t, y=year: normalized(f"ZIP Calidad del aire. Datos horarios desde 2001. {y}") in normalized(t),
        )
        print(f"[air] {year}: {url}")
        payload = download_bytes(url)
        extract_air_quality_zip(payload, year)


def download_meteo() -> None:
    cards = fetch_download_cards("https://datos.madrid.es/dataset/300352-0-meteorologicos-horarios/downloads")
    for year in (2021, 2022, 2023):
        yy = str(year)[-2:]
        for month_name, month_abbr in MONTHS.items():
            target = ROOT / "meteo" / str(year) / f"{month_abbr}_meteo{yy}.csv"
            if target.exists() and target.stat().st_size > 0:
                print(f"[meteo] {year} {month_name}: already present")
                continue
            text, url = find_first(
                cards,
                lambda t, y=year, m=month_name: normalized(
                    f"CSV Datos meteorológicos. Datos horarios desde 2019. {y}. {m}"
                ) in normalized(t),
            )
            print(f"[meteo] {year} {month_name}: {url}")
            save_bytes(target, download_bytes(url))


def download_traffic_measurements() -> None:
    cards = fetch_download_cards("https://datos.madrid.es/dataset/208627-0-transporte-ptomedida-historico/downloads")
    for year in (2021, 2022, 2023):
        for month_num, month_name in enumerate(MONTHS.keys(), start=1):
            target = ROOT / "trafico" / "mediciones" / str(year) / f"{month_num:02d}-{year}.csv"
            if target.exists() and target.stat().st_size > 0:
                print(f"[traffic] {year} {month_name}: already present")
                continue
            text, url = find_first(
                cards,
                lambda t, y=year, m=month_name: all(
                    token in normalized(t)
                    for token in ("zip", normalized(m), str(y), "tráfico", "histórico")
                ),
            )
            print(f"[traffic] {year} {month_name}: {url}")
            extract_single_csv_from_zip(download_bytes(url), target)


def download_traffic_locations() -> None:
    cards = fetch_download_cards("https://datos.madrid.es/dataset/202468-0-intensidad-trafico/downloads")
    required = [
        (2021, "31/01/2021"),
        (2021, "31/03/2021"),
        (2021, "30/06/2021"),
        (2021, "30/09/2021"),
        (2021, "31/12/2021"),
        (2022, "31/03/2022"),
        (2022, "30/06/2022"),
        (2022, "30/09/2022"),
        (2022, "31/12/2022"),
        (2023, "31/03/2023"),
        (2023, "30/06/2023"),
        (2023, "30/09/2023"),
        (2023, "31/12/2023"),
    ]
    for year, date_text in required:
        mm, yyyy = date_text[3:5], date_text[-4:]
        target = ROOT / "trafico" / "ubicaciones" / f"pmed_ubicacion_{mm}-{yyyy}.csv"
        if target.exists() and target.stat().st_size > 0:
            print(f"[location] {date_text}: already present")
            continue
        text, url = find_first(
            cards,
            lambda t, y=year, d=date_text: all(
                token in normalized(t)
                for token in ("csv", "ubicación", "puntos de medida", str(y), normalized(d))
            ),
        )
        print(f"[location] {date_text}: {url}")
        save_bytes(target, download_bytes(url))


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    download_air_quality()
    download_meteo()
    download_traffic_measurements()
    download_traffic_locations()
    print("Download completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

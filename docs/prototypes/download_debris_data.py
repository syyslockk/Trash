import requests
import pandas as pd
import numpy as np

# --- Константы ---
GM_EARTH = 398600.4418     # км^3/с^2, гравитационный параметр Земли
R_EARTH = 6378.137         # км, экваториальный радиус Земли (WGS84)
SECONDS_PER_DAY = 86400

GROUPS = {
    "active": "active_satellite",
    "cosmos-1408-debris": "cosmos_1408_debris",
    "fengyun-1c-debris": "fengyun_1c_debris",
    "iridium-33-debris": "iridium_33_debris",
    "cosmos-2251-debris": "cosmos_2251_debris",
}

BASE_URL = "https://celestrak.org/NORAD/elements/gp.php"

# Стандартные поля OMM JSON от CelesTrak, которые нам нужны
REQUIRED_FIELDS = [
    "OBJECT_NAME", "OBJECT_ID", "NORAD_CAT_ID", "EPOCH",
    "MEAN_MOTION", "ECCENTRICITY", "INCLINATION", "RA_OF_ASC_NODE",
]


def fetch_group(group: str) -> pd.DataFrame:
    """Скачивает одну группу с CelesTrak в формате OMM/JSON."""
    params = {"GROUP": group, "FORMAT": "json"}
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    if df.empty:
        print(f"[!] Группа '{group}' вернула 0 объектов")
        return df

    missing = [f for f in REQUIRED_FIELDS if f not in df.columns]
    if missing:
        raise ValueError(
            f"В ответе для группы '{group}' нет полей {missing}. "
            f"Похоже, CelesTrak изменил формат — проверьте вручную: "
            f"{BASE_URL}?GROUP={group}&FORMAT=json"
        )
    return df


def add_orbital_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Считает большую полуось и высоты перигея/апогея из mean motion
    и эксцентриситета (третий закон Кеплера).
    """
    df = df.copy()

    # mean motion приходит в оборотах/сутки -> переводим в рад/с
    n_rad_s = df["MEAN_MOTION"].astype(float) * 2 * np.pi / SECONDS_PER_DAY

    # третий закон Кеплера: n^2 * a^3 = GM  =>  a = (GM / n^2)^(1/3)
    df["semi_major_axis_km"] = (GM_EARTH / n_rad_s**2) ** (1 / 3)

    ecc = df["ECCENTRICITY"].astype(float)
    df["perigee_alt_km"] = df["semi_major_axis_km"] * (1 - ecc) - R_EARTH
    df["apogee_alt_km"] = df["semi_major_axis_km"] * (1 + ecc) - R_EARTH
    df["mean_alt_km"] = (df["perigee_alt_km"] + df["apogee_alt_km"]) / 2

    df["inclination_deg"] = df["INCLINATION"].astype(float)
    df["raan_deg"] = df["RA_OF_ASC_NODE"].astype(float)
    df["eccentricity"] = ecc

    return df


def main():
    frames = []
    for group, category in GROUPS.items():
        print(f"Скачиваю группу: {group} ...")
        df = fetch_group(group)
        if df.empty:
            continue
        df["category"] = category
        df = add_orbital_features(df)
        frames.append(df)

    if not frames:
        print("Не удалось скачать ни одной группы.")
        return

    full = pd.concat(frames, ignore_index=True)

    cols = [
        "OBJECT_NAME", "OBJECT_ID", "NORAD_CAT_ID", "EPOCH", "category",
        "mean_alt_km", "perigee_alt_km", "apogee_alt_km",
        "inclination_deg", "eccentricity", "raan_deg",
    ]
    result = full[cols].rename(columns={
        "OBJECT_NAME": "object_name",
        "OBJECT_ID": "object_id",
        "NORAD_CAT_ID": "norad_id",
        "EPOCH": "epoch",
    })

    # Фильтр на LEO (примерно 160-2000 км)
    leo = result[
        (result["mean_alt_km"] >= 160) & (result["mean_alt_km"] <= 2000)
    ].reset_index(drop=True)

    print(f"\nВсего объектов скачано: {len(result)}")
    print(f"Из них в диапазоне LEO (160-2000 км): {len(leo)}")
    print("\nПо категориям (в LEO):")
    print(leo["category"].value_counts())

    result.to_csv("orbital_dataset_full.csv", index=False)
    leo.to_csv("orbital_dataset_leo.csv", index=False)
    print("\nСохранено: orbital_dataset_full.csv, orbital_dataset_leo.csv")


if __name__ == "__main__":
    main()
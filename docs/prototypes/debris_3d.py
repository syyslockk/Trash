
import argparse
import json
import os
import webbrowser

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from scipy.ndimage import gaussian_filter
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# Константы
# ----------------------------------------------------------------------------
GM_EARTH = 398600.4418   # км^3/с^2
R_EARTH = 6378.137       # км

# Сетка для расчёта риска: высота 160-2000 км (шаг 20), наклонение 0-180 град (шаг 2)
ALT_MIN, ALT_MAX, ALT_STEP = 160, 2000, 20
INC_MIN, INC_MAX, INC_STEP = 0, 180, 2

SIGMA_KM2 = 1e-5         # условное сечение столкновения (на относительный индекс не влияет)
N_NODE_STEPS = 72        # шагов усреднения по разности долгот узлов
SMOOTH_SIGMA = 1.0       # сглаживание плотности мусора (в клетках сетки)
SHELL_THRESHOLD = 0.25   # высотные слои с риском выше 25% от максимума рисуем как оболочки
MAX_SHELLS = 5

ACTIVE_NAME = "active_satellite"   # метка действующих спутников в колонке category

SHORT_NAMES = {
    "active_satellite": "спутники",
    "fengyun_1c_debris": "обломки Fengyun-1C",
    "iridium_33_debris": "обломки Iridium-33",
    "cosmos_2251_debris": "обломки Cosmos-2251",
    "cosmos_1408_debris": "обломки Cosmos-1408",
}
GROUP_NAMES = {True: "Действующие спутники", False: "Мусор (обломки)"}

COAST_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
             "master/geojson/ne_110m_coastline.geojson")
COAST_CACHE = "ne_110m_coastline.geojson"

# ----------------------------------------------------------------------------
# Оформление (тёмная тема: фон - космос)
# ----------------------------------------------------------------------------
SURFACE = "#1a1a19"
INK_PRIMARY = "#ffffff"
INK_SECONDARY = "#c3c2b7"
INK_MUTED = "#898781"
BTN_BG = "#2c2c2a"
BTN_BORDER = "#383835"
EARTH_COLOR = "#1d2f45"
GRATICULE_COLOR = "#2e4560"
COAST_COLOR = "#7d8a99"
SHELL_COLOR = "#9ec5f4"

# Категориальные цвета в фиксированном порядке (тёмные шаги палитры)
CAT_COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500",
              "#d55181", "#008300", "#9085e9", "#e66767"]
COLOR_OTHER = INK_MUTED
COLOR_NOISE = "#52514e"
# Шкала опасности: тёмный = низкая, светлый = высокая (на тёмном фоне ярче = заметнее)
RISK_SCALE = [[0.0, "#184f95"], [0.25, "#256abf"], [0.5, "#3987e5"],
              [0.75, "#9ec5f4"], [1.0, "#cde2fb"]]


# ----------------------------------------------------------------------------
# 1. Загрузка данных
# ----------------------------------------------------------------------------
def load_data(path):
    df = pd.read_csv(path)
    need = ["category", "mean_alt_km", "inclination_deg"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"В файле {path} нет колонок {missing}. "
                         "Проверьте, что файл создан скриптом загрузки данных.")

    n_all = len(df)
    if "norad_id" in df.columns:
        df = df.drop_duplicates(subset="norad_id", keep="first")
    df = df[(df["mean_alt_km"] >= ALT_MIN) & (df["mean_alt_km"] < ALT_MAX)
            & (df["inclination_deg"] >= INC_MIN) & (df["inclination_deg"] < INC_MAX)]
    df = df.reset_index(drop=True)

    if "object_name" not in df.columns:
        df["object_name"] = [f"OBJ-{i}" for i in range(len(df))]
    if "raan_deg" not in df.columns:
        print("[!] В файле нет raan_deg - долгота узла будет случайной")
        df["raan_deg"] = np.random.default_rng(0).uniform(0, 360, len(df))

    is_active = df["category"] == ACTIVE_NAME
    print(f"Загружено строк: {n_all}, после очистки и фильтра по LEO: {len(df)}")
    print(f"  действующих спутников: {int(is_active.sum())}, "
          f"объектов-обломков: {int((~is_active).sum())}")
    if is_active.sum() == 0 or (~is_active).sum() == 0:
        raise SystemExit("Нужны объекты обоих типов: active_satellite и *_debris")
    return df


# ----------------------------------------------------------------------------
# 2. Кластеризация
# ----------------------------------------------------------------------------
def run_clustering(df, min_cluster_size, min_samples):
    X = StandardScaler().fit_transform(df[["mean_alt_km", "inclination_deg"]].to_numpy())
    try:
        import hdbscan
        model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    except ImportError:
        from sklearn.cluster import HDBSCAN
        model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = model.fit_predict(X)

    mask = labels != -1
    n_clusters = len(set(labels[mask]))
    n_noise = int((~mask).sum())
    print(f"HDBSCAN: кластеров {n_clusters}, шум {n_noise} ({n_noise / len(labels):.1%})")
    if n_clusters > 1:
        sil = silhouette_score(X[mask], labels[mask],
                               sample_size=min(10000, int(mask.sum())), random_state=0)
        dbi = davies_bouldin_score(X[mask], labels[mask])
        print(f"  Silhouette: {sil:.3f} (ближе к 1 - лучше), "
              f"Davies-Bouldin: {dbi:.3f} (ближе к 0 - лучше)")
    return labels


def validate_against_known_events(df):
    """Какая доля обломков каждого известного события попала в один кластер."""
    print("Проверка по известным событиям (доля обломков в одном кластере):")
    for cat, g in df[df["category"] != ACTIVE_NAME].groupby("category"):
        counts = g["cluster"].value_counts()
        top_label = counts.index[0]
        where = "шум" if top_label == -1 else f"кластер {top_label}"
        name = SHORT_NAMES.get(cat, cat.replace("_", " "))
        print(f"  {name}: {counts.iloc[0] / len(g):.0%} ({where}), объектов: {len(g)}")


# ----------------------------------------------------------------------------
# 3. Расчёт риска
# ----------------------------------------------------------------------------
def make_grid():
    alt_edges = np.arange(ALT_MIN, ALT_MAX + ALT_STEP, ALT_STEP)
    inc_edges = np.arange(INC_MIN, INC_MAX + INC_STEP, INC_STEP)
    alt_c = (alt_edges[:-1] + alt_edges[1:]) / 2
    inc_c = (inc_edges[:-1] + inc_edges[1:]) / 2
    return alt_edges, inc_edges, alt_c, inc_c


def vrel_factor_matrix(inc_c):
    """Средний множитель 2*sin(D/2) для каждой пары наклонений (усреднение по dRAAN)."""
    i = np.radians(inc_c)
    i1, i2 = i[:, None, None], i[None, :, None]
    d_raan = np.linspace(0, 2 * np.pi, N_NODE_STEPS, endpoint=False)[None, None, :]
    cos_d = np.clip(np.cos(i1) * np.cos(i2) + np.sin(i1) * np.sin(i2) * np.cos(d_raan), -1, 1)
    return np.sqrt(2.0 * (1.0 - cos_d)).mean(axis=2)


def compute_risk(df):
    alt_edges, inc_edges, alt_c, inc_c = make_grid()
    is_active = df["category"] == ACTIVE_NAME

    def counts(sub):
        h, _, _ = np.histogram2d(sub["mean_alt_km"], sub["inclination_deg"],
                                 bins=[alt_edges, inc_edges])
        return h

    n_active, n_debris = counts(df[is_active]), counts(df[~is_active])

    v_circ = np.sqrt(GM_EARTH / (R_EARTH + alt_c))
    volume = 4 * np.pi * (R_EARTH + alt_c) ** 2 * ALT_STEP
    dens_deb = gaussian_filter(n_debris, sigma=SMOOTH_SIGMA, mode="constant") / volume[:, None]

    hazard = (dens_deb @ vrel_factor_matrix(inc_c).T) * v_circ[:, None] * SIGMA_KM2
    exposure = hazard * n_active
    hazard_rel = hazard / hazard.max() if hazard.max() > 0 else hazard
    exposure_rel = exposure / exposure.max() if exposure.max() > 0 else exposure

    # опасность в клетке, где находится каждый объект
    a_idx = np.clip(np.digitize(df["mean_alt_km"], alt_edges) - 1, 0, hazard_rel.shape[0] - 1)
    i_idx = np.clip(np.digitize(df["inclination_deg"], inc_edges) - 1, 0, hazard_rel.shape[1] - 1)
    df["hazard_rel"] = hazard_rel[a_idx, i_idx]

    # таблица самых опасных клеток
    aa, ii = np.meshgrid(alt_c, inc_c, indexing="ij")
    grid = pd.DataFrame({"alt_center_km": aa.ravel(), "inc_center_deg": ii.ravel(),
                         "n_active": n_active.ravel().astype(int),
                         "n_debris": n_debris.ravel().astype(int),
                         "exposure_rel": exposure_rel.ravel()})
    top = grid[grid["exposure_rel"] > 0].sort_values("exposure_rel", ascending=False).head(10)
    top = top.assign(
        alt_range_km=top["alt_center_km"].map(lambda a: f"{a - ALT_STEP / 2:.0f}-{a + ALT_STEP / 2:.0f}"),
        inc_range_deg=top["inc_center_deg"].map(lambda i: f"{i - INC_STEP / 2:.0f}-{i + INC_STEP / 2:.0f}"),
    )[["alt_range_km", "inc_range_deg", "n_active", "n_debris", "exposure_rel"]]

    profile = exposure_rel.sum(axis=1)
    profile = profile / profile.max() if profile.max() > 0 else profile
    return top, profile, alt_c


def find_risk_bands(profile, alt_c):
    """Непрерывные диапазоны высот, где риск выше порога - для оболочек на 3D-модели."""
    idx = np.flatnonzero(profile >= SHELL_THRESHOLD)
    if len(idx) == 0:
        return []
    bands = []
    for grp in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
        w = profile[grp]
        bands.append({"centre": float(np.average(alt_c[grp], weights=w)),
                      "lo": float(alt_c[grp[0]] - ALT_STEP / 2),
                      "hi": float(alt_c[grp[-1]] + ALT_STEP / 2),
                      "peak": float(w.max())})
    return sorted(bands, key=lambda b: b["peak"], reverse=True)[:MAX_SHELLS]


def describe_clusters(df):
    rows = []
    for cid, g in df[df["cluster"] != -1].groupby("cluster"):
        dom = g["category_short"].value_counts()
        rows.append({
            "cluster": int(cid), "n_objects": len(g),
            "dominant": dom.index[0], "dominant_share": round(dom.iloc[0] / len(g), 2),
            "median_alt_km": round(g["mean_alt_km"].median(), 1),
            "median_inc_deg": round(g["inclination_deg"].median(), 1),
            "mean_hazard_rel": round(g["hazard_rel"].mean(), 4),
        })
    return pd.DataFrame(rows).sort_values("n_objects", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------
# 4. 3D-модель
# ----------------------------------------------------------------------------
def latlon_to_xyz(lon_deg, lat_deg, radius):
    lon, lat = np.radians(lon_deg), np.radians(lat_deg)
    return (radius * np.cos(lat) * np.cos(lon),
            radius * np.cos(lat) * np.sin(lon),
            radius * np.sin(lat))


def orbital_positions(df, seed=42):
    """
    Положение объектов в геоцентрической системе (ось Z - ось вращения Земли).
    Высота, наклонение и долгота узла - из данных, положение вдоль орбиты - случайное.
    """
    rng = np.random.default_rng(seed)
    r = R_EARTH + df["mean_alt_km"].to_numpy()
    inc = np.radians(df["inclination_deg"].to_numpy())
    raan = np.radians(df["raan_deg"].to_numpy())
    u = rng.uniform(0, 2 * np.pi, len(df))     # аргумент широты
    cu, su, co, so, ci, si = np.cos(u), np.sin(u), np.cos(raan), np.sin(raan), np.cos(inc), np.sin(inc)
    x = r * (co * cu - so * su * ci)
    y = r * (so * cu + co * su * ci)
    z = r * (su * si)
    return np.round(x, 1), np.round(y, 1), np.round(z, 1)


def load_coastlines():
    try:
        if os.path.exists(COAST_CACHE):
            with open(COAST_CACHE, encoding="utf-8") as f:
                data = json.load(f)
        else:
            resp = requests.get(COAST_URL, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            with open(COAST_CACHE, "w", encoding="utf-8") as f:
                json.dump(data, f)
    except Exception as exc:
        print(f"[!] Береговые линии недоступны ({exc}) - Земля будет без континентов")
        return None
    lines = []
    for feat in data.get("features", []):
        geom = feat.get("geometry") or {}
        if geom.get("type") == "LineString":
            lines.append(geom["coordinates"])
        elif geom.get("type") == "MultiLineString":
            lines.extend(geom["coordinates"])
    return lines


def lines_trace(lines_lonlat, radius, color, width):
    xs, ys, zs = [], [], []
    for line in lines_lonlat:
        arr = np.asarray(line, dtype=float)
        x, y, z = latlon_to_xyz(arr[:, 0], arr[:, 1], radius)
        xs += x.tolist() + [None]
        ys += y.tolist() + [None]
        zs += z.tolist() + [None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=color, width=width),
                        hoverinfo="skip", showlegend=False)


def sphere_surface(radius, color, opacity=1.0, n_lat=60, n_lon=120, lighting=True):
    lon, lat = np.meshgrid(np.linspace(-180, 180, n_lon), np.linspace(-90, 90, n_lat))
    x, y, z = latlon_to_xyz(lon, lat, radius)
    return go.Surface(
        x=x, y=y, z=z, surfacecolor=np.zeros_like(x), cmin=0, cmax=1,
        colorscale=[[0, color], [1, color]], showscale=False, opacity=opacity,
        hoverinfo="skip", showlegend=False,
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0) if lighting else None,
    )


def scatter_objects(x, y, z, custom, name, color, size, visible):
    return go.Scatter3d(
        x=x, y=y, z=z, mode="markers", name=name, visible=visible, customdata=custom,
        marker=dict(size=size, color=color, opacity=1.0),
        hovertemplate=("<b>%{customdata[0]}</b><br>группа: %{customdata[1]}"
                       "<br>высота: %{customdata[2]} км<br>наклонение: %{customdata[3]}°"
                       "<br>кластер: %{customdata[4]}<br>индекс опасности: %{customdata[5]}"
                       "<extra></extra>"),
    )


def build_figure(df, bands, cluster_table):
    x, y, z = orbital_positions(df)
    custom = np.column_stack([
        df["object_name"].astype(str), df["category_short"],
        df["mean_alt_km"].round(0).astype(int), df["inclination_deg"].round(1),
        np.where(df["cluster"] == -1, "шум", df["cluster"].astype(str)),
        df["hazard_rel"].round(3),
    ]).astype(object)

    fig = go.Figure()

    # --- Земля: поверхность, сетка координат, континенты ---
    fig.add_trace(sphere_surface(R_EARTH, EARTH_COLOR))
    grid_lines = ([[(lon, lat) for lat in np.linspace(-90, 90, 61)] for lon in range(-180, 180, 30)]
                  + [[(lon, lat) for lon in np.linspace(-180, 180, 121)] for lat in range(-60, 61, 30)])
    fig.add_trace(lines_trace(grid_lines, R_EARTH * 1.001, GRATICULE_COLOR, 1))
    coast = load_coastlines()
    if coast:
        fig.add_trace(lines_trace(coast, R_EARTH * 1.002, COAST_COLOR, 2))

    # --- Объекты: три режима окраски ---
    obj_mode = []   # режим, которому принадлежит каждая трасса объектов
    first_obj = len(fig.data)

    def add(trace_mode, *args, **kwargs):
        fig.add_trace(scatter_objects(*args, visible=(trace_mode == 0), **kwargs))
        obj_mode.append(trace_mode)

    # режим 0: тип объектов
    is_active = (df["category"] == ACTIVE_NAME).to_numpy()
    for flag, color, size in ((True, CAT_COLORS[0], 2.6), (False, CAT_COLORS[1], 2.2)):
        m = is_active == flag
        add(0, x[m], y[m], z[m], custom[m], f"{GROUP_NAMES[flag]} ({int(m.sum())})", color, size)

    # режим 1: кластеры (8 крупнейших - своими цветами, остальные - серым)
    top_ids = cluster_table["cluster"].head(len(CAT_COLORS)).tolist()
    for rank, cid in enumerate(top_ids):
        row = cluster_table[cluster_table["cluster"] == cid].iloc[0]
        mixed = "" if row["dominant_share"] >= 0.8 else " + др."
        label = (f"К{cid}: {row['dominant']}{mixed}, ~{row['median_alt_km']:.0f} км, "
                 f"{row['median_inc_deg']:.0f}° ({row['n_objects']})")
        m = (df["cluster"] == cid).to_numpy()
        add(1, x[m], y[m], z[m], custom[m], label, CAT_COLORS[rank], 2.4)
    other = (~df["cluster"].isin(top_ids) & (df["cluster"] != -1)).to_numpy()
    if other.any():
        add(1, x[other], y[other], z[other], custom[other],
            f"Прочие кластеры ({int(other.sum())})", COLOR_OTHER, 2.0)
    noise = (df["cluster"] == -1).to_numpy()
    if noise.any():
        add(1, x[noise], y[noise], z[noise], custom[noise],
            f"Шум, вне кластеров ({int(noise.sum())})", COLOR_NOISE, 1.8)

    # режим 2: индекс опасности (логарифмическая шкала, как на тепловой карте)
    risk_color = np.log10(np.clip(df["hazard_rel"].to_numpy(), 1e-3, 1.0))
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z, mode="markers", name="Индекс опасности", showlegend=False,
        visible=False, customdata=custom,
        marker=dict(size=2.4, color=risk_color, colorscale=RISK_SCALE, cmin=-3, cmax=0, opacity=1.0,
                    colorbar=dict(title=dict(text="Индекс опасности", font=dict(color=INK_SECONDARY)),
                                  tickvals=[-3, -2, -1, 0], ticktext=["0.001", "0.01", "0.1", "1"],
                                  tickfont=dict(color=INK_SECONDARY), len=0.5, thickness=12,
                                  outlinewidth=0, x=0.97)),
        hovertemplate=("<b>%{customdata[0]}</b><br>высота: %{customdata[2]} км"
                       "<br>наклонение: %{customdata[3]}°<br>кластер: %{customdata[4]}"
                       "<br>индекс опасности: %{customdata[5]}<extra></extra>"),
    ))
    obj_mode.append(2)
    obj_idx = list(range(first_obj, len(fig.data)))

    # --- Оболочки риска (рисуются последними, чтобы не закрывать объекты) ---
    first_shell = len(fig.data)
    for b in bands:
        r = R_EARTH + b["centre"]
        fig.add_trace(sphere_surface(r, SHELL_COLOR, opacity=0.10 + 0.12 * b["peak"],
                                     n_lat=45, n_lon=90, lighting=False))
        fig.add_trace(go.Scatter3d(
            x=[r], y=[0], z=[0], mode="text", text=[f"зона риска {b['lo']:.0f}-{b['hi']:.0f} км"],
            textfont=dict(color=INK_SECONDARY, size=11), hoverinfo="skip", showlegend=False))
    shell_idx = list(range(first_shell, len(fig.data)))

    # --- Кнопки ---
    mode_labels = ["Тип объектов", "Кластеры", "Индекс опасности"]
    menus = [dict(
        type="buttons", direction="right", x=0.01, y=1.0, xanchor="left", yanchor="bottom",
        bgcolor=BTN_BG, bordercolor=BTN_BORDER, font=dict(color=INK_PRIMARY, size=12), active=0,
        buttons=[dict(label=lbl, method="restyle",
                      args=[{"visible": [m == k for m in obj_mode]}, obj_idx])
                 for k, lbl in enumerate(mode_labels)])]
    if shell_idx:
        menus.append(dict(
            type="buttons", direction="right", x=0.5, y=1.0, xanchor="left", yanchor="bottom",
            bgcolor=BTN_BG, bordercolor=BTN_BORDER, font=dict(color=INK_PRIMARY, size=12), active=0,
            buttons=[dict(label="Оболочки риска: вкл", method="restyle",
                          args=[{"visible": True}, shell_idx]),
                     dict(label="Оболочки риска: выкл", method="restyle",
                          args=[{"visible": False}, shell_idx])]))

    fig.update_layout(
        title=dict(text="Космический мусор на низких орбитах: кластеры и зоны риска",
                   x=0.01, y=0.97, font=dict(color=INK_PRIMARY, size=18)),
        paper_bgcolor=SURFACE,
        scene=dict(bgcolor=SURFACE, aspectmode="data",
                   xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                   camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))),
        legend=dict(x=0.01, y=0.95, bgcolor="rgba(26,26,25,0.7)", itemsizing="constant",
                    font=dict(color=INK_SECONDARY, size=12)),
        hoverlabel=dict(bgcolor=BTN_BG, font=dict(color=INK_PRIMARY)),
        margin=dict(l=0, r=0, t=95, b=0),
        updatemenus=menus,
        annotations=[dict(
            xref="paper", yref="paper", x=0.01, y=0.0, xanchor="left", yanchor="bottom",
            showarrow=False, align="left", font=dict(color=INK_MUTED, size=11),
            text=("Ось Z - ось вращения Земли. Положение объектов вдоль орбиты случайное "
                  "(в данных его нет);<br>высота, наклонение и долгота узла - реальные. "
                  "Индекс опасности относительный (0..1)."))],
    )
    return fig


# ----------------------------------------------------------------------------
# Запуск
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Кластеризация космического мусора и 3D-карта риска")
    ap.add_argument("csv", nargs="?", default="orbital_dataset_leo.csv",
                    help="CSV из скрипта загрузки данных (по умолчанию orbital_dataset_leo.csv)")
    ap.add_argument("--min-cluster-size", type=int, default=15, help="минимальный размер кластера")
    ap.add_argument("--min-samples", type=int, default=5, help="параметр плотности HDBSCAN")
    ap.add_argument("--output", default="debris_3d_model.html", help="имя выходного HTML-файла")
    ap.add_argument("--no-open", action="store_true", help="не открывать браузер")
    args = ap.parse_args()

    print("[1/4] Загрузка данных")
    df = load_data(args.csv)

    print("\n[2/4] Кластеризация")
    df["cluster"] = run_clustering(df, args.min_cluster_size, args.min_samples)
    df["category_short"] = df["category"].map(lambda c: SHORT_NAMES.get(c, c.replace("_", " ")))
    validate_against_known_events(df)

    print("\n[3/4] Расчёт риска")
    top, profile, alt_c = compute_risk(df)
    bands = find_risk_bands(profile, alt_c)
    cluster_table = describe_clusters(df)
    if cluster_table.empty:
        raise SystemExit("HDBSCAN не нашёл кластеров. Уменьшите --min-cluster-size.")
    print("Самые опасные зоны (по риску для действующих спутников):")
    print(top.to_string(index=False))
    print("\nКластеры (по размеру):")
    print(cluster_table.head(15).to_string(index=False))

    df.drop(columns=["category_short"]).rename(columns={"cluster": "cluster_hdbscan"}) \
        .to_csv("clustered_risk_dataset.csv", index=False)
    top.to_csv("risk_top_zones.csv", index=False)

    print("\n[4/4] Построение 3D-модели")
    fig = build_figure(df, bands, cluster_table)
    fig.write_html(args.output, include_plotlyjs=True, full_html=True,
                   config={"displaylogo": False, "responsive": True})
    path = os.path.abspath(args.output)
    print(f"Готово: {path}")
    print("Сохранено также: clustered_risk_dataset.csv, risk_top_zones.csv")
    if not args.no_open:
        webbrowser.open("file://" + path)


if __name__ == "__main__":
    main()

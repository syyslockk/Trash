"""Общие настройки проекта: пути, адреса источников, физические константы,
параметры сетки риска и кластеризации.

Все модули берут константы отсюда, чтобы GUI и CLI работали с одними значениями.
"""
from __future__ import annotations

import logging
from pathlib import Path

# --- Пути ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"              # результат loader
PROCESSED_DIR = DATA_DIR / "processed"  # результаты processing и analysis
RESOURCES_DIR = ROOT_DIR / "resources"
WEBUI_DIR = ROOT_DIR / "webui"

# --- Источник: CelesTrak (без регистрации) ---
# Полный каталог SATCAT одним файлом, обновляется раз в сутки.
SATCAT_URL = "https://celestrak.org/pub/satcat.csv"
# Элементы орбит GP/OMM в JSON по группам. CelesTrak просит не скачивать
# одни и те же данные чаще, чем раз в 2 часа, поэтому loader кэширует файлы.
GP_URL = "https://celestrak.org/NORAD/elements/gp.php"

# Группа CelesTrak -> категория объектов в нашем наборе
GP_GROUPS = {
    "active": "active_satellite",
    "cosmos-1408-debris": "cosmos_1408_debris",
    "fengyun-1c-debris": "fengyun_1c_debris",
    "iridium-33-debris": "iridium_33_debris",
    "cosmos-2251-debris": "cosmos_2251_debris",
}

SATCAT_FILE = "satcat.csv"
META_FILE = "meta.json"
OBJECTS_FILE = "objects.csv"   # результат processing в data/processed

# Группа GP с действующими спутниками; остальные группы - облака обломков
GP_ACTIVE_GROUP = "active"


def gp_filename(group: str) -> str:
    return f"gp_{group}.json"


def gp_url(group: str) -> str:
    return f"{GP_URL}?GROUP={group}&FORMAT=json"


SATCAT_REQUIRED_COLUMNS = [
    "OBJECT_NAME", "OBJECT_ID", "NORAD_CAT_ID", "OBJECT_TYPE",
    "OPS_STATUS_CODE", "DECAY_DATE", "PERIOD", "INCLINATION",
    "APOGEE", "PERIGEE", "ORBIT_CENTER",
]
GP_REQUIRED_FIELDS = [
    "OBJECT_NAME", "OBJECT_ID", "NORAD_CAT_ID", "EPOCH",
    "MEAN_MOTION", "ECCENTRICITY", "INCLINATION", "RA_OF_ASC_NODE",
]

# Коды OPS_STATUS_CODE, которые CelesTrak считает действующими:
# + работает, P частично, B резерв, S запасной, X продлённая миссия
ACTIVE_OPS_CODES = ("+", "P", "B", "S", "X")

# --- Сеть ---
HTTP_TIMEOUT_S = 60
HTTP_RETRIES = 3
HTTP_BACKOFF_S = 2.0   # пауза перед повтором, удваивается с каждой попыткой
USER_AGENT = "debris-risk-map/0.1 (educational project)"

# --- Физика ---
GM_EARTH = 398600.4418   # км^3/с^2
R_EARTH = 6378.137       # км (WGS84)
SECONDS_PER_DAY = 86400

# --- Область и сетка карты риска ---
ALT_MIN, ALT_MAX, ALT_STEP = 160, 2000, 20   # км
INC_MIN, INC_MAX, INC_STEP = 0, 180, 2       # градусы

# --- Кластеризация (HDBSCAN по mean_alt_km и inclination_deg) ---
CLUSTER_FEATURES = ["mean_alt_km", "inclination_deg"]
HDBSCAN_MIN_CLUSTER_SIZE = 15
HDBSCAN_MIN_SAMPLES = 5
SILHOUETTE_SAMPLE = 10000   # silhouette по случайной выборке, иначе считается долго
RANDOM_SEED = 0

# --- Проверка кластеров по известным облакам обломков ---
CLOUD_NAMES = {
    "fengyun_1c_debris": "Fengyun-1C (2007)",
    "iridium_33_debris": "Iridium-33 (2009)",
    "cosmos_2251_debris": "Cosmos-2251 (2009)",
    "cosmos_1408_debris": "Cosmos-1408 (2021)",
}
MIN_EVENT_OBJECTS = 20   # облака, где обломков меньше, не проверяем

# --- Модель риска ---
SIGMA_KM2 = 1e-5        # условное сечение, на относительный индекс не влияет
N_NODE_STEPS = 72       # шагов усреднения по разности RAAN
SMOOTH_SIGMA = 1.0      # gaussian_filter плотности обломков (в клетках)
SHELL_THRESHOLD = 0.25  # слои с риском > 25% от максимума -> «оболочки» на 3D
MAX_SHELLS = 5
TOP_N = 10              # строк в таблицах топ-зон и топ-объектов
HAZARD_ZONE_MIN_TARGETS = 5   # в топ зон по hazard - только клетки, где летает хотя бы столько целей
# Модель считает орбиты круговыми, поэтому сильно вытянутые в риск не берём
RISK_MAX_ECCENTRICITY = 0.1
# Источник опасности - обломки (object_type DEB).
# Цели - действующие спутники и ракетные ступени (R/B).

# --- Файлы analysis в data/processed ---
CLUSTERED_FILE = "clustered.csv"
RISK_GRID_FILE = "risk_grid.csv"
RISK_PROFILE_FILE = "risk_profile.csv"
TOP_ZONES_FILE = "top_zones.csv"
TOP_HAZARD_ZONES_FILE = "top_hazard_zones.csv"
TOP_OBJECTS_FILE = "top_objects.csv"
CLUSTERS_FILE = "clusters.csv"
VALIDATION_FILE = "validation.json"

# --- Логирование ---
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
LOG_DATEFMT = "%H:%M:%S"


def setup_logging(level: int = logging.INFO) -> None:
    """Настраивает вывод лога в консоль. GUI добавит свой обработчик сам."""
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=LOG_DATEFMT)

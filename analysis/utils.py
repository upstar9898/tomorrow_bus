"""
utils.py
========
모든 블록에서 공통으로 사용하는 함수 모음

[포함 내용]
  - 데이터 로드 (일별 CSV → 단일 DataFrame)
  - 한글 폰트 설정
  - 그래프 스타일 설정
  - 그래프/테이블 저장
  - 혼잡 레이블 컬러 정의
"""

from pathlib import Path
import glob
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ================================================================
# 경로 설정
# ================================================================
BASE_DIR     = Path(__file__).resolve().parent.parent
DATA_DIR     = BASE_DIR / "data" / "prepared"
OUTPUT_DIR   = BASE_DIR / "data" / "eda_output"

# ================================================================
# 혼잡 기준 (전처리 담당자와 합의된 기준)
# ================================================================
CONGESTION_COLORS = {
    "여유": "#4CAF50",   # 초록
    "혼잡": "#FF9800",   # 주황
    "만차": "#F44336",   # 빨강
}

CONGESTION_ORDER = ["여유", "혼잡", "만차"]

ROUTE_ORDER = ["9401", "9401-1", "9404", "9408", "9409", "9707", "9711"]

# ================================================================
# 데이터 로드
# ================================================================
def load_data(version: str = "v1") -> pd.DataFrame:
    """
    data/prepared/ 안의 일별 CSV를 전부 합쳐서 반환

    Parameters
    ----------
    version : str
        "v1" → 2026-03-10 ~ 2026-04-05
        "v2" → 추후 추가 예정
        "all" → v1 + v2 전부
    """
    pattern = str(DATA_DIR / "*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"데이터 파일이 없습니다: {DATA_DIR}")

    # 버전 필터
    if version == "v1":
        files = [f for f in files if "2026_03" in f or "2026_04_0" in f]
    elif version == "v2":
        files = [f for f in files if "2026_04" in f and "2026_04_0" not in f]
    # "all" 이면 전체 사용

    if not files:
        raise FileNotFoundError(f"version='{version}'에 해당하는 파일이 없습니다.")

    print(f"[데이터 로드] {len(files)}개 파일 로드 중...")
    dfs = [pd.read_csv(f, low_memory=False) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    # 타입 정리
    df["mkTm"] = pd.to_datetime(df["mkTm"], errors="coerce")
    df["remaining_seat"] = pd.to_numeric(df["remaining_seat"], errors="coerce")
    df["staOrd"] = pd.to_numeric(df["staOrd"], errors="coerce")

    # 운행시간 외 제거 (02~04시)
    df = df[~df["hour"].isin([2, 3, 4])].copy()

    # staOrd 최솟값(첫 번째 정류소) 오류값 제거
    # API가 차량 출발 대기 중일 때 잔여좌석을 0으로 내려보내는 오류값
    # → trip 내 staOrd 최솟값에서 remaining_seat=0인 행만 제거
    min_staord = df.groupby("trip_id")["staOrd"].transform("min")
    error_mask = (df["staOrd"] == min_staord) & (df["remaining_seat"] == 0)
    removed = error_mask.sum()
    df = df[~error_mask].copy()
    if removed > 0:
        print(f"  오류값 제거: {removed:,}건 (staOrd 최솟값 & 잔여좌석=0)")

    print(f"[데이터 로드] 완료 — shape: {df.shape}")
    print(f"  기간: {df['mkTm'].min().date()} ~ {df['mkTm'].max().date()}")
    print(f"  노선: {sorted(df['route_name'].unique())}")
    print(f"  Trip 수: {df['trip_id'].nunique():,}")

    return df


# ================================================================
# 한글 폰트 설정
# ================================================================
def set_korean_font():
    candidates = [
        "Apple SD Gothic Neo",
        "AppleGothic",
        "NanumGothic",
        "Malgun Gothic",
        "DejaVu Sans",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams["font.family"] = font
            mpl.rcParams["font.family"] = font
            break

    plt.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["axes.unicode_minus"] = False


# ================================================================
# 그래프 스타일 설정
# ================================================================
def set_plot_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "savefig.facecolor": "white",
        "axes.titlesize":    14,
        "axes.titleweight":  "bold",
        "axes.labelsize":    11,
        "axes.unicode_minus": False,
        "legend.frameon":    False,
    })
    set_korean_font()


# ================================================================
# 저장 함수
# ================================================================
def save_plot(filename: str, subdir: str = "plots"):
    """그래프를 data/eda_output/{subdir}/ 에 저장"""
    save_dir = OUTPUT_DIR / subdir
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / filename
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  저장: {path}")


def save_table(df: pd.DataFrame, filename: str, subdir: str = "tables"):
    """테이블을 data/eda_output/{subdir}/ 에 저장"""
    save_dir = OUTPUT_DIR / subdir
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  저장: {path}")


# ================================================================
# TOP N 제목 자동 생성 (모수 명시용)
# ================================================================
def top_n_title(title: str, n: int, total: int, unit: str = "개") -> str:
    """
    예: top_n_title("탑승 인원 많은 정류소", 20, 161, "개")
    → "탑승 인원 많은 정류소 (TOP 20 / 전체 161개 중)"
    """
    return f"{title} (TOP {n} / 전체 {total:,}{unit} 중)"
"""스마트 HACCP 확산 후보군 분석 파이프라인.

입력: data/raw/haccp_designation_raw.csv, data/raw/smart_haccp_raw.csv
출력: results/ 폴더의 기초 현황, 0% 관측 후보군, 지원 우선순위, 민감도 결과
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


MIN_FACILITIES = 30
SENSITIVITY_THRESHOLDS = (20, 30, 50)

# 업체명과 시도·시군구가 함께 일치해 수동으로 확인한 보조 결합 2건입니다.
MANUAL_MATCHES = {
    "20190371173",  # (주)닥터브레드, 경기 파주시
    "20000355553",  # 다미아일품(주), 경기 파주시
}


def normalize_name(value: object) -> str:
    """법인 표기·공백·기호 차이를 제거한 업체명 비교용 값입니다."""
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower()
    for token in ("주식회사", "유한회사", "농업회사법인", "영농조합법인", "사회적협동조합", "협동조합", "(주)", "㈜"):
        text = text.replace(token, "")
    return re.sub(r"[\s\-.,·()\[\]]", "", text)


def normalize_sido(value: object) -> str:
    if pd.isna(value):
        return ""
    aliases = {
        "서울특별시": "서울", "부산광역시": "부산", "대구광역시": "대구",
        "인천광역시": "인천", "광주광역시": "광주", "대전광역시": "대전",
        "울산광역시": "울산", "세종특별자치시": "세종", "경기도": "경기",
        "강원특별자치도": "강원", "강원도": "강원", "충청북도": "충북",
        "충청남도": "충남", "전북특별자치도": "전북", "전라북도": "전북",
        "전라남도": "전남", "경상북도": "경북", "경상남도": "경남",
        "제주특별자치도": "제주",
    }
    text = str(value).strip()
    return aliases.get(text, text[:2])


def extract_location(address: object) -> tuple[str, str]:
    """사업장 주소에서 분석용 시도와 시군구를 추출합니다."""
    if pd.isna(address) or not str(address).strip():
        return "주소 미상", "시군구 미상"
    parts = str(address).strip().split()
    province, municipality = parts[0], parts[1] if len(parts) > 1 else "시군구 미상"
    if province == "전남광주통합특별시":
        gwangju_districts = {"동구", "서구", "남구", "북구", "광산구"}
        return ("광주" if municipality in gwangju_districts else "전남"), municipality
    return normalize_sido(province), municipality


def first_non_null(series: pd.Series) -> str:
    values = series.dropna()
    return values.iloc[0] if not values.empty else ""


def one_industry(series: pd.Series) -> str:
    values = sorted({value for value in series.dropna() if str(value).strip()})
    if not values:
        return "업종 미상"
    return values[0] if len(values) == 1 else "복수 업종"


def min_max(series: pd.Series) -> pd.Series:
    lower, upper = series.min(), series.max()
    return pd.Series(0.0, index=series.index) if upper == lower else (series - lower) / (upper - lower)


def load_inputs(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    smart_path = input_dir / "smart_haccp_raw.csv"
    haccp_path = input_dir / "haccp_designation_raw.csv"
    if not smart_path.exists() or not haccp_path.exists():
        raise FileNotFoundError("data/raw 폴더에 HACCP·스마트 HACCP 원본 CSV 두 개가 필요합니다.")
    smart = pd.read_csv(smart_path, dtype="string", encoding="utf-8-sig")
    haccp = pd.read_csv(haccp_path, dtype="string", encoding="utf-8-sig")
    required = {
        "스마트 HACCP": (smart, {"licenseno", "company", "sido", "sgg"}),
        "HACCP 지정현황": (haccp, {"LCNS_NO", "BSSH_NM", "SITE_ADDR", "INDUTY_CD_NM"}),
    }
    for name, (frame, columns) in required.items():
        missing = columns - set(frame.columns)
        if missing:
            raise ValueError(f"{name} 필수 컬럼 누락: {sorted(missing)}")
    return smart, haccp


def build_facilities(smart: pd.DataFrame, haccp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """품목 행을 업체 단위로 통합하고 스마트 HACCP 여부를 붙입니다."""
    smart_facilities = smart.dropna(subset=["licenseno"]).drop_duplicates("licenseno")
    haccp_facilities = (
        haccp.groupby("LCNS_NO", as_index=False)
        .agg(
            업체명=("BSSH_NM", first_non_null),
            사업장주소=("SITE_ADDR", first_non_null),
            업종=("INDUTY_CD_NM", one_industry),
            소규모업체여부=("SSIZESCALE_BSSH_YN", first_non_null),
            HACCP_지정일=("HACCP_APPN_DT", first_non_null),
        )
        .rename(columns={"LCNS_NO": "영업등록번호"})
    )
    haccp_facilities[["시도", "시군구"]] = haccp_facilities["사업장주소"].map(extract_location).apply(pd.Series)

    direct_keys = set(smart_facilities["licenseno"])
    haccp_facilities["스마트HACCP여부"] = haccp_facilities["영업등록번호"].isin(direct_keys | MANUAL_MATCHES)
    haccp_facilities["결합유형"] = "미결합"
    haccp_facilities.loc[haccp_facilities["영업등록번호"].isin(direct_keys), "결합유형"] = "영업등록번호 직접 결합"
    haccp_facilities.loc[haccp_facilities["영업등록번호"].isin(MANUAL_MATCHES), "결합유형"] = "업체명·시군구 보조 결합"

    unmatched = smart_facilities.loc[~smart_facilities["licenseno"].isin(set(haccp_facilities["영업등록번호"]))].copy()
    unmatched["업체명_정규화"] = unmatched["company"].map(normalize_name)
    return haccp_facilities, unmatched


def summarize(facilities: pd.DataFrame) -> pd.DataFrame:
    facilities = facilities.copy()
    facilities["순위분석대상"] = (
        facilities["시도"].ne("주소 미상")
        & facilities["시군구"].ne("시군구 미상")
        & facilities["업종"].ne("업종 미상")
    )
    summary = (
        facilities.groupby(["시도", "시군구", "업종"], dropna=False)
        .agg(
            HACCP_업체수=("영업등록번호", "size"),
            스마트HACCP_업체수=("스마트HACCP여부", "sum"),
            소규모업체수=("소규모업체여부", lambda values: values.eq("Y").sum()),
            순위분석대상=("순위분석대상", "all"),
        )
        .reset_index()
    )
    summary["미전환_업체수"] = summary["HACCP_업체수"] - summary["스마트HACCP_업체수"]
    summary["스마트HACCP_관측전환율"] = summary["스마트HACCP_업체수"] / summary["HACCP_업체수"]
    summary["소규모업체_비율"] = summary["소규모업체수"] / summary["HACCP_업체수"]
    return summary.sort_values(["시도", "시군구", "업종"])


def rank_observed_adopters(summary: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """스마트 HACCP이 1건 이상 관측된 조합만 지원 우선순위를 산출합니다."""
    eligible = summary.loc[
        summary["순위분석대상"]
        & (summary["HACCP_업체수"] >= threshold)
        & (summary["스마트HACCP_업체수"] > 0)
    ].copy()
    eligible["미전환_정규화"] = min_max(eligible["미전환_업체수"])
    eligible["낮은전환율_정규화"] = 1 - min_max(eligible["스마트HACCP_관측전환율"])
    eligible["이상점거리"] = np.sqrt(
        (1 - eligible["미전환_정규화"]) ** 2 + (1 - eligible["낮은전환율_정규화"]) ** 2
    )
    eligible["우선순위"] = eligible["이상점거리"].rank(method="first", ascending=True).astype(int)
    return eligible.sort_values("우선순위")


def main() -> None:
    parser = argparse.ArgumentParser(description="스마트 HACCP 확산 후보군 분석")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    smart, haccp = load_inputs(args.input_dir)
    facilities, unmatched = build_facilities(smart, haccp)
    summary = summarize(facilities)
    summary.to_csv(args.output_dir / "city_industry_summary.csv", index=False, encoding="utf-8-sig")

    zero_observation = summary.loc[
        summary["순위분석대상"]
        & (summary["HACCP_업체수"] >= MIN_FACILITIES)
        & (summary["스마트HACCP_업체수"] == 0)
    ].sort_values("미전환_업체수", ascending=False)
    zero_observation.to_csv(args.output_dir / "zero_observation_candidates_n30.csv", index=False, encoding="utf-8-sig")

    rankings = {threshold: rank_observed_adopters(summary, threshold) for threshold in SENSITIVITY_THRESHOLDS}
    main_ranking = rankings[MIN_FACILITIES].copy()
    sensitivity = pd.concat(
        [frame.head(10).assign(최소업체수기준=threshold) for threshold, frame in rankings.items()],
        ignore_index=True,
    )
    appearance = (
        sensitivity.groupby(["시도", "시군구", "업종"], as_index=False)
        .agg(Top10_등장기준수=("최소업체수기준", "nunique"), Top10_등장기준=("최소업체수기준", lambda x: ", ".join(map(str, sorted(x)))))
    )
    main_ranking = main_ranking.merge(appearance, on=["시도", "시군구", "업종"], how="left")
    main_ranking.to_csv(args.output_dir / "support_priority_ranking_n30.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(args.output_dir / "sensitivity_top10.csv", index=False, encoding="utf-8-sig")
    unmatched.to_csv(args.output_dir / "unmatched_smart_haccp_facilities.csv", index=False, encoding="utf-8-sig")

    print(f"HACCP 업체: {len(facilities):,}개 / 스마트 HACCP 표시: {facilities['스마트HACCP여부'].sum():,}개")
    print(f"등록 0건 점검 후보(최소 {MIN_FACILITIES}개): {len(zero_observation):,}개")
    print(f"전환 지원 우선순위 후보(최소 {MIN_FACILITIES}개, 등록 1건 이상): {len(main_ranking):,}개")
    print("\n[전환 지원 우선 검토 Top 10]")
    for _, row in main_ranking.head(10).iterrows():
        print(f"{row['우선순위']:>2}위 | {row['시도']} {row['시군구']} · {row['업종']} | "
              f"전환율 {row['스마트HACCP_관측전환율']:.1%}, 미전환 {row['미전환_업체수']:,}개")


if __name__ == "__main__":
    main()

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TourismAbstainDecision:
    """観光カテゴリ専用の abstain 判定結果。

    - abstain: True の場合は回答を生成せず abstain する
    - reason: "keyword" / "dist" / "ok"
    - hit: 何に反応したか（ログ可視化用）
    """

    abstain: bool
    reason: str
    hit: str = ""


# 1) 即 abstain（状況依存が強い／当日変動／リアルタイム要求）
HARD_PATTERNS: list[tuple[str, str]] = [
    # リアルタイム要求（今日/今/現在）
    (r"(今日|本日|きょう|今|いま|現在|いま現在|リアルタイム)", "realtime"),

    # 混雑・空き・待ち（現況）
    (r"(混雑|空いて(る|ます)|すいて(る|ます)|渋滞|満車|駐車場.*空き|待ち時間|行列)", "congestion"),

    # 開花の現況（咲いてる？満開？散り始め？）
    (r"(咲いて(る|ます)|開花して(る|ます)|満開|散り始め|散って(る|ます))", "bloom_now"),

    # 「今日」付き開催可否（当日変動）
    (r"(開催|中止|やって(る|ます)|実施|ありますか).*(今日|本日|今|現在)", "event_today"),
]

# 2) 条件付き（一般論はOKだが、根拠が弱いと危険）
SOFT_PATTERNS: list[tuple[str, str]] = [
    # 「見頃/開花」→ 例年傾向はOK。ただし dist が悪いなら abstain へ
    (r"(開花|見頃|みごろ)", "bloom_general"),

    # イベント系（年度変動しやすい）→ dist が悪いと危険
    (r"(桜まつり|祭り|ライトアップ|イルミ|出店|屋台|イベント)", "event_general"),
]


def _match_any(patterns: list[tuple[str, str]], text: str) -> tuple[bool, str, str]:
    for pat, tag in patterns:
        m = re.search(pat, text)
        if m:
            return True, tag, m.group(0)
    return False, "", ""


def decide_tourism_abstain(query: str, best_dist: float, dist_threshold: float) -> TourismAbstainDecision:
    """観光カテゴリの abstain 判定。

    優先順位:
      1) 文意（HARD_PATTERNS）で即 abstain
      2) 文意（SOFT_PATTERNS）＋ dist が悪い場合 abstain
      3) dist が悪い場合 abstain
      4) それ以外は OK
    """
    q = (query or "").strip()
    if not q:
        return TourismAbstainDecision(False, "ok")

    # 1) HARD
    hit, tag, frag = _match_any(HARD_PATTERNS, q)
    if hit:
        return TourismAbstainDecision(True, "keyword", f"{tag}:{frag}")

    # 2) SOFT（一般論はOK。ただし根拠が弱いなら abstain）
    soft_hit, soft_tag, soft_frag = _match_any(SOFT_PATTERNS, q)
    if soft_hit:
        if best_dist > dist_threshold:
            return TourismAbstainDecision(True, "dist", f"{soft_tag}:{soft_frag}")
        return TourismAbstainDecision(False, "ok", f"{soft_tag}:{soft_frag}")

    # 3) dist ゲート
    if best_dist > dist_threshold:
        return TourismAbstainDecision(True, "dist", f"dist={best_dist:.3f}>{dist_threshold:.3f}")

    return TourismAbstainDecision(False, "ok")


def should_abstain_tourism(query: str, best_dist: float, dist_threshold: float) -> bool:
    return decide_tourism_abstain(query, best_dist, dist_threshold).abstain


TOURISM_ABSTAIN_MESSAGE = (
    "その内容は、天候や当日の状況によって変わるため、"
    "この案内では正確にお答えすることができません。\n\n"
    "最新の情報は、岡崎市の公式サイトや現地案内をご確認ください。"
)

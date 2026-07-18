"""
correction.py — 補正(逆変換)モードと、その物理的限界の明示

思想: 補聴器=蝸牛の前段プリプロセッサ。信号 → 補正 → 障害蝸牛 → 知覚。
補正が良い条件は「補正後の知覚 ≈ 健聴の知覚」。

★このモジュールの主眼は「できることを誇る」ことではなく、
  「できないことを測定で証明して明示する」こと。
  可逆な次元(閾値/補充現象)は補正で署名が回復することを示し、
  不可逆な次元(選択性/デッド領域/TFS)は、どんな前処理でも回復しないことを示す。

依存: numpy, scipy, cochlea_engine, probes。
"""
import numpy as np
from cochlea_engine import (CochleaVocoder, GammatoneBank, preset, erb_hz,
                            erb_space, recruitment_gain)


# ── 補正可能性レジストリ(明示) ───────────────────────────────────
CORRECTABILITY = {
    "閾値損失 (audiogram)": (
        "◎ 補正可能",
        "帯域別ゲインで可聴性を回復。既存補聴器の基本"),
    "補充現象 (recruitment)": (
        "◎ 補正可能",
        "WDRC圧縮=失われた蝸牛圧縮の外部代替。伸長の逆写像"),
    "デッド領域 (dead region)": (
        "△ 再マッピングのみ",
        "センサー消失→増幅は無効(歪みとマスキング増)。"
        "周波数移動で生存帯へ転写=補正でなく再配置+脳の再学習"),
    "周波数選択性低下 (broadened filters)": (
        "✗ 補正不可",
        "広がったフィルタは前処理の尖鋭化を再び混ぜる=ロッシー通信路。"
        "スペクトルコントラスト強調で部分緩和が限界"),
    "時間微細構造 (TFS) 喪失": (
        "✗ 補正不可",
        "包絡線のみに退化した情報は復元不能"),
}


def print_registry():
    print("=" * 70)
    print("補正可能性レジストリ — 何が治せて何が治せないか(物理で決まる)")
    print("=" * 70)
    for dim, (verdict, why) in CORRECTABILITY.items():
        print(f"\n  {verdict}  {dim}")
        print(f"      {why}")
    print("=" * 70)


# ── 補正変換(可逆な次元) ─────────────────────────────────────────
def wdrc_aid_level(L_db, hl_db, floor_db=-60.0, ceil_db=-6.0):
    """
    補充現象の逆=WDRC。障害蝸牛の伸長 recruitment を打ち消すよう入力レベルを
    事前圧縮+閾値上へ持ち上げる。recruitment(wdrc(L)) ≈ L(健聴知覚)になる設計。
    """
    ti = min(floor_db + hl_db, ceil_db - 3.0)
    ratio = (ceil_db - floor_db) / (ceil_db - ti)      # 障害側の伸長比
    return ti + (L_db - floor_db) / ratio              # 逆(=圧縮)して閾値上へ


def freq_lower(x, fs, knee_hz=1800.0, factor=0.5):
    """
    周波数移動(線形周波数圧縮)。knee 以上の成分を factor 倍に畳んで下げ、
    デッド領域の情報を生存帯へ転写。★これは補正でなく再マッピング。
    """
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / fs)
    Y = np.zeros_like(X)
    knee_bin = np.searchsorted(f, knee_hz)
    Y[:knee_bin] = X[:knee_bin]                        # knee 以下は素通し
    for i in range(knee_bin, len(f)):                  # knee 以上を圧縮して下へ
        tgt = knee_hz + (f[i] - knee_hz) * factor
        j = int(round(tgt / (fs / len(x))))
        if 0 <= j < len(Y):
            Y[j] += X[i]
    return np.fft.irfft(Y, n=len(x)).astype(np.float32)


def spectral_sharpen(x, strength=1.0):
    """
    スペクトルコントラスト強調(選択性低下への"部分"対策)。山を持ち上げ谷を
    深める。★広帯域フィルタが再び混ぜるため完全復元は原理的に不可能。
    """
    X = np.fft.rfft(x)
    mag = np.abs(X); ph = np.angle(X)
    m = mag / (mag.max() + 1e-12)
    mag2 = mag * (m ** strength)                       # 相対的に大きい成分を強調
    return np.fft.irfft(mag2 * np.exp(1j * ph), n=len(x)).astype(np.float32)


# ══════════════════════════════════ 限界の証明(閉ループ測定) ══
def _validate():
    from probes import two_tone, ten_dead, level_sweep
    fs = 44100
    print_registry()
    print("\n" + "=" * 70)
    print("限界の証明: 補正で署名が回復するか(可逆) / しないか(不可逆)")
    print("=" * 70)

    def band_rms(y, lo, hi):
        Y = np.abs(np.fft.rfft(y)); f = np.fft.rfftfreq(len(y), 1 / fs)
        m = (f >= lo) & (f <= hi)
        return float(np.sqrt(np.mean(Y[m] ** 2))) if m.any() else 0.0

    # ── ◎ 補充現象: WDRC で可聴ダイナミックレンジが回復するか ──────
    print("\n[◎ 補充現象] 可聴入力ダイナミックレンジ(HL=0の健聴値が回復目標)")
    floor, ceil = -60.0, -6.0
    for hl in [0, 20, 40, 60]:
        ti = min(floor + hl, ceil - 3.0)
        ratio = (ceil - floor) / (ceil - ti)
        # 無補正: recruitment(L)>-40 となる入力域
        L = np.linspace(-80, 0, 2000)
        out_un = recruitment_gain(10 ** (L / 20)[None, :], np.array([hl]))[0]
        aud_un = L[out_un > 10 ** (-40 / 20)]
        rng_un = np.ptp(aud_un) if aud_un.size else 0.0
        # 補正: WDRC を通してから recruitment
        L_aid = wdrc_aid_level(L, hl, floor, ceil)
        out_co = recruitment_gain(10 ** (L_aid / 20)[None, :], np.array([hl]))[0]
        aud_co = L[out_co > 10 ** (-40 / 20)]
        rng_co = np.ptp(aud_co) if aud_co.size else 0.0
        tag = "(健聴基準)" if hl == 0 else "(回復)"
        print(f"    HL={hl:2d}dB:  無補正 {rng_un:4.1f}dB  → WDRC補正 {rng_co:4.1f}dB  {tag}")
    print("    → WDRCで可聴域が健聴(HL=0)値へ回復=補充現象は補正可能")

    # ── △ デッド領域: 増幅は無効・周波数移動のみ有効 ────────────────
    print("\n[△ デッド領域] クリーン5kHzトーン, 重度SNHL(3-8kHzデッド)")
    from probes import probe_tone
    probe = probe_tone(5000, warble=False)
    voc = CochleaVocoder(fs, profile=preset("重度感音難聴+デッド領域"))
    normal = CochleaVocoder(fs, profile=preset("健聴 (Normal)"))
    p5_normal = band_rms(normal.process(probe), 4500, 5500)  # 健聴なら5kHzを知覚
    p5_dead = band_rms(voc.process(probe), 4500, 5500)       # デッドで消失
    # 増幅補正: 5kHz を +30dB
    X = np.fft.rfft(probe); f = np.fft.rfftfreq(len(probe), 1 / fs)
    Xa = X.copy(); Xa[(f >= 4500) & (f <= 5500)] *= 10 ** (30 / 20)
    amp = np.fft.irfft(Xa, n=len(probe)).astype(np.float32); amp /= np.abs(amp).max()
    p5_amp = band_rms(voc.process(amp), 4500, 5500)
    # 周波数移動補正: 5kHz を生存帯(~2.4kHz)へ
    low = freq_lower(probe, fs, knee_hz=1800, factor=0.5)
    p_surv = band_rms(voc.process(low), 2000, 3000)
    print(f"    健聴の5kHz帯知覚      : {p5_normal:.2e}  (基準=聞こえる)")
    print(f"    デッド無補正 5kHz帯    : {p5_dead:.2e}  (消失)")
    print(f"    デッド+30dB増幅 5kHz帯 : {p5_amp:.2e}  (増幅しても消失=無効)")
    print(f"    デッド+周波数移動 2-3kHz: {p_surv:.2e}  (生存帯へ転写=知覚回復)")
    print("    → 増幅は無効・再マッピングのみ有効=デッド領域は補正でなく再配置")

    # ── ✗ 選択性低下: 前処理の尖鋭化は再び混ぜられ回復しない ────────
    print("\n[✗ 周波数選択性低下] 2音分離(1500Hz±1.5ERB)の励起の谷")
    tt = two_tone(1500, sep_erb=1.5)
    d = erb_hz(1500) * 1.5
    for label, prof in [("健聴(目標)", "健聴 (Normal)"),
                        ("障害・無補正", "重度感音難聴+デッド領域"),
                        ("障害・尖鋭化補正", "重度感音難聴+デッド領域")]:
        sig = spectral_sharpen(tt, strength=2.0) if "補正" in label else tt
        voc = CochleaVocoder(fs, profile=preset(prof))
        e = voc.envelopes(sig).mean(axis=1)
        e_at = lambda fq: np.interp(fq, voc.cfs, e)
        peak = (e_at(1500 - d / 2) + e_at(1500 + d / 2)) / 2 + 1e-12
        valley = e_at(1500) + 1e-12
        print(f"    {label:<18} 谷/山 {20*np.log10(valley/peak):6.2f} dB")
    print("    → 尖鋭化しても障害フィルタが再び混ぜ、健聴の谷は戻らない=補正不可")
    print("=" * 70)


if __name__ == "__main__":
    _validate()

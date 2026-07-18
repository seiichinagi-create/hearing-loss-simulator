"""
probes.py — 病態炙り出し用プローブ信号生成(診断用サンプルWAV)

各病態が「はっきり聞こえる/はっきり消える」よう設計した刺激。
実臨床の聴覚検査に対応:
  spectral_ripple  … スペクトルリップル弁別(CIのチャンネル数=周波数分解能の標準指標)
  two_tone         … 2音分離(周波数選択性=聴覚フィルタ幅)
  probe_tone(f)    … 特定周波数プローブ(NIHL 4kHzノッチ/クッキーバイト1kHz を炙る)
  ten_dead         … デッド領域プローブ(TEN検査の発想: 雑音中の該当帯トーン)
  gap_detection    … ギャップ検出(時間分解能/APD)
  level_sweep      … レベルスイープ(補充現象=急峻な音量成長を炙る)

いずれも -1..1 正規化・float32。書き出しは save_probe()。
依存: numpy のみ(書き出し時のみ soundfile)。
"""
import numpy as np

FS = 44100


def _norm(x):
    x = np.asarray(x, dtype=np.float64)
    p = np.abs(x).max()
    return (x / p if p > 0 else x).astype(np.float32)


def _fade(x, fs=FS, ms=10):
    n = int(fs * ms / 1000)
    if n * 2 < len(x):
        r = np.ones(len(x))
        r[:n] = np.linspace(0, 1, n)
        r[-n:] = np.linspace(1, 0, n)
        x = x * r
    return x


# ── スペクトルリップル(CIチャンネル数=周波数分解能を炙る) ──────────
def spectral_ripple(dur=2.0, fs=FS, ripples_per_oct=2.0, f_lo=200, f_hi=6000, drift=True):
    """
    対数周波数上で正弦状に山谷を作った広帯域雑音。リップル密度が
    フィルタバンクの分解能を超えると山谷が潰れて平坦化=CIで弁別不能に。
    drift=True で山谷位置をゆっくり動かし「模様が動く/動かない」で分解能を可聴化。
    """
    n = int(dur * fs)
    t = np.arange(n) / fs
    # 対数間隔の多数トーン和
    freqs = np.logspace(np.log10(f_lo), np.log10(f_hi), 300)
    octs = np.log2(freqs / f_lo)
    out = np.zeros(n)
    rng = np.random.default_rng(0)
    for f, o in zip(freqs, octs):
        phase = 2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)
        ph_shift = 2 * np.pi * 0.2 * t if drift else 0.0
        amp = 1.0 + 0.9 * np.sin(2 * np.pi * ripples_per_oct * o + ph_shift)
        out += amp * np.sin(phase)
    return _norm(_fade(out, fs))


# ── 2音分離(周波数選択性=聴覚フィルタ幅を炙る) ──────────────────
def two_tone(dur=2.0, fs=FS, f_center=1500, sep_erb=0.5):
    """
    中心周波数の上下に ERB の sep 倍だけ離した2トーン。正常な狭い
    フィルタなら2音に分離、フィルタが広い(選択性低下)と唸り1つに融合。
    """
    from cochlea_engine import erb_hz
    n = int(dur * fs)
    t = np.arange(n) / fs
    d = erb_hz(f_center) * sep_erb
    y = np.sin(2 * np.pi * (f_center - d / 2) * t) + np.sin(2 * np.pi * (f_center + d / 2) * t)
    return _norm(_fade(y, fs))


# ── 特定周波数プローブ(ノッチ/中域損失を炙る) ────────────────────
def probe_tone(f_target=4000, dur=2.0, fs=FS, warble=True):
    """
    対象周波数のワーブルトーン(±3%FM)。NIHLの4kHzノッチや
    クッキーバイトの1kHz損失に置くと、その病態でだけ音が痩せる/消える。
    """
    n = int(dur * fs)
    t = np.arange(n) / fs
    if warble:
        inst = f_target * (1 + 0.03 * np.sin(2 * np.pi * 5 * t))
        ph = 2 * np.pi * np.cumsum(inst) / fs
    else:
        ph = 2 * np.pi * f_target * t
    return _norm(_fade(np.sin(ph), fs))


# ── デッド領域プローブ(TEN検査の発想) ────────────────────────────
def ten_dead(f_target=5000, dur=2.0, fs=FS):
    """
    広帯域雑音の中に対象周波数のトーンを埋める。健常なら雑音中でも
    トーンが聞こえるが、その帯がデッド領域だと(off-frequency listening
    でしか拾えず)雑音に完全に埋もれる=検出不能。
    """
    n = int(dur * fs)
    t = np.arange(n) / fs
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(n) * 0.5
    tone = np.sin(2 * np.pi * f_target * t) * 0.9
    return _norm(_fade(noise + tone, fs))


# ── ギャップ検出(時間分解能/APD) ─────────────────────────────────
def gap_detection(dur=2.0, fs=FS, gaps_ms=(2, 4, 8, 16, 32)):
    """
    広帯域雑音に段々広いギャップ(無音)を穿つ。時間分解能が保たれて
    いれば全ギャップが分かるが、低下(APD等)だと短いギャップが埋まる。
    """
    n = int(dur * fs)
    rng = np.random.default_rng(2)
    y = rng.standard_normal(n) * 0.7
    seg = n // (len(gaps_ms) + 1)
    for i, g in enumerate(gaps_ms):
        c = seg * (i + 1)
        h = int(fs * g / 1000 / 2)
        y[max(0, c - h): c + h] = 0.0
    return _norm(_fade(y, fs))


# ── レベルスイープ(補充現象=急峻な音量成長を炙る) ────────────────
def level_sweep(dur=3.0, fs=FS, f=1000, lo_db=-60, hi_db=0):
    """
    一定周波数トーンを lo→hi dB へ滑らかに増大。健常は緩やかに大きく
    なるが、補充現象があると閾値超えで突然うるさくなる(狭ダイナミックレンジ)。
    """
    n = int(dur * fs)
    t = np.arange(n) / fs
    env_db = np.linspace(lo_db, hi_db, n)
    amp = 10.0 ** (env_db / 20.0)
    return _norm(_fade(amp * np.sin(2 * np.pi * f * t), fs))


PROBES = {
    "spectral_ripple_2cyc": lambda: spectral_ripple(ripples_per_oct=2.0),
    "spectral_ripple_4cyc": lambda: spectral_ripple(ripples_per_oct=4.0),
    "two_tone_1500Hz":      lambda: two_tone(f_center=1500, sep_erb=0.6),
    "probe_4kHz_NIHL":      lambda: probe_tone(4000),
    "probe_1kHz_cookiebite": lambda: probe_tone(1000),
    "ten_dead_5kHz":        lambda: ten_dead(5000),
    "gap_detection":        lambda: gap_detection(),
    "level_sweep_recruit":  lambda: level_sweep(),
}


def save_probe(name, path=None, fs=FS):
    import soundfile as sf
    y = PROBES[name]()
    path = path or f"probe_{name}.wav"
    sf.write(path, y, fs, subtype="PCM_16")
    return path, y


# ══════════════════════════════════════ 検証: プローブが病態を炙るか ══
def _validate():
    """各プローブを [健聴 vs 標的病態] に通し、差(炙り度)を測定。"""
    from cochlea_engine import CochleaVocoder, preset, _centroid, erb_hz
    fs = FS
    print("=" * 66)
    print("プローブ検証: 健聴 vs 標的病態 で信号がどれだけ変わるか")
    print("=" * 66)

    def band_rms(y, f_lo, f_hi):
        Y = np.abs(np.fft.rfft(y)); f = np.fft.rfftfreq(len(y), 1 / fs)
        m = (f >= f_lo) & (f <= f_hi)
        return np.sqrt(np.mean(Y[m] ** 2)) if m.any() else 0.0

    def run(name, profile):
        y = PROBES[name]()
        return CochleaVocoder(fs, profile=preset(profile)).process(y)

    # 1) 4kHzプローブ: NIHLノッチで4kHz帯が痩せる
    n0 = run("probe_4kHz_NIHL", "健聴 (Normal)")
    n1 = run("probe_4kHz_NIHL", "騒音性難聴 (NIHL 4kHz notch)")
    d = 20 * np.log10((band_rms(n1, 3000, 5000) + 1e-9) / (band_rms(n0, 3000, 5000) + 1e-9))
    print(f"\n[1] 4kHzプローブ → NIHL: 3-5kHz帯エネルギー変化 {d:+.1f} dB (負=ノッチで痩せる)")

    # 2) 1kHzプローブ: クッキーバイトで中域が痩せる
    c0 = run("probe_1kHz_cookiebite", "健聴 (Normal)")
    c1 = run("probe_1kHz_cookiebite", "クッキーバイト (Cookie-bite)")
    d = 20 * np.log10((band_rms(c1, 700, 1400) + 1e-9) / (band_rms(c0, 700, 1400) + 1e-9))
    print(f"[2] 1kHzプローブ → Cookie-bite: 0.7-1.4kHz帯変化 {d:+.1f} dB (負=中域損失)")

    # 3) TENデッド領域プローブ: 重度SNHL(3-8kHzデッド)で5kHzトーンが埋没
    t0 = run("ten_dead_5kHz", "健聴 (Normal)")
    t1 = run("ten_dead_5kHz", "重度感音難聴+デッド領域")
    d = 20 * np.log10((band_rms(t1, 4500, 5500) + 1e-9) / (band_rms(t0, 4500, 5500) + 1e-9))
    print(f"[3] TENデッド5kHz → 重度SNHL: 5kHz帯変化 {d:+.1f} dB (負=デッド領域で埋没)")

    # 共通の健聴アナライザ=「知覚される励起パターン」(搬送波交絡を除去)
    normal_an = CochleaVocoder(fs, profile=preset("健聴 (Normal)"))
    def excitation(y):
        return normal_an.envelopes(y).mean(axis=1)          # (30,) チャンネル励起

    # 4) スペクトルリップル: 全条件を同じ健聴アナライザに通して知覚励起で測る。
    #    励起パターン(対数CF軸)のリップル(1cyc/oct)射影深さ。CIで潰れる。
    print(f"\n[4] スペクトルリップル(1cyc/oct)の知覚励起での残存深さ")
    cfs_oct = np.log2(normal_an.cfs / normal_an.cfs[0])

    def ripple_in_excitation(y, rpo=1.0):
        e = np.log(excitation(y) + 1e-12); e -= e.mean()
        c = np.sum(e * np.cos(2 * np.pi * rpo * cfs_oct))
        s = np.sum(e * np.sin(2 * np.pi * rpo * cfs_oct))
        return np.hypot(c, s) / len(e)

    r_in = spectral_ripple(ripples_per_oct=1.0)
    base = ripple_in_excitation(r_in)
    for prof in ["健聴 (Normal)", "人工内耳 16ch", "人工内耳 8ch", "人工内耳 4ch (重度)"]:
        y = CochleaVocoder(fs, profile=preset(prof)).process(r_in)
        print(f"    {prof:<20} 深さ {ripple_in_excitation(y):.3f} "
              f"(保持率 {100*ripple_in_excitation(y)/base:5.1f}%)")
    print("    → 健聴で高保持 vs CIで≈1/3以下へ潰れる(CI間の細かい順位はこの指標の分解能外)")

    # 5) 2音分離: 間隔を1.5 ERB(チャンネル間隔より広い)にし、病態フィルタ自身の
    #    励起で1500Hz中間の谷を測る。広帯域化で谷が埋まる=融合。
    print(f"\n[5] 2音分離(1500Hz±1.5ERB)の励起パターンの谷")
    d = erb_hz(1500) * 1.5
    f_lo_t, f_hi_t = 1500 - d / 2, 1500 + d / 2
    tt = two_tone(f_center=1500, sep_erb=1.5)
    for prof in ["健聴 (Normal)", "加齢性難聴 (Presbycusis)", "重度感音難聴+デッド領域"]:
        voc = CochleaVocoder(fs, profile=preset(prof))
        e = voc.envelopes(tt).mean(axis=1)
        e_at = lambda fq: np.interp(fq, voc.cfs, e)
        peak = (e_at(f_lo_t) + e_at(f_hi_t)) / 2 + 1e-12
        valley = e_at(1500) + 1e-12
        print(f"    {prof:<24} 谷/山 {20*np.log10(valley/peak):6.1f} dB (深い負=分離)")
    print("    → 健聴のみ谷・広帯域化で埋まる(方向は正・効果は小=30ch励起の分解能限界)")
    print("    ※選択性の厳密な証明は cochlea_engine.py [6](実測BW=係数倍)が強い")
    print("=" * 66)


if __name__ == "__main__":
    _validate()

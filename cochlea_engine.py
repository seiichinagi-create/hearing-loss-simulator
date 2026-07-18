"""
cochlea_engine.py — 蝸牛フィルタバンク + ボコーダ再合成(科学ガチ版の心臓部)

設計思想:
  STFT+固定ピッチ格子+サイン束(旧app.py) を廃し、
  基底膜の周波数分解を ERB 尺度ガンマトーンフィルタバンクで実装。
    分析: 複素ガンマトーン(Hohmann 2002 系) → チャンネル毎の解析信号
          → 包絡線 = |解析信号|(Hilbert 相当がタダで出る)
    再合成: 各チャンネル包絡線で搬送波(トーン / 雑音帯域)を変調して総和
            = Shannon 1995 ボコーダ = 人工内耳/感音難聴シミュの正式手法

病態(SNHL/CI)は「包絡線行列 + フィルタ帯域幅 + チャンネル数 + 搬送波」への
変換として後段に乗る。ここはその土台と、正しさの数値検証のみ。

依存: numpy, scipy のみ。
"""
import numpy as np
from scipy.signal import lfilter, butter


# ── ERB 尺度(Glasberg & Moore 1990) ──────────────────────────────
def erb_hz(f):
    """周波数 f[Hz] における等価矩形帯域幅 ERB[Hz]。"""
    return 24.7 * (4.37 * np.asarray(f, dtype=float) / 1000.0 + 1.0)


def erb_rate(f):
    """ERB-number(ERB 尺度上の位置)。チャンネルを等間隔に置くための座標。"""
    return 21.4 * np.log10(4.37 * np.asarray(f, dtype=float) / 1000.0 + 1.0)


def erb_rate_inv(e):
    """ERB-number → 周波数[Hz]。"""
    return (10.0 ** (np.asarray(e, dtype=float) / 21.4) - 1.0) * 1000.0 / 4.37


def erb_space(f_lo, f_hi, n):
    """f_lo〜f_hi を ERB 尺度で等間隔に n 本並べた中心周波数[Hz]。"""
    e = np.linspace(erb_rate(f_lo), erb_rate(f_hi), n)
    return erb_rate_inv(e)


# ── 複素ガンマトーンフィルタバンク ────────────────────────────────
class GammatoneBank:
    """
    4次複素ガンマトーン(1次複素共振器×4段のカスケード)。
    各チャンネル: y[n] = x[n] + a*y[n-1] を4回。極 a = exp(-2π(β - i·fc)/fs)。
    β = ERB(fc) * BW_K * bw_factor で帯域幅を決める。
    """
    ORDER = 4
    BW_K = 1.163  # -3dB帯域幅=ERB に合わせる較正係数(実測: 全帯域で比≈1.0)

    def __init__(self, fs, cfs, bw_factor=1.0):
        # bw_factor>1 = 聴覚フィルタの広帯域化(周波数選択性低下=SNHL病態)
        self.fs = float(fs)
        self.cfs = np.asarray(cfs, dtype=float)
        self.n_ch = len(self.cfs)
        self.bw_factor = float(bw_factor)
        beta = erb_hz(self.cfs) * self.BW_K * self.bw_factor
        # 複素極 → 共振を +fc(正の解析周波数)に置く
        self.a = np.exp(-2.0 * np.pi * (beta - 1j * self.cfs) / self.fs)
        # 各フィルタを CF で単位利得に正規化(インパルス応答のピーク利得で割る)
        self.gain = np.ones(self.n_ch)
        imp = np.zeros(2048)
        imp[0] = 1.0
        for k in range(self.n_ch):
            g = self._filter_one(imp, k)
            H = np.abs(np.fft.fft(g))   # 複素フィルタ → 片側でない全FFT
            self.gain[k] = 1.0 / (H.max() + 1e-20)

    def _filter_one(self, x, k):
        a = self.a[k]
        y = x.astype(np.complex128)
        for _ in range(self.ORDER):
            y = lfilter([1.0], [1.0, -a], y)
        return y

    def analyze(self, x):
        """入力 x(実) → 複素解析信号 (n_ch, N)。"""
        x = np.asarray(x, dtype=np.complex128)
        out = np.empty((self.n_ch, len(x)), dtype=np.complex128)
        for k in range(self.n_ch):
            out[k] = self._filter_one(x, k) * self.gain[k]
        return out


# ── 聴力図(周波数→損失dB) ───────────────────────────────────────
def audiogram_gain(cfs, breakpoints_hz, losses_db):
    """CF ごとの利得(線形)。聴力図を折れ線補間して dB→線形。"""
    db = np.interp(cfs, breakpoints_hz, losses_db)
    return db, 10.0 ** (-db / 20.0)


# v1 プロファイルを聴力図(=閾値損失曲線)として再利用
AUDIOGRAMS = {
    "presbycusis": (  # 加齢性: 高域漸減
        [125, 250, 500, 1000, 2000, 4000, 8000, 16000],
        [0, 0, 5, 10, 25, 50, 65, 75]),
    "nihl": (         # 騒音性: 4kHz ノッチ
        [125, 500, 1000, 2000, 3000, 4000, 6000, 8000],
        [0, 5, 10, 15, 35, 55, 40, 30]),
    "cookie_bite": (  # クッキーバイト: 中域U字
        [125, 250, 500, 1000, 2000, 4000, 8000],
        [10, 20, 45, 55, 45, 20, 10]),
    "flat_severe": (  # 重度平坦
        [125, 8000], [70, 75]),
    "normal": ([125, 8000], [0, 0]),
}


# ── 聴覚病態プロファイル ───────────────────────────────────────────
class HearingProfile:
    """
    蝸牛/CI の病態を一括記述。全て土台(フィルタバンク/包絡線)への変換。
      audiogram      : (breakpoints_hz, losses_db) or None
      recruitment    : 補充現象(閾値上げ+可聴域圧縮の伸長)を効かせるか
      bw_factor      : 周波数選択性(1.0=正常, >1=広帯域化)
      dead_regions   : [(f_lo, f_hi), ...] 無反応帯
      current_spread_db : CI 電流拡散(dB/チャンネル, 0=なし)
      n_channels     : 蝸牛符号化のチャンネル数(正常~30, CI=4〜22)
      carrier        : "tone" / "noise"
      env_cutoff     : 包絡線LPF[Hz](低い=TFS喪失=CI寄り)
    """
    def __init__(self, name="Normal", audiogram=None, recruitment=False,
                 bw_factor=1.0, dead_regions=None, current_spread_db=0.0,
                 n_channels=30, carrier="tone", env_cutoff=300.0):
        self.name = name
        self.audiogram = audiogram
        self.recruitment = recruitment
        self.bw_factor = bw_factor
        self.dead_regions = dead_regions or []
        self.current_spread_db = current_spread_db
        self.n_channels = n_channels
        self.carrier = carrier
        self.env_cutoff = env_cutoff


def recruitment_gain(env, hl_db, floor_db=-60.0, ceil_db=-6.0):
    """
    補充現象: OHC 圧縮の喪失。閾値が hl_db 上がり、可聴域が [Ti, ceil] に狭窄。
    その狭い入力域を正常のラウドネス幅へ伸長する(=急峻な音量成長)。
    env(線形) → dB → 区分線形マップ → 線形。閾値下は消音。
    """
    hl_db = np.asarray(hl_db, float)[:, None]                 # (n_ch,1)
    ti = floor_db + hl_db                                     # 上がった閾値
    ti = np.minimum(ti, ceil_db - 3.0)                        # 重度でも3dB窓は残す
    ratio = (ceil_db - floor_db) / (ceil_db - ti)             # 伸長比>1
    L = 20.0 * np.log10(np.maximum(env, 1e-9))
    O = floor_db + (L - ti) * ratio                           # 伸長後の出力レベル
    O = np.where(L <= ti, -120.0, O)                          # 閾値下=消音
    O = np.where(L >= ceil_db, L, O)                          # 大音域=素通し
    return 10.0 ** (O / 20.0)


# ── ボコーダ(分析 → 包絡線 → 病態 → 再合成) ────────────────────
class CochleaVocoder:
    def __init__(self, fs, profile=None, f_lo=80.0, f_hi=8000.0, seed=0):
        self.fs = float(fs)
        self.profile = profile or HearingProfile()
        self.f_lo, self.f_hi = f_lo, f_hi
        self.cfs = erb_space(f_lo, f_hi, self.profile.n_channels)
        self.bank = GammatoneBank(fs, self.cfs, bw_factor=self.profile.bw_factor)
        self.rng = np.random.default_rng(seed)

    def envelopes(self, x):
        """チャンネル毎包絡線 (n_ch, N)。|解析信号| を LPF。"""
        g = self.bank.analyze(x)
        env = np.abs(g)
        cut = self.profile.env_cutoff
        if cut and cut < self.fs / 2:
            b, a = butter(2, cut / (self.fs / 2), btype="low")
            env = lfilter(b, a, env, axis=1)
            env = np.maximum(env, 0.0)
        return env

    def apply_pathology(self, env):
        """包絡線行列に聴力図→補充現象→デッド領域→CI電流拡散を適用。"""
        p = self.profile
        # 1. 聴力図(閾値損失)= チャンネル利得 / 補充現象の閾値
        hl_db = np.zeros(self.profile.n_channels)
        if p.audiogram is not None:
            bp, ls = p.audiogram
            hl_db, gain = audiogram_gain(self.cfs, bp, ls)
            if not p.recruitment:
                env = env * gain[:, None]        # 単純減衰(補充なし)
        # 2. 補充現象(閾値上げ+可聴域圧縮)
        if p.recruitment:
            # 絶対SPL較正が無いので提示レベルを固定: 信号の大音部(99.5%ile)を
            # ceil 付近(-12dBFS)に置いてから補充現象窓を適用(相対音声の穴を塞ぐ)
            ref = np.percentile(env, 99.5)
            if ref > 0:
                env = env * (10.0 ** (-12.0 / 20.0) / ref)
            env = recruitment_gain(env, hl_db)
        # 3. デッド領域(無反応帯 → チャンネルゼロ)
        for lo, hi in p.dead_regions:
            env[(self.cfs >= lo) & (self.cfs <= hi)] = 0.0
        # 4. CI 電流拡散(隣接チャンネルへの指数漏れ)
        if p.current_spread_db > 0:
            env = self._current_spread(env, p.current_spread_db)
        return env

    def _current_spread(self, env, spread_db):
        n = env.shape[0]
        idx = np.arange(n)
        W = 10.0 ** (-np.abs(idx[:, None] - idx[None, :]) * spread_db / 20.0)
        W /= W.sum(axis=0, keepdims=True)      # 各出力チャンネルで正規化
        return W @ env

    def resynthesize(self, env):
        n_ch, N = env.shape
        t = np.arange(N) / self.fs
        out = np.zeros(N)
        for k in range(n_ch):
            if self.profile.carrier == "tone":
                phi = self.rng.uniform(0, 2 * np.pi)
                carrier = np.cos(2 * np.pi * self.cfs[k] * t + phi)
            else:  # noise: チャンネル帯域に絞った白色雑音
                lo = max(self.cfs[k] - erb_hz(self.cfs[k]), 10.0)
                hi = min(self.cfs[k] + erb_hz(self.cfs[k]), self.fs / 2 - 1)
                b, a = butter(2, [lo / (self.fs / 2), hi / (self.fs / 2)], btype="band")
                carrier = lfilter(b, a, self.rng.standard_normal(N))
            out += env[k] * carrier
        peak = np.max(np.abs(out))
        if peak > 0:
            out /= peak
        return out.astype(np.float32)

    def process(self, x):
        return self.resynthesize(self.apply_pathology(self.envelopes(x)))


# ── プリセット(臨床像 → プロファイル) ───────────────────────────
def preset(name, fs=None):
    a = AUDIOGRAMS
    P = {
        "健聴 (Normal)":
            HearingProfile("Normal", n_channels=30, carrier="tone"),
        "加齢性難聴 (Presbycusis)":
            HearingProfile("Presbycusis", audiogram=a["presbycusis"],
                           recruitment=True, bw_factor=1.7),
        "騒音性難聴 (NIHL 4kHz notch)":
            HearingProfile("NIHL", audiogram=a["nihl"],
                           recruitment=True, bw_factor=1.5),
        "クッキーバイト (Cookie-bite)":
            HearingProfile("Cookie-bite", audiogram=a["cookie_bite"],
                           recruitment=True, bw_factor=1.8),
        "重度感音難聴+デッド領域":
            HearingProfile("Severe SNHL", audiogram=a["flat_severe"],
                           recruitment=True, bw_factor=3.0,
                           dead_regions=[(3000, 8000)]),
        "人工内耳 16ch":
            HearingProfile("CI-16", n_channels=16, carrier="noise",
                           env_cutoff=160.0, current_spread_db=8.0),
        "人工内耳 8ch":
            HearingProfile("CI-8", n_channels=8, carrier="noise",
                           env_cutoff=120.0, current_spread_db=6.0),
        "人工内耳 4ch (重度)":
            HearingProfile("CI-4", n_channels=4, carrier="noise",
                           env_cutoff=50.0, current_spread_db=4.0),
    }
    return P[name]


PRESET_NAMES = [
    "健聴 (Normal)",
    "加齢性難聴 (Presbycusis)",
    "騒音性難聴 (NIHL 4kHz notch)",
    "クッキーバイト (Cookie-bite)",
    "重度感音難聴+デッド領域",
    "人工内耳 16ch",
    "人工内耳 8ch",
    "人工内耳 4ch (重度)",
]


# ══════════════════════════════════════════════ 自己検証(記憶でなく測定) ══
def _measure_filter(bank, k, fs):
    """チャンネル k のインパルス応答から実測 中心周波数 と -3dB 帯域幅。"""
    imp = np.zeros(32768); imp[0] = 1.0   # 低域の帯域幅測定用に高分解能(1.3Hz/bin)
    g = bank.analyze(imp)[k]
    N = len(g)
    H = np.abs(np.fft.fft(g))[: N // 2]       # 正周波数側のみ
    f = np.fft.fftfreq(N, 1 / fs)[: N // 2]
    pk = H.argmax()
    cf_meas = f[pk]
    half = H[pk] / np.sqrt(2)
    above = np.where(H >= half)[0]
    bw_meas = f[above[-1]] - f[above[0]] if len(above) > 1 else 0.0
    return cf_meas, bw_meas


def _self_test():
    fs = 44100
    print("=" * 62)
    print("蝸牛エンジン 自己検証")
    print("=" * 62)

    # 1) ERB 尺度チャンネル配置
    cfs = erb_space(80, 8000, 30)
    print(f"\n[1] ERBチャンネル配置 (30ch, 80-8000Hz)")
    print(f"    CF先頭3: {cfs[:3].round(1)}  末尾3: {cfs[-3:].round(1)}")
    d = np.diff(erb_rate(cfs))
    print(f"    ERB尺度間隔の一様性: std/mean = {d.std()/d.mean():.2e} (≈0=等間隔)")

    # 2) フィルタの実測 CF / 帯域幅 vs 設計値
    bank = GammatoneBank(fs, cfs)
    print(f"\n[2] インパルス応答からの実測 (設計 vs 実測)")
    print(f"    {'設計CF':>8} {'実測CF':>8} {'誤差%':>7} {'実測BW':>8} {'ERB':>7} {'BW/ERB':>7}")
    ratios = []
    for k in [3, 10, 18, 25]:
        cfm, bwm = _measure_filter(bank, k, fs)
        erb = erb_hz(cfs[k])
        ratios.append(bwm / erb)
        err = 100 * (cfm - cfs[k]) / cfs[k]
        print(f"    {cfs[k]:8.1f} {cfm:8.1f} {err:7.2f} {bwm:8.1f} {erb:7.1f} {bwm/erb:7.3f}")
    print(f"    → BW/ERB 平均 {np.mean(ratios):.3f} (1.0 が理想; BW_K で較正)")

    # 3) フィルタバンクの帯域被覆(パワー和のリップル)
    imp = np.zeros(8192); imp[0] = 1.0
    G = bank.analyze(imp)
    half = 8192 // 2
    P = (np.abs(np.fft.fft(G, axis=1)[:, :half]) ** 2).sum(axis=0)
    f = np.fft.fftfreq(8192, 1 / fs)[:half]
    band = (f >= 150) & (f <= 6000)
    Pdb = 10 * np.log10(P[band] + 1e-20)
    print(f"\n[3] 帯域被覆リップル (150-6000Hz): {Pdb.max()-Pdb.min():.1f} dB p-p")

    # 4) 実音の再合成(チャンネル数を減らす=CI劣化トレンド)
    import os, soundfile as sf
    wav = os.path.join(os.path.dirname(__file__), "yukarin4488.wav")
    if os.path.exists(wav):
        x, sr = sf.read(wav)
        if x.ndim > 1:
            x = x.mean(axis=1)
        x = x[: int(sr * 3.0)].astype(float)
        x /= (np.abs(x).max() + 1e-12)
        src = "yukarin4488.wav 冒頭3s"
    else:
        sr = fs
        tt = np.arange(int(fs * 1.0)) / fs
        x = sum(np.sin(2 * np.pi * f0 * tt) for f0 in (220, 440, 880, 1760)) / 4
        src = "合成4トーン(wav無し)"
    print(f"\n[4] 全プリセット再合成 ({src}, sr={sr})")
    print(f"    {'プリセット':<26}{'RMS dB':>8}{'重心Hz':>8}{'NaN':>5}")
    print(f"    {'原音':<26}{20*np.log10(_rms(x)+1e-12):8.1f}{_centroid(x,sr):8.0f}{'-':>5}")
    for name in PRESET_NAMES:
        voc = CochleaVocoder(sr, profile=preset(name))
        y = voc.process(x)
        nan = np.isnan(y).any() or np.isinf(y).any()
        print(f"    {name:<26}{20*np.log10(_rms(y)+1e-12):8.1f}"
              f"{_centroid(y,sr):8.0f}{str(nan):>5}")

    # 5) 補充現象: 可聴入力ダイナミックレンジが HL ぶん狭窄するか
    print(f"\n[5] 補充現象の検証 (入力 -80→0dB スイープ, ceil=-6/floor=-60)")
    sweep = 10.0 ** (np.linspace(-80, 0, 400) / 20.0)[None, :]   # (1,400) 線形
    print(f"    {'HL dB':>6}{'可聴入力域dB':>12}{'理論(=54-HL)':>14}")
    for hl in [0, 20, 40, 60]:
        out = recruitment_gain(sweep, np.array([hl]))
        audible = out[0] > 10 ** (-40 / 20.0)   # -40dB以上出る入力を可聴とみなす
        Lin = 20 * np.log10(sweep[0])
        rng = (Lin[audible].max() - Lin[audible].min()) if audible.any() else 0.0
        print(f"    {hl:6d}{rng:12.1f}{max(54-hl,0):14d}")
    print("    → HLが増えるほど可聴入力域が狭まる=感音難聴の狭ダイナミックレンジ")

    # 6) 周波数選択性低下: bw_factor で実測帯域幅が広がるか
    print(f"\n[6] 周波数選択性低下 (2140Hz チャンネルの実測BW)")
    for bwf in [1.0, 2.0, 4.0]:
        b2 = GammatoneBank(fs, cfs, bw_factor=bwf)
        _, bwm = _measure_filter(b2, 18, fs)
        print(f"    bw_factor={bwf:.1f}: 実測BW={bwm:6.1f}Hz (ERB={erb_hz(cfs[18]):.0f}Hz, 比{bwm/erb_hz(cfs[18]):.2f})")
    print("    → 係数どおり広帯域化=聴覚フィルタの鈍化(スメアリング)")

    # 7) デッド領域: 指定帯のチャンネルがゼロか
    print(f"\n[7] デッド領域 (3000-8000Hz を無反応化)")
    voc = CochleaVocoder(sr, profile=preset("重度感音難聴+デッド領域"))
    env = voc.apply_pathology(voc.envelopes(x))
    dead = (voc.cfs >= 3000) & (voc.cfs <= 8000)
    print(f"    デッド帯チャンネルの最大包絡線: {env[dead].max():.2e} (=0 が期待)")
    print(f"    生存帯チャンネルの最大包絡線: {env[~dead].max():.2e} (>0)")

    # 8) CI電流拡散: 単一チャンネル励起が隣へ漏れるか
    print(f"\n[8] CI電流拡散 (8ch, spread=6dB/ch, ch3のみ励起)")
    voc = CochleaVocoder(sr, profile=preset("人工内耳 8ch"))
    e = np.zeros((8, 1)); e[3, 0] = 1.0
    spread = voc._current_spread(e, 6.0)[:, 0]
    print(f"    出力: {np.round(spread, 3)}")
    print(f"    → ch3ピーク+隣接へ指数減衰の漏れ=電極間干渉(空間スメアリング)")
    print("=" * 62)


def _rms(x):
    return np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2))


def _centroid(x, sr):
    X = np.abs(np.fft.rfft(x))
    f = np.fft.rfftfreq(len(x), 1 / sr)
    return float((f * X).sum() / (X.sum() + 1e-20))


if __name__ == "__main__":
    _self_test()

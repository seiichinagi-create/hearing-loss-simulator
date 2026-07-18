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
    各チャンネル: y[n] = x[n] + a*y[n-1] を4回。極 a = exp(-2π(β + i·fc)/fs)。
    β = ERB(fc) * BW_K で帯域幅を ERB に一致させる(BW_K は測定で較正)。
    """
    ORDER = 4
    BW_K = 1.163  # -3dB帯域幅=ERB に合わせる較正係数(実測: 全帯域で比≈1.0)
                  # 「周波数選択性低下(SNHL)」病態はこの係数を数倍する

    def __init__(self, fs, cfs):
        self.fs = float(fs)
        self.cfs = np.asarray(cfs, dtype=float)
        self.n_ch = len(self.cfs)
        beta = erb_hz(self.cfs) * self.BW_K
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


# ── ボコーダ(分析 → 包絡線 → 病態 → 再合成) ────────────────────
class CochleaVocoder:
    def __init__(self, fs, n_channels=30, f_lo=80.0, f_hi=8000.0,
                 env_cutoff=300.0, carrier="tone", seed=0):
        self.fs = float(fs)
        self.cfs = erb_space(f_lo, f_hi, n_channels)
        self.bank = GammatoneBank(fs, self.cfs)
        self.env_cutoff = env_cutoff        # 包絡線 LPF[Hz](低いほど TFS 喪失=CI 寄り)
        self.carrier = carrier              # "tone"(正弦) or "noise"(雑音帯域)
        self.rng = np.random.default_rng(seed)

    def envelopes(self, x):
        """チャンネル毎包絡線 (n_ch, N)。|解析信号| を LPF。"""
        g = self.bank.analyze(x)
        env = np.abs(g)
        if self.env_cutoff and self.env_cutoff < self.fs / 2:
            b, a = butter(2, self.env_cutoff / (self.fs / 2), btype="low")
            env = lfilter(b, a, env, axis=1)
            env = np.maximum(env, 0.0)
        return env

    def resynthesize(self, env):
        n_ch, N = env.shape
        t = np.arange(N) / self.fs
        out = np.zeros(N)
        for k in range(n_ch):
            if self.carrier == "tone":
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
        return self.resynthesize(self.envelopes(x))


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
    print(f"\n[4] 実音再合成 ({src}, sr={sr})")
    cen0 = _centroid(x, sr)
    print(f"    {'条件':<22}{'RMS dB':>8}{'ピーク':>7}{'重心Hz':>8}{'NaN':>5}")
    print(f"    {'原音':<22}{20*np.log10(_rms(x)+1e-12):8.1f}{np.abs(x).max():7.2f}{cen0:8.0f}{'-':>5}")
    for n_ch, carrier, cut, label in [
        (30, "tone",  300, "健聴30ch/tone"),
        (30, "noise", 300, "健聴30ch/noise"),
        (8,  "noise", 160, "CI 8ch/noise"),
        (4,  "noise", 50,  "CI 4ch/noise(重度)"),
    ]:
        voc = CochleaVocoder(sr, n_channels=n_ch, carrier=carrier, env_cutoff=cut)
        y = voc.process(x)
        nan = np.isnan(y).any() or np.isinf(y).any()
        print(f"    {label:<22}{20*np.log10(_rms(y)+1e-12):8.1f}"
              f"{np.abs(y).max():7.2f}{_centroid(y,sr):8.0f}{str(nan):>5}")
    print("\n    期待: chを減らすと重心・微細構造が崩れる=CI符号化の劣化(=正しい挙動)")
    print("=" * 62)


def _rms(x):
    return np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2))


def _centroid(x, sr):
    X = np.abs(np.fft.rfft(x))
    f = np.fft.rfftfreq(len(x), 1 / sr)
    return float((f * X).sum() / (X.sum() + 1e-20))


if __name__ == "__main__":
    _self_test()

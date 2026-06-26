import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d
import soundfile as sf
import librosa
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sounddevice as sd
import os
import tempfile

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

TMPWAV = os.path.join(tempfile.gettempdir(), "hlsim_preview.wav")

SIM_MODES = [
    "なし（原音）",
    "1. 加齢性難聴  Presbycusis",
    "2. 騒音性難聴  NIHL (4kHz notch)",
    "3. 聴覚処理障害  APD",
    "4. メニエール病  Meniere's",
    "5. クッキーバイト型  Cookie-bite",
]

SIM_DESCS = {
    SIM_MODES[0]: "変調なし — 原音をそのまま再合成",
    SIM_MODES[1]: "1kHz以上の高域が徐々に減衰 | ISO 7029準拠モデル",
    SIM_MODES[2]: "4kHz にガウシアンノッチ＋高域損失 | 騒音・コンサート難聴",
    SIM_MODES[3]: "閾値正常・時間分解能と周波数選択性が低下 | 聞こえるが理解できない",
    SIM_MODES[4]: "低音域が 0.4Hz 周期で変動減衰 | 内リンパ水腫モデル",
    SIM_MODES[5]: "中音域（~1kHz）にガウシアン型損失 | U字型・先天性に多い",
}


class HearingLossSimulatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("機能性難聴シミュレータ  —  Hearing Loss Simulator v1")
        self.root.geometry("980x790")

        self.file_path = None
        self.sr_src = None
        self.time_frames = None
        self.vol_matrices_raw = []   # list[(84, T)] — STFT生データ（変更しない）
        self.is_stereo = False

        self.base_midi      = 33
        self.semitone_step  = 0.25                          # 1/4半音刻み
        self.num_notes      = int(84 / self.semitone_step)  # 336ビン
        midi_vals = self.base_midi + np.arange(self.num_notes) * self.semitone_step
        self.note_frequencies = 440.0 * (2.0 ** ((midi_vals - 69) / 12.0))
        self._smooth_sigma  = 0.0                           # 時間軸平滑化 sigma

        self._create_widgets()

    # ═══════════════════════════════════════════════════ UI ═══
    def _create_widgets(self):
        # 1. ファイル
        f1 = tk.LabelFrame(self.root, text="1. 音声ファイル", padx=10, pady=8)
        f1.pack(fill="x", padx=15, pady=(8, 3))
        tk.Button(f1, text="ファイルを選択", command=self._load_file).pack(side="left")
        self.lbl_file = tk.Label(f1, text="未選択", fg="gray")
        self.lbl_file.pack(side="left", padx=10)
        gpu_txt = "GPU(CuPy)✓" if HAS_CUPY else "CPU  ※GPU化: pip install cupy-cuda12x"
        tk.Label(f1, text=gpu_txt, fg="green" if HAS_CUPY else "gray").pack(side="right")

        # 2. 波形 & 実行
        f2 = tk.LabelFrame(self.root, text="2. 波形設定 & 実行", padx=10, pady=8)
        f2.pack(fill="x", padx=15, pady=3)

        tk.Label(f2, text="波形:").pack(side="left")
        self.wave_type = tk.StringVar(value="サイン波")
        ttk.Combobox(
            f2, textvariable=self.wave_type,
            values=["サイン波", "三角波", "鋸歯状波", "矩形波"],
            state="readonly", width=10,
        ).pack(side="left", padx=(2, 12))

        self.btn_process = tk.Button(
            f2, text="分析（STFT）", command=self._start_analysis_thread,
            state="disabled", bg="#d0f0c0",
        )
        self.btn_process.pack(side="left")

        tk.Frame(f2, width=10).pack(side="left")

        self.btn_play = tk.Button(
            f2, text="▶ 再合成して再生", command=self._start_play_thread,
            state="disabled", bg="#c0d8f0",
        )
        self.btn_play.pack(side="left", padx=2)

        self.btn_stop = tk.Button(f2, text="■ 停止", command=self._stop, state="disabled")
        self.btn_stop.pack(side="left", padx=2)

        tk.Frame(f2, width=10).pack(side="left")

        self.btn_save = tk.Button(f2, text="保存", command=self._start_save_thread, state="disabled")
        self.btn_save.pack(side="left", padx=2)

        self.btn_csv = tk.Button(f2, text="CSV", command=self._export_csv, state="disabled")
        self.btn_csv.pack(side="left", padx=2)

        # 3. 難聴シミュレーションモード
        f3 = tk.LabelFrame(self.root, text="3. 難聴シミュレーション", padx=10, pady=8)
        f3.pack(fill="x", padx=15, pady=3)

        tk.Label(f3, text="モード:").pack(side="left")
        self.sim_mode = tk.StringVar(value=SIM_MODES[0])
        ttk.Combobox(
            f3, textvariable=self.sim_mode,
            values=SIM_MODES, state="readonly", width=36,
        ).pack(side="left", padx=(4, 10))

        self.sim_desc = tk.Label(f3, text=SIM_DESCS[SIM_MODES[0]], fg="#444", font=("", 8))
        self.sim_desc.pack(side="left")

        # モード変更 → プロット自動更新（STFT済みの場合）
        self.sim_mode.trace_add("write", lambda *_: self._on_sim_change())

        # 4. セル消去エフェクト
        f4 = tk.LabelFrame(
            self.root,
            text="4. セル消去エフェクト  ─  半音×0.1秒升、4方向相関≥3で音高方向に1升ランダム消去",
            padx=10, pady=8,
        )
        f4.pack(fill="x", padx=15, pady=3)

        self.enable_erase = tk.BooleanVar(value=False)
        tk.Checkbutton(f4, text="有効", variable=self.enable_erase).pack(side="left")

        tk.Label(f4, text="相関感度:").pack(side="left", padx=(14, 2))
        self.sensitivity = tk.DoubleVar(value=0.5)
        tk.Scale(
            f4, from_=0.05, to=0.95, resolution=0.05,
            orient="horizontal", variable=self.sensitivity, length=200,
        ).pack(side="left")
        tk.Label(f4, text="← 弱  強 →", fg="gray", font=("", 8)).pack(side="left", padx=4)

        tk.Label(f4, text="繰返:").pack(side="left", padx=(14, 2))
        self.erase_iter = tk.IntVar(value=1)
        tk.Spinbox(f4, from_=1, to=20, textvariable=self.erase_iter, width=4).pack(side="left")

        # プログレス
        fp = tk.Frame(self.root)
        fp.pack(fill="x", padx=15, pady=3)
        self.progress = ttk.Progressbar(fp, orient="horizontal", length=360, mode="determinate")
        self.progress.pack(side="left")
        self.lbl_status = tk.Label(fp, text="", fg="blue")
        self.lbl_status.pack(side="left", padx=10)

        # 5. スペクトログラム
        f5 = tk.LabelFrame(self.root, text="5. スペクトログラム（音高×時間、シミュレーション後）", padx=8, pady=8)
        f5.pack(fill="both", expand=True, padx=15, pady=(3, 10))
        self.fig, self.ax = plt.subplots(figsize=(7, 3.2))
        self.fig.tight_layout(pad=1.5)
        self.canvas = FigureCanvasTkAgg(self.fig, master=f5)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _on_sim_change(self):
        self.sim_desc.config(text=SIM_DESCS.get(self.sim_mode.get(), ""))
        self._refresh_plot()

    # ═══════════════════════════════════════════════════ ファイル ═══
    def _load_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.ogg")]
        )
        if path:
            self.file_path = path
            self.lbl_file.config(text=path.replace("\\", "/").split("/")[-1], fg="black")
            self.btn_process.config(state="normal")

    def _set_status(self, text):
        self.lbl_status.config(text=text)

    # ═══════════════════════════════════════════════════ STFT解析 ═══
    def _start_analysis_thread(self):
        self.btn_process.config(state="disabled")
        for btn in (self.btn_play, self.btn_stop, self.btn_save, self.btn_csv):
            btn.config(state="disabled")
        threading.Thread(target=self._analysis_worker, daemon=True).start()

    def _build_volume_matrix(self, y_mono, sr):
        hop = int(sr * 0.01)
        n_fft = int(2 ** np.ceil(np.log2(sr * 0.04)))
        mag = np.abs(librosa.stft(y_mono, n_fft=n_fft, hop_length=hop, window="hann"))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        tf = librosa.frames_to_time(np.arange(mag.shape[1]), sr=sr, hop_length=hop)

        vm = np.zeros((self.num_notes, mag.shape[1]))
        for i, f0 in enumerate(self.note_frequencies):
            diffs = np.abs(freqs - f0)
            idxs = np.argsort(diffs)[:3]
            w = 1.0 / (diffs[idxs] + 1e-8)
            w /= w.sum()
            vm[i] = (mag[idxs, :] * w[:, None]).sum(axis=0)
        return vm, tf

    def _analysis_worker(self):
        def gui(fn): self.root.after(0, fn)
        try:
            gui(lambda: self.progress.config(value=5))
            gui(lambda: self._set_status("ロード中..."))

            y_raw, sr = librosa.load(self.file_path, sr=None, mono=False)
            self.sr_src = sr
            self.is_stereo = y_raw.ndim == 2
            channels = [y_raw[0], y_raw[1]] if self.is_stereo else [y_raw]

            gui(lambda: self._set_status("STFT解析中..."))
            raw_vms, tf = [], None
            for ci, y_ch in enumerate(channels):
                vm, tf = self._build_volume_matrix(y_ch, sr)
                raw_vms.append(vm)
                v = 10 + 40 * (ci + 1)
                gui(lambda v=v: self.progress.config(value=v))

            self.time_frames = tf
            self.vol_matrices_raw = raw_vms

            gui(lambda: self._refresh_plot())
            gui(lambda: [btn.config(state="normal")
                         for btn in (self.btn_play, self.btn_stop, self.btn_save, self.btn_csv)])
            gui(lambda: self.progress.config(value=100))
            gui(lambda: self._set_status("STFT完了 — モードを選んで▶再生"))

        except Exception as e:
            err = str(e)
            gui(lambda: messagebox.showerror("エラー", err))
            gui(lambda: self.progress.config(value=0))
            gui(lambda: self._set_status("エラー"))
        finally:
            gui(lambda: self.btn_process.config(state="normal"))

    # ═══════════════════════════════════════════════════ 難聴シミュレーション ═══
    def _apply_simulation(self, vol_mat):
        """
        5種類の難聴プロファイルを vol_mat に適用して返す。
        生データ (vol_matrices_raw) は変更しない。
        """
        mode = self.sim_mode.get()
        if mode == SIM_MODES[0]:
            return vol_mat.copy()

        result = vol_mat.copy()
        n_notes, n_frames = result.shape
        hz = self.note_frequencies   # shape (84,)

        def atten_db_to_gain(db_arr):
            return 10.0 ** (-np.abs(np.asarray(db_arr, dtype=np.float64)) / 20.0)

        # ── 1. 加齢性難聴 Presbycusis ─────────────────────────────
        if "Presbycusis" in mode:
            # ISO 7029 典型値に基づくスロープ
            breakpoints = np.array([125, 250, 500, 1000, 2000, 4000, 8000, 16000])
            losses_db   = np.array([  0,   0,   5,   10,   25,   50,   65,    75])
            for i, f in enumerate(hz):
                db = float(np.interp(f, breakpoints, losses_db))
                result[i] *= atten_db_to_gain(db)

        # ── 2. 騒音性難聴 NIHL ────────────────────────────────────
        elif "NIHL" in mode:
            notch_hz = 4000.0
            for i, f in enumerate(hz):
                st_dist  = 12 * np.log2(max(f, 1.0) / notch_hz)
                notch_db = 55 * np.exp(-0.5 * (st_dist / 3.5) ** 2)
                hf_db    = float(np.interp(f, [1500, 8000], [0, 20])) if f > 1500 else 0.0
                result[i] *= atten_db_to_gain(max(notch_db, hf_db))

        # ── 3. 聴覚処理障害 APD ───────────────────────────────────
        elif "APD" in mode:
            # 時間ギャップ: ランダムな 20〜30ms の脱落
            rng = np.random.default_rng(seed=42)
            n_gaps = max(1, n_frames // 12)
            gap_centers = rng.choice(n_frames, size=n_gaps, replace=False)
            for gc in gap_centers:
                t0, t1 = max(0, gc - 1), min(n_frames, gc + 3)
                result[:, t0:t1] *= 0.07
            # 周波数選択性低下: 隣接ビンへのスメア
            result = gaussian_filter1d(result, sigma=4.8, axis=0)  # 1/4半音ビン換算

        # ── 4. メニエール病 Meniere's ─────────────────────────────
        elif "Meniere" in mode:
            hop   = int(self.sr_src * 0.01)
            t_sec = np.arange(n_frames) * hop / self.sr_src
            # 0.4Hz 変動（秒オーダーで揺れる低音損失）
            fluc  = 0.5 + 0.5 * np.sin(2 * np.pi * 0.4 * t_sec)   # (n_frames,) 0→1

            breakpoints_hz = np.array([62,  125, 250, 500, 1000])
            base_db_vals   = np.array([45,   40,  30,  15,     0], dtype=float)
            fluc_db_vals   = np.array([25,   22,  15,   6,     0], dtype=float)

            for i, f in enumerate(hz):
                base_db = float(np.interp(f, breakpoints_hz, base_db_vals))
                fluc_db = float(np.interp(f, breakpoints_hz, fluc_db_vals))
                atten   = base_db + fluc_db * fluc          # (n_frames,)
                result[i] *= 10.0 ** (-atten / 20.0)

        # ── 5. クッキーバイト型 Cookie-bite ──────────────────────
        elif "Cookie" in mode:
            mid_hz = 1000.0
            for i, f in enumerate(hz):
                st_dist = 12 * np.log2(max(f, 1.0) / mid_hz)
                db      = 50 * np.exp(-0.5 * (st_dist / 8.0) ** 2)
                result[i] *= atten_db_to_gain(db)

        return result

    # ═══════════════════════════════════════════════════ セル消去エフェクト ═══
    def _apply_erase_effect(self, vol_mat):
        n_notes, n_frames = vol_mat.shape
        bsize    = 10  # 0.1s @ 10ms hop
        n_tblk   = n_frames // bsize
        threshold = 1.0 - self.sensitivity.get()
        result   = vol_mat.copy()

        for _ in range(self.erase_iter.get()):
            ba = result[:, :n_tblk * bsize].reshape(n_notes, n_tblk, bsize).mean(axis=2)

            def _corr(a, b):
                ma = np.maximum(a, b)
                return np.where(ma > 1e-8, np.minimum(a, b) / (ma + 1e-8), 0.0) >= threshold

            c_up    = np.zeros((n_notes, n_tblk), dtype=bool)
            c_down  = np.zeros((n_notes, n_tblk), dtype=bool)
            c_left  = np.zeros((n_notes, n_tblk), dtype=bool)
            c_right = np.zeros((n_notes, n_tblk), dtype=bool)
            c_up[:-1, :]    = _corr(ba[:-1, :], ba[1:, :])
            c_down[1:, :]   = _corr(ba[1:, :],  ba[:-1, :])
            c_left[:, 1:]   = _corr(ba[:, 1:],  ba[:, :-1])
            c_right[:, :-1] = _corr(ba[:, :-1], ba[:, 1:])

            total = (c_up.astype(np.int8) + c_down.astype(np.int8)
                     + c_left.astype(np.int8) + c_right.astype(np.int8))

            erase = np.zeros((n_notes, n_frames), dtype=bool)
            for ni, ti in np.argwhere((total >= 3) & (ba > 1e-8)):
                tn = int(ni) + np.random.choice([-1, 1])
                if 0 <= tn < n_notes:
                    ts, te = int(ti) * bsize, min((int(ti) + 1) * bsize, n_frames)
                    erase[tn, ts:te] = True
            result[erase] = 0.0

        return result

    def _get_processed_matrices(self):
        """raw → simulation → erase(optional) → temporal smoothing の順に適用"""
        mats = [self._apply_simulation(vm) for vm in self.vol_matrices_raw]
        if self.enable_erase.get():
            mats = [self._apply_erase_effect(vm) for vm in mats]
        # 時間軸平滑化: sigma>0 のときだけ適用（0のとき skip）
        sigma = self._smooth_sigma
        if sigma > 0.0:
            mats = [gaussian_filter1d(m, sigma=sigma, axis=1) for m in mats]
        return mats

    # ═══════════════════════════════════════════════════ 合成 ═══
    def _synthesize(self, vol_mat, sr):
        if HAS_CUPY:
            try:
                return self._synth_gpu(vol_mat, sr)
            except Exception:
                pass
        return self._synth_cpu(vol_mat, sr)

    def _synth_gpu(self, vol_mat, sr):
        """チャンク分割GPU合成 — 1344ビン×長尺でもVRAM溢れしない"""
        hop         = int(sr * 0.01)
        n_T         = vol_mat.shape[1]
        num_samples = int((self.time_frames[-1] + 0.01) * sr)
        CHUNK       = int(sr * 6)   # 6秒チャンク（VRAM ~4GB/chunk）
        wt          = self.wave_type.get()

        # vol_mat と freqs はチャンク間で共有 → VRAMに常駐
        vm_gpu    = cp.array(vol_mat, dtype=cp.float32)
        freqs_gpu = cp.array(self.note_frequencies, dtype=cp.float32)
        # 全オシレータにランダム初期位相を付与 → t=0の全波同位相フランジング防止
        rng_phase = np.random.default_rng()
        phase_offsets = cp.array(
            rng_phase.uniform(0.0, 2.0 * np.pi, len(self.note_frequencies)),
            dtype=cp.float32,
        )

        chunks = []
        for c_start in range(0, num_samples, CHUNK):
            c_end = min(c_start + CHUNK, num_samples)

            t_gpu  = cp.arange(c_start, c_end, dtype=cp.float32) / sr
            t_idx  = t_gpu * (sr / hop)
            idx_lo = cp.clip(cp.floor(t_idx).astype(cp.int32), 0, n_T - 1)
            idx_hi = cp.clip(idx_lo + 1, 0, n_T - 1)
            frac   = (t_idx - idx_lo).astype(cp.float32)
            amps   = vm_gpu[:, idx_lo] * (1.0 - frac) + vm_gpu[:, idx_hi] * frac

            phase = (2.0 * np.pi * freqs_gpu[:, None] * t_gpu[None, :]
                     + phase_offsets[:, None]).astype(cp.float32)

            if wt == "サイン波":
                waves = cp.sin(phase)
            elif wt == "三角波":
                waves = (2.0 / np.pi) * cp.arcsin(cp.clip(cp.sin(phase), -1.0, 1.0))
            elif wt == "鋸歯状波":
                waves = (phase / np.pi) % 2.0 - 1.0
            else:
                waves = cp.sign(cp.sin(phase))

            chunk_out = (waves * amps).sum(axis=0)
            chunks.append(cp.asnumpy(chunk_out).astype(np.float32))

            del t_gpu, t_idx, idx_lo, idx_hi, frac, amps, phase, waves, chunk_out
            cp.get_default_memory_pool().free_all_blocks()

        del vm_gpu, freqs_gpu
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

        out  = np.concatenate(chunks)
        peak = np.max(np.abs(out))
        if peak > 0:
            out /= peak
        return out

    def _synth_cpu(self, vol_mat, sr):
        num_samples = int((self.time_frames[-1] + 0.01) * sr)
        t   = np.arange(num_samples, dtype=np.float64) / sr
        out = np.zeros(num_samples)
        wt  = self.wave_type.get()
        rng_phase = np.random.default_rng()
        phase_offsets = rng_phase.uniform(0.0, 2.0 * np.pi, len(self.note_frequencies))
        for i, freq in enumerate(self.note_frequencies):
            amp   = np.interp(t, self.time_frames, vol_mat[i])
            theta = 2 * np.pi * freq * t + phase_offsets[i]
            if wt == "サイン波":
                w = np.sin(theta)
            elif wt == "三角波":
                w = signal.sawtooth(theta, 0.5)
            elif wt == "鋸歯状波":
                w = signal.sawtooth(theta, 1.0)
            else:
                w = signal.square(theta)
            out += w * amp
        peak = np.max(np.abs(out))
        if peak > 0:
            out /= peak
        return out.astype(np.float32)

    def _build_output(self, vol_matrices):
        """ステレオ入力→(N,2)、モノ入力→(N,)"""
        sigs = [self._synthesize(vm, self.sr_src) for vm in vol_matrices]
        if len(sigs) == 2:
            n = min(len(s) for s in sigs)
            return np.stack([sigs[0][:n], sigs[1][:n]], axis=1)
        return sigs[0]

    # ═══════════════════════════════════════════════════ 再生 ═══
    def _start_play_thread(self):
        if not self.vol_matrices_raw:
            return
        self.btn_play.config(state="disabled")
        threading.Thread(target=self._play_worker, daemon=True).start()

    def _play_worker(self):
        def gui(fn): self.root.after(0, fn)
        try:
            engine = "GPU" if HAS_CUPY else "CPU"
            gui(lambda: self._set_status(f"再合成中({engine})..."))
            gui(lambda: self.progress.config(value=20))

            audio = self._build_output(self._get_processed_matrices())

            gui(lambda: self.progress.config(value=85))
            gui(lambda: self._set_status("WAV書き出し中..."))
            sf.write(TMPWAV, audio, self.sr_src, subtype="PCM_16")

            mode_short = self.sim_mode.get().split("  ")[0]
            wt = self.wave_type.get()
            gui(lambda: self.progress.config(value=100))
            gui(lambda: self._set_status(f"再生中 [{wt} / {mode_short}]"))
            sd.play(audio, self.sr_src)

        except Exception as e:
            err = str(e)
            gui(lambda: messagebox.showerror("再生エラー", err))
            gui(lambda: self._set_status("エラー"))
        finally:
            gui(lambda: self.btn_play.config(state="normal"))

    def _stop(self):
        sd.stop()
        self._set_status("停止")

    # ═══════════════════════════════════════════════════ 保存 ═══
    def _start_save_thread(self):
        if not self.vol_matrices_raw:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".wav", filetypes=[("WAV files", "*.wav")]
        )
        if not path:
            return
        self.btn_save.config(state="disabled")
        threading.Thread(target=self._save_worker, args=(path,), daemon=True).start()

    def _save_worker(self, path):
        def gui(fn): self.root.after(0, fn)
        try:
            gui(lambda: self._set_status("保存用再合成中..."))
            audio = self._build_output(self._get_processed_matrices())
            sf.write(path, audio, self.sr_src, subtype="PCM_16")
            gui(lambda: self._set_status("保存完了"))
            gui(lambda: messagebox.showinfo("保存完了", path))
        except Exception as e:
            err = str(e)
            gui(lambda: messagebox.showerror("保存エラー", err))
        finally:
            gui(lambda: self.btn_save.config(state="normal"))

    # ═══════════════════════════════════════════════════ CSV ═══
    def _export_csv(self):
        if not self.vol_matrices_raw:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV files", "*.csv")]
        )
        if not path:
            return
        vm = self._get_processed_matrices()[0]
        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["MIDI_note"] + [f"{t:.3f}s" for t in self.time_frames])
            for i in range(self.num_notes):
                w.writerow([self.base_midi + i * self.semitone_step] + list(vm[i]))
        messagebox.showinfo("CSV出力完了", path)

    # ═══════════════════════════════════════════════════ プロット ═══
    def _refresh_plot(self):
        if not self.vol_matrices_raw:
            return
        # プロットにはシミュレーションのみ適用（消去なし・安定表示）
        vm = self._apply_simulation(self.vol_matrices_raw[0])
        self.ax.clear()
        data   = 20 * np.log10(vm + 1e-6)
        extent = [self.time_frames[0], self.time_frames[-1],
                  self.base_midi, self.base_midi + self.num_notes]
        self.ax.imshow(data, aspect="auto", origin="lower", extent=extent, cmap="magma")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("MIDI Note (336 bins / 1/4st / 7oct)")
        mode_short = self.sim_mode.get().split("  ")[0]
        title = mode_short
        if self.is_stereo:
            title += " [Lch]"
        self.ax.set_title(title)
        self.fig.tight_layout(pad=1.5)
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = HearingLossSimulatorApp(root)
    root.mainloop()

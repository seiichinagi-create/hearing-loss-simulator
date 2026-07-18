"""
app_cochlea.py — v2 蝸牛/人工内耳シミュレータ GUI(tkinter)

v1(app.py・STFT+サイン格子)は非破壊で残置。こちらは cochlea_engine の
ガンマトーン+ボコーダを使う科学版フロントエンド。

できること:
  ・音声ファイル or 診断プローブ信号を入力
  ・病態プリセット(健聴/加齢性/騒音性/クッキーバイト/重度SNHL/人工内耳4-16ch)で
    「その人に聞こえている音」を再合成
  ・原音(A) ⇄ シミュ(B) のA/B試聴
  ・補正: 周波数移動トグル(デッド領域用)+補正可能性レジストリを画面に明示
  ・シミュ出力のスペクトログラム表示・WAV保存

処理は別スレッド(重い機械でもUIが固まらない)。
依存: numpy scipy soundfile sounddevice librosa matplotlib + cochlea_engine/probes/correction
"""
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import soundfile as sf
import sounddevice as sd
import librosa
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from cochlea_engine import CochleaVocoder, preset, PRESET_NAMES
from probes import PROBES
try:
    from correction import freq_lower, CORRECTABILITY
    HAS_CORRECTION = True
except Exception:
    HAS_CORRECTION = False
    CORRECTABILITY = {}


# ── 純ロジック(GUIなしでもテスト可能) ──────────────────────────────
def simulate(audio, sr, preset_name, freq_lowering=False):
    """audio(mono float) → 病態プリセットで知覚される音を再合成。"""
    x = audio
    if freq_lowering and HAS_CORRECTION:
        x = freq_lower(x, sr, knee_hz=1800, factor=0.5)
    return CochleaVocoder(sr, profile=preset(preset_name)).process(x)


def _registry_text():
    if not CORRECTABILITY:
        return "correction.py 未読込"
    lines = []
    for dim, (verdict, _why) in CORRECTABILITY.items():
        lines.append(f"{verdict}  {dim}")
    return "   |   ".join(lines)


class CochleaSimApp:
    def __init__(self, root):
        self.root = root
        root.title("蝸牛/人工内耳シミュレータ v2  —  Cochlear & CI Simulator")
        root.geometry("1000x820")

        self.audio = None          # mono float
        self.sr = None
        self.src_name = "未選択"
        self.sim_cache = None       # 直近シミュ出力(保存用)

        self._build()

    # ─────────────────────────────────────────────────── UI 構築 ──
    def _build(self):
        # 1. 入力(ファイル / プローブ)
        f1 = tk.LabelFrame(self.root, text="1. 入力", padx=10, pady=8)
        f1.pack(fill="x", padx=15, pady=(8, 3))
        tk.Button(f1, text="音声ファイルを選択", command=self._load_file).pack(side="left")
        tk.Label(f1, text="  または診断プローブ:").pack(side="left")
        self.probe_var = tk.StringVar(value="(選択)")
        ttk.Combobox(f1, textvariable=self.probe_var, values=list(PROBES.keys()),
                     state="readonly", width=22).pack(side="left", padx=4)
        tk.Button(f1, text="プローブ生成", command=self._load_probe).pack(side="left", padx=2)
        self.lbl_src = tk.Label(f1, text="未選択", fg="gray")
        self.lbl_src.pack(side="left", padx=10)

        # 2. 病態プリセット
        f2 = tk.LabelFrame(self.root, text="2. 病態プリセット", padx=10, pady=8)
        f2.pack(fill="x", padx=15, pady=3)
        tk.Label(f2, text="モード:").pack(side="left")
        self.preset_var = tk.StringVar(value=PRESET_NAMES[0])
        ttk.Combobox(f2, textvariable=self.preset_var, values=PRESET_NAMES,
                     state="readonly", width=28).pack(side="left", padx=(4, 12))

        # 3. 補正(限界を画面に明示)
        f3 = tk.LabelFrame(self.root, text="3. 補正(できること/できないことを明示)", padx=10, pady=8)
        f3.pack(fill="x", padx=15, pady=3)
        self.freq_low = tk.BooleanVar(value=False)
        tk.Checkbutton(f3, text="周波数移動(デッド領域を生存帯へ転写)",
                       variable=self.freq_low).pack(side="left")
        tk.Label(f3, text=_registry_text(), fg="#555", font=("", 8)).pack(side="left", padx=10)

        # 4. 試聴(A/B)
        f4 = tk.LabelFrame(self.root, text="4. 試聴  A=原音 / B=シミュ", padx=10, pady=8)
        f4.pack(fill="x", padx=15, pady=3)
        self.btn_a = tk.Button(f4, text="▶ A 原音", command=self._play_original,
                               state="disabled", bg="#c0d8f0")
        self.btn_a.pack(side="left", padx=2)
        self.btn_b = tk.Button(f4, text="▶ B シミュ再合成", command=self._play_sim,
                               state="disabled", bg="#d0f0c0")
        self.btn_b.pack(side="left", padx=2)
        tk.Button(f4, text="■ 停止", command=self._stop).pack(side="left", padx=2)
        tk.Frame(f4, width=16).pack(side="left")
        self.btn_save = tk.Button(f4, text="シミュをWAV保存", command=self._save,
                                  state="disabled")
        self.btn_save.pack(side="left", padx=2)

        # 進捗
        fp = tk.Frame(self.root)
        fp.pack(fill="x", padx=15, pady=3)
        self.progress = ttk.Progressbar(fp, orient="horizontal", length=340, mode="indeterminate")
        self.progress.pack(side="left")
        self.lbl_status = tk.Label(fp, text="", fg="blue")
        self.lbl_status.pack(side="left", padx=10)

        # 5. スペクトログラム(シミュ出力)
        f5 = tk.LabelFrame(self.root, text="5. シミュ出力スペクトログラム", padx=8, pady=8)
        f5.pack(fill="both", expand=True, padx=15, pady=(3, 10))
        self.fig, self.ax = plt.subplots(figsize=(7, 3.2))
        self.fig.tight_layout(pad=1.5)
        self.canvas = FigureCanvasTkAgg(self.fig, master=f5)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # ─────────────────────────────────────────────────── 入力 ──
    def _set_audio(self, y, sr, name):
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = y.astype(float)
        p = np.abs(y).max()
        if p > 0:
            y = y / p
        self.audio, self.sr, self.src_name = y, sr, name
        self.lbl_src.config(text=f"{name}  ({sr}Hz, {len(y)/sr:.1f}s)", fg="black")
        for b in (self.btn_a, self.btn_b, self.btn_save):
            b.config(state="normal")

    def _load_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Audio", "*.wav *.mp3 *.flac *.ogg")])
        if not path:
            return
        try:
            y, sr = librosa.load(path, sr=None, mono=False)
            if y.ndim > 1:
                y = y.T if y.shape[0] < y.shape[1] else y  # librosaは(ch,N)
                y = y.mean(axis=0) if y.ndim > 1 else y
            self._set_audio(np.atleast_1d(y), sr, os.path.basename(path))
        except Exception as e:
            messagebox.showerror("読込エラー", str(e))

    def _load_probe(self):
        name = self.probe_var.get()
        if name not in PROBES:
            messagebox.showinfo("プローブ", "プローブを選択してください")
            return
        try:
            y = PROBES[name]()
            self._set_audio(y, 44100, f"probe:{name}")
        except Exception as e:
            messagebox.showerror("プローブ生成エラー", str(e))

    # ─────────────────────────────────────────────────── 処理 ──
    def _busy(self, on, msg=""):
        self.lbl_status.config(text=msg)
        if on:
            self.progress.start(12)
        else:
            self.progress.stop()

    def _play_original(self):
        if self.audio is None:
            return
        sd.stop()
        self._busy(False, f"再生: 原音 [{self.src_name}]")
        sd.play(self.audio, self.sr)

    def _play_sim(self):
        if self.audio is None:
            return
        self.btn_b.config(state="disabled")
        self._busy(True, "シミュ再合成中(重い機械では時間がかかります)...")
        threading.Thread(target=self._sim_worker, args=(True,), daemon=True).start()

    def _sim_worker(self, play):
        def gui(fn):
            self.root.after(0, fn)
        try:
            y = simulate(self.audio, self.sr, self.preset_var.get(), self.freq_low.get())
            self.sim_cache = (y, self.sr)
            gui(lambda: self._refresh_plot(y))
            if play:
                gui(lambda: self._busy(False, f"再生: シミュ [{self.preset_var.get()}]"))
                gui(lambda: sd.play(y, self.sr))
            else:
                gui(lambda: self._busy(False, "シミュ完了"))
        except Exception as e:
            err = str(e)
            gui(lambda: self._busy(False, "エラー"))
            gui(lambda: messagebox.showerror("シミュエラー", err))
        finally:
            gui(lambda: self.btn_b.config(state="normal"))

    def _stop(self):
        sd.stop()
        self._busy(False, "停止")

    # ─────────────────────────────────────────────────── 保存 ──
    def _save(self):
        if self.audio is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".wav", filetypes=[("WAV", "*.wav")])
        if not path:
            return
        self._busy(True, "保存用シミュ再合成中...")
        threading.Thread(target=self._save_worker, args=(path,), daemon=True).start()

    def _save_worker(self, path):
        def gui(fn):
            self.root.after(0, fn)
        try:
            y = simulate(self.audio, self.sr, self.preset_var.get(), self.freq_low.get())
            sf.write(path, y, self.sr, subtype="PCM_16")
            gui(lambda: self._busy(False, "保存完了"))
            gui(lambda: messagebox.showinfo("保存完了", path))
        except Exception as e:
            err = str(e)
            gui(lambda: self._busy(False, "エラー"))
            gui(lambda: messagebox.showerror("保存エラー", err))

    # ─────────────────────────────────────────────────── 描画 ──
    def _refresh_plot(self, y):
        self.ax.clear()
        S = np.abs(librosa.stft(y.astype(float), n_fft=2048, hop_length=512))
        SdB = librosa.amplitude_to_db(S + 1e-9, ref=np.max)
        f = librosa.fft_frequencies(sr=self.sr, n_fft=2048)
        t = librosa.frames_to_time(np.arange(S.shape[1]), sr=self.sr, hop_length=512)
        self.ax.pcolormesh(t, f, SdB, cmap="magma", shading="auto")
        self.ax.set_ylim(0, min(8000, self.sr / 2))
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Hz")
        self.ax.set_title(f"シミュ出力: {self.preset_var.get()}")
        self.fig.tight_layout(pad=1.5)
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = CochleaSimApp(root)
    root.mainloop()

"""
app_cochlea.py — v2 蝸牛/人工内耳シミュレータ GUI(tkinter, EN/JA 切替・既定EN)

v1(app.py・STFT+サイン格子)は非破壊で残置。こちらは cochlea_engine の
ガンマトーン+ボコーダを使う科学版フロントエンド。

言語は右上で English / 日本語 を切替(既定=English)。切替時はUIを再構築するが、
読込済み音声・選択プリセット・チェック状態は保持される。
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
import matplotlib
# matplotlib既定フォントは日本語グリフ非対応(グラフ内が豆腐化)→ Windows標準の和文フォントへ
matplotlib.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from cochlea_engine import CochleaVocoder, preset, PRESET_NAMES
from probes import PROBES
try:
    from correction import freq_lower
    HAS_CORRECTION = True
except Exception:
    HAS_CORRECTION = False


# ── 翻訳テーブル ───────────────────────────────────────────────────
STR = {
    "en": {
        "title": "Cochlear & CI Simulator v2",
        "lang": "Language:",
        "sec_input": "1. Input",
        "btn_loadfile": "Load audio file",
        "or_probe": "  or diagnostic probe:",
        "btn_genprobe": "Generate probe",
        "src_none": "none",
        "sec_preset": "2. Pathology preset",
        "lbl_mode": "Mode:",
        "sec_correction": "3. Correction (what can / cannot be fixed — made explicit)",
        "chk_freqlower": "Frequency lowering (transpose dead region to surviving band)",
        "registry": "◎ correctable: threshold / recruitment    △ remap only: dead region    "
                    "✗ not correctable: selectivity / TFS",
        "sec_listen": "4. Listen   A = original / B = simulated",
        "btn_a": "▶ A  Original",
        "btn_b": "▶ B  Simulated",
        "btn_stop": "■ Stop",
        "btn_save": "Save simulated WAV",
        "sec_spec": "5. Simulated-output spectrogram",
        "st_play_a": "Playing: original [{name}]",
        "st_play_b": "Playing: simulated [{name}]",
        "st_sim": "Resynthesizing (may take time on a loaded machine)...",
        "st_sim_done": "Simulation done",
        "st_saving": "Resynthesizing for save...",
        "st_saved": "Saved",
        "st_stopped": "Stopped",
        "st_error": "Error",
        "spec_title": "Simulated output: {name}",
        "err_load": "Load error",
        "err_probe": "Probe generation error",
        "err_sim": "Simulation error",
        "err_save": "Save error",
        "info_saved": "Saved",
        "info_probe_sel": "Please select a probe",
    },
    "ja": {
        "title": "蝸牛/人工内耳シミュレータ v2",
        "lang": "言語:",
        "sec_input": "1. 入力",
        "btn_loadfile": "音声ファイルを選択",
        "or_probe": "  または診断プローブ:",
        "btn_genprobe": "プローブ生成",
        "src_none": "未選択",
        "sec_preset": "2. 病態プリセット",
        "lbl_mode": "モード:",
        "sec_correction": "3. 補正(できること/できないことを明示)",
        "chk_freqlower": "周波数移動(デッド領域を生存帯へ転写)",
        "registry": "◎補正可能: 閾値/補充現象    △再マッピングのみ: デッド領域    "
                    "✗補正不可: 選択性/TFS",
        "sec_listen": "4. 試聴   A=原音 / B=シミュ",
        "btn_a": "▶ A  原音",
        "btn_b": "▶ B  シミュ再合成",
        "btn_stop": "■ 停止",
        "btn_save": "シミュをWAV保存",
        "sec_spec": "5. シミュ出力スペクトログラム",
        "st_play_a": "再生: 原音 [{name}]",
        "st_play_b": "再生: シミュ [{name}]",
        "st_sim": "シミュ再合成中(重い機械では時間がかかります)...",
        "st_sim_done": "シミュ完了",
        "st_saving": "保存用シミュ再合成中...",
        "st_saved": "保存完了",
        "st_stopped": "停止",
        "st_error": "エラー",
        "spec_title": "シミュ出力: {name}",
        "err_load": "読込エラー",
        "err_probe": "プローブ生成エラー",
        "err_sim": "シミュエラー",
        "err_save": "保存エラー",
        "info_saved": "保存完了",
        "info_probe_sel": "プローブを選択してください",
    },
}

# プリセット内部キー(preset()呼び出し用) → 表示ラベル(EN/JA)
PRESET_LABELS = {
    "健聴 (Normal)": {"en": "Normal hearing", "ja": "健聴 (Normal)"},
    "加齢性難聴 (Presbycusis)": {"en": "Presbycusis (age-related)", "ja": "加齢性難聴 (Presbycusis)"},
    "騒音性難聴 (NIHL 4kHz notch)": {"en": "NIHL (4 kHz notch)", "ja": "騒音性難聴 (NIHL 4kHz notch)"},
    "クッキーバイト (Cookie-bite)": {"en": "Cookie-bite (mid-freq)", "ja": "クッキーバイト (Cookie-bite)"},
    "重度感音難聴+デッド領域": {"en": "Severe SNHL + dead region", "ja": "重度感音難聴+デッド領域"},
    "人工内耳 16ch": {"en": "Cochlear implant 16 ch", "ja": "人工内耳 16ch"},
    "人工内耳 8ch": {"en": "Cochlear implant 8 ch", "ja": "人工内耳 8ch"},
    "人工内耳 4ch (重度)": {"en": "Cochlear implant 4 ch (severe)", "ja": "人工内耳 4ch (重度)"},
}


# ── 純ロジック(GUIなしでもテスト可能) ──────────────────────────────
def simulate(audio, sr, preset_key, freq_lowering=False):
    """audio(mono float) → 病態プリセットで知覚される音を再合成。preset_keyは内部キー。"""
    x = audio
    if freq_lowering and HAS_CORRECTION:
        x = freq_lower(x, sr, knee_hz=1800, factor=0.5)
    return CochleaVocoder(sr, profile=preset(preset_key)).process(x)


class CochleaSimApp:
    def __init__(self, root, lang="en"):
        self.root = root
        self.lang = lang if lang in STR else "en"

        # 言語切替をまたいで保持する状態
        self.audio = None
        self.sr = None
        self.src_name = None
        self._src_info = None          # (name, sr, nsamples) or None
        self.sim_cache = None
        self.preset_key = PRESET_NAMES[0]
        self.probe_var = tk.StringVar(value="(select)")
        self.freq_low = tk.BooleanVar(value=False)

        self._build()

    def t(self, key, **fmt):
        s = STR[self.lang][key]
        return s.format(**fmt) if fmt else s

    def _plabel(self, key):
        return PRESET_LABELS.get(key, {}).get(self.lang, key)

    # ─────────────────────────────────────────────────── UI 構築 ──
    def _build(self):
        self.root.title(self.t("title"))
        self.root.geometry("1010x830")

        # 言語切替(右上)
        top = tk.Frame(self.root)
        top.pack(fill="x", padx=15, pady=(6, 0))
        tk.Label(top, text=self.t("lang")).pack(side="right")
        self.lang_var = tk.StringVar(value=("English" if self.lang == "en" else "日本語"))
        cb = ttk.Combobox(top, textvariable=self.lang_var, values=["English", "日本語"],
                          state="readonly", width=9)
        cb.pack(side="right", padx=4)
        cb.bind("<<ComboboxSelected>>", self._on_lang)

        # 1. 入力
        f1 = tk.LabelFrame(self.root, text=self.t("sec_input"), padx=10, pady=8)
        f1.pack(fill="x", padx=15, pady=(6, 3))
        tk.Button(f1, text=self.t("btn_loadfile"), command=self._load_file).pack(side="left")
        tk.Label(f1, text=self.t("or_probe")).pack(side="left")
        ttk.Combobox(f1, textvariable=self.probe_var, values=list(PROBES.keys()),
                     state="readonly", width=22).pack(side="left", padx=4)
        tk.Button(f1, text=self.t("btn_genprobe"), command=self._load_probe).pack(side="left", padx=2)
        self.lbl_src = tk.Label(f1, text="", fg="black")
        self.lbl_src.pack(side="left", padx=10)
        self._render_src()

        # 2. 病態プリセット
        f2 = tk.LabelFrame(self.root, text=self.t("sec_preset"), padx=10, pady=8)
        f2.pack(fill="x", padx=15, pady=3)
        tk.Label(f2, text=self.t("lbl_mode")).pack(side="left")
        self.preset_display = tk.StringVar(value=self._plabel(self.preset_key))
        values = [self._plabel(k) for k in PRESET_NAMES]
        pcb = ttk.Combobox(f2, textvariable=self.preset_display, values=values,
                           state="readonly", width=30)
        pcb.pack(side="left", padx=(4, 12))
        pcb.bind("<<ComboboxSelected>>", self._on_preset)

        # 3. 補正(限界を明示)
        f3 = tk.LabelFrame(self.root, text=self.t("sec_correction"), padx=10, pady=8)
        f3.pack(fill="x", padx=15, pady=3)
        tk.Checkbutton(f3, text=self.t("chk_freqlower"), variable=self.freq_low).pack(side="left")
        tk.Label(f3, text=self.t("registry"), fg="#555", font=("", 8)).pack(side="left", padx=10)

        # 4. 試聴
        f4 = tk.LabelFrame(self.root, text=self.t("sec_listen"), padx=10, pady=8)
        f4.pack(fill="x", padx=15, pady=3)
        self.btn_a = tk.Button(f4, text=self.t("btn_a"), command=self._play_original,
                               bg="#c0d8f0")
        self.btn_a.pack(side="left", padx=2)
        self.btn_b = tk.Button(f4, text=self.t("btn_b"), command=self._play_sim, bg="#d0f0c0")
        self.btn_b.pack(side="left", padx=2)
        tk.Button(f4, text=self.t("btn_stop"), command=self._stop).pack(side="left", padx=2)
        tk.Frame(f4, width=16).pack(side="left")
        self.btn_save = tk.Button(f4, text=self.t("btn_save"), command=self._save)
        self.btn_save.pack(side="left", padx=2)
        self._apply_enabled()

        # 進捗
        fp = tk.Frame(self.root)
        fp.pack(fill="x", padx=15, pady=3)
        self.progress = ttk.Progressbar(fp, orient="horizontal", length=340, mode="indeterminate")
        self.progress.pack(side="left")
        self.lbl_status = tk.Label(fp, text="", fg="blue")
        self.lbl_status.pack(side="left", padx=10)

        # 5. スペクトログラム
        f5 = tk.LabelFrame(self.root, text=self.t("sec_spec"), padx=8, pady=8)
        f5.pack(fill="both", expand=True, padx=15, pady=(3, 10))
        self.fig, self.ax = plt.subplots(figsize=(7, 3.2))
        self.fig.tight_layout(pad=1.5)
        self.canvas = FigureCanvasTkAgg(self.fig, master=f5)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        if self.sim_cache is not None:
            self._refresh_plot(self.sim_cache[0])

    # ─────────────────────────────────────────────────── 言語切替 ──
    def _on_lang(self, _evt=None):
        self.lang = "en" if self.lang_var.get() == "English" else "ja"
        for w in list(self.root.children.values()):
            w.destroy()
        self._build()

    def _on_preset(self, _evt=None):
        label = self.preset_display.get()
        for k in PRESET_NAMES:
            if self._plabel(k) == label:
                self.preset_key = k
                break

    def _apply_enabled(self):
        state = "normal" if self.audio is not None else "disabled"
        for b in (self.btn_a, self.btn_b, self.btn_save):
            b.config(state=state)

    def _render_src(self):
        if self._src_info is None:
            self.lbl_src.config(text=self.t("src_none"), fg="gray")
        else:
            name, sr, n = self._src_info
            self.lbl_src.config(text=f"{name}  ({sr}Hz, {n/sr:.1f}s)", fg="black")

    # ─────────────────────────────────────────────────── 入力 ──
    def _set_audio(self, y, sr, name):
        if y.ndim > 1:
            y = y.mean(axis=0)
        y = np.atleast_1d(y).astype(float)
        p = np.abs(y).max()
        if p > 0:
            y = y / p
        self.audio, self.sr, self.src_name = y, sr, name
        self._src_info = (name, sr, len(y))
        self._render_src()
        self._apply_enabled()

    def _load_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Audio", "*.wav *.mp3 *.flac *.ogg")])
        if not path:
            return
        try:
            y, sr = librosa.load(path, sr=None, mono=True)
            self._set_audio(y, sr, os.path.basename(path))
        except Exception as e:
            messagebox.showerror(self.t("err_load"), str(e))

    def _load_probe(self):
        name = self.probe_var.get()
        if name not in PROBES:
            messagebox.showinfo(self.t("sec_input"), self.t("info_probe_sel"))
            return
        try:
            self._set_audio(PROBES[name](), 44100, f"probe:{name}")
        except Exception as e:
            messagebox.showerror(self.t("err_probe"), str(e))

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
        self._busy(False, self.t("st_play_a", name=self.src_name))
        sd.play(self.audio, self.sr)

    def _play_sim(self):
        if self.audio is None:
            return
        self.btn_b.config(state="disabled")
        self._busy(True, self.t("st_sim"))
        threading.Thread(target=self._sim_worker, daemon=True).start()

    def _sim_worker(self):
        def gui(fn):
            self.root.after(0, fn)
        try:
            y = simulate(self.audio, self.sr, self.preset_key, self.freq_low.get())
            self.sim_cache = (y, self.sr)
            gui(lambda: self._refresh_plot(y))
            gui(lambda: self._busy(False, self.t("st_play_b", name=self._plabel(self.preset_key))))
            gui(lambda: sd.play(y, self.sr))
        except Exception as e:
            err = str(e)
            gui(lambda: self._busy(False, self.t("st_error")))
            gui(lambda: messagebox.showerror(self.t("err_sim"), err))
        finally:
            gui(lambda: self.btn_b.config(state="normal"))

    def _stop(self):
        sd.stop()
        self._busy(False, self.t("st_stopped"))

    # ─────────────────────────────────────────────────── 保存 ──
    def _save(self):
        if self.audio is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".wav", filetypes=[("WAV", "*.wav")])
        if not path:
            return
        self._busy(True, self.t("st_saving"))
        threading.Thread(target=self._save_worker, args=(path,), daemon=True).start()

    def _save_worker(self, path):
        def gui(fn):
            self.root.after(0, fn)
        try:
            y = simulate(self.audio, self.sr, self.preset_key, self.freq_low.get())
            sf.write(path, y, self.sr, subtype="PCM_16")
            gui(lambda: self._busy(False, self.t("st_saved")))
            gui(lambda: messagebox.showinfo(self.t("info_saved"), path))
        except Exception as e:
            err = str(e)
            gui(lambda: self._busy(False, self.t("st_error")))
            gui(lambda: messagebox.showerror(self.t("err_save"), err))

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
        self.ax.set_title(self.t("spec_title", name=self._plabel(self.preset_key)))
        self.fig.tight_layout(pad=1.5)
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = CochleaSimApp(root, lang="en")   # 既定=英語
    root.mainloop()

# DTCHull プロジェクト（計算データ集約）

DTC Hull（Duisburg Test Case 船体ベンチマーク）に関する GeoPT / Transolver の
**入力データ・学習済みモデル・推論結果・スクリプト・ログ**を 1 か所に集約したもの。
集約日: 2026-07-07。元は GeoPT リポジトリ直下に散在していた。

DTC = Duisburg Test Case（コンテナ船。フィールド名は OpenFOAM 由来）。

---

## 1. ディレクトリ構成

```
DTCHull_project/
├── README.md                この文書
├── data/                    学習用 npy（x_/y_/cond_ 各130、計390 / 約2.1GB）← 旧 dtchull_npys/
├── preprocess/
│   └── DTCHull_process.py   元VTK → npy 前処理（HF DL + 座標変換 + SDF + box crop）
├── pipeline/
│   └── _run_dtchull_pipeline.sh   DL+前処理 → GeoPT 200ep → scratch 200ep の一括実行
├── scripts/                 学習/評価スクリプト（run.py への引数set）
│   ├── finetune/            GeoPT ファインチューン（GeoPT_dtchull.sh / _eval / .bak）
│   ├── scratch_normal_cond/ スクラッチ Transolver・normal条件（Transolver_dtchull.sh / _eval / .bak）
│   └── scratch_geopt_cond/  スクラッチ・GeoPT条件（Transolver_dtchull.sh）
├── checkpoints/             学習済み重み（2 run、各約15MB）
├── training_logs/           train/test loss 履歴（npy、2 run × 2）
├── results/                 推論結果・可視化
│   ├── dtchull_transolver_8layers_normal_cond/  scratch run（x_raw/y_raw/out_raw npy + pred/gt/error PDF）
│   ├── hull_geopt_transolver_new_data_correct/  GeoPT finetune run（同上）
│   ├── paraview_vtp/        ParaView 用点群（p_gt/p_pred/p_err 保持）
│   └── paraview_png/        レンダリング画像
├── viz/                     可視化スクリプト（_render_dtchull_surface.py / _render_pressure_paraview.py / _make_convergence_plot.py）
└── logs/                    実行ログ（*.failed は失敗試行）
```

### 集約方法（重要）
- `run.py` / `exp/steady_cond.py` は出力先を **リポジトリ直下固定**（`./checkpoints`, `./results/<name>`,
  `./training_logs`）にハードコードしており、全データセット共有コードのため変更していない。
- そこで **実データは本フォルダに移動**し、リポジトリ直下の `checkpoints/`・`results/` には
  **本フォルダを指すシンボリックリンク**を残した。→ 再学習・再評価しても中身は本フォルダに入る。
- `training_logs/` の 4 npy は元ファイルが root 所有で移動できなかったため、**コピー**を本フォルダに置いた
  （元も残存。再学習時はリポジトリ直下の root 所有ファイルに書かれる点に注意）。
- 前処理・学習・可視化スクリプトのパスは本構成に合わせて修正済み（`--data_path ./DTCHull_project/data` 等）。

---

## 2. データ仕様（1サンプル = 1ケース、計 130 ケース）

出典: `preprocess/DTCHull_process.py`

| ファイル | shape | 内容 |
|---|---|---|
| `x_{i}.npy` | (N, 7) | `[x, y, z, sdf, nx, ny, nz]` — 幾何情報のみ |
| `y_{i}.npy` | (N, 4) | `[p, Ux, Uy, Uz]` — 圧力と速度（**時間平均場** `p_rghMean`/`UMean`） |
| `cond_{i}.npy` | (1,) | スカラー1個。**斜航角[度]**（ファイル名の rad 値を度へ変換） |

- N（点数）は約 22.9万〜24.7万／ケース（船体表面 + 近傍ボリューム box）。
- `sdf`: 船体表面までの符号付き距離。**表面点は厳密に 0**。
- `cond` の分布: **-9.96° 〜 +9.83°**、130 個すべて異なる値（斜航角スイープ）。
- 座標変換: 軸入替 + 固定スケール `5/7`（waterline heuristic）+ box crop
  （`x∈[-4,5]`, 鉛直 `y∈[-1,1]`, `z∈[-1.5,1.5]`）。

学習設定（`scripts/*/Transolver_dtchull.sh`）: `--task steady_cond --space_dim 3 --fun_dim 8 --out_dim 4`
（入力8 = 幾何7 + cond1、出力4 = p,Ux,Uy,Uz）。

---

## 3. モデル 2 系統と結果

いずれも Transolver 8層 / n_hidden=256 / slice_num=32 / 200 epoch / ntrain=100, ntest=20。

| run | save_name | 内容 | best test loss | final test | rel_err（SUMMARY） |
|---|---|---|---|---|---|
| GeoPT ファインチューン | `hull_geopt_transolver_new_data_correct` | 事前学習あり | 0.1267 @160 | 0.1289 | **0.1282**（優） |
| スクラッチ | `dtchull_transolver_8layers_normal_cond` | 事前学習なし | 0.1362 @171 | 0.1377 | 0.1360 |

→ DTCHull では **GeoPT（事前学習）が全指標で優位**（両者とも 200ep の公平比較）。
　詳細な指標・注意点はリポジトリ直下 `SUMMARY_GeoPT_results.md` を参照。

---

## 4. ⚠ 波（自由表面変動）は現仕様では計算できない

新規で「波あり」を始めるにあたっての結論。**現在の入出力仕様は静水中・定常のサロゲートであり、
波を表現・予測する口が無い**。根拠:

1. **条件入力 `cond` がスカラー1個（斜航角のみ）**。波高 H / 波長 λ / 周期・周波数 T,ω /
   波向き β / 出会い周波数 といった波パラメータが入力に存在しない。「どんな波か」を指定できない。
2. **出力が時間平均場**（`p_rghMean`, `UMean`）。波は非定常・振動現象で、平均化により波成分が消える。
   時間軸・位相の情報も無い。
3. **自由表面がどこにも変数として無い**。出力は `[p,Ux,Uy,Uz]` の 4ch のみで、
   VOF/`alpha.water`（気液界面）も波高（自由表面変位）チャネルも無い。
   水線は固定ヒューリスティック（`transform()` の `scale=5/7` と固定 crop box）として焼き込まれ、
   **自由表面を「解く量」ではなく「固定平面」として扱っている**。
4. 学習タスク名が明示的に **`steady_cond`（定常＋スカラー条件）**。船体姿勢も固定
   （喫水/トリム/沈下・波浪中動揺 heave/pitch 無し）。

**波対応に最低限必要な変更**: ①条件に波パラメータ（H, λ or ω, β）を追加、
②出力に自由表面（`alpha.water` or 波高場）と時間/位相情報を追加、③固定水線ヒューリスティックの撤廃。

### 元 VTK 調査結果（2026-07-07、ローカルキャッシュを直接検査）
`hf_cache/.../DTCHull/hull_*/{DTCHull_500,hull_500}_*.vtk`（OpenFOAM interFoam の二相VOF、19GBキャッシュ済）
のフィールドは以下:

| フィールド | 種別 | npy 化 |
|---|---|---|
| `p_rghMean`, `UMean` | 時間平均（圧力・速度） | ✅ 使用中 |
| `p_rgh` | **瞬時**圧力 | ❌ 破棄 |
| `alpha.water` | **VOF 体積率＝自由表面** | ❌ 破棄 |
| `nut` | 乱流粘性 | ❌ 破棄 |

- **朗報**: 自由表面 `alpha.water` と瞬時場 `p_rgh` は**元データに存在**する。
  → 前処理（`preprocess/DTCHull_process.py` の `--p_field`/`--u_field` 相当を拡張）で
  **船の自航波（造波・Kelvin 波系）を出力チャネルに追加するのは比較的容易**。
- **注意**: 各ケースは **1 スナップショットのみ**（時系列なし）で、変動条件は**斜航角だけ**。
  したがって「入射波（規則波・不規則波）中の耐航性（seakeeping）」＝真の波浪中応答は、
  **時間軸・波パラメータ・出会い周波数がそもそも無い**ため、この既存データからは学習できない。
  → 入射波を扱うには **波条件を振った新規 CFD（時系列出力）の生成が必須**。

---

## 5. 実行方法（すべてリポジトリ直下 `~/projects/GeoPT` から）

```bash
# フル: DL+前処理 → GeoPT 200ep → scratch 200ep
bash DTCHull_project/pipeline/_run_dtchull_pipeline.sh

# 単発（NGC コンテナ経由）
bash _run_finetune.sh DTCHull_project/scripts/finetune/GeoPT_dtchull.sh geopt_dtchull
bash _run_finetune.sh DTCHull_project/scripts/scratch_normal_cond/Transolver_dtchull.sh transolver_dtchull_scratch

# 評価のみ
bash DTCHull_project/scripts/scratch_normal_cond/Transolver_dtchull_eval.sh

# 圧力分布の再描画（ParaView, 1〜9 のサンプル番号）
tools/mamba-root/envs/pv/bin/pvbatch --force-offscreen-rendering \
    DTCHull_project/viz/_render_pressure_paraview.py 1 2 3
```

> 注: 再学習/再評価の出力は `exp/steady_cond.py` が **リポジトリ直下** の `./checkpoints`・`./results/<name>`・
> `./training_logs` に書く。`checkpoints/`・`results/` はここへのシンボリックリンクを張ってあるので中身は
> 本フォルダに入るが、`training_logs/` はリンクしていない（元 root 所有）ため手動でコピーが必要。

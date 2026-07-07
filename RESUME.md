# GeoPT 再開メモ

## コンテナ起動
cd ~/projects/GeoPT
docker run --rm -it --gpus all --shm-size=8g \
  -v $HOME/projects/GeoPT:/workspace/GeoPT \
  -w /workspace/GeoPT \
  nvcr.io/nvidia/pytorch:25.10-py3 bash

## コンテナ内で
pip install h5py timm matplotlib einops pyvista vtk huggingface_hub "scipy>=1.11"

## 学習(20エポック設定済み)
bash scripts/finetune/GeoPT_craft.sh

## 評価のみ
bash scripts/finetune/GeoPT_craft_eval.sh

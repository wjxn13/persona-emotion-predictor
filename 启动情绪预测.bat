@echo off
set HF_ENDPOINT=https://hf-mirror.com
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set HF_HUB_CACHE=D:\.kaggle\.hf_cache
python "D:\.kaggle\文本情绪反应预测\desktop_app.py"
pause
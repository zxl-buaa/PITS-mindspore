@echo off
setlocal enabledelayedexpansion

REM 定义静态参数
set "ds_pretrain=etth1"
set "ds_finetune=etth1"
set "ep_ft_head=5"
set "ep_ft_entire=10"
set "num_patches=%context_points%/%patch_len%"
set "context_points=512"
set "d_model=128"
set "patch_len=12"
set "stride=12"

REM 预测长度列表
set pred_lens=96 192 336 720

REM 遍历预测长度并执行Python脚本
for %%i in (%pred_lens%) do (
    echo 正在处理预测长度: %%i
    python PITS_finetune.py --dset_pretrain "%ds_pretrain%" --dset_finetune "%ds_finetune%" ^
        --n_epochs_finetune_head %ep_ft_head% --n_epochs_finetune_entire %ep_ft_entire% ^
        --target_points %%i --num_patches %num_patches% --context_points %context_points% ^
        --d_model %d_model% --patch_len %patch_len% --stride %stride% --is_finetune 1
    if errorlevel 1 (
        echo 预测长度 %%i 处理时出现错误
        pause
    )
)

endlocal

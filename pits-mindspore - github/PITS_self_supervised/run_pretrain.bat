@echo off
REM 设置模型训练参数
set "ds_pretrain=etth1"
set "ds_finetune=etth1"
set "d_model=128"
set "context_points=512"
set "patch_len=12"
set "stride=12"
set "num_patches=%context_points%/%patch_len%"
set /A num_patches=%num_patches% 2> nul
if %errorlevel% neq 0 set num_patches=0
set "ep_ft_head=5"
set /A ep_ft_entire=ep_ft_head * 2

REM 执行Python脚本，传入参数
echo 开始执行预训练...
python PITS_pretrain.py ^
    --dset_pretrain "%ds_pretrain%" ^
    --context_points %context_points% ^
    --d_model %d_model% ^
    --patch_len %patch_len% ^
    --stride %stride%

echo 预训练完成。
pause
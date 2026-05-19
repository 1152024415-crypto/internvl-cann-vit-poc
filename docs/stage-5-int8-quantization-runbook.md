# Stage 5: INT8 量化 + OMG 转换 Runbook

Last updated: 2026-05-19

基于 `success_demo/` 的成功案例，将 InternVL ViT + Projector FP32 ONNX 量化为
INT8，再用 OMG + `--compress_conf` 转成 NPU 可执行的 OM。

FP32 直连 OMG 会导致所有算子回退 CPU（588 个 op 不被 NPUCL 支持）。
INT8 量化后算子模式改变，NPU 可以原生执行。

## 总流程

```text
Step 0: 环境准备
Step 1: dopt_onnx_py3 无训练量化 FP32 ONNX → INT8 ONNX + compress_conf
Step 2: OMG 转换 INT8 ONNX + compress_conf → OM
Step 3: 下载 OM → rawfile → 设备验证
```

## 前置条件

```text
Linux 服务器: 100.102.192.199 (x00516260)
GPU: 6x V100-PCIE-32GB
Conda 环境: cann_lm_310 (Python 3.10 + PyTorch 2.6 + CUDA)
DDK: ~/internvl-omg/ddk/ (HiAI_DDK_100.600.020.010)
校准集: ~/internvl-omg/calib_images/ (156 张 jpg)
```

---

## Step 0: 环境准备

### 0.1 SSH 登录 + 激活 conda

```bash
ssh x00516260@100.102.192.199
conda activate cann_lm_310
```

### 0.2 安装依赖

```bash
pip install onnx opencv-python-headless protobuf==3.20.3
```

注意: DDK 的 `dopt/common_utils/notrain_pb2.py` 是旧版 protoc 生成的，
必须用 `protobuf==3.20.x`，否则报 `Descriptors cannot be created directly`。

### 0.3 校准集

从 Windows 上传:

```powershell
scp "D:\proj\internvl-cann-vit-poc\success_demo\校准集\calib_images.zip" x00516260@100.102.192.199:~/internvl-omg/
```

Linux 上解压（注意避免双层目录）:

```bash
cd ~/internvl-omg
unzip calib_images.zip
# 如果出现 calib_images/calib_images/ 双层，修复:
# mv calib_images/calib_images/* calib_images/ && rmdir calib_images/calib_images
ls calib_images/ | wc -l  # 预期 156
```

### 0.4 上传 success_demo ONNX

从 Windows 上传:

```powershell
scp "D:\proj\internvl-cann-vit-poc\success_demo\待量化的ONNX模型\split_internvit_projector_fixed_b1_no_resize.onnx" x00516260@100.102.192.199:~/internvl-omg/onnx/
```

注意: success_demo 的 ONNX 输出节点名是 `visual_features`，
我们项目自己的 ONNX 输出节点名是 `visual_tokens`。
如果用自己的 ONNX 量化，`--out_nodes` 参数要改为 `visual_tokens`。

### 0.5 验证 dopt 可运行

```bash
python ~/internvl-omg/ddk/tools/tools_dopt/dopt_onnx_py3/dopt_so.py --help
```

如果报 `undefined symbol: Py_EnterRecursiveCall` → Python 版本不对，需 3.10。

---

## Step 1: INT8 量化（已验证成功）

### 1.1 创建工作目录 + 配置文件

```bash
mkdir -p ~/internvl-omg/quantized_v6

cat > ~/internvl-omg/quantized_v6/int8_8_config_v6.prototxt << 'EOF'
strategy: 'Quant_INT8-8'
device: USE_CPU
exclude_op: '/vision_model/embeddings/patch_embedding/Conv'
preprocess_parameter:
{
    input_type: IMAGE
    image_format: RGB
    input_file_path: './calib_images'
    mean_value: 0.485
    mean_value: 0.456
    mean_value: 0.406
    standard_deviation: 0.229
    standard_deviation: 0.224
    standard_deviation: 0.225
}
EOF
```

配置说明:
- `strategy: Quant_INT8-8` — 权重 INT8 + 激活 INT8
- `device: USE_CPU` — 校准用 CPU
- `exclude_op` — 首层 Conv 保持 FP32（精度敏感）
- `preprocess_parameter` — dopt 自动从图片做 ImageNet normalize

### 1.2 执行量化（success_demo 版 ONNX）

```bash
cd ~/internvl-omg

python ddk/tools/tools_dopt/dopt_onnx_py3/dopt_so.py \
  --framework 5 \
  -m 0 \
  --model ./onnx/split_internvit_projector_fixed_b1_no_resize.onnx \
  --cal_conf ./quantized_v6/int8_8_config_v6.prototxt \
  --output ./quantized_v6/split_internvit_v6_exclude_conv_int8_8.onnx \
  --input_shape 'pixel_values:1,3,448,448' \
  --out_nodes 'visual_features' \
  --compress_conf ./quantized_v6/split_internvit_v6_exclude_conv_int8_8_param \
  --device_idx 0 \
  2>&1 | tee ./quantized_v6/dopt_quantize.log
```

预期成功标志: `Quantize model success`

产出:
```text
quantized_v6/split_internvit_v6_exclude_conv_int8_8.onnx        ← INT8 ONNX
quantized_v6/split_internvit_v6_exclude_conv_int8_8_param/       ← 量化参数
quantized_v6/dopt_quantize.log                                   ← 日志
```

### 1.3 (备选) 用项目自己的 ONNX 量化

```bash
python ddk/tools/tools_dopt/dopt_onnx_py3/dopt_so.py \
  --framework 5 \
  -m 0 \
  --model ./onnx/internvl3_5_vit_projector_fp32_opset18_staticpos.onnx \
  --cal_conf ./quantized_v6/int8_8_config_v6.prototxt \
  --output ./quantized_v6/internvl3_5_vit_int8_8.onnx \
  --input_shape 'pixel_values:1,3,448,448' \
  --out_nodes 'visual_tokens' \
  --compress_conf ./quantized_v6/internvl3_5_vit_int8_8_param \
  --device_idx 0 \
  2>&1 | tee ./quantized_v6/dopt_quantize.log
```

---

## Step 2: OMG 转换 INT8 ONNX → OM

### 2.1 准备 OMG 环境

```bash
cd ~/internvl-omg
export DDK_PATH=$PWD/ddk
export OMG_DIR=$DDK_PATH/tools/tools_omg/master
export GLIBC_DIR=$OMG_DIR/x86_64-pc-linux-gnu-6.3.0
export CONDA_LIB=/srv/workspace/x00516260/anaconda3/lib
export PLATFORM_DIR=$DDK_PATH/tools/platform/kirin9030
export LD_LIBRARY_PATH=$CONDA_LIB:$GLIBC_DIR:$OMG_DIR/lib64:$PLATFORM_DIR/lib64:$DDK_PATH/ddk/tbe/lib64:$DDK_PATH/ddk/ascendcpp/lib64
export SOC_VERSION=kirin9030
export PATH=$DDK_PATH/ddk/ccec_compiler/bin:/usr/bin:$PATH
export PYTHONPATH=$DDK_PATH/ddk/tbe/ops:$PLATFORM_DIR/ops/impl:$DDK_PATH/ddk/tbe/data
export ASCEND_OPP_PATH=$DDK_PATH/ddk
export KIRIN_CONF_JSON_PATH=$DDK_PATH/ddk/ascendcpp/conf/
```

### 2.2 执行 OMG 转换

```bash
# success_demo ONNX 版
$GLIBC_DIR/ld-linux-x86-64.so.2 --library-path $LD_LIBRARY_PATH \
  $OMG_DIR/omg \
  --model ./quantized_v6/split_internvit_v6_exclude_conv_int8_8.onnx \
  --framework 5 \
  --input_shape "pixel_values:1,3,448,448" \
  --compress_conf=./quantized_v6/split_internvit_v6_exclude_conv_int8_8_param \
  --platform kirin9030 \
  --output ./quantized_v6/split_internvit_v6_exclude_conv_int8_8 \
  2>&1 | tee ./quantized_v6/omg_convert.log
```

关键: `--compress_conf` 是 INT8 OM 的核心参数，没有它 OMG 不知道如何处理量化算子。

### 2.3 检查结果

```bash
ls -lh ./quantized_v6/split_internvit_v6_exclude_conv_int8_8.om*
grep -iE 'partition type|OMG generate' ./quantized_v6/omg_convert.log
```

判断:
- `partition type NPU:1` → NPU 放置成功
- `partition type NPU:0, CPU:1` → 仍然 CPU，需排查

---

## Step 3: 下载 OM + 设备验证

### 3.1 下载到 Windows

```powershell
scp x00516260@100.102.192.199:~/internvl-omg/quantized_v6/split_internvit_v6_exclude_conv_int8_8.om `
  D:\proj\internvl-cann-vit-poc\demo\entry\src\main\resources\rawfile\internvl3_5_vit_projector_fp32_opset18_staticpos.om
```

如果扩展名是 `.omc`，在 Linux 上先 `cp xxx.omc xxx.om`。

### 3.2 设备端验证

1. DevEco 打开 `demo/`，Build & Run
2. Official Smoke → NN Runtime 可用
3. Load Model → OM 编译通过
4. dog/cat Run Once → cosine >= 0.99, device = ACCELERATOR
5. 20-run Stability → 20/20

---

## 已踩过的坑

### 1. Python 版本

```text
dopt.so 是 Python 3.10 编译的
系统 Python 3.6 会报 undefined symbol: Py_EnterRecursiveCall
修复: conda activate cann_lm_310
```

### 2. protobuf 版本

```text
DDK 的 notrain_pb2.py 是旧版 protoc 生成的
protobuf >= 4.x 报 Descriptors cannot be created directly
修复: pip install protobuf==3.20.3
```

### 3. 缺少 cv2

```text
dopt 校准时需要读取图片
修复: pip install opencv-python-headless
```

### 4. 校准集双层目录

```text
unzip calib_images.zip 可能产生 calib_images/calib_images/ 双层
dopt 把内层目录当图片读，报 IsADirectoryError
修复: mv calib_images/calib_images/* calib_images/ && rmdir calib_images/calib_images
```

### 5. FP32 直连 OMG 全部回退 CPU

```text
FP32 BatchMatMul 不被 NPUCL elementary_lib 支持 (588 个 op)
NPUCL 分区 > 9 → rollback to CPU → 生成 CPU-only .omc
必须走 INT8 量化路径，--compress_conf 让 OMG 理解 INT8 算子
```

### 6. OMG 需要 ld-linux 包装

```text
Ubuntu 18.04 glibc 2.27，DDK OMG 需要 glibc 2.35
必须用 DDK 自带的 ld-linux + glibc 运行 OMG
LD_LIBRARY_PATH 需要 anaconda libstdc++ (GLIBCXX_3.4.33)
```

### 7. kirin9030 平台插件缺失

```text
DDK 自带 ddk/tools/platform/ 不存在
需要从 CANN_LLM 项目复制: cp -r /srv/.../cann_lm_engine/tools/platform/kirin9030 ddk/tools/platform/
并把 libai_npucore_ascendc_kernel.so 等复制到 master/lib64/
```

---

## success_demo 对照表

| success_demo 文件 | 说明 |
|---|---|
| `待量化的ONNX模型/split_internvit_projector_fixed_b1_no_resize.onnx` | 1.2G, 输出节点 `visual_features` |
| `校准集/calib_images.zip` | 156 张校准图片 |
| `说明文档.docx` | 成功案例的完整命令和配置 |

success_demo 使用的命令:
```bash
python3 ddk/tools/tools_dopt/dopt_onnx_py3/dopt_so.py \
  --framework 5 -m 0 \
  --model ./onnx/split_internvit_projector_fixed_b1_no_resize.onnx \
  --cal_conf ./quantized_v6/int8_8_config_v6.prototxt \
  --output ./quantized_v6/split_internvit_v6_exclude_conv_int8_8.onnx \
  --input_shape 'pixel_values:1,3,448,448' \
  --out_nodes 'visual_features' \
  --compress_conf ./quantized_v6/split_internvit_v6_exclude_conv_int8_8_param \
  --device_idx 0
```

success_demo 的 OMG 命令:
```bash
./omg --model split_internvit_v6_exclude_conv_int8_8.onnx \
  --output split_internvit_v6_exclude_conv_int8_8 \
  --framework 5 \
  --input_shape='pixel_values:1,3,448,448' \
  --compress_conf=./split_internvit_v6_exclude_conv_int8_8_param
```

success_demo 的设备端 dump 验证:
```bash
# push tools_sysdbg 到设备
hdc file send tools_sysdbg /data/local/tmp/vlm1b_vit/ViT/
# 在设备上运行
export LD_LIBRARY_PATH=.
./model_run_tool_internal -h
```

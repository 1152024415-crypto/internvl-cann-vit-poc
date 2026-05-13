#!/usr/bin/env bash
set +e

echo "--- identity ---"
whoami
pwd

echo "--- project files ---"
ls -lh /root/internvl-cann-vit-poc/artifacts/onnx
ls -lh /root/internvl-cann-vit-poc/scripts

echo "--- omg in PATH ---"
command -v omg
echo "command -v omg exit=$?"

echo "--- common env vars ---"
env | grep -E 'CANN|ASCEND|OMG' | sort

echo "--- filesystem search ---"
find /root /opt /usr/local /usr -path '*tools_omg/omg' -type f 2>/dev/null | head -n 50
echo "find exit=$?"

echo "--- conversion script ---"
cd /root/internvl-cann-vit-poc || exit 10
./scripts/convert_onnx_to_om.sh
echo "convert script exit=$?"

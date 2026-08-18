#!/usr/bin/env bash

# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e
export swift_dir=/workspace/Megatron-LM
mkdir -p /workspace/build_logs
export log_path=/workspace/build_logs
mkdir -p /workspace/upload
upload_path=/workspace/upload

python -m pip config --user set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip config --user set global.trusted-host pypi.tuna.tsinghua.edu.cn

swift_tar (){
    cd /workspace
    # Megatron-LM.tar only include the main branch
    if [ -n "$BRANCH" ] && [ "$BRANCH" = "main" ]; then
        echo "Checkout branch $BRANCH"
        tar -zcf Megatron-LM.tar.gz Megatron-LM/
        mv Megatron-LM.tar.gz ${upload_path}/
    else
        echo "No BRANCH specified, skip checkout"
    fi
}

swift_build (){
    cd $swift_dir
    rm -rf build/
    rm -rf dist/
    rm -rf ms_swift.egg-info/

    python -m pip install -r requirements.txt
    python setup.py bdist_wheel

    echo "install_megatron_develop_whl"
    python -m pip install --upgrade pip
    python -m pip install --ignore-installed dist/megatron-*.whl --no-cache-dir --force-reinstall --no-dependencies
    echo "waiting for import megatron..."
    python -c "import megatron; print('megatron version:', megatron.__version__)"
    python -c "import megatron; print('megatron version:', megatron.__version__)" >> ${log_path}/commit_info.txt

    commit=${COMMIT_ID:-unknown}
    commit=${commit:0:7}

    whl_file=$(ls $swift_dir/dist/megatron-*.whl)
    base_name=$(basename $whl_file)
    new_name=$(echo $base_name | sed "s/\.dev0/&+${commit}/")
    echo "commit whl: $new_name"
    cp "$whl_file" "${upload_path}/${new_name}"
    cp "$whl_file" "${upload_path}/${base_name}"

    zero_name=$(echo $base_name | sed "s/^megatron-[^-]*-/megatron-0.0.0-/")
    echo "0.0.0 whl: $zero_name"
    cp "$whl_file" "${upload_path}/${zero_name}"
}

# main
cd ${swift_dir}
echo -e "\033[32m ---- make Megatron-LM.tar.gz  \033[0m"
swift_tar
echo -e "\033[32m ---- build Megatron-LM whl  \033[0m"
swift_build


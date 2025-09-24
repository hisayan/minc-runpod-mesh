FROM runpod/pytorch:1.0.0-cu1281-torch271-ubuntu2204

# 出力を強制的に表示
RUN python --version && echo "Python version check completed"
RUN python -m pip --version && echo "Pip version check completed"
RUN uname -m && echo "Architecture check completed"

RUN python -m pip install --upgrade pip setuptools wheel

RUN python -m pip --version && echo "Pip version check completed"

# PyPI 上でバージョン確認
RUN python -m pip index versions open3d

RUN pip install --no-cache-dir open3d==0.19.0

# パッケージリストを更新
RUN apt-get update

# 基本パッケージをインストール
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    curl \
    ca-certificates

# Open3D用の基本依存関係
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libc6 \
    libstdc++6 \
    libgcc-s1 \
    libgomp1

# OpenGLライブラリ（段階的インストール）
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libgl1 \
    libgl1-mesa-dri \
    libglx-mesa0 \
    libglx0 \
    libopengl0

# X11ライブラリ
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libx11-6 \
    libxau6 \
    libxdmcp6

# EGL/GLESライブラリ（ヘッドレスレンダリング用）
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libegl1 \
    libgles2

# クリーンアップ
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python依存関係をインストール
COPY builder/requirements.txt .
RUN pip3 install --no-cache-dir --break-system-packages --timeout 600 -r requirements.txt

# # Node.js依存関係もインストール（splat-transformのため）
# COPY package.json ./
# RUN npm install

# Pythonワーカーファイルをコピー
COPY src/handler.py .

# テスト用ファイルをコピー
COPY test_input.json .

ENTRYPOINT ["python3", "-u", "handler.py"]
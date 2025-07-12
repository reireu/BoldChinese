# ベースイメージの指定
FROM python:3.9

# システムの更新とビルドツールのインストール
# CMake, build-essential (gcc, g++など), git はPythonライブラリのビルドに必要な場合があります
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリの設定
# コンテナ内の /app ディレクトリを現在の作業ディレクトリとします
WORKDIR /app

# プロジェクトのrequirements.txtをコンテナにコピーし、依存関係をインストール
# Dockerfileと同じ階層にrequirements.txtがあることを想定しています
COPY requirements.txt /app/

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# PaddlePaddle（CPU版）をインストール
RUN pip install --no-cache-dir paddlepaddle==2.6.2 -f https://www.paddlepaddle.org.cn/whl/paddlepaddle_cpu.html

# paddlespeech インストール

RUN pip install opencc==1.1.9

RUN pip install git+https://github.com/PaddlePaddle/PaddleAudio.git

RUN pip install --no-cache-dir paddlespeech==1.4.1
# aistudio-sdkのインストール
RUN pip install --no-cache-dir aistudio_sdk==0.3.5
# boldchineseプロジェクトのbackendディレクトリの内容をコンテナの/appにコピー
# これにより、/app/app/main.py が利用可能になります
COPY ./backend /app

# FastAPIアプリケーションを起動
# uvicorn app.main:app は /app ディレクトリ内の app/main.py から 'app' オブジェクトを探します
# --host 0.0.0.0 でコンテナ外からのアクセスを許可
# --port 8000 でポートを指定
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


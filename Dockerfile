FROM python:3.9-slim

# 避免交互式提示
ENV DEBIAN_FRONTEND=noninteractive

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    # OCR 相關
    tesseract-ocr \
    tesseract-ocr-chi-tra \
    # PDF 處理相關
    poppler-utils \
    # Selenium 相關
    chromium \
    chromium-driver \
    # 其他必要工具
    wget \
    gnupg \
    # 中文字體
    fonts-noto-cjk \
    # 清理 apt 快取
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 設置 Tesseract 數據路徑
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

# 設置 Chrome 和 ChromeDriver 路徑
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver

# 設置工作目錄
WORKDIR /app

# 複製依賴文件
COPY requirements.txt .

# 安裝 Python 依賴
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用程式代碼
COPY . .

# 設置環境變數
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# 暴露端口
EXPOSE 8080

# 啟動命令
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--threads", "4", "--timeout", "120", "app:app"] 
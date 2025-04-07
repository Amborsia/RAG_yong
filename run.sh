#!/bin/bash

# 로깅 함수
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# 에러 로깅 함수
error() {
    log "ERROR: $1"
    exit 1
}

# 환경 변수 로드
if [ -f .env ]; then
    set -a
    source .env
    set +a
    log "✅ 성공적으로 로드된 환경 변수:"
    grep -v '^#' .env | while read line; do echo "   ${line%%=*}"; done
else
    log ".env 파일을 찾을 수 없습니다. 기본 포트 8502를 사용합니다."
    PORT=8502
fi
log "🌐 서버 포트: $PORT"

PYENV_ROOT="$HOME/.pyenv"
STREAMLIT_LOG="streamlit.log"
STREAMLIT_PID="streamlit.pid"

# pyenv 환경을 설정하는 함수
setup_pyenv() {
        source ~/.bashrc
#       export PYENV_ROOT="$HOME/.pyenv"
#       export PATH="$PYENV_ROOT/bin:$PATH"
#       eval "$(pyenv init --path)"
#       eval "$(pyenv init -)"
#       eval "$(pyenv virtualenv-init -)"
}

# Streamlit 실행 함수
start_streamlit() {
    log "Starting Streamlit..."

    # pyenv 환경 활성화
    setup_pyenv

    # .python-version이 있을 경우 해당 환경 활성화
    if [ -f .python-version ]; then
        PYENV_ENV=$(cat .python-version)
        pyenv activate "$PYENV_ENV"
        log "Activated pyenv environment: $PYENV_ENV"
    else
        log "No .python-version found. Running without pyenv."
    fi

    # Streamlit을 백그라운드로 실행
    nohup streamlit run app.py --server.address 0.0.0.0 --server.port $PORT > "$STREAMLIT_LOG" 2>&1 &

    # 실행된 프로세스 ID 저장
    echo $! > "$STREAMLIT_PID"
    log "Streamlit started (PID: $(cat $STREAMLIT_PID))"
}

# Streamlit 종료 함수
stop_streamlit() {
    if [ -f "$STREAMLIT_PID" ]; then
        PID=$(cat "$STREAMLIT_PID")
        log "Stopping Streamlit (PID: $PID)..."
        kill $PID
        rm "$STREAMLIT_PID"
        log "Streamlit stopped."
    else
        log "Streamlit is not running or PID file not found."
    fi
}

# PDF를 이미지로 변환하는 함수
convert_pdf_to_images() {
    local pdf_path="$1"
    
    # PDF 파일 존재 확인
    if [ ! -f "$pdf_path" ]; then
        error "PDF 파일을 찾을 수 없습니다: $pdf_path"
    fi
    
    # 출력 디렉토리 생성
    local output_dir="data/ebs/pages/$(basename "$pdf_path" .pdf)"
    mkdir -p "$output_dir"
    
    log "Converting PDF to images: $pdf_path"
    python services/pdf_preprocessor.py "$pdf_path"
    
    if [ $? -eq 0 ]; then
        log "✅ PDF 변환 완료: $output_dir"
    else
        error "PDF 변환 실패"
    fi
}

# 모든 PDF 파일 변환
convert_all_pdfs() {
    local pdf_dir="data/ebs/pdfs"
    
    if [ ! -d "$pdf_dir" ]; then
        error "PDF 디렉토리를 찾을 수 없습니다: $pdf_dir"
    fi
    
    log "Converting all PDFs in $pdf_dir"
    
    for pdf in "$pdf_dir"/*.pdf; do
        if [ -f "$pdf" ]; then
            convert_pdf_to_images "$pdf"
        fi
    done
}

# 텍스트 파일을 JSON으로 변환하는 함수
convert_texts_to_json() {
    local text_dir="data/ebs/texts"
    
    if [ ! -d "$text_dir" ]; then
        error "텍스트 디렉토리를 찾을 수 없습니다: $text_dir"
    fi
    
    log "텍스트 파일을 JSON으로 변환 중..."
    python utils/ebs_json_converter.py
    
    if [ $? -eq 0 ]; then
        log "✅ 텍스트 파일 변환 완료"
    else
        error "텍스트 파일 변환 실패"
    fi
}

# 인자 처리
case "$1" in
    start)
        start_streamlit
        ;;
    stop)
        stop_streamlit
        ;;
    convert)
        if [ -z "$2" ]; then
            convert_all_pdfs
        else
            convert_pdf_to_images "$2"
        fi
        ;;
    convert-texts)
        convert_texts_to_json
        ;;
    *)
        echo "Usage: $0 {start|stop|convert [pdf_path]|convert-texts}"
        echo "  start: Streamlit 서버 시작"
        echo "  stop: Streamlit 서버 종료"
        echo "  convert [pdf_path]: PDF를 이미지로 변환 (pdf_path 생략시 모든 PDF 변환)"
        echo "  convert-texts: 텍스트 파일들을 JSON으로 변환"
        exit 1
        ;;
esac

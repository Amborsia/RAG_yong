#!/bin/bash

# 환경 변수 로드
if [ -f .env ]; then
    set -a
    source .env
    set +a
    echo "✅ 성공적으로 로드된 환경 변수:"
    grep -v '^#' .env | while read line; do echo "   ${line%%=*}"; done
else
    echo ".env 파일을 찾을 수 없습니다. 기본 포트 8502를 사용합니다."
    PORT=8502
fi
echo "🌐 서버 포트: $PORT"

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
        echo "Starting Streamlit..."

        # pyenv 환경 활성화
        setup_pyenv

        # .python-version이 있을 경우 해당 환경 활성화
        if [ -f .python-version ]; then
                PYENV_ENV=$(cat .python-version)
                pyenv activate "$PYENV_ENV"
                echo "Activated pyenv environment: $PYENV_ENV"
        else
                echo "No .python-version found. Running without pyenv."
        fi

        # Streamlit을 백그라운드로 실행
        nohup streamlit run app.py --server.address 0.0.0.0 --server.port $PORT > "$STREAMLIT_LOG" 2>&1 &

        # 실행된 프로세스 ID 저장
        echo $! > "$STREAMLIT_PID"
        echo "Streamlit started (PID: $(cat $STREAMLIT_PID))"
}

# Streamlit 종료 함수
stop_streamlit() {
        if [ -f "$STREAMLIT_PID" ]; then
                PID=$(cat "$STREAMLIT_PID")
                echo "Stopping Streamlit (PID: $PID)..."
                kill $PID
                rm "$STREAMLIT_PID"
                echo "Streamlit stopped."
        else
                echo "Streamlit is not running or PID file not found."
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
        *)
                echo "Usage: $0 {start|stop}"
                exit 1
        ;;
esac

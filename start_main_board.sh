#!/usr/bin/env bash
#
# start_main_board.sh — start/stop the services this Rock 5C (the LLM board)
# hosts, as background daemons that survive an SSH disconnect:
#
#   rkllama     :8080   NPU LLM backend          (LLM/rkllama/venv)
#   llm_server  :8765   orchestrator (-> TTS)    (LLM/)
#   stt         :8767   faster-whisper STT (CPU) (STT/)
#
# The TTS node lives on the OTHER board, and the coordinator is an interactive
# client — neither is started here.
#
# Usage:
#   ./start_main_board.sh [start|stop|status|restart] [service]
#   ./start_main_board.sh                # start everything
#   ./start_main_board.sh status
#   ./start_main_board.sh restart stt    # just one service
#
# Logs + pids go to ./logs/ (gitignored).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR="$REPO/logs"
mkdir -p "$LOGDIR"

RKLLAMA_DIR="$REPO/LLM/rkllama"
RKLLAMA_MODELS="$(ls -d "$RKLLAMA_DIR"/venv/lib/python*/site-packages/rkllama/config/models 2>/dev/null | head -1)"

# service order matters: rkllama must be ready before llm_server starts.
SERVICES=(rkllama llm_server stt)

svc_port()    { case "$1" in rkllama) echo 8080;; llm_server) echo 8765;; stt) echo 8767;; esac; }
svc_dir()     { case "$1" in rkllama) echo "$RKLLAMA_DIR";; llm_server) echo "$REPO/LLM";; stt) echo "$REPO/STT";; esac; }
svc_cmd()     {
  case "$1" in
    rkllama)    echo "$RKLLAMA_DIR/venv/bin/rkllama_server --models $RKLLAMA_MODELS";;
    llm_server) echo "uv run python llm_server.py";;
    stt)        echo "uv run python stt_server.py";;
  esac
}

C_G="\033[32m"; C_R="\033[31m"; C_Y="\033[33m"; C_0="\033[0m"

port_up() { ss -ltn 2>/dev/null | grep -q ":$1 "; }

pid_on_port() { ss -ltnp 2>/dev/null | grep ":$1 " | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2; }

wait_port() {  # name port timeout_s
  local name=$1 port=$2 t=${3:-30} i=0
  while ! port_up "$port"; do
    sleep 1; i=$((i+1))
    if [ "$i" -ge "$t" ]; then
      printf "${C_R}failed (see logs/%s.log)${C_0}\n" "$name"
      tail -n 5 "$LOGDIR/$name.log" 2>/dev/null | sed 's/^/      /'
      return 1
    fi
  done
  printf "${C_G}up${C_0}\n"
}

start_one() {
  local name=$1 port dir cmd
  port=$(svc_port "$name"); dir=$(svc_dir "$name"); cmd=$(svc_cmd "$name")
  if port_up "$port"; then
    printf "  ${C_Y}[skip]${C_0}  %-11s already running on :%s\n" "$name" "$port"
    return 0
  fi
  printf "  [start] %-11s :%s … " "$name" "$port"
  # setsid + </dev/null detaches fully so it survives this shell / SSH closing.
  ( cd "$dir" && setsid bash -c "exec $cmd" >"$LOGDIR/$name.log" 2>&1 </dev/null & )
  # rkllama must answer /models before llm_server (which queries it at startup).
  if [ "$name" = "rkllama" ]; then
    wait_port "$name" "$port" 60 || return 1
    local i=0
    until curl -fsS "http://127.0.0.1:$port/models" >/dev/null 2>&1; do
      sleep 1; i=$((i+1)); [ "$i" -ge 30 ] && break
    done
  else
    wait_port "$name" "$port" 40 || return 1
  fi
}

stop_one() {
  local name=$1 port pid
  port=$(svc_port "$name"); pid=$(pid_on_port "$port")
  if [ -z "$pid" ]; then
    printf "  ${C_Y}[skip]${C_0}  %-11s not running\n" "$name"; return 0
  fi
  printf "  [stop]  %-11s :%s (pid %s) … " "$name" "$port" "$pid"
  kill "$pid" 2>/dev/null || true
  for _ in 1 2 3 4 5; do port_up "$port" || break; sleep 1; done
  port_up "$port" && { kill -9 "$pid" 2>/dev/null || true; sleep 1; }
  port_up "$port" && printf "${C_R}still up${C_0}\n" || printf "${C_G}stopped${C_0}\n"
}

status_one() {
  local name=$1 port pid; port=$(svc_port "$name"); pid=$(pid_on_port "$port")
  if [ -n "$pid" ]; then
    printf "  %-11s ${C_G}● up${C_0}   :%s  pid %s\n" "$name" "$port" "$pid"
  else
    printf "  %-11s ${C_R}○ down${C_0} :%s\n" "$name" "$port"
  fi
}

action="${1:-start}"
filter="${2:-}"
targets=("${SERVICES[@]}")
[ -n "$filter" ] && targets=("$filter")

case "$action" in
  start)   echo "Starting Aven services on this board:";     for s in "${targets[@]}"; do start_one "$s"; done ;;
  stop)    echo "Stopping Aven services:";  for s in "${targets[@]}"; do stop_one "$s"; done ;;
  restart) echo "Restarting Aven services:";
           for s in "${targets[@]}"; do stop_one "$s"; done
           for s in "${targets[@]}"; do start_one "$s"; done ;;
  status)  echo "Aven services on this board:"; for s in "${SERVICES[@]}"; do status_one "$s"; done ;;
  *) echo "usage: $0 [start|stop|status|restart] [service]" >&2; exit 2 ;;
esac

[ "$action" != "status" ] && { echo; echo "Logs: $LOGDIR/<service>.log   ·   status: $0 status"; }
exit 0

#!/usr/bin/env bash
#
# start_main_board.sh — start/stop the Aven services on this Rock 5C as
# background daemons that survive an SSH disconnect.
#
#   rkllama      :8080   NPU LLM backend            (LLM/rkllama/venv)
#   llm_server   :8765   orchestrator (-> TTS board)(LLM/)
#   stt          :8767   faster-whisper STT (CPU)   (STT/)
#   coordinator  (proc)  voice loop: wakeword + mic + STT/LLM clients + speaker
#                        playback (coordinator/) — needs a USB mic + speakers
#
# The TTS node lives on the OTHER board, so it isn't started here.
#
# Usage:
#   ./start_main_board.sh [start|stop|status|restart] [service]
#   ./start_main_board.sh                  # start everything
#   ./start_main_board.sh status
#   ./start_main_board.sh restart coordinator
#
# Logs go to ./logs/ (gitignored).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR="$REPO/logs"
mkdir -p "$LOGDIR"

# Load local secrets (e.g. WEATHERAPI_KEY) so the daemons inherit them.
[ -f "$REPO/.env.local" ] && { set -a; . "$REPO/.env.local"; set +a; }

RKLLAMA_DIR="$REPO/LLM/rkllama"
RKLLAMA_MODELS="$(ls -d "$RKLLAMA_DIR"/venv/lib/python*/site-packages/rkllama/config/models 2>/dev/null | head -1)"

# start order matters: backend before the coordinator that talks to it.
SERVICES=(rkllama llm_server stt coordinator)

svc_kind() { case "$1" in coordinator) echo proc;; *) echo port;; esac; }
svc_port() { case "$1" in rkllama) echo 8080;; llm_server) echo 8765;; stt) echo 8767;; *) echo "";; esac; }
svc_dir()  { case "$1" in rkllama) echo "$RKLLAMA_DIR";; llm_server) echo "$REPO/LLM";; stt) echo "$REPO/STT";; coordinator) echo "$REPO/coordinator";; esac; }
svc_cmd()  {
  case "$1" in
    rkllama)     echo "$RKLLAMA_DIR/venv/bin/rkllama_server --models $RKLLAMA_MODELS";;
    llm_server)  echo "uv run python llm_server.py";;
    stt)         echo "uv run python stt_server.py";;
    coordinator) echo "uv run python coordinator.py";;
  esac
}

C_G="\033[32m"; C_R="\033[31m"; C_Y="\033[33m"; C_0="\033[0m"

port_up()  { ss -ltn 2>/dev/null | grep -q ":$1 "; }
pid_port() { ss -ltnp 2>/dev/null | grep ":$1 " | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2 || true; }

# proc services are tracked by a pidfile holding the setsid session-leader pid
# (so we never pgrep a command line — which could match an unrelated process).
pidfile()  { echo "$LOGDIR/$1.pid"; }
proc_pid() { local f; f=$(pidfile "$1"); [ -f "$f" ] && cat "$f" 2>/dev/null || true; }

is_running() {
  local name=$1
  if [ "$(svc_kind "$name")" = port ]; then port_up "$(svc_port "$name")"
  else local p; p=$(proc_pid "$name"); [ -n "$p" ] && kill -0 "$p" 2>/dev/null; fi
}
pids_of() {
  local name=$1
  if [ "$(svc_kind "$name")" = port ]; then pid_port "$(svc_port "$name")"
  else is_running "$name" && proc_pid "$name"; fi
  return 0
}

mic_present() { arecord -l 2>/dev/null | grep -q '^card'; }

# Which LLM brain config.yaml selects ("claude" or "rkllama"); defaults to claude.
llm_backend() {
  local b
  b=$(grep -E "^[[:space:]]*backend:[[:space:]]*" "$REPO/config.yaml" 2>/dev/null \
        | head -1 | sed -E 's/.*backend:[[:space:]]*"?([A-Za-z]+)"?.*/\1/')
  echo "${b:-claude}"
}

launch() {  # name
  local name=$1 dir cmd pidf
  dir=$(svc_dir "$name"); cmd=$(svc_cmd "$name"); pidf=$(pidfile "$name")
  # setsid + </dev/null fully detaches so it survives this shell / SSH closing.
  # The new session leader records its own pid (= process-group id) so stop can
  # signal the whole group. PYTHONUNBUFFERED keeps logs live.
  ( cd "$dir" && PYTHONUNBUFFERED=1 setsid bash -c "echo \$\$ > '$pidf'; exec $cmd" \
      >"$LOGDIR/$name.log" 2>&1 </dev/null & )
}

start_one() {
  local name=$1
  if is_running "$name"; then
    printf "  ${C_Y}[skip]${C_0}  %-11s already running\n" "$name"; return 0
  fi
  if [ "$name" = coordinator ] && ! mic_present; then
    printf "  ${C_Y}[skip]${C_0}  %-11s no capture device — connect the USB mic, then: %s start coordinator\n" \
           "$name" "$0"; return 0
  fi
  # The local NPU model is only needed for the rkllama backend; skip it on claude.
  if [ "$name" = rkllama ] && [ "$(llm_backend)" = claude ]; then
    printf "  ${C_Y}[skip]${C_0}  %-11s llm.backend=claude (local LLM not needed)\n" "$name"; return 0
  fi
  # rkllama's prompt cache can be left corrupt by an unclean shutdown, which
  # silently makes the model emit one token then stop (empty replies, no error).
  # Clear it on each start; it's regenerated automatically on first inference.
  if [ "$name" = rkllama ] && [ -n "$RKLLAMA_MODELS" ]; then
    rm -rf "$RKLLAMA_MODELS"/*/cache/* 2>/dev/null || true
  fi
  printf "  [start] %-11s … " "$name"
  launch "$name"
  if [ "$(svc_kind "$name")" = port ]; then
    local port; port=$(svc_port "$name")
    local t=40; [ "$name" = rkllama ] && t=60
    local i=0
    while ! port_up "$port"; do
      sleep 1; i=$((i+1))
      if [ "$i" -ge "$t" ]; then
        printf "${C_R}failed (logs/%s.log)${C_0}\n" "$name"
        tail -n 5 "$LOGDIR/$name.log" 2>/dev/null | sed 's/^/      /'; return 1
      fi
    done
    # rkllama must answer /models before llm_server starts.
    if [ "$name" = rkllama ]; then
      i=0; until curl -fsS "http://127.0.0.1:$port/models" >/dev/null 2>&1; do
        sleep 1; i=$((i+1)); [ "$i" -ge 30 ] && break; done
    fi
    printf "${C_G}up${C_0} (:%s)\n" "$port"
  else
    sleep 3
    if is_running "$name"; then printf "${C_G}up${C_0}\n"
    else printf "${C_R}exited (logs/%s.log)${C_0}\n" "$name"
         tail -n 5 "$LOGDIR/$name.log" 2>/dev/null | sed 's/^/      /'; fi
  fi
}

stop_one() {
  local name=$1 kind pid
  kind=$(svc_kind "$name")
  if ! is_running "$name"; then printf "  ${C_Y}[skip]${C_0}  %-11s not running\n" "$name"; return 0; fi
  if [ "$kind" = port ]; then pid=$(pid_port "$(svc_port "$name")"); else pid=$(proc_pid "$name"); fi
  printf "  [stop]  %-11s (pid %s) … " "$name" "$pid"
  if [ "$kind" = proc ]; then kill -- -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
  else kill "$pid" 2>/dev/null || true; fi
  for _ in 1 2 3 4 5; do is_running "$name" || break; sleep 1; done
  if is_running "$name"; then
    [ "$kind" = proc ] && kill -9 -- -"$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
    sleep 1
  fi
  if is_running "$name"; then printf "${C_R}still up${C_0}\n"
  else [ "$kind" = proc ] && rm -f "$(pidfile "$name")"; printf "${C_G}stopped${C_0}\n"; fi
}

status_one() {
  local name=$1 pids extra=""
  pids=$(pids_of "$name" | tr '\n' ',' | sed 's/,$//')
  [ "$(svc_kind "$name")" = port ] && extra=" :$(svc_port "$name")"
  if [ -n "$pids" ]; then printf "  %-11s ${C_G}● up${C_0}  %s pid %s\n" "$name" "$extra" "$pids"
  else printf "  %-11s ${C_R}○ down${C_0}%s\n" "$name" "$extra"; fi
}

action="${1:-start}"; filter="${2:-}"
targets=("${SERVICES[@]}"); [ -n "$filter" ] && targets=("$filter")

case "$action" in
  start)   echo "Starting Aven services on this board:"; for s in "${targets[@]}"; do start_one "$s"; done ;;
  stop)    echo "Stopping Aven services:"; for s in "${targets[@]}"; do stop_one "$s"; done ;;
  restart) echo "Restarting Aven services:"
           for s in "${targets[@]}"; do stop_one "$s"; done
           for s in "${targets[@]}"; do start_one "$s"; done ;;
  status)  echo "Aven services on this board:"; for s in "${SERVICES[@]}"; do status_one "$s"; done ;;
  *) echo "usage: $0 [start|stop|status|restart] [service]" >&2; exit 2 ;;
esac

[ "$action" != status ] && { echo; echo "Logs: $LOGDIR/<service>.log   ·   status: $0 status"; }
exit 0

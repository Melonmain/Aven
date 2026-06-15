# Host config for Spotify (not auto-applied — copied here for reproducibility)

These live on the host, outside the repo's runtime. To reproduce on a board:

    # 1. Share the Pebble between apps via ALSA dmix (cross-user via ipc_perm 0666)
    sudo cp deploy/asound.conf /etc/asound.conf

    # 2. Let Raspotify/librespot share the host dmix IPC (drop PrivateUsers sandbox)
    sudo mkdir -p /etc/systemd/system/raspotify.service.d
    sudo cp deploy/raspotify-override.conf /etc/systemd/system/raspotify.service.d/override.conf
    sudo systemctl daemon-reload && sudo systemctl restart raspotify

`/etc/raspotify/conf` also needs `LIBRESPOT_NAME="Aven"`, `LIBRESPOT_BACKEND="alsa"`,
`LIBRESPOT_DEVICE="default"`. The coordinator uses ALSA `default` (config
`coordinator.output_device: null`).

## Auto-start on boot (systemd)

`aven.service` runs `start_main_board.sh` as user `melon` at boot, bringing up
rkllama, llm_server, stt, and the coordinator. Install once:

    sudo cp deploy/aven.service /etc/systemd/system/aven.service
    sudo systemctl daemon-reload
    sudo systemctl enable --now aven.service

Manage it like any unit:

    sudo systemctl start|stop|restart aven      # whole stack
    systemctl status aven                        # cgroup shows all 4 services

It's a `Type=oneshot`/`RemainAfterExit` wrapper, so per-service control still
goes through the script (`./start_main_board.sh restart coordinator`). An
`ExecStartPre` waits up to 30s for the USB mic so the coordinator isn't skipped
when USB enumerates slightly after boot. The unit calls the script's `stop` on
shutdown. Logs stay in `logs/<service>.log`; `journalctl -u aven` shows the
start/stop wrapper output.

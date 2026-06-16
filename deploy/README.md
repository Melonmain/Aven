# Host config for Spotify (not auto-applied — copied here for reproducibility)

These live on the host, outside the repo's runtime. To reproduce on a board:

    # 1. Share the Pebble between apps via ALSA dmix (cross-user via ipc_perm 0666)
    sudo cp deploy/asound.conf /etc/asound.conf

    # 2. librespot: share the dmix IPC (drop PrivateUsers) AND launch with
    #    explicit flags so it logs in with cached credentials on boot.
    sudo mkdir -p /etc/systemd/system/raspotify.service.d
    sudo cp deploy/raspotify-override.conf /etc/systemd/system/raspotify.service.d/override.conf
    sudo systemctl daemon-reload && sudo systemctl restart raspotify

The override's `ExecStart` runs librespot with `--name Aven --disable-discovery
--system-cache /var/cache/raspotify` (so it bypasses `/etc/raspotify/conf` env
vars). The coordinator uses ALSA `default` (config `coordinator.output_device: null`).

### Spotify device sign-in (no phone, survives reboots)

`play_music` finds the `Aven` device via Spotify's Web API. A discovery-mode
librespot only appears there after a phone activates it, and that's lost on every
reboot ("the Aven speaker isn't available"). Instead, sign in **once** with a
browser via librespot's OAuth flow, which caches `credentials.json`; with
`--disable-discovery` librespot then logs in on every boot and stays in the API —
no phone, ever.

    sudo bash deploy/spotify_device_auth.sh

It prints a URL; open it in any browser, log in, authorize. The redirect goes to
`http://127.0.0.1:5588`, so if your browser is on another machine, forward the
port first: `ssh -L 5588:localhost:5588 <user>@<board>`. Once `credentials.json`
is written it restarts raspotify, and `Aven` is available to play/stop/resume.

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

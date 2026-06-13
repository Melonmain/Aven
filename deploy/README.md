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

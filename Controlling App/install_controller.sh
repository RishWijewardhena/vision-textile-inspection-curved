#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# install.sh  —  A&C Textile Inspection System Service Control Panel
# Tested on Ubuntu 24.04 LTS
# Run with:  bash install.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

APP_DIR="/opt/thread_control"
AUTOSTART_DIR="$HOME/.config/autostart"
SCRIPT_NAME="system_controller.py"
DESKTOP_NAME="system-control.desktop"
SUDOERS_FILE="/etc/sudoers.d/thread-control"
CURRENT_USER="$(whoami)"
SYSTEMCTL_PATH="$(command -v systemctl)"

if [ -z "$SYSTEMCTL_PATH" ]; then
    echo "ERROR: systemctl not found."
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  A&C Textile Inspection System — Installer"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 1. Install python3-tk if missing
echo "[1/5] Checking python3-tk..."
if ! python3 -c "import tkinter" &>/dev/null; then
    echo "      Installing python3-tk..."
    sudo apt-get update
    sudo apt-get install -y python3-tk
else
    echo "      python3-tk is already installed. ✓"
fi

# 2. Copy app to /opt
echo "[2/5] Installing app to $APP_DIR..."
sudo mkdir -p "$APP_DIR"
sudo cp "$SCRIPT_NAME" "$APP_DIR/"
sudo chmod +x "$APP_DIR/$SCRIPT_NAME"
echo "      Done. ✓"

# 3. Set up passwordless sudo for Thread.service only
echo "[3/5] Configuring sudoers for Thread.service..."
SUDOERS_LINE="$CURRENT_USER ALL=(ALL) NOPASSWD: $SYSTEMCTL_PATH start Thread.service, $SYSTEMCTL_PATH stop Thread.service"

TMPFILE="$(mktemp)"
echo "$SUDOERS_LINE" > "$TMPFILE"

if sudo visudo -c -f "$TMPFILE" &>/dev/null; then
    sudo cp "$TMPFILE" "$SUDOERS_FILE"
    sudo chmod 440 "$SUDOERS_FILE"
    echo "      Sudoers rule added for user '$CURRENT_USER'. ✓"
else
    echo "      WARNING: sudoers validation failed — skipping. You may need to add manually."
fi
rm -f "$TMPFILE"

# 4. Add to GNOME autostart
echo "[4/5] Setting up autostart..."
mkdir -p "$AUTOSTART_DIR"

cat > "$AUTOSTART_DIR/$DESKTOP_NAME" <<EOF
[Desktop Entry]
Type=Application
Name=A&C Textile Inspection System
Comment=Service control panel for Thread.service
Exec=/usr/bin/python3 /opt/thread_control/system_controller.py
Icon=utilities-system-monitor
Terminal=false
Categories=Utility;System;
X-GNOME-Autostart-enabled=true
X-GNOME-Autostart-Delay=15
StartupNotify=false
EOF

echo "      Autostart entry created at $AUTOSTART_DIR/$DESKTOP_NAME ✓"

# 5. Launch the app immediately
echo "[5/5] Launching app..."
nohup /usr/bin/python3 "$APP_DIR/$SCRIPT_NAME" &>/dev/null &
echo "      App launched (PID $!) ✓"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Installation complete!"
echo ""
echo "  ▸ Sudoers rule:  $SUDOERS_FILE"
echo "  ▸ App installed: $APP_DIR/$SCRIPT_NAME"
echo "  ▸ Autostart:     $AUTOSTART_DIR/$DESKTOP_NAME"
echo ""
echo "  The app will auto-start on every login."
echo "  START / STOP buttons require no password."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
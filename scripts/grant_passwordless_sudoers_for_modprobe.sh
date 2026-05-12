#!/bin/bash
set -euo pipefail

# This script installs a sudoers drop-in that permits the system user
# to run specific modprobe commands without a password.
# Run as root: sudo ./create_sudoers_thread_modprobe.sh

if [ "$EUID" -ne 0 ]; then
  echo "This script must be run as root (sudo)." >&2
  exit 1
fi

# Determine the non-root invoking user (prefer SUDO_USER)
if [ -n "${SUDO_USER:-}" ] && [ "${SUDO_USER:-}" != "root" ]; then
  target_user="$SUDO_USER"
else
  target_user="$(logname 2>/dev/null || id -un)"
fi

echo "Installing sudoers rule for user: $target_user"

TMPFILE="$(mktemp)"
cat >"$TMPFILE" <<EOF
${target_user} ALL=(root) NOPASSWD: /sbin/modprobe -r cdc_acm, /sbin/modprobe cdc_acm, /sbin/modprobe -r usb_storage, /sbin/modprobe usb_storage, /sbin/modprobe -r uvcvideo, /sbin/modprobe uvcvideo
EOF

# Validate syntax with visudo
if ! visudo -c -f "$TMPFILE" >/dev/null 2>&1; then
  echo "Visudo validation failed. Aborting." >&2
  rm -f "$TMPFILE"
  exit 1
fi

DEST="/etc/sudoers.d/thread-modprobe"
# Backup existing if present
if [ -f "$DEST" ]; then
  cp "$DEST" "$DEST.bak.$(date +%s)"
fi

install -m 0440 -o root -g root "$TMPFILE" "$DEST"
rm -f "$TMPFILE"

echo "Installed $DEST for user $target_user (mode 0440)."
echo "You can verify with: sudo visudo -c -f $DEST && sudo cat $DEST"

exit 0

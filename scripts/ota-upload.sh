#!/usr/bin/env bash
set -euo pipefail

OTA_URL="${OTA_URL:-http://192.168.22.102:18792}"
FIRMWARE="${1:-build/xiaozhi.bin}"
VERSION="${2:-}"

if [ ! -f "$FIRMWARE" ]; then
    echo "❌ Firmware not found: $FIRMWARE"
    echo "Usage: $0 [path-to-firmware.bin] [version]"
    echo ""
    echo "Examples:"
    echo "  $0                                        # build/xiaozhi.bin, auto version"
    echo "  $0 build/xiaozhi.bin 2.3.1                # explicit version"
    echo "  $0 /tmp/firmware.bin                       # custom path"
    exit 1
fi

SIZE=$(stat -c%s "$FIRMWARE")
echo "📦 Firmware: $FIRMWARE ($(numfmt --to=iec $SIZE))"

if [ -n "$VERSION" ]; then
    echo "🔖 Version: $VERSION"
    RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$OTA_URL/api/firmware/upload" \
        -F "file=@$FIRMWARE" \
        -F "version=$VERSION")
else
    RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$OTA_URL/api/firmware/upload" \
        -F "file=@$FIRMWARE")
fi

HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | head -n -1)

if [ "$HTTP_CODE" = "200" ]; then
    echo "✅ Upload OK ($HTTP_CODE)"
    echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
else
    echo "❌ Upload failed (HTTP $HTTP_CODE)"
    echo "$BODY"
    exit 1
fi

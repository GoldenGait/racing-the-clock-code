#!/bin/bash
# Find the Go2 over Wi-Fi and point DDS at it. The wireless analogue of
# go2_link.sh (which handles the USB ethernet adapter).
#
#   ./go2_wifi_link.sh            # discover + patch
#   ./go2_wifi_link.sh 192.168.12.1   # skip discovery, use this address
#
# Prints the --iface to pass to go2_check.py / go2_client.py.
#
# Why unicast: DDS discovery multicasts to 239.255.0.1, and access points
# routinely drop or rate-limit multicast. Over Wi-Fi the SDK will subscribe
# happily and then receive nothing. Patching a unicast Peer avoids that.
set -u
HINT="${1:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"

# 1. which device is Wi-Fi
WIF=$(networksetup -listallhardwareports 2>/dev/null \
      | awk '/Hardware Port: Wi-Fi/{getline; print $2}')
[ -n "$WIF" ] || { echo "no Wi-Fi device found (is this a Mac?)"; exit 1; }
SSID=$(networksetup -getairportnetwork "$WIF" 2>/dev/null | sed 's/^.*: //')
MYIP=$(ipconfig getifaddr "$WIF" 2>/dev/null)
echo "Wi-Fi device : $WIF"
echo "SSID         : ${SSID:-<not associated>}"
echo "my address   : ${MYIP:-<none>}"
[ -n "$MYIP" ] || { echo "not on a Wi-Fi network. Join the dog's AP (Unitree_Go2_...) or your LAN first."; exit 1; }

# 2. candidate addresses for the robot, best guess first.
#    AP mode : the dog IS the gateway.
#    Wi-Fi mode: the gateway is your router and the dog is some other lease,
#    so sweep the subnet and report what answers.
GW=$(route -n get -ifscope "$WIF" default 2>/dev/null | awk '/gateway/{print $2}')
CANDS="$HINT $GW 192.168.12.1 192.168.123.161"
echo "trying       : $(echo $CANDS | xargs)"

DOG=""
for ip in $CANDS; do
  [ -n "$ip" ] || continue
  # Only trust the gateway when it IS the dog, i.e. AP mode. On a router-based
  # network the gateway answers every time and is not the robot.
  if [ "$ip" = "$GW" ] && [ "$ip" != "$HINT" ] && [ "$ip" != "192.168.12.1" ]; then
    echo "  $ip is your router (gateway), skipping - the dog is elsewhere on this LAN"
    continue
  fi
  printf "  %-16s " "$ip"
  if ping -c 2 -t 2 "$ip" >/dev/null 2>&1; then
    echo "responds"; DOG="$ip"; break
  fi
  echo "no reply"
done
if [ -z "$DOG" ]; then
  echo
  echo "none of the usual addresses answered. Sweeping the subnet for the dog"
  echo "(this is the Wi-Fi Mode case, where the gateway is your router)..."
  BASE=$(echo "$MYIP" | cut -d. -f1-3)
  for i in $(seq 1 254); do (ping -c 1 -t 1 "$BASE.$i" >/dev/null 2>&1 &) ; done
  sleep 4
  echo "  live hosts on $BASE.0/24:"
  arp -a -i "$WIF" 2>/dev/null | awk '{print "    "$2" "$4}' | tr -d '()'
  echo
  echo "  The Go2 is the one whose MAC starts with a Unitree OUI, or read the"
  echo "  robot IP straight off the Unitree app. Then re-run:"
  echo "    $0 <that-ip>"
  exit 1
fi

# 3. unicast DDS at it
echo
echo "patching cyclonedds to unicast at $DOG"
python "$HERE/fix_dds_macos.py" --peer "$DOG" || exit 1

echo
echo "READY: robot at $DOG"
echo
echo "  python go2_check.py state  --iface $MYIP"
echo "  python go2_check.py camera --iface $MYIP"
echo
echo "(--iface is this machine IP, not $WIF: the patched cyclonedds config"
echo " selects the interface by address.)"
echo
echo "If DDS still sees nothing, quit Terminal (Cmd-Q) and reopen: macOS binds"
echo "the Local Network permission per process, and a stale shell stays deaf."

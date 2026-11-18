#!/bin/bash
# Find whichever interface the Go2 is actually on, give it the right IP, print
# the name to pass as --iface. The USB ethernet adapter renumbers constantly
# (en20 -> en8 -> en22 -> ...), so hardcoding the name never survives a replug.
#
#   sudo ./go2_link.sh
#
# Prints e.g.  READY: --iface en9
set -u
DOG=192.168.123.161
MYIP=192.168.123.99

if [ "$(id -u)" != "0" ]; then
  echo "needs sudo (tcpdump + ipconfig): sudo $0"; exit 1
fi

# candidate wired interfaces: UP+RUNNING, not loopback/wifi/tunnels
CANDS=$(ifconfig | awk '/^en[0-9]+:/{name=substr($1,1,length($1)-1)}
                        /status: active/{if(name!="")print name; name=""}')
if [ -z "$CANDS" ]; then
  echo "no active wired interfaces found. is the adapter plugged in?"; exit 1
fi
echo "active interfaces: $CANDS"

FOUND=""
for IF in $CANDS; do
  printf "  probing %s ... " "$IF"
  OUT=$(mktemp)
  tcpdump -i "$IF" -c 1 -n "host $DOG" > "$OUT" 2>/dev/null &
  PID=$!
  sleep 4
  kill $PID 2>/dev/null; wait $PID 2>/dev/null
  if grep -q "$DOG" "$OUT" 2>/dev/null; then
    echo "DOG FOUND"; FOUND=$IF; rm -f "$OUT"; break
  fi
  echo "nothing"; rm -f "$OUT"
done

if [ -z "$FOUND" ]; then
  echo "the dog is not visible on any interface. check the cable and that the robot is powered on."
  exit 1
fi

CUR=$(ifconfig "$FOUND" | awk '/inet /{print $2}' | head -1)
if [ "$CUR" != "$MYIP" ]; then
  echo "  setting $MYIP on $FOUND (was ${CUR:-none})"
  ipconfig set "$FOUND" MANUAL "$MYIP" 255.255.255.0
  sleep 1
fi
ifconfig "$FOUND" | grep "inet " | sed 's/^/  /'
echo
echo "READY: robot on $FOUND"
echo
# Which value --iface wants depends on the cyclonedds config that is installed,
# and the error for guessing wrong is identical either way:
#   "X: does not match an available interface"
if grep -q '__ATTR__="\$__IF_NAME__\$"' "$(python -c 'import unitree_sdk2py.core.channel_config as c; print(c.__file__)' 2>/dev/null)" 2>/dev/null; then
  echo "  cyclonedds config: PATCHED (selects interface by address)"
  echo "  python go2_check.py state  --iface $MYIP"
  echo "  python go2_check.py camera --iface $MYIP"
else
  echo "  cyclonedds config: original (selects interface by name)"
  echo "  python go2_check.py state  --iface $FOUND"
  echo "  python go2_check.py camera --iface $FOUND"
fi
echo
echo "NB the Go2 publishes multicast-only (239.255.0.1), so the original config"
echo "is what you want on the cable. fix_dds_macos.py disables multicast and"
echo "will make DDS subscribe successfully and receive nothing."
echo "if python still gets no DDS, quit Terminal (Cmd-Q) and reopen -- macOS binds"
echo "the Local Network permission per process and a stale shell stays deaf."

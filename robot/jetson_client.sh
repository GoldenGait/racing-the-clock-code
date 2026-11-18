#!/bin/bash
# Run the navigation client ON the dog's expansion computer (Jetson), so the DDS motion
# commands and the RealSense stay on the dog's internal network and only the model RPC
# crosses WiFi. Copy robot/go2_client.py, robot/rs_server.py and this file to the home directory on the
# Jetson (scp -o BindAddress=192.168.123.99 ... unitree@192.168.123.18:~/ -- /tmp is wiped on reboot), then:
#
#   MODEL_HOST=<model-server ip> MODEL_PORT=8877 bash $HOME/jetson_client.sh --category chair --budget 400
#
# If the Jetson cannot reach the model host directly, open a reverse tunnel from the laptop
# first (laptop -> Jetson): ssh -R 18877:localhost:18877 unitree@192.168.123.18, with the
# laptop's own ssh -L 18877:localhost:8877 <model host> running, and use
# MODEL_HOST=127.0.0.1 MODEL_PORT=18877.
set -u
MODEL_HOST=${MODEL_HOST:-127.0.0.1}
MODEL_PORT=${MODEL_PORT:-18877}
cd "$HOME"
pgrep -f "python3 $HOME/rs_server.py" >/dev/null || \
  { nohup python3 $HOME/rs_server.py --dev 4 > $HOME/rs_server.log 2>&1 < /dev/null & sleep 2; }
curl -s http://127.0.0.1:8090/info >/dev/null || { echo "rs_server not answering; see $HOME/rs_server.log"; exit 1; }
exec python3 $HOME/go2_client.py --host "$MODEL_HOST" --port "$MODEL_PORT" --iface eth0 \
  --cam http://127.0.0.1:8090/frame.jpg "$@"

# Go2 deployment

Running the budget-conditioned VLN on a Unitree Go2 EDU. No ROS: DDS to the robot
via `unitree_sdk2py`, JSON-lines TCP to an inference server on a GPU box.

```
laptop ──DDS over ethernet── Go2 (192.168.123.161)
   └────JSON/TCP :8877────── nav_server.py on the GPU host
```

## Pieces

| file | what it does |
|---|---|
| `nav_server.py` | the model behind a TCP API. Wraps the bench's `Model` class, so a real run uses the exact code path every frozen-400 number came from. Holds the budget countdown, the training-parity instruction, and full sim-schema trace logging. |
| `go2_client.py` | the robot loop. Capture at ~1 fps, send frames, execute the first `--exec-n` of the four returned primitives, repeat. |
| `go2_check.py` | link, camera and motion checks. Run these before anything walks. |
| `go2_jog.py` | manual positioning, one-shot or interactive `drive` mode. |
| `run_uninavid.py` | standalone inference over a video file. No robot, no habitat. |
| `go2_link.sh` | finds the renumbering USB adapter, sets `192.168.123.99`. |
| `go2_wifi_link.sh` | same idea over Wi-Fi, for AP mode or a shared LAN. |
| `fix_dds_macos.py` | switches cyclonedds to unicast discovery. See the warning below. |
| `start_servers.sh` | brings up both servers: ours on :8877, base Uni-NaVid on :8878. |

## Running it

```bash
# on the GPU host
bash start_servers.sh

# on the laptop, wired to the dog
sudo ./go2_link.sh                       # sets the static IP, finds the adapter
python go2_check.py state                # DDS alive?
python go2_check.py camera               # frame.jpg should be the room
python go2_client.py --host <gpu-host> --port 8877 --category chair --budget 400
```

If the GPU host is not directly routable, tunnel it:
`ssh -f -N -L 8877:localhost:8877 -L 8878:localhost:8878 <gpu-host>`.

## Things that cost hours

**The Go2 publishes DDS multicast-only.** Every sample goes to `239.255.0.1` on
ports 7400/7401. `fix_dds_macos.py` disables multicast in favour of a unicast
peer, which makes DDS subscribe successfully and then receive nothing. It exists
for the opposite failure, where macOS refuses to join the multicast group. On a
working wired link, leave the config alone (`--restore`). Confirm which case you
are in with `sudo tcpdump -i <if> -n "udp and portrange 7400-7600"`.

**`--iface` means different things depending on that config**, and guessing wrong
gives the same message either way (`X: does not match an available interface`):
the original config selects the interface by name (`en23`), the patched one by
address (`192.168.123.99`). `--iface auto` sidesteps it by finding whichever
interface currently holds a `192.168.123.x` address, which also survives the USB
adapter renumbering itself on every replug.

**Frame/action interleaving has to match the simulator.** One observation is
captured *before* every primitive, and a decision consumes the frames
accumulated since the last one. Grabbing N frames from a standstill and then
executing N actions hands the model N copies of the same pose and no motion
history.

**The budget counts what was executed, not what was predicted.** The model
returns four primitives and the client executes two, so the countdown has to
follow the client.

**A `stop` in the tail of a prediction is a forecast, not a command.** Acting on
one beyond `--exec-n` ends the episode early. Only a stop inside the executed
slice, or an exhausted budget, should stop the run.

**Camera FOV.** The Go2 front camera is a wide fisheye; the model was trained at
79 degrees. `--src-hfov 110` is the measured best: at 110 the model
discriminates chair from toilet on real frames, at 130/150 it emits the same
actions regardless of the target.

**Obstacle avoidance is on by default.** The model has no collision sense at all
and walked into a wall on the first unguarded run. `--no-avoid` turns the guard
off and drives `SportClient` raw.

**macOS binds Local Network permission per process.** A shell that was denied
once stays deaf no matter how correct the config is. Quit Terminal entirely
(Cmd-Q) and reopen before debugging anything else.

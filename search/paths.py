"""Where the data lives.

Every script reads scenes, traces, datasets and checkpoints from one root directory.
Set it with the RTC_DATA environment variable; docs/INSTALL.md lists the layout.
"""
import os

DATA_ROOT = os.environ.get('RTC_DATA', os.path.expanduser('~/rtc_data'))

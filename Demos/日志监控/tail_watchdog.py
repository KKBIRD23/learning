import watchdog
import time
import os
import sys


class ProcessTransientFile(pyinotify.ProcessEvent):
    def process_in_modify(self, event):
        line = file.readline()
        if line:
            print(line, end='')


if __name__ == '__main__':
    filename = sys.argv[1]
    file = open(filename, 'r')
    st_results = os.stat(filename)
    st_size = st_results[6]
    file.seek(st_size)

    wm = pyinotify.WatchManager()
    notifier = pyinotify.Notifier(wm)
    wm.watch_transient_file(filename, pyinotify.IN_MODIFY, ProcessTransientFile)
    notifier.loop()

#!/bin/bash
#sudo bash -c 'echo -1 > /sys/module/usbcore/parameters/autosuspend'
echo "Vision program starting..."

until ./vision.py -c 0 -v ; do
    if [ $? -eq 1 ]; then
        echo "Vision program exited with code $? (safe shutdown via signal). Closing wrapper."
        exit 1
    fi
    echo "Vision program crashed with code $?. Respawning..."
    sleep 1
done

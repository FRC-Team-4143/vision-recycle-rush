#!/bin/bash
echo -1 > /sys/kernel/debug/tegra_hdmi/hotplug
echo 4 > /sys/class/graphics/fb0/blank
echo -1 > /sys/module/usbcore/parameters/autosuspend

sleep 2

cd /home/ubuntu/vision-recycle-rush
echo "Vision program starting..."

until ./vision.py -c 0 -v ; do
    if [ $? -eq 1 ]; then
        echo "Vision program exited with code $? (safe shutdown via signal). Closing wrapper."
        exit 1
    fi
    echo "Vision program crashed with code $?. Respawning..."
    sleep 1
done

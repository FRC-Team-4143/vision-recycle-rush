#!/bin/bash

echo -1 > /sys/module/usbcore/parameters/autosuspend
echo -1 > /sys/kernel/debug/tegra_hdmi/hotplug
echo 4 > /sys/class/graphics/fb0/blank

uvcdynctrl --set='Exposure, Auto' 1 

#uvcdynctrl --set='Exposure (Absolute)' 5 # Min
#uvcdynctrl --set='Exposure (Absolute)' 20000 # Max
uvcdynctrl --set='Exposure (Absolute)' 10

#uvcdynctrl --set='Brightness' 30   # Min
#uvcdynctrl --set='Brightness' 255   # Max
uvcdynctrl --set='Brightness' 30

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

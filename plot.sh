#!/bin/sh

logdir=./logs/$1
if [ "${logdir}" = "./logs/" ]; then
    echo "Usage: ./plot.sh < tensorboard log file date in ./logs/ folder, eg: 2021-11-28 > <optional: port>"
    return
fi
if [ -d ${logdir} ]; then
    echo "Tensorboard logdir = '${logdir}'"
    if [ " $2" = " " ]; then 
        tensorboard --logdir=${logdir}
    else
        tensorboard --logdir=${logdir} --port=$2
    fi
else
    echo "Directory '${logdir}' not found."
fi
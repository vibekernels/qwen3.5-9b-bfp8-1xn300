# Reference System Info

This directory contains some system info captured from the system delivering the tok/s rates in the repo README. It can be helpful if your system is slow and you're not sure why.

## Capturing System Info

```bash
tt-smi -f tt-smi-snapshot.json
lscpu > lscpu.txt
cat /proc/meminfo > meminfo.txt
mount > mount.txt
dmesg > dmesg.txt
```

# sudo sh -c "ulimit -n 65535 && exec su $LOGNAME"
sudo service tgt stop
mkdir -p results
mkdir -p logs
mkdir -p imgs
sudo mkdir -p /dev/mqueue 
sudo mount -t mqueue none /dev/mqueue
echo 8192 | sudo tee /proc/sys/fs/mqueue/msg_max
echo 8192 | sudo tee /proc/sys/fs/mqueue/msgsize_max
sudo insmod /home/weiweij/workshop/CNEX-INFO-new/codebase/trunk_wl/nexus_ftl/nexus.ko
sudo insmod /home/weiweij/workshop/CNEX-INFO-new/codebase/trunk_wl/nexus_ftl/nexus_disk.ko

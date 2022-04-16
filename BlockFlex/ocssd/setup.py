#!/usr/bin/python3

from macros import *

def set_target(ids):
    for id in ids:
        log_msg(f"###### Setup iSCSI target {id}")
        log_msg(ISCSI_TARGET_IMGS)
        TGT_BS=ISCSI_TARGET_IMGS[id]
        ISCSI_TARGET_NAME=ISCSI_TARGET_NAMES[id]

        run_command(f"mkdir -p {IMG_DIR}")
        run_command(f"fallocate -l {DUMMY_DISK_SIZE} {TGT_BS}")

        run_command(f"sudo {PFE_USR_DIR}/tgtadm  --lld iscsi --op new --mode target --tid {id} -T {ISCSI_TARGET_NAME}")
        run_command(f"sudo {PFE_USR_DIR}/tgtadm  --lld iscsi --op new --mode logicalunit --tid {id} --lun 1 -b {TGT_BS}")
        run_command(f"sudo {PFE_USR_DIR}/tgtadm   --lld iscsi --op bind --mode target --tid {id} -I ALL")
    run_command(f"sudo {PFE_USR_DIR}/tgtadm   --lld iscsi --op show --mode target")

def set_initiator(ids):
    run_command(f"sudo service open-iscsi stop && sudo service open-iscsi start")
    run_command(f"sudo iscsiadm --mode discovery --type sendtargets --portal {ISCSI_TARGET_IP}")
    for id in ids:
        ISCSI_TARGET_NAME=ISCSI_TARGET_NAMES[id]
        run_command(f"sudo iscsiadm --mode node --targetname {ISCSI_TARGET_NAME} --portal {ISCSI_TARGET_IP}:3260 --login")

def run_tgtd():
    run_command(f"mkdir -p {PFE_LOG_DIR}")
    run_command(f"sudo rm {IO_HEAD_LOG} {IO_DATA_LOG} {IO_HEAD_SPLIT_LOG} {ERR_BLOCK_LOG} {TGTD_LOG}")
    run_command(f"touch {IO_HEAD_LOG} {IO_DATA_LOG} {IO_HEAD_SPLIT_LOG} {ERR_BLOCK_LOG} {TGTD_LOG}")
    run_command(f"sudo chmod 666 {IO_HEAD_LOG} {IO_DATA_LOG} {IO_HEAD_SPLIT_LOG} {ERR_BLOCK_LOG} {TGTD_LOG}")

    # Run tgtd
    run_command(f"sudo {PFE_USR_DIR}/tgtd -f --pfe-io-header-log {IO_HEAD_LOG} --pfe-fail-type-tgtd 0 --pfe-err-blk {ERR_BLOCK_LOG} --pfe-io-data-log {IO_DATA_LOG} --pfe-enable-record 0 >  {TGTD_LOG} 2>&1 &")
    # run_command(f"sudo {PFE_USR_DIR}/tgtd --pfe-io-header-log {IO_HEAD_LOG} --pfe-fail-type-tgtd 0 --pfe-err-blk {ERR_BLOCK_LOG} --pfe-io-data-log {IO_DATA_LOG} --pfe-enable-record 0 >  {TGTD_LOG} 2>&1 ")

def kill_tgtd():
    run_command("sudo killall -9 tgtd")

def clean_logs():
    #LOGS = [IO_HEAD_LOG, IO_DATA_LOG, IO_HEAD_SPLIT_LOG, ERR_BLOCK_LOG, TGTD_LOG]
    LOGS = [IO_HEAD_LOG, IO_DATA_LOG, IO_HEAD_SPLIT_LOG, ERR_BLOCK_LOG]
    for LOG in LOGS:
        if os.path.exists(LOG):
            run_command(f"cat /dev/null > {LOG}")

def start_log():
    run_command(f"touch {PFE_USR_DIR}/start_log")

def stop_log():
    run_command(f"rm -f {PFE_USR_DIR}/start_log")

def start_replay():
    run_command(f"touch {PFE_USR_DIR}/start_replay")

def stop_replay():
    run_command(f"rm -f {PFE_USR_DIR}/start_replay")

def clear_queues_and_flags():
    for queue in QUEUES:
        run_command(f"rm -f {queue}")
        run_command(f"touch {queue}")
    run_command(f"rm -f {HARVEST_FLAG}")


def obtain_disk_id(ids):
    disk_ids = []
    raw_output = run_command("sudo lsblk -d", output=True)
    raw_output = raw_output['stdout']
    for line in raw_output.strip().split("\n"):
        line = list(filter(lambda x: len(x) > 0, line.strip().split(" ")))
        if line[3] == DUMMY_DISK_SIZE_STRING:
            disk_ids.append(line[0])

    # FIXME: find a way to restart the script instead of exit
    assert(len(disk_ids) == len(ids))
        

    log_msg("Virtual disk ids", disk_ids)

    return disk_ids

def kill_vms(vms):
    for vm in vms:
        run_command(f"sudo virsh shutdown {vm}")

    # check if all vms are shutdown  
    while True:
        raw_output = run_command("sudo virsh list --all --state-shutoff", output=True)
        raw_output = raw_output['stdout']
        all_shutdown = True
        for vm in vms:
            if vm not in raw_output:
                all_shutdown = False
                break
        if not all_shutdown:
            time.sleep(3)
        else:
            break

def start_vms(vms):
    for vm in vms:
        run_command(f"sudo virsh start {vm}")

    # check if all vms are running
    while True:
        raw_output = run_command("sudo virsh list --all --state-running", output=True)
        raw_output = raw_output['stdout']
        all_running = True
        for vm in vms:
            if vm not in raw_output:
                all_running = False
                break
        if not all_running:
            time.sleep(3)
        else:
            break
    
    # test ssh connection
    log_msg("Waiting for ssh connections (may take a while)")
    for vm in vms:
        while True:
            output = run_command(f"ssh {vm} 'ls'", error=True, verbose=False)
            output = output['stderr']
            if "No route to host" in output:
                time.sleep(3)
            else:
                break

def attach_device(disk_ids, vms):
    for i, vm in enumerate(vms):
        run_command(f"sudo virsh attach-disk {vm} /dev/{disk_ids[i]} --target vdb")

def detach_device(vms):
    for vm in vms:
        raw_output = run_command(f"ssh {vm} 'ls /dev/vd*'", output=True)
        raw_output = raw_output['stdout']
        ## FIXME: by default we assume the last disk in VM is the virtual disk 
        disk_id = raw_output.strip().split("\n")[-1].split("/")[-1]
        ## we only find the default disk -- no additional virtual disks are attached yet
        if "vda" in disk_id:
            continue

        run_command(f"sudo virsh detach-disk {vm} --target {disk_id}")

def run_test_workloads(vm):
    if vm:
        run_command("ssh vm0 'ls' ")
    else:
        run_command("sudo dd if=/dev/zero of=/dev/sde bs=1M")


if __name__ == "__main__":
    ids = [1,2]
    vms = ["vm0", "vm1"]
    VERBOSE = True
    WITH_VM = False

    if WITH_VM:
        detach_device(vms)
        kill_vms(vms)

    kill_tgtd()
    clean_logs()
    stop_log()
    stop_replay()
    clear_queues_and_flags()

    run_tgtd()
    start_log()
    set_target(ids)
    set_initiator(ids)

    disk_ids = obtain_disk_id(ids)

    if WITH_VM:
        start_vms(vms)
        attach_device(disk_ids, vms)

    # run_test_workloads(WITH_VM)

    start_replay()

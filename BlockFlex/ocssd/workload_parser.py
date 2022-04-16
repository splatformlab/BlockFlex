import time
from joblib import Parallel, delayed
import os
from macros import *

def open_trace(filename):
    traces = []
    with open(filename) as open_f:
        for i, line in enumerate(open_f):
            if i >= 5000000:
                break
            
            raw = [_ for _ in line.strip().split(" ") if _ != ""]
            
            ts, mode, off, size = float(raw[3]), raw[6][0], int(raw[7]) * 512, int(raw[9]) * 512
            traces.append((ts, mode, off, size))
    return traces

def run_trace(filename, disk_id, vm_type):
    log_msg("Start parsing", vm_type)
    traces = open_trace(filename)
    log_msg("Finished parsing", vm_type)
    start = time.time()
    writes, write_size = 0, 0
    reads, read_size = 0, 0
    total_s = 0
    PAGE_SIZE = 16384
    assert(disk_id not in ['/dev/sda', '/dev/sdb', '/dev/sdc', '/dev/sdd'])
    disk = open(disk_id,'r+b')

    # traces = [(0, "W", 0, 32768), (0, "R", 0, 32768)]
    #traces = [(0, "R", 0, 32768)]
    if vm_type == 'regular':
        while not os.path.exists(HARVEST_FLAG):
            time.sleep(3)
    
    log_msg("Start replaying", vm_type)
    aggregated_read = None
    for t in traces:
        if t[1] == 'R':
            reads += 1
            read_size += t[3]
            if aggregated_read is None:
                aggregated_read = [t[0], "R", t[2], t[3]]
            else:
                if aggregated_read[2] + aggregated_read[3] != t[2]:
                    # flush previous read
                    disk.seek(aggregated_read[2])
                    data = disk.read(aggregated_read[3])
                    # print(aggregated_read[2]//4096, aggregated_read[3]//4096)
                    aggregated_read = [t[0], "R", t[2], t[3]]
                else:
                    aggregated_read[3] += t[3]
            
            if aggregated_read[3] >= PAGE_SIZE:
                disk.seek(aggregated_read[2])
                data = disk.read(aggregated_read[3])
                # print(aggregated_read[2]//4096, aggregated_read[3]//4096)
                aggregated_read = None

        elif t[1] == "W":
            writes += 1
            write_size += t[3]
            s0 = time.time()
            disk.seek(t[2])
            disk.write(bytearray(t[3]))
            e0 = time.time()
            total_s += e0 - s0

    disk.close()
    end = time.time()

    log_msg(disk_id, writes, write_size, reads, read_size, end-start, total_s)


# TRACE_FILE = "/home/js39/software/hs_js39/pagerank_offset.trace"
TRACE_FILES = ["/home/js39/software/hs_js39/ml_prep_offset.trace", "/home/js39/software/hs_js39/ycsb_offset.trace"]
DISK_IDS = ['/dev/sde', '/dev/sdf']
TYPES = ['harvest', 'regular']

results = Parallel(n_jobs=2)(delayed(run_trace)(TRACE_FILES[i], DISK_IDS[i], TYPES[i]) for i in range(2))
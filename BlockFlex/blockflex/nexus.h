/********************************************************************
* FILE NAME: utest.h
*
*
* PURPOSE: headerfile of this application. 
*
* 
* NOTES:
*
* 
* DEVELOPMENT HISTORY: 
* 
* Date Author Release  Description Of Change 
* ---- ----- ---------------------------- 
*2014.6.27, dxu, initial coding.
*
****************************************************************/
#ifndef _NEXUS_H
#define _NEXUS_H

#include <linux/types.h>
#include <errno.h>
#include <stdio.h>
//#include "../../cfg/flash_cfg.h"
#include "/home/breidys2/CNEX-INFO-new/codebase/cfg/flash_cfg.h"

#define NEXUS_VER "R010 D20150505 14:30"

#ifndef OK
#define OK 0
#endif

#ifndef ERROR
#define ERROR (-1)
#endif

#define GOOD_PPA 1
#define BAD_PPA  0

#define pdebug

#ifdef pdebug
#define PRINT printf
#else
#define PRINT 
#endif

//just for compile reason
#ifndef FLASH_SKHYNIX
#define SB_BITS 0
#define WL_BITS 0
#define SB_MIN 0
#define SB_NUM 0
#define WL_NUM 0
#define CFG_NAND_BLOCK_OFFSET 0
#endif

#define DISPLAY(fmt, arg...) printf(fmt"\n", ##arg)

#define CHANNEL_COUNT           0x4098c
#define MTDSZ           0x2104


#define MAX_CMD_LEN 10
#define MAX_FILE_NAME 100

#define LINE_BYTES 0x10
#define LINE_BYTES_MASK (LINE_BYTES - 1)

#define META_SIZE 16
#define META_RAWSIZE (256+48)  // 304, this value used to true.  moving forward it is dynamically  read from HW . see METARAW_SIZE (dev->raw_size)
#define PAGE_SIZE  0x1000
#define BLOCK_SIZE 0x1000
#define CQE_SIZE 16
#define SQE_SIZE 64
#define MAX_Q_DEPTH 10240

#define BURST_DWORD_NUM     128
#define DDR_ADDR_BUS_NUM    29
#define DDR_ADDR_BUS        ((1 << DDR_ADDR_BUS_NUM) -1)      // 29 address bus
#define DDR_DATA_BUS_NUM    32
#define DDR_DATA_BUS        0xffffffff                        // 32 data bus
#define DDR_ADDR_DWORD      (DDR_ADDR_BUS - 3) 				  // dword, 0 and 1 bit is 0
#define PATTERN_NUM         8

// from "../ktest/ktest_func_test.h"
#define CH_INC 0
#define EP_INC  1
#define PL_INC  2
#define ADDR_FIELDS_SHIFT_CH 0 
#define ADDR_FIELDS_SHIFT_EP CH_BITS

//#define BLOCK_NUM (1 << BL_BITS)
#define LUN_NUM   (1 << LN_BITS)
#define CH_NUM    (1 << CH_BITS)
#define PL_NUM    (1 << PL_BITS)
#define EP_NUM    (1 << EP_BITS)
#define BL_BITS_NUM (32 - CH_BITS - EP_BITS - PL_BITS - LN_BITS - PG_BITS)
#define SECTORS_PER_CELL (1 << (PL_BITS + EP_BITS))  // 4x4 = 16 or 2x4 = 8

#define PAGE255   255
#define EP3       3
#define BAD_ADDR_NUM  0x80000
#define KTEST_DEV "/dev/ktest"
#define KTEST_DEV0 "/dev/ktest0"
#define NEXUS_DEV  "/dev/nexus"
#define DEVICE_INDEX 0
#define DEV_NAME_SIZE  16
#define HOST_MAC_ADDR       0x2120
#define NAMESPACE_MAC_BASE  0x10000
#define LOCATION_BIT        48

#define NEXUS_MAX_QUEUE       30

typedef signed char s8;
typedef unsigned char u8;
typedef signed short s16;
typedef unsigned short u16;
typedef signed int s32;
typedef unsigned int u32;
typedef signed long s64;
typedef unsigned long u64;


/* nexus commands include nexus standard and cnex specify cmd*/
enum nexus_opcode {
    nvme_cmd_flush          = 0x00,
    nvme_cmd_write          = 0x81,    //0x01
    nvme_cmd_read           = 0x02,
    nvme_cmd_write_uncor    = 0x04,
    nvme_cmd_compare        = 0x05,
    nvme_cmd_dsm            = 0x09,
    nvme_cmd_rdlbatoecpu    = 0x82,
    nvme_cmd_rdppatoecpu    = 0x86,
    nvme_cmd_ersppa   = 0x90,
    nvme_cmd_wrppa    = 0x91,
    nvme_cmd_rdppa    = 0x92,
    nvme_cmd_dealloc  = 0x94,
    nvme_cmd_wrpparaw = 0x95,
    nvme_cmd_rdpparaw = 0x96,
    nvme_cmd_wrmem    = 0x99,
    nvme_cmd_rdmem    = 0x9a,
    nvme_cmd_wrraid   = 0x9d,
    nvme_cmd_loadraid = 0x9e,
    nvme_cmd_wrxordata = 0xa1,    
    nvme_cmd_rdraid   = 0xa2,
    nvme_cmd_rdpparaid = 0xb2,
    nvme_cmd_rdppatomem = 0xb6,
    nvme_cmd_svvpc    = 0xd1,
    nvme_cmd_ldvpc    = 0xd2,
    nvme_cmd_manuinit = 0xe0,
    nvme_cmd_pwron    = 0xe4,
    nvme_cmd_pwrdwn   = 0xe8,
    nvme_cmd_pwrdwnf  = 0xec,
    nvme_cmd_ldftl    = 0xea,
    nvme_cmd_wrlbamem = 0xed,
    nvme_cmd_rdlbamem = 0xee,
    nvme_cmd_wrppamem = 0xf5,
    nvme_cmd_rdppamem = 0xf6,
    nvme_cmd_ldraid   = 0xfe,
};

struct nvme_identify {
    __u8            opcode;
    __u8            flags;
    __u16           command_id;
    __le32          nsid;
    __le16          prp1_offt;
    __le16          prp2_offt;
    __u32           rsvd2[3];
    __le64          prp1;
    __le64          prp2;
    __le32          cns;
    __u32           rsvd11[5];
};

struct nvme_features {
    __u8            opcode;
    __u8            flags;
    __u16           command_id;
    __le32          nsid;
    __u64           rsvd2[2];
    __le64          prp1;
    __le64          prp2;
    __u8            fid;
    __u8            sel;
    __u16           sv;
    __le32          dword11;
    __u32           filelen;
    __le16          prp1_offt;
    __le16          prp2_offt;
    __u32           rsvd12[2];
};

struct nvme_create_cq {
    __u8            opcode;
    __u8            flags;
    __u16           command_id;
    __u32           rsvd1[5];
    __le64          prp1;
    __u64           rsvd8;
    __le16          cqid;
    __le16          qsize;
    __le16          cq_flags;
    __le16          irq_vector;
    __u32           rsvd12[4];
};

struct nvme_create_sq {
    __u8            opcode;
    __u8            flags;
    __u16           command_id;
    __u32           rsvd1[5];
    __le64          prp1;
    __u64           rsvd8;
    __le16          sqid;
    __le16          qsize;
    __le16          sq_flags;
    __le16          cqid;
    __u32           rsvd12[4];
};

struct nvme_delete_queue {
    __u8            opcode;
    __u8            flags;
    __u16           command_id;
    __u32           rsvd1[9];
    __le16          qid;
    __u16           rsvd10;
    __u32           rsvd11[5];
};

struct nvme_abort_cmd {
    __u8            opcode;
    __u8            flags;
    __u16           command_id;
    __u32           rsvd1[9];
    __le16          sqid;
    __le16          cid;
    __u32           rsvd11[5];
};

struct nvme_asye_cmd {
    __u8            opcode;
    __u8            flags;
    __u16           command_id;
    __u32           rsvd1[15];
};

struct nvme_download_firmware {
    __u8            opcode;
    __u8            flags;
    __u16           command_id;
    __u32           rsvd1[5];
    __le64          prp1;
    __le64          prp2;
    __le32          ndw;
    __le32          offset;
    __le32          prp1_offt;
    __u32           rsvd12[3];
};

struct nvme_activate_firmware{
    __u8            opcode;
    __u8            flags;
    __u16           command_id;
    __u32           rsvd1[9];
    __le16          aafs;
    __u16           rsvd10;
    __u32           rsvd11[5];    
};

struct nvme_format_cmd {
    __u8            opcode;
    __u8            flags;
    __u16           command_id;
    __le32          nsid;
    __u64           rsvd2[4];
    __le32          cdw10;
    __u32           rsvd11[5];
};

struct nvme_ppa_command {
    __u8            opcode ;
    __u8            flags;          /*data transfer   PRP or SGL*/
    __u16           command_id;
    __le32          nsid;
    __le32          cdw2[2];
    __le64          metadata;
    __le64          prp1;
    __le64          prp2;
    __le64          start_list;
    __le16          nlb;    
    __le16          control;       /*CDW12 [16:31] PRINFO FUA LR */
    __le32          dsmgmt;       
    __le32          reftag;
    __le16          apptag;
    __le16          appmask;
};

struct nvme_get_log_page{
    __u8            opcode;
    __u8            flags;
    __u16           command_id;
    __le32          nsid;    
    __le16          prp1_offt; 
    __le16          prp2_offt; 
    __u32           rsvd2[3];
    __le64          prp1;
    __le64          prp2;
    __u16           lid;
    __u16           ndw;
    __u32           rsvd11[5];    
};

struct nvme_rw_regspace {        /* RW controller register && DDR space */
    __u8            opcode;
    __u8            flags;
    __u16           command_id;
    __le32          nsid;
    __le64          rsvd[2];
    __le64          prp1;
    __le64          rsvd2;          /* MAX4KB   prp2 is rsvd */
    __le64          start_addr;
    __le32          ndw;          /* number of Dword */
    __le32          qid;
    __le32          rsvd3[2];
};

struct cmd_func_stru
{
    char* cmd;
    int (*fn)(int, char**);
};

struct nvme_read_memory {
    __u32    mem_addr;
    __u32    length;
    __u32*   pdata; 
};

struct nvme_write_memory {
    __u32    mem_addr;
    __u32    length;
    __u32*   pdata; 
};

struct rdmem_stru {
    u32      mem_addr;
    u32      length;
    u32*     pdata; /* pointer to the data */
};

struct wrmem_stru {
    u32      mem_addr;
    u32      length;
    u32*     pdata; /* pointer to the data */
};

struct nvme_read_reg64 {
    __u32    mem_addr;
    __u64*    pdata; 
};

struct nvme_write_reg64 {
    __u32    mem_addr;
    __u64*    pdata; 
};


struct rdreg64_stru {
    u32      mem_addr;
    u64*     pdata; /* pointer to the data */
};

struct wrreg64_stru {
    u32      mem_addr;
    u64*     pdata; /* pointer to the data */
};


//
struct nvme_read_cq {
    u32    length;
    u32    qid;
    void   *pdata; /* pointer to the data */
};


struct ppa_performance {
    int         type;	
    int         nlb;
    int         block;
	int         addr_field;
    u16*        file;
    u32         length;
    u16         index;
    u16         qid;
    u32         nsid;
	u16         chmask_sw;	
	u16         pgmask_sw;
	u32         xor_id;	
	u16         ctrl;
	u8			random;
    u8          line_cnt;
};

struct ppa_cmd {
    u32      nsid;
    u16      qid;
    u16      nlb;
    u64      addr;
    u64      meta_id;
    u64      data_id;
    __le64   metadata;
    __le64   prp1;
    __le64   prp2;
    u32      dsmgmt;
    u16      addr_field;  //which ppa addr field is given priority
    u8       opcode;
    u8       localdram;
    u32      file_len;   // file length
    u16      index;
    u16      xorid;	
    u16      xornum;
    u16      ctrl;
    u16      *file;      // bad block mark 
};

struct nvme_reg {
    u64    cap;         /* Controller Capabilities */
    u32    vs;          /* Version */
    u32    intms;       /* Interrupt Mask Set */
    u32    intmc;       /* Interrupt Mask Clear */
    u32    cc;          /* Controller Configuration */
    u32    rsvd1;       /* Reserved */
    u32    csts;        /* Controller Status */
    u32    rsvd2;       /* Reserved */
    u32    aqa;         /* Admin Queue Attributes */
    u64    asq;         /* Admin SQ Base Address */
    u64    acq;         /* Admin CQ Base Address */
    u64    rsvd[473];
    u64    cmdss[32];
    u32    sq0_tdb;     /*DB Start: 1000h*/
    u32    cq0_hdb;
    u32    sq1_tdb;
    u32    cq1_hdb;

    u32    sq2_tdb;
    u32    cq2_hdb;    
    u32    sq3_tdb;
    u32    cq3_hdb;
        
    u32    queue_db[1048];  /*DB End: 2079h*/
    u64    nfet;            /*2080h*/
    u32    nfvt;            /*2088h*/
    u32    sqpcr;           /*208ch*/
    u32    rsvd3[12];       
    u64    sbar;           /* 20c0 */
    u64    ebar;           /* 20c8 */
};

struct nvme_bar_space {
    struct nvme_reg nvmereg;
    u32             sqfc0;
    u32             sqfc1;
    u32             sqfc2;
	u32             sqfc3;
	
    u32             sqcc0;
    u32             sqcc1;
    u32             sqcc2;
	u32             sqcc3;
};

struct pci_header {
    u16 pci_vendor_id;
    u16 pci_device_id;
    u16 pci_command;
    u16 pci_status;
    u8  pci_rev_id;
    u32 pci_class_code;  /* 24 bit Class Code  8 bit	Cache Line Size */    
    u8  pci_cache_line_size;
    u8  pci_lat_tim;
    u8  pci_header_t;
    u8  pci_bist;
    u32 pci_bar[6];
    u32 ccptr;
    u16 pci_sub_vid;
    u16 pci_sub_id;
    u32 pci_base_rom;
    u8  pci_cap;
    u8  rsvd[7];
    u8  pci_irq_line;
    u8  pci_irq_pin;
};

struct nexus_dumpinfo{
    struct nvme_read_memory rd_mem;
    u16 queue_cnt;
    u16 queue[NEXUS_MAX_QUEUE];
    u32 *nsid_active;
};
extern int usage(int argc, char* argv[]);
void* parse_read_file(char* filename, u32 *file_len);
int parse_write_file(char* filename, void* memaddr, u32 length, int width);
void print_format(void *adrs, int n, int type );
void parse_cmd_returnval(int cmd, int retval, u16 devid, u16 qid);
int read_nvme_reg32(int devid, u32 offset, u32 *regval);
int write_nvme_reg32(int devid, u32 offset, u32 regval);
extern int parse_temptest_cmdline(int argc, char* argv[]);
extern int parse_rdreg_cmdline(int argc, char* argv[]);
extern int parse_wrreg_cmdline(int argc, char* argv[]);
extern int parse_rdreg64_cmdline(int argc, char* argv[]);
extern int parse_wrreg64_cmdline(int argc, char* argv[]);
extern int parse_dumppci_cmdline(int argc, char* argv[]);
extern int parse_dumpnexus_cmdline(int argc, char* argv[]);
extern int parse_checkcq_cmdline(int argc, char* argv[]);
extern int parse_checksq_cmdline(int argc, char* argv[]);

extern int parse_delsq_cmdline(int argc, char* argv[]);
extern int parse_crtsq_cmdline(int argc, char* argv[]);
extern int parse_delcq_cmdline(int argc, char* argv[]);
extern int parse_crtcq_cmdline(int argc, char* argv[]);
extern int parse_idn_cmdline(int argc, char* argv[]);
extern int parse_abort_cmdline(int argc, char* argv[]);
extern int parse_setft_cmdline(int argc, char* argv[]);
extern int parse_getft_cmdline(int argc, char* argv[]);
extern int parse_fwdown_cmdline(int argc, char* argv[]);
extern int parse_fwactv_cmdline(int argc, char* argv[]);
extern int parse_rstns_cmdline(int argc, char* argv[]);
extern int parse_getlp_cmdline(int argc, char* argv[]);
extern int parse_fmtnvm_cmdline(int argc, char* argv[]);
extern int parse_asyner_cmdline(int argc, char* argv[]);

extern int parse_ppapf_cmdline(int argc, char* argv[]);

extern int parse_wrppa_sync_cmdline(int argc, char* argv[]);
extern int parse_rdppa_sync_cmdline(int argc, char* argv[]);
extern int parse_wrpparaw_sync_cmdline(int argc, char* argv[]);
extern int parse_rdpparaw_sync_cmdline(int argc, char* argv[]);
extern int parse_ersppa_sync_cmdline(int argc, char* argv[]);

extern int parse_wrregspa_cmdline(int argc, char* argv[]);
extern int parse_rdregspa_cmdline(int argc, char* argv[]);
extern int parse_memcheck_cmdline(int argc, char* argv[]);
extern int parse_datacheck_cmdline(int argc, char* argv[]);
extern int parse_addrcheck_cmdline(int argc, char* argv[]);
extern int parse_rddword_cmdline(int argc, char* argv[]);
extern int parse_wrdword_cmdline(int argc, char* argv[]);
extern int parse_autotest_cmdline(int argc, char* argv[]);
extern int parse_badblock_cmdline(int argc, char* argv[]);
extern int parse_set_mac_addr_cmdline(int argc, char* argv[]);
extern int parse_get_mac_addr_cmdline(int argc, char* argv[]);
extern int parse_set_db_switch_cmdline(int argc, char* argv[]);
extern int parse_get_db_switch_cmdline(int argc, char* argv[]);
extern int parse_erase_disk_cmdline(int argc, char* argv[]);
extern int parse_erase_raidblock_cmdline(int argc, char* argv[]);

extern int dump_command(int argc, char* argv[]);

//E:open ktest      N:nexus
/* debug */
#define NEXUS_IOCTL_RD_REG                  _IOWR('N', 0x80, struct nvme_read_memory)
#define NEXUS_IOCTL_WR_REG                  _IOWR('N', 0x81, struct nvme_write_memory)
#define NEXUS_IOCTL_RD_REG64                _IOWR('N', 0x82, struct nvme_read_reg64)
#define NEXUS_IOCTL_WR_REG64                _IOWR('N', 0x83, struct nvme_write_reg64)
#define NEXUS_IOCTL_CHECK_CQ                _IOWR('N', 0x84, struct nvme_read_cq)
#define NEXUS_IOCTL_CHECK_SQ                _IOWR('N', 0x85, struct nvme_read_cq)
#define NEXUS_IOCTL_DUMPPCI              	_IOWR('N', 0x8B, struct pci_header)
#define NEXUS_IOCTL_DUMPNVME                _IOWR('N', 0x8C, struct nvme_read_memory)

/* Admin */
#define NEXUS_IOCTL_CRT_SQ                  _IOWR('N', 0x90, struct nvme_create_sq)
#define NEXUS_IOCTL_DEL_SQ                  _IOWR('N', 0x91, struct nvme_delete_queue)
#define NEXUS_IOCTL_GET_LP                  _IOWR('N', 0x92, struct nvme_get_log_page)
#define NEXUS_IOCTL_DEL_CQ                  _IOWR('N', 0x93, struct nvme_delete_queue)
#define NEXUS_IOCTL_CRT_CQ                  _IOWR('N', 0x94, struct nvme_create_cq)
#define NEXUS_IOCTL_IDN                     _IOWR('N', 0x95, struct nvme_identify)
#define NEXUS_IOCTL_ABORT                   _IOWR('N', 0x96, struct nvme_abort_cmd)
#define NEXUS_IOCTL_SET_FT                  _IOWR('N', 0x97, struct nvme_features)
#define NEXUS_IOCTL_GET_FT                  _IOWR('N', 0x98, struct nvme_features)
#define NEXUS_IOCTL_ASN_NER                 _IO('N', 0x99)
#define NEXUS_IOCTL_FMT_NVM                 _IOWR('N', 0x9A, struct nvme_format_cmd)
#define NEXUS_IOCTL_FWDOWN                  _IOWR('N', 0x9B, struct nvme_download_firmware)
#define NEXUS_IOCTL_FWACTV                  _IOWR('N', 0x9C, struct nvme_activate_firmware)

#define NEXUS_IOCTL_RD_REG_SPACE            _IOWR('N', 0xB6, struct nvme_rw_regspace)
#define NEXUS_IOCTL_WR_REG_SPACE            _IOWR('N', 0xB7, struct nvme_rw_regspace)
#define NEXUS_IOCTL_RD_DWORD                _IOWR('N', 0x86, struct nvme_read_memory)
#define NEXUS_IOCTL_WR_DWORD                _IOWR('N', 0x87, struct nvme_read_memory)
#define NEXUS_IOCTL_BWR_DWORD               _IOWR('N', 0x88, struct nvme_read_memory)
#define NEXUS_IOCTL_BRD_DWORD               _IOWR('N', 0x89, struct nvme_read_memory)
#define NEXUS_IOCTL_AUTO_TEST               _IOWR('N', 0x8A, struct nvme_read_memory)
#define NEXUS_IOCTL_GET_DB_SWITCH           _IOWR('N', 0xA0, int)
#define NEXUS_IOCTL_SET_DB_SWITCH           _IOWR('N', 0xA1, int)

#define NEXUS_IOCTL_PPA_SYNC                _IOWR('N', 0x40, struct nvme_ppa_command)
#define NEXUS_IOCTL_RST_NS                  _IOWR('N', 0xB4, u32)

/* Ktest ioctl interface */
#define IOCTL_PPA_PF            		 _IOWR('E', 0x71, struct ppa_performance)


#endif /* _UTEST_H */


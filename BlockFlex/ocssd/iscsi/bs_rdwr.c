/*
 * Synchronous I/O file backing store routine
 *
 * Copyright (C) 2006-2007 FUJITA Tomonori <tomof@acm.org>
 * Copyright (C) 2006-2007 Mike Christie <michaelc@cs.wisc.edu>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, version 2 of the
 * License.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA
 * 02110-1301 USA
 */
#define _XOPEN_SOURCE 600

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <linux/fs.h>
#include <sys/epoll.h>

#include "list.h"
#include "util.h"
#include "tgtd.h"
#include "scsi.h"
#include "spc.h"
#include "bs_thread.h"

#include "powerfault.h"
#include "../ocssd_queue.h"

static void set_medium_error(int *result, uint8_t *key, uint16_t *asc)
{
    printf("PFE:PFE: void set_medium_error()\n");

	*result = SAM_STAT_CHECK_CONDITION;
	*key = MEDIUM_ERROR;
	*asc = ASC_READ_ERROR;

    printf("PFE: result = 0x%x\n", *result);
    printf("PFE: key = 0x%x\n", *key);
    printf("PFE: asc = 0x%x\n", *asc);
    fflush(stdout);
}

static void bs_sync_sync_range(struct scsi_cmd *cmd, uint32_t length,
			       int *result, uint8_t *key, uint16_t *asc)
{
    printf("PFE:PFE: void bs_sync_sync_range()\n");
    fflush(stdout);

	int ret;

	ret = fdatasync(cmd->dev->fd);
    printf("PFE: %d = fdatasync(%d)\n", ret, cmd->dev->fd);
	if (ret)
		set_medium_error(result, key, asc);
}

/*mai: important!!!*/
//could be called by multiple threads!
static void bs_rdwr_request(struct scsi_cmd *cmd)
{
	// fprintf(stderr, "PFE:PFE: bs_rdwr_request(struct scsi_cmd *cmd): *cmd= %p, cmd->scb[0]= %x \n", cmd, cmd->scb[0]);
	// fflush(stderr);
    // pfe_print_scsi_cmd(cmd);

	int ret, fd = cmd->dev->fd;
	uint32_t length;

	int result = SAM_STAT_GOOD;//error handling code, should be 0x 
	uint8_t key;//error handling code, 0x
	uint16_t asc;//error handling code, 0x

	uint32_t info = 0;
	char *tmpbuf;
	size_t blocksize;
	uint64_t offset = cmd->offset;
	uint32_t tl     = cmd->tl;
	int do_verify = 0;
	int i;
	char *ptr;

    //mai: need to corrupt buf, so rm const
	//const char *write_buf = NULL;
	char *write_buf = NULL;

    //mai added: also record reads
	char *read_buf = NULL;

	ret = length = 0;
	key = asc = 0;

	int status;
	struct stat sbuf;

	switch (cmd->scb[0])
	{
	case ORWRITE_16:
        printf("PFE: cmd.scb[0] = %x: ORWRITE_16\n", cmd->scb[0]);
		length = scsi_get_out_length(cmd);

		tmpbuf = malloc(length);
		if (!tmpbuf) {
			result = SAM_STAT_CHECK_CONDITION;
			key = HARDWARE_ERROR;
			asc = ASC_INTERNAL_TGT_FAILURE;
			break;
		}
		ret = pread64(fd, tmpbuf, length, offset);
        printf("PFE: %d = pread64(%d, buffer, %u, %"PRIu64")\n", ret, fd, length, offset);
        fflush(stdout);

		if (ret != length) {
            //pread failed error handling
			set_medium_error(&result, &key, &asc);
			free(tmpbuf);
			break;
		}

		ptr = scsi_get_out_buffer(cmd);
		for (i = 0; i < length; i++)
			ptr[i] |= tmpbuf[i];

		free(tmpbuf);

		write_buf = scsi_get_out_buffer(cmd);
		goto write;

	case COMPARE_AND_WRITE:
        printf("PFE: cmd.scb[0] = %x: COMPARE_AND_WRITE\n", cmd->scb[0]);
		/* Blocks are transferred twice, first the set that
		 * we compare to the existing data, and second the set
		 * to write if the compare was successful.
		 */
		length = scsi_get_out_length(cmd) / 2;
		if (length != cmd->tl) {
			result = SAM_STAT_CHECK_CONDITION;
			key = ILLEGAL_REQUEST;
			asc = ASC_INVALID_FIELD_IN_CDB;
			break;
		}

		tmpbuf = malloc(length);
		if (!tmpbuf) {
			result = SAM_STAT_CHECK_CONDITION;
			key = HARDWARE_ERROR;
			asc = ASC_INTERNAL_TGT_FAILURE;
			break;
		}
		ret = pread64(fd, tmpbuf, length, offset);
        printf("PFE: %d = pread64(%d, buffer, %u, %"PRIu64")\n", ret, fd, length, offset);
        fflush(stdout);

		if (ret != length) { //mai: error handling
			set_medium_error(&result, &key, &asc);
			free(tmpbuf);
			break;
		}

		if (memcmp(scsi_get_out_buffer(cmd), tmpbuf, length)) {
			uint32_t pos = 0;
			char *spos = scsi_get_out_buffer(cmd);
			char *dpos = tmpbuf;

			/*
			 * Data differed, this is assumed to be 'rare'
			 * so use a much more expensive byte-by-byte
			 * comparasion to find out at which offset the
			 * data differs.
			 */
			for (pos = 0; pos < length && *spos++ == *dpos++;
			     pos++)
				;
			info = pos;
			result = SAM_STAT_CHECK_CONDITION;
			key = MISCOMPARE;
			asc = ASC_MISCOMPARE_DURING_VERIFY_OPERATION;
			free(tmpbuf);
			break;
		}

		if (cmd->scb[1] & 0x10){
			posix_fadvise(fd, offset, length,
				      POSIX_FADV_NOREUSE);
            printf("PFE: posix_fadvise(%d, %"PRIu64", %"PRIu64", POSIX_FADV_NOREUSE)\n", fd, offset, length); 
        }

		free(tmpbuf);

		write_buf = scsi_get_out_buffer(cmd) + length;
		goto write;

	case SYNCHRONIZE_CACHE:
	case SYNCHRONIZE_CACHE_16:
        printf("PFE: cmd.scb[0] = %x: SYNCHRONIZE_CACHE, SYNCHRONIZE_CACHE_16\n", cmd->scb[0]);
		/* TODO */
		length = (cmd->scb[0] == SYNCHRONIZE_CACHE) ? 0 : 0;

		if (cmd->scb[1] & 0x2) {
			result = SAM_STAT_CHECK_CONDITION;
			key = ILLEGAL_REQUEST;
			asc = ASC_INVALID_FIELD_IN_CDB;
		} else{
			pfe_log_io_req(cmd, 0, 0, NULL);
			bs_sync_sync_range(cmd, length, &result, &key, &asc);
		}
		break;

	case WRITE_VERIFY:
	case WRITE_VERIFY_12:
	case WRITE_VERIFY_16:
        printf("PFE: cmd.scb[0] = %x: WRITE_VERIFY, WRITE_VERIFY_12, WRITE_VERIFY_16\n", cmd->scb[0]);
		do_verify = 1;
	case WRITE_6:
	case WRITE_10:
	case WRITE_12:
	case WRITE_16:
        // printf("PFE: cmd.scb[0] = %x: WRITE_6, _10, _12, _16\n", cmd->scb[0]);
		length = scsi_get_out_length(cmd);
		write_buf = scsi_get_out_buffer(cmd);
write:

        /////////////////////////////////////////////////////
        //filter cmds by cmd type
        /*if((cmd->scb[0] == WRITE_10 || cmd->scb[0] == WRITE_6 ||
            cmd->scb[0] == WRITE_12 || cmd->scb[0] == WRITE_16)) {


           printf("PFE: interesting write cmd %x.\n", cmd->scb[0]);
           //ret = pwrite64(fd, write_buf, length, offset);
           //if(ENABLE_RECORD){
           if(pfe_enable_record != 0){
                printf("pfe_enable_record = %d\n", pfe_enable_record);
                pfe_log_io_req(cmd, offset, length, write_buf);
           }
        }
        else {//original processing
            printf("PFE: uninteresting write cmd\n");
            //ret = pwrite64(fd, write_buf, length, offset);
        }*/
        
        //do not filer cmds; log all write cmds
        // printf("PFE: interesting write cmd %x.\n", cmd->scb[0]);
        //ret = pwrite64(fd, write_buf, length, offset);
        //if(ENABLE_RECORD){
        if(pfe_enable_record != 0){
            fprintf(stdout, "pfe_enable_record = %d\n", pfe_enable_record);
            pfe_log_io_req(cmd, offset, length, write_buf);
        }


		// fprintf(stdout, "%lu %u\n", offset, length);
		// fflush(stdout);

        // direct the normal write to message queue
		// ret = pwrite64(fd, write_buf, length, offset);
		
		char* path = concat(USR_DIR, "iscsi/start_replay");
		if(stat(path, &sbuf) == 0){
			Req message = {.vssd_id = cmd->c_target->tid, .mode=1, .offset = offset, .length=length};
			strcpy(message.data, "harvest request");

			// printf("%s %d\n", cmd->dev->path, cmd->c_target->tid);

			if(cmd->c_target->tid == 1) mqfd = mqfd_0;
			else if(cmd->c_target->tid == 2) mqfd = mqfd_1;

			if(mqfd == -1) {
				fprintf(stderr, "Client mq_open failure");
				fflush(stderr);
			}
			else{
				status = mq_send(mqfd, (char*) &message, sizeof(message), 0);
				// if (status == -1) fprintf(stderr, "mq_send failure\n");
				// else fprintf(stdout, "Client is sending request: {%d, %d}\n", message.offset, message.length);
			}
		}
		free(path);
		// fflush(stdout);
		// fflush(stderr);
		
		// mq_close(mqfd);

        //appear on every sync write
        //printf("PFE: write: %d = pwrite64(%d, write_buf, %u, %"PRIu64")\n", 
        //        ret, fd, length, offset);
        /////////////////////////////////////////////////////////////
        
		/*
			 * it would be better not to access to pg
			 * directy.
			 */

		/* if (ret == length) {
			struct mode_pg *pg;

			
			pg = find_mode_page(cmd->dev, 0x08, 0);
			if (pg == NULL) {
				result = SAM_STAT_CHECK_CONDITION;
				key = ILLEGAL_REQUEST;
				asc = ASC_INVALID_FIELD_IN_CDB;
				break;
			}
			if (((cmd->scb[0] != WRITE_6) && (cmd->scb[1] & 0x8)) ||
			    !(pg->mode_data[0] & 0x04))
				bs_sync_sync_range(cmd, length, &result, &key,
						   &asc);
		} else {//write error handling
 			set_medium_error(&result, &key, &asc);
        }

		if ((cmd->scb[0] != WRITE_6) && (cmd->scb[1] & 0x10)){
			posix_fadvise(fd, offset, length,
				      POSIX_FADV_NOREUSE);
            printf("PFE: posix_fadvise(%d, %u, %"PRIu64", POSIX_FADV_NOREUSE)\n", fd, offset, length); 
        } */
		if (do_verify)
			goto verify;
		break;
	case WRITE_SAME:
	case WRITE_SAME_16:
        printf("PFE: cmd.scb[0] = %x: WRITE_SAME, WRITE_SAME_16\n", cmd->scb[0]);
		/* WRITE_SAME used to punch hole in file */
		if (cmd->scb[1] & 0x08) {
			ret = unmap_file_region(fd, offset, tl);
			if (ret != 0) {
				eprintf("Failed to punch hole for WRITE_SAME"
					" command\n");
				result = SAM_STAT_CHECK_CONDITION;
				key = HARDWARE_ERROR;
				asc = ASC_INTERNAL_TGT_FAILURE;
				break;
			}
			break;
		}
		while (tl > 0) {
			blocksize = 1 << cmd->dev->blk_shift;
			tmpbuf = scsi_get_out_buffer(cmd);

			switch(cmd->scb[1] & 0x06) {
			case 0x02: /* PBDATA==0 LBDATA==1 */
				put_unaligned_be32(offset, tmpbuf);
				break;
			case 0x04: /* PBDATA==1 LBDATA==0 */
				/* physical sector format */
				put_unaligned_be64(offset, tmpbuf);
				break;
			}

			ret = pwrite64(fd, tmpbuf, blocksize, offset);
            //printf("PFE: write: %d = pwrite64(fd, tmpbuf, %u, %"PRIu64")\n", 
            //    ret, blocksize, offset);

			if (ret != blocksize)  //error handling
				set_medium_error(&result, &key, &asc);

			offset += blocksize;
			tl     -= blocksize;
		}
		break;

	case READ_6:
	case READ_10:
	case READ_12:
	case READ_16:
        // printf("PFE: cmd.scb[0] = %x: READ_6, _10, _12, _16\n", cmd->scb[0]);
		length = scsi_get_in_length(cmd);
        
        /////////////////////////////////////////////////////////////////
        /*//emulate failure in read cmds

        if((cmd->scb[0] == READ_6 || cmd->scb[0] == READ_10 ||
            cmd->scb[0] == READ_12 || cmd->scb[0] == READ_16) &&
            (pfe_err_blk_fd >= 0) ){ //err_blk_fd opened if emulate fail type 5, 7, 8

            printf("PFE: interesting read cmd %x.\n", cmd->scb[0]);
            //ret = pread64(fd, scsi_get_in_buffer(cmd), length, offset);
            ret = pfe_read(fd, scsi_get_in_buffer(cmd), length, offset, cmd);

        }
        else {//original processing
            ret = pread64(fd, scsi_get_in_buffer(cmd), length, offset);
        }
       */ 

        //log all read cmds
        // printf("PFE: interesting read cmd %x.\n", cmd->scb[0]);
        // if(pfe_enable_record != 0){
        //     printf("pfe_enable_record = %d\n", pfe_enable_record);
        //     read_buf=scsi_get_in_buffer(cmd);
        //     pfe_log_io_req(cmd, offset, length, read_buf);
        // }

		//if(length != 4096){
			// fprintf(stdout, "PFE: interesting read cmd %x. <%lu, %u, %d, %lu>\n", cmd->scb[0], offset, length, cmd->attribute, cmd->tag);
			// fflush(stdout);
		//}

    
        //original processing
        // ret = pread64(fd, scsi_get_in_buffer(cmd), length, offset);

        // printf("PFE: %d = pread64(%d, buffer, %u, %"PRIu64")\n", ret, fd, length, offset);
        // fflush(stdout);

		// filter non-workload commands
    	if(length != 4096){
			Req read_message = {.vssd_id = cmd->c_target->tid, .mode=0, .offset = offset, .length=length};
			strcpy(read_message.data, "harvest request");

			// fprintf(stdout, "%s\n", QUEUE_NAME);
			// fflush(stdout);

			// write/create - attr needed because of O_CREAT
			
			if(cmd->c_target->tid == 1) mqfd = mqfd_0;
			else if(cmd->c_target->tid == 2) mqfd = mqfd_1;
			
			if(mqfd == -1) {
				fprintf(stderr, "Client mq_open failure");
				fflush(stderr);
			}
			else{
				status = mq_send(mqfd, (char*) &read_message, sizeof(read_message), 0);
				// if (status == -1) fprintf(stderr, "mq_send failure\n");
				// else fprintf(stdout, "Client is sending request: {%d, %d}\n", message.offset, message.length);
			}
		}

        /////////////////////////////////////////////////////////////////


		// if (ret != length){//error handling; metadata corruption falls into here  
		// 	set_medium_error(&result, &key, &asc);
        // }

		// if ((cmd->scb[0] != READ_6) && (cmd->scb[1] & 0x10)){
		// 	posix_fadvise(fd, offset, length,
		// 		      POSIX_FADV_NOREUSE);
        //     // printf("PFE: posix_fadvise(%d, %d, %"PRIu64", POSIX_FADV_NOREUSE)\n", fd, offset, length); 
        // }

		break;

	case PRE_FETCH_10:
	case PRE_FETCH_16:
        printf("PFE: cmd.scb[0] = %x: PRE_FETCH_10, _16\n", cmd->scb[0]);
		ret = posix_fadvise(fd, offset, cmd->tl,
				POSIX_FADV_WILLNEED);
        printf("PFE: %d = posix_fadvise(%d, %"PRIu64", %d, POSIX_FADV_NOREUSE)\n", 
                ret, fd, offset, cmd->tl); 

		if (ret != 0)
			set_medium_error(&result, &key, &asc);
		break;
	case VERIFY_10:
	case VERIFY_12:
	case VERIFY_16:
        printf("PFE: cmd.scb[0] = %x: VERIFY_10, _12, _16\n", cmd->scb[0]);
verify:
		length = scsi_get_out_length(cmd);

		tmpbuf = malloc(length);
		if (!tmpbuf) {
			result = SAM_STAT_CHECK_CONDITION;
			key = HARDWARE_ERROR;
			asc = ASC_INTERNAL_TGT_FAILURE;
			break;
		}
		ret = pread64(fd, tmpbuf, length, offset);
        printf("PFE: %d = pread64(%d, buffer, %u, %"PRIu64")\n", ret, fd, length, offset);
        fflush(stdout);


		if (ret != length){  //error handling
			set_medium_error(&result, &key, &asc);
        }
		else if (memcmp(scsi_get_out_buffer(cmd), tmpbuf, length)) {
			result = SAM_STAT_CHECK_CONDITION;
			key = MISCOMPARE;
			asc = ASC_MISCOMPARE_DURING_VERIFY_OPERATION;
		}

		if (cmd->scb[1] & 0x10){
			posix_fadvise(fd, offset, length,
				      POSIX_FADV_NOREUSE);
            printf("PFE: posix_fadvise(%d, %u, %"PRIu64", POSIX_FADV_NOREUSE)\n", fd, offset, length); 
        }

		free(tmpbuf);
		break;
	case UNMAP:
        printf("PFE: cmd.scb[0] = %x: UNMAP\n", cmd->scb[0]);
		if (!cmd->dev->attrs.thinprovisioning) {
			result = SAM_STAT_CHECK_CONDITION;
			key = ILLEGAL_REQUEST;
			asc = ASC_INVALID_FIELD_IN_CDB;
			break;
		}

		length = scsi_get_out_length(cmd);
		tmpbuf = scsi_get_out_buffer(cmd);

		if (length < 8)
			break;

		length -= 8;
		tmpbuf += 8;

		while (length >= 16) {
			offset = get_unaligned_be64(&tmpbuf[0]);
			offset = offset << cmd->dev->blk_shift;

			tl = get_unaligned_be32(&tmpbuf[8]);
			tl = tl << cmd->dev->blk_shift;

			if (offset + tl > cmd->dev->size) {
				eprintf("UNMAP beyond EOF\n");
				result = SAM_STAT_CHECK_CONDITION;
				key = ILLEGAL_REQUEST;
				asc = ASC_LBA_OUT_OF_RANGE;
				break;
			}

			if (tl > 0) {
				if (unmap_file_region(fd, offset, tl) != 0) {
					eprintf("Failed to punch hole for"
						" UNMAP at offset:%" PRIu64
						" length:%d\n",
						offset, tl);
					result = SAM_STAT_CHECK_CONDITION;
					key = HARDWARE_ERROR;
					asc = ASC_INTERNAL_TGT_FAILURE;
					break;
				}
			}

			length -= 16;
			tmpbuf += 16;
		}
		break;
	default:
        printf("PFE: cmd.scb[0] = %x: default\n", cmd->scb[0]);
		break;
	}

	dprintf("io done %p %x %d %u\n", cmd, cmd->scb[0], ret, length);

	/*printf("PFE: bs_rdwr_request() io done %p %x %d %u\n", cmd, cmd->scb[0], ret, length);
    printf("PFE: ret = %d\n", ret);
    printf("PFE: length = %d\n", length);
    printf("PFE: result = 0x%x\n", result);
    printf("PFE: key = 0x%x\n", key);
    printf("PFE: asc = 0x%x\n", asc);
    printf("\n");
    fflush(stdout);*/

    //set cmd->result = result
	scsi_set_result(cmd, result);

	if (result != SAM_STAT_GOOD) {
		eprintf("io error %p %x %d %d %" PRIu64 ", %m\n",
			cmd, cmd->scb[0], ret, length, offset);

		printf("PFE: bs_rdwr_request() io error %p %x %d %d %" PRIu64 ", %m\n",
			cmd, cmd->scb[0], ret, length, offset);
        printf("\n");
        fflush(stdout);

        //set cmd->sense_buffer and cmd->sense_len
		sense_data_build(cmd, key, asc);
	}
}

static int bs_rdwr_open(struct scsi_lu *lu, char *path, int *fd, uint64_t *size)
{
	printf("PFE:PFE: static int bs_rdwr_open(struct scsi_lu *lu, char *path, int *fd, uint64_t *size)\n");
    printf("PFE: path = '%s'\n", path);
    printf("PFE: *size = %"PRIu64" \n", *size);

	uint32_t blksize = 0;

	*fd = backed_file_open(path, O_RDWR|O_LARGEFILE|lu->bsoflags, size,
				&blksize);
	/* If we get access denied, try opening the file in readonly mode */
	if (*fd == -1 && (errno == EACCES || errno == EROFS)) {
		*fd = backed_file_open(path, O_RDONLY|O_LARGEFILE|lu->bsoflags,
				       size, &blksize);
		lu->attrs.readonly = 1;
	}
	if (*fd < 0)
		return *fd;

	if (!lu->attrs.no_auto_lbppbe)
		update_lbppbe(lu, blksize);

	return 0;
}

static void bs_rdwr_close(struct scsi_lu *lu)
{
	close(lu->fd);
}

int nr_iothreads = 16;

static tgtadm_err bs_rdwr_init(struct scsi_lu *lu)
{
	struct bs_thread_info *info = BS_THREAD_I(lu);

    //mai: worker threads execute bs_rdwr_request() 
    printf("PFE: bs_rdwr_init(): nr_iothreads = %d\n", nr_iothreads);
	return bs_thread_open(info, bs_rdwr_request, nr_iothreads);
}

static void bs_rdwr_exit(struct scsi_lu *lu)
{
	struct bs_thread_info *info = BS_THREAD_I(lu);

	bs_thread_close(info);
}

static struct backingstore_template rdwr_bst = {
	.bs_name		= "rdwr",
	.bs_datasize		= sizeof(struct bs_thread_info),
	.bs_open		= bs_rdwr_open,
	.bs_close		= bs_rdwr_close,
	.bs_init		= bs_rdwr_init,
	.bs_exit		= bs_rdwr_exit,
	.bs_cmd_submit		= bs_thread_cmd_submit,
	.bs_oflags_supported    = O_SYNC | O_DIRECT,
};

__attribute__((constructor)) static void bs_rdwr_constructor(void)
{
	register_backingstore_template(&rdwr_bst);
}

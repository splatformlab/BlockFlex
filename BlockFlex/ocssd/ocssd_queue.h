#include <stdio.h>
#include <stdlib.h>
#include <stdint.h> 
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <sys/wait.h>
#include <mqueue.h>

#define PMODE 0655
#define MAX_MQUEUE_MSG_SIZE 8192
#define MAX_MSG 8192
#define QUEUE_NAME_0 "/harvest0"
#define QUEUE_NAME_1 "/harvest1"

extern mqd_t mqfd_1, mqfd_0, mqfd;

typedef struct Req
{
  uint64_t vssd_id;
  int mode;
  uint64_t offset;
  uint32_t length;
  char data[1024];
} Req;
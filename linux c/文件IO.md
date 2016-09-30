## c标准函数和系统函数

### 区别
c标准函数，有缓冲区，默认为8kb。夸平台。   
系统函数没有缓冲区，只有内核I/O缓冲区

## linux 文件函数
### files_sturuct 
1. 文件由这个结构体来描述。用户程序不能直接访问内核中的文件描述符表。只能使用文件描述表的索引（即，0，1，2这些数字），索引也称子这文件描述符。
2. 当调用open打开一个文件或创建一个新文件时，内核分配一个文件描述符并返回给用户程序，该文件描述符表项中的指针指向新打开的文件。当读写文件时，用户程序把文件描述符传给read或write，内核根据文件描述符找到相应的表项，再通过表项中的指针找到相应的文件


### 文件描述符
1. 一个进程默认打开了3个文件描述符  

		STDIN_FILENO 0 //标准输入
		STDOUT_FILENO 1 //标准输出
		STDERR_FILENO 2  //标准错误
2. 新打开文件返回文件描述符表中未使用的最小文件描述符

### 打开和关才文件 open
1. 函数定义 

		/**
		 * 打开一个文件
		 * @param char *pathname 文件名
		 * @param int flags 打开标识
		 * @param int mode 如果是创建文件，设置权限。8进制数据
		 * @return int 文件描述符，出错返回 -1,并设置errno
		 */
		int open(const char *pathname, int flags, int mode);
		
		/**
		 * 关闭文件
		 * @param int fd 文件描述符
		 * @return 返回值：成功返回0，出错返回-1并设置errno
		*/	
		int close(int fd);

2. flags 说明   

	标识|说明
	-|-
	O_RDONLY|只读打开
	O_WRONLY|只写打开
	O_RDWR|读写打开
	O_APPEND|追加，如果文件已有内容， 打开后指针文件末尾
	O_CREAT|若此文件不存在则创建它，要设置第三个参数，文件权限
	O_EXCL|如果同时指定了O_CREAT，并且文件已存在，则出错返回
	O_TRUNC|如果文件已存在，并且以只写或可读可写方式打开，则将其长度截断
	O_NONBLOCK|对于设备文件，以O_NONBLOCK方式打开可以做非阻塞I/O


3. 文件权限由open的mode参数和当前进程的umask掩码共同决定。    

		真实权限 = mode &~umask 
4. 例子，打开错误

		#include <stdio.h>
		#include <stdlib.h>
		#include <sys/types.h>
		#include <sys/stat.h>
		#include <sys/fcntl.h>
		
		int main(void)
		{
		    char *name = "test.log";
		    int fd = open(name, O_RDONLY);
		    if (fd == -1) { //只读打开，文件不存在，设置error
		        perror("文件打开失败：");
		    } else {
		        printf("打开成功 fd = %d\n", fd);
		    }
		}
5. 不存在创建  

		#include <stdio.h>
		#include <stdlib.h>
		#include <sys/types.h>
		#include <sys/stat.h>
		#include <sys/fcntl.h>
		
		int main(void)
		{
		    char *name = "test.log";
		    int fd = open(name, O_RDONLY | O_CREAT, 0755);
		    if (fd == -1) {
		        perror("文件打开失败：");
		    } else {
		        printf("打开成功 fd = %d\n", fd);
		    }
		}
6. 当一个进程终止时，内核对该进程所有尚未关闭的文件描述符调用close关闭，所以即使用户程序不调用close，在终止时内核也会自动关闭它打开的所有文件。但是对于一个长年累月运行的程序（比如网络服务器），打开的文件描述符一定要记得关闭，否则随着打开的文件越来越多，会占用大量文件描述符和系统资源


### read/write
1. 函数定义  
		
		/**
		 * read函数从打开的设备或文件中读取数据
		 * @param int fd 文件描述符
		 * @param void *buf 保存读取的字符
		 * @param size_t count 读取的字符数
		 * @return 返回实际读取的字节数，出错返回-1并设置errno，
		 *         如果在调read之前已到达文件末尾，则这次read返回0
		 *
		 *         返回字节有可能小于count.
		 *         常规文件到文件末尾返回
		 *         终端设备读到换行返回
		 *         网络读，和协议相关
		 */
		ssize_t read(int fd, void *buf, size_t count);
		
		/**
		 * read函数从打开的设备或文件中读取数据
		 * @param int fd 文件描述符
		 * @param void *buf 保存读取的字符
		 * @param size_t count 读取的字符数
		 * @return 返回实际读取的字节数，出错返回-1并设置errno，
		 *        写常规文件时，write的返回值通常等于请求写的字节数count，而向终端设备或网络写则不一定
		 */
		ssize_t write(int fd, const void *buf, size_t count);
2. 阻塞和非阻塞  
	1. 阻塞（Block）。当进程调用一个阻塞的系统函数时，该进程被置于睡眠（Sleep）状态，这时内核调度其它进程运行，直到该进程等待的事件发生了。运行态表示程序正在被执行或等待cpu执行
	2. 常规文件读写是不会阻塞的，一定会在有限的时候内返回。
	3. 设置文件默认是阻塞读取。  

			#include <stdio.h>
			#include <stdlib.h>
			#include <sys/types.h>
			#include <sys/stat.h>
			#include <sys/fcntl.h>
			#include <unistd.h>
			#include <errno.h>
			
			
			int main(void)
			{
			    char buf[10];
			    int fd, n , i;
			    //非阻塞方式打开当前终端
			    fd = open("/dev/tty", O_RDONLY | O_NONBLOCK);
			    if (fd < 0) { //打开失败
			        perror("open /dev/tty");
			        exit(1);
			    }
			    for (i = 0; i < 5; ++i) {
			        n = read(fd, buf, 10);
			        if (n >= 0) { //读取到数据了
			            break;
			        }
			        // 设备暂时没有数据可读就返回-1，同时置errno为EWOULDBLOCK（或者EAGAIN，这两个宏定义的值相同）
			        if (errno != EAGAIN) {
			            perror("read /dev/tty");
			        }
			        sleep(1);
			        write(STDOUT_FILENO, "try again\n", 10);
			    }
			    if(i==5) {
			        write(STDOUT_FILENO, "timeout\n", 9);
			    } else {
			        write(STDOUT_FILENO, buf, n);
			    }
			    close(fd); return 0;
			}
	4. 非阻塞，通常会以while(true) sleep 不停轮询的方式读取数据，会存大量的无用查询和sleep。效率不高。

### lseek 移动位置
1. 原型  
		
		/**
		 * 每个打开的文件都记录着当前读写位置。lseek用于移动这个位置
		 * @param int fd 文件描述符
		 * @param off_t offset 偏移字节数
		 * @param int whence 表示相对位置
		 *        SEEK_SET 开始位置
		 *        SEEK_CUR 当前位置
		 *        SEEK_END 结束位置
		 * @return int 成功时返回0失败时返回-1。 设备是否可以设置偏移量
		 *
		 *
		 * 偏移量允许超过文件末尾，这种情况下对该文件的下一次写操作将延长文件，中间空洞的部分读出来都是0
		 */
		off_t lseek(int fd, off_t offset, int whence);

### fcntl 
1. 定义

		/**
		 * 获取或修改文件属性
		 * 可以重新设置读、写、追加、非阻塞等标志
		 *
		 * @param int fd 文件描述符
		 * @param int cmd 命令类型
		 *        F_GETFL 获取
		 *        F_SETFL 设置
		 * @param long|struct flock* 可选参数
		 * @return 新的标志,失败返回-1
		 */
		int fcntl(int fd, int cmd, struct flock *lock);
2. 例子

		int main(void)
		{
		    char buf[10];
		    int n;
		    int flags;
		
		    //先获取当前标识
		    flags = fcntl(STDIN_FILENO, F_GETFL);
		    flags |=  O_NONBLOCK;
		
		    //在设置标识
		    if (fcntl(STDIN_FILENO, F_SETFL, flags) == -1) {
		        perror("fcntl");
		        exit(1);
		    }
		    try:
		        n = read(STDIN_FILENO, buf, 10);
		    if (n < 0) {
		        if (errno == EAGAIN) {
		            sleep(1);
		            write(STDOUT_FILENO, "try\n", 4);
		            goto try;
		        } else {
		            perror("read stdin");
		            exit(1);
		        }
		    }
		    write(STDOUT_FILENO, strcat(buf, "\n"), n + 1);
		    return 0;
		}

## 文件系统
### 文件类型

编号|标识|说明|宏
-|-|-|-
0||未知文件|DT_UNKNOWN
1|-|普通文件|DT_REG 
2|d|目标|DT_DIR
3|c|字符设备文件|DT_CHAR
4|b|块设备文件|DT_BLK
5|n|有名管道|DT_FIFO
6|s|socket文件|DT_SOCK
7|l|符号链接|DT_LNK

### stat 函数系统
1. 文件状态结构  

		struct stat {
		    dev_t st_dev; /* 文件设备id */
		    ino_t st_ino; /* inode 号 */
		    mode_t st_mode; /* 文件类型和权限 */
		    nlink_t st_nlink; /* 链向此文件的连接数(硬连接 */
		    uid_t st_uid; /* 用户ID */
		    gid_t st_gid; /* 组ID */
		    dev_t st_rdev; /* 设备号，针对设备文件 */
		    off_t st_size; /* 文件大小，字节为单位 */
		    blksize_t st_blksize; /*  I/O -系统块的大小 */
		    blkcnt_t st_blocks; /* 文件所占块数 */
		    time_t st_atime; /* 最后访问时间 */
		    time_t st_mtime; /* 最后修改时间 */
		    time_t st_ctime; /* 节点状态的最后更改时间 */
		};
2. 函数定义  

		/**
		 * @param char *path 文件名
		 * @param struct stat *buf 文件状态数据结构体指针
		 * @return 成功返回0， 失败返回 -1
		 *
		 * @desc stat 跟踪符号连接
		 *       lstat 不跟踪符号链接
		 *       fstat 第一个参数为一个文件描述符
		 */
		int stat(const char *path, struct stat *buf);
3. 代码  

		int main(void)
		{
		    struct stat buf;
		    stat("test.log", &buf);
		    printf("%d\n", (int)buf.st_mtime);
		}

### access 文件权限检查
1. 定义  

		/**
		 * access函数取出文件inode中的st_mode字段
		 * @param char *pathname 文件名
		 * @param int mode 模式，可以同时是多个
		 *          R_OK 是否有读权限
		 *          W_OK 是否有写权限
		 *          X_OK 是否有执行权限
		 *          F_OK 测试一个文件是否存在
		 * @return 如果所有权限都有返回 0，否则返回-1，并设置错误码
		 */
		int access(const char *pathname, int mode);
2. 代码  

		int main(void)
		{
		    int flag = access("1.log", F_OK);
		    printf("%d\n", flag);
		}

### chmod 修改文件权限
1. 定义  


		/**
		 * 函数改变文件的访问权限，也就是修改inode中的st_mode字段
		 * @param char *path 文件名
		 * @param mode_t mode 权限 ，8进制数据
		 * @return 成功返回0，失败返回-1
		 *
		 * @desc fchmode 第一个参数为 fd
		 */
		int chmod(const char *path, mode_t mode);
2. 代码  

		int main(void)
		{
		    int flag = chmod("a.out", 0777);
		    if (flag < 0) {
		        perror("chmod");
		    }
		    printf("%d\n", flag);
		}
### chown
1. 定义  

		/**
		 * 修改文件所有者。
		 * @param char *path 文件路径
		 * @param uid_t owner 所有者ID
		 * @param gid_t group 所属组ID
		 * @return 成功返回0， 失败返回-1
		 *
		 * @desc  fchown 第一个参数为fd
		 *        lchonw 不跟踪符号链接
		 */
		int chown(const char *path, uid_t owner, gid_t group);

### 文件链接函数
1. 定义

		/**
		 * 创建一个硬链接，指向同一个inode
		 * 当rm删除文件时，只是删除了目录下的记录项和把inode硬链接计数减1，
		 * 当硬链接计数减为0时，才会真正的删除文件
		 *
		 * @param char *oldpath 源文件
		 * @param char *newpath 目标文件
		 * @return 成功返回0， 失败返回-1
		 *
		 * @desc symlink 创建一个软链接，参数和link 一样
		 */
		int link(const char *oldpath, const char *newpath);

		/**
		 * 删除一个链接
		 * 1. 如果是符号链接，删除符号链接，不影响指向的文件
		 *
		 * 2. 如果是硬链接，硬链接数减1，当减为0时，释放数据块和inode
		 *
		 * 3. 如果文件硬链接数为0，但有进程已打开该文件，并持有文件描述符，
		 *    则等该进程关闭该文件时，kernel才真正去删除该文件
		 *    利用该特性创建临时文件，先open或creat创建一个文件，马上unlink此文件
		 * @param char * pathname 文件名
		 * @return 成功返回0， 失败返回 -1
		 */
		int int unlink(const char *pathname)

### 文件重命名
1. 定义


		/**
		 * 文件重命名。同目录，重命名。不同目录，移动后重命名
		 * 
		 * 如果原文件名和新文件名不在一个目录下则需要从原目录数据
		 * 块中清除一条记录然后添加到新目录的数据块中
		 * @param char *oldpath 原名称
		 * @param char *newpath 新名称
		 * @return 成功返回0， 失败返回-1
		 */
		int rename(const char *oldpath, const char *newpath);

## 目录函数

### 工作目录相关
1. 定义


		/**
		 * 修改当前工作目录。
		 * @param char *path 目录名
		 * @return 成功返回0，推向返回 -1
		 */
		int chdir(const char *path);
		
		/**
		 * @param char *buf 保存目录
		 * @param size_t size buff的大小。如果size小于目录大于，返回null
		 * @return 返回buf指针。出错返回NULL
		 */
		char * getcwd(char * buf,size_t size);
2. 代码   

		int main(void)
		{
		    int flag = chdir("/home/xuen");
		    if (flag < 0) {
		        perror("chdir");
		        exit(1);
		    }
		    char buf[100];
		    printf("%s\n", getcwd(buf, sizeof(buf)));
		    open("test.log", O_RDWR | O_CREAT, 0755); //此是创建的文件位于/home/xuen下
		}
### 目录操作
1. 定义  
		
		/**
		 * 创建一个目录
		 * @param char *pathname 目录名
		 * @param mode_t mode 目录权限
		 * @return 成功返回 0， 失败返回-1
		 */
		int mkdir(const char *pathname, mode_t mode);
		
		/**
		 * 删除一个目录
		 * @param char *pathname 目录名
		 * @return 成功返回 0， 失败返回-1
		 */
		int rmdir(const char *pathname);
		
		/**
		 * @param char * name 目录名称
		 * @return 成功返回DIR流。 接下来对目录的读取和搜索都要使用此返回值
		 */
		DIR *opendir(const char *name);
		
		/**
		 * 读取目录中的一个文件 ，并移动指针到下一个文件
		 * @param Dir * drip DIR流
		 * @return struct dirent 如果到未尾返回NULL
		 */
		struct dirent *readdir(DIR *dirp);
		
		/**
		 * 把目录指针恢复到目录的起始位置
		 */
		void rewinddir(DIR *dirp);
		
		/**
		 * 关才目录指针
		 */
		int closedir(DIR *dirp);
2. dirent结构体 

		struct dirent {
		    ino_t d_ino; /* inode 编号 */
		    off_t d_off; /* 下一个 dirent偏移量 */
		    unsigned short d_reclen; /* 记录长度 */
		    unsigned char d_type; /* 文件类型 */
		    char d_name[256]; /* 文件名称 */
		};
3. 递归遍历目录  

		void mydir(char *baseDir, char (*np)[1024], int i)
		{
		    DIR *dp = opendir(baseDir);
		    if (dp == NULL) {
		        printf("%s not exists\n", baseDir);
		        exit(1);
		    }
		    struct dirent *ptr = NULL;
		    struct stat dirInfo;
		    while ((ptr = readdir(dp)) != NULL) {
		        if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0) {
		            continue;
		        }
		        char tmp[1024];
		        sprintf(tmp, "%s/%s", baseDir, ptr->d_name);
		        lstat(tmp, &dirInfo);
		        if (S_ISDIR(dirInfo.st_mode)) {
		            mydir(tmp, np, i);
		        } else {
		            strcpy(np[i], tmp);
		            i++;
		        }
		    }
		}
		int main(void)
		{
		    char baseDir[20] = "/data/wwwroot/c";
		    char np[1024][1024] = {'\0'};
		    mydir(baseDir, np, 0);
		    for (int i = 0; i < 1024; ++i) {
		        if (strcmp(np[i], "") != 0) {
		            printf("%s\n", np[i]);
		        }
		    }
		}

### dup dup2
1. 定义  

		/**
		 * 复制一个现存的文件描述符，使两个文件描述符指向同一个file结构体。
		 *      如果两个文件描述符指向同一个file结构体，File Status Flag和读写位置只保存一
		 *      份在file结构体中，并且file结构体的引用计数是2
		 *
		 * open 两次open同一文件得到两个文件描述符，则每个描述符对应一个不同的file结构体
		 * @param int oldfd 原文件描述符
		 * @return int 新的文件描述符
		 */
		int dup(int oldfd);
		
		/**
		 * 复制文件描述符。
		 * newfd当前已经打开，先关闭newfd,在将newfd指向oldfd可用于重定向。
		 * 如果newfd = oldfd 则什么也不做。
		 * 
		 * @param int oldfd 原文件描述符
		 * @param int newfd 新的文件描述符
		 * @param return 新的文件描述符 失败返回-1
		 */
		int dup2(int oldfd, int newfd);
2. 代码  


		int main(void)
		{
		    int fd, save_fd;
		    char *msg = "This is a test\n";
		    fd = open("test.log", O_RDWR | O_CREAT, 0755);
		
		    save_fd = dup(STDOUT_FILENO); //save_fd STDOUT_FILENO指向同一个file结构体
		    dup2(fd, STDOUT_FILENO); // STDOUT_FILENO，重新指向fd
		    close(fd); //关闭了fd, 但是STDIN_FILENO仍然指向test.log
		    write(STDOUT_FILENO, msg, strlen(msg)); // 向test.log写数据
		
		    dup2(save_fd, STDOUT_FILENO); //STDOUT_FILENO重新指向标准输出
		    write(STDOUT_FILENO, msg, strlen(msg));
		    close(save_fd);
		}
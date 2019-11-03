#pragma once

#include	<cublas_v2.h>
#include	<stdio.h>
#include	<stdlib.h>
#include	<string.h>
#include	<errno.h>
#include	<sys/time.h>
#include	<time.h>

#ifndef		__PTR_STACK_MAXSIZE
#define		__PTR_STACK_MAXSIZE		32
#endif

static struct {
    void	*addr;
    char	*name;
} __ptr_stack[__PTR_STACK_MAXSIZE];

unsigned __ptr_stack_sz = 0;

#define		__ptr_stack_push(a,n)			do {																						\
                                                if (__ptr_stack_sz == __PTR_STACK_MAXSIZE) {											\
                                                    fprintf(stderr, "%s:%d: pointer stack full\n", __FILE__, __LINE__);					\
                                                    __ptr_stack_abort();																\
                                                }																						\
                                                                                                                                        \
                                                __ptr_stack[__ptr_stack_sz].addr = (a);													\
                                                __ptr_stack[__ptr_stack_sz].name = (n);													\
                                                                                                                                        \
                                                ++__ptr_stack_sz;																		\
                                            } while (0)

#define		__ptr_stack_remove(i)			do {																						\
                                                if (__ptr_stack[i].name != NULL) {														\
                                                    if (fclose((FILE *) __ptr_stack[i].addr))											\
                                                        fprintf(stderr, "%s:%d: error closing file %s: %s\n", __FILE__, __LINE__,		\
                                                            __ptr_stack[i].name, strerror(errno));										\
                                                                                                                                        \
                                                    free(__ptr_stack[i].name);															\
                                                } else {																				\
                                                    free(__ptr_stack[i].addr);															\
                                                }																						\
                                                                                                                                        \
                                                if ((i) != __ptr_stack_sz - 1)															\
                                                    __ptr_stack[i] = __ptr_stack[__ptr_stack_sz - 1];									\
                                                                                                                                        \
                                                --__ptr_stack_sz;																		\
                                            } while (0)

#define		__ptr_stack_abort()				do {																						\
                                                for (int i = __ptr_stack_sz - 1; i >= 0; --i) 											\
                                                    __ptr_stack_remove(i);																\
                                                                                                                                        \
                                                cudaDeviceReset();																		\
                                                exit(EXIT_FAILURE);																		\
                                            } while (0)

#define		open_file(fp, fname, perm)		do {																						\
                                                if (((fp) = fopen((fname), (perm))) == NULL) {											\
                                                    fprintf(stderr, "%s:%d: error opening file %s: %s\n", __FILE__, __LINE__,			\
                                                        (fname), strerror(errno));														\
                                                                                                                                        \
                                                    __ptr_stack_abort();																\
                                                } else {																				\
                                                    char *tmp_name;																		\
                                                                                                                                        \
                                                    if ((tmp_name = (char *) malloc(strlen(fname) + 1)) == NULL) {						\
                                                        fprintf(stderr, "%s:%d: error allocating memory: %s\n", __FILE__, __LINE__,		\
                                                            strerror(errno));															\
                                                                                                                                        \
                                                        __ptr_stack_abort();															\
                                                    }																					\
                                                                                                                                        \
                                                    strcpy(tmp_name, fname);															\
                                                    __ptr_stack_push(fp, tmp_name);														\
                                                }																						\
                                            } while (0)

#define		close_file(fp)					do {																						\
                                                for (int i = 0; i < __ptr_stack_sz; ++i)												\
                                                    if (__ptr_stack[i].addr == (fp)) {													\
                                                        __ptr_stack_remove(i);															\
                                                        break;																			\
                                                    }																					\
                                            } while (0)

#define		host_alloc(ptr, type, nelem)		do {																				    \
                                                if (((ptr) = (type *) malloc((nelem) * sizeof(type))) == NULL) {					    \
                                                    fprintf(stderr, "%s:%d: memory allocation failure: %s\n", __FILE__, __LINE__,	    \
                                                        strerror(errno));																\
                                                                                                                                        \
                                                    __ptr_stack_abort();																\
                                                } else {																				\
                                                    __ptr_stack_push(ptr, NULL);														\
                                                }																						\
                                            } while (0)

#define		host_free(ptr)					do {																						\
                                                for (int i = 0; i < __ptr_stack_sz; ++i)												\
                                                    if (__ptr_stack[i].addr == (ptr)) {													\
                                                        __ptr_stack_remove(i);															\
                                                        break;																			\
                                                    }																					\
                                            } while (0)

#define		write_file(ptr, sz, cnt, fid)	do {																						\
                                                if (fwrite((ptr), (sz), (cnt), (fid)) != (cnt)) {										\
                                                    fprintf(stderr, "%s:%d: error writing to file\n", __FILE__, __LINE__);				\
                                                    __ptr_stack_abort();																\
                                                }																						\
                                            } while (0)

#define		read_file(ptr, sz, cnt, fid)	do {																						\
                                                if (fread((ptr), (sz), (cnt), (fid)) != (cnt)) {										\
                                                    fprintf(stderr, "%s:%d: error reading from file\n", __FILE__, __LINE__);			\
                                                    __ptr_stack_abort();																\
                                                }																						\
                                            } while (0)

#define		cuda_exec(func_call)			do {																						\
                                                cudaError_t	error = (func_call);														\
                                                                                                                                        \
                                                if (error != cudaSuccess) {																\
                                                    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error));		\
                                                                                                                                        \
                                                    __ptr_stack_abort();																\
                                                }																						\
                                            } while (0)

#define		cublas_exec(func_call)			do {																						\
                                                cublasStatus_t error = (func_call);														\
                                                                                                                                        \
                                                if (error != CUBLAS_STATUS_SUCCESS) {													\
                                                    fprintf(stderr, "%s:%d: CUBLAS call error\n", __FILE__, __LINE__);					\
                                                    __ptr_stack_abort();																\
                                                }																						\
                                            } while (0)


double	timer()
{
    struct	timeval		tp;

    gettimeofday(&tp, NULL);

    return ((double) tp.tv_sec + (double) tp.tv_usec * 1.0e-6);
}

template <typename T>
void	init_matrix(T *m, int nx, int ny, int lda)
{
    srand(time(0));

    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j)
            m[j] = (double) rand() / RAND_MAX;

        m += lda;
    }
}


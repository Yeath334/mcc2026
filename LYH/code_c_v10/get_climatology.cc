#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <omp.h>

#include <hip/hip_runtime.h>
#include "hdf5.h"

#define HIP_CHECK(cmd) do { \
    hipError_t _e = (cmd); \
    if (_e != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d: %s (rc=%d)\n", \
                __FILE__, __LINE__, hipGetErrorString(_e), (int)_e); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define GPU_THREADS 256

#define START_YR 1991
#define END_YR   2020
#define DELTA_DAY 5
#define N_YEARS  30
#define N_OFFSETS 11
#define SAMPLE_TOTAL (N_YEARS * N_OFFSETS)

#define DEFAULT_SAVE_PATH "/public/home/mcc20262029/lyh/code_c_v9/ERA5/Climatology/"
#define DEFAULT_NC_PATH "/public/home/achwjznh4b/Newdata/"

#define DATA_OFFSET 23488

#define LON_NUM      1440
#define LAT_NUM      721
#define GRID_SIZE    ((size_t)LON_NUM * LAT_NUM)
#define FILE_ELEM_SZ 8
#define RAW_BYTES    (GRID_SIZE * FILE_ELEM_SZ)
#define FLT_BYTES    (GRID_SIZE * sizeof(float))

#define HIST_BINS 128
#define P90_FIXED_MIN -2.0f
#define P90_FIXED_MAX 35.0f
#define P90_FIXED_INV_WIDTH ((float)HIST_BINS / (P90_FIXED_MAX - P90_FIXED_MIN))
#define P90_FIXED_BIN_WIDTH ((P90_FIXED_MAX - P90_FIXED_MIN) / (float)HIST_BINS)
#define MAX_IO_THREADS 64

static void gen_lon(double *lon) {
    #pragma omp simd
    for (int i = 0; i < LON_NUM; i++)
        lon[i] = -180.0 + i * (360.0 / (double)LON_NUM);
}
static void gen_lat(double *lat) {
    #pragma omp simd
    for (int i = 0; i < LAT_NUM; i++)
        lat[i] =  -90.0 + i * (180.0 / (double)(LAT_NUM - 1));
}

__attribute__((noinline)) __device__ static int is_finite_f(float val) {
    volatile union { float f; unsigned int u; } u;
    u.f = val;
    return (u.u & 0x7F800000) != 0x7F800000;
}

__global__ __launch_bounds__(256, 2) void compute_stats_kernel(
    const float * __restrict__ sst_data,
    double * __restrict__ Clim,
    double * __restrict__ P90,
    const int * __restrict__ d_slot_valid,
    size_t grid_size,
    int n_valid)
{
    __shared__ unsigned short tile_hist[256][HIST_BINS];

    (void)d_slot_valid;

    unsigned long long gid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int valid = (gid < (unsigned long long)grid_size);
    const unsigned long long stride = grid_size;

    if (n_valid == SAMPLE_TOTAL) {
        #pragma unroll
        for (int b = 0; b < HIST_BINS; b++)
            tile_hist[tid][b] = 0;

        if (valid) {
            double dsum = 0.0;
            unsigned long long idx = gid;

            #pragma unroll 8
            for (int k = 0; k < SAMPLE_TOTAL; k++) {
                float val = sst_data[idx];
                dsum += (double)val;

                int bin = (int)((val - P90_FIXED_MIN) * P90_FIXED_INV_WIDTH);
                bin = (bin >= 0) ? bin : 0;
                bin -= (bin >= HIST_BINS) * (bin - (HIST_BINS - 1));
                tile_hist[tid][bin]++;

                idx += stride;
            }
            Clim[gid] = dsum / (double)SAMPLE_TOTAL;

            int target = (int)ceilf(0.9f * (float)SAMPLE_TOTAL);
            int acc = 0, p90_bin = HIST_BINS - 1;
            #pragma unroll 4
            for (int b = 0; b < HIST_BINS; b++) {
                acc += tile_hist[tid][b];
                if (acc >= target) { p90_bin = b; break; }
            }
            P90[gid] = (double)(P90_FIXED_MIN + ((float)p90_bin + 0.5f) * P90_FIXED_BIN_WIDTH);
        }
    } else {
        if (valid) {
            unsigned short hist[HIST_BINS] = {0};
            double dsum = 0.0;
            int count = 0;

            unsigned long long idx = gid;
            #pragma unroll 8
            for (int k = 0; k < SAMPLE_TOTAL; k++) {
                float val = sst_data[idx];
                if (is_finite_f(val)) {
                    dsum += (double)val;
                    count++;

                    int bin = (int)((val - P90_FIXED_MIN) * P90_FIXED_INV_WIDTH);
                    bin = (bin >= 0) ? bin : 0;
                    bin -= (bin >= HIST_BINS) * (bin - (HIST_BINS - 1));
                    hist[bin]++;
                }
                idx += stride;
            }

            if (count == 0) { Clim[gid] = NAN; P90[gid] = NAN; return; }
            Clim[gid] = dsum / (double)count;

            int target = (int)ceilf(0.9f * (float)count);
            int acc = 0, p90_bin = HIST_BINS - 1;
            #pragma unroll 4
            for (int b = 0; b < HIST_BINS; b++) {
                acc += hist[b];
                if (acc >= target) { p90_bin = b; break; }
            }
            P90[gid] = (double)(P90_FIXED_MIN + ((float)p90_bin + 0.5f) * P90_FIXED_BIN_WIDTH);
        }
    }
}

static const char *nc_path = DEFAULT_NC_PATH;

#define H5_CHECK(x) do { \
    if ((x) < 0) { fprintf(stderr, "HDF5 error at line %d\n", __LINE__); return; } \
} while (0)

static void doy_to_month_day(int doy, int *month, int *day) {
    int mdays[12] = {31,28,31,30,31,30,31,31,30,31,30,31};
    for (int m = 0; m < 12; m++) {
        if (doy <= mdays[m]) { *month = m + 1; *day = doy; return; }
        doy -= mdays[m];
    }
}

static void make_file_path(int year, int doy, char *out) {
    int m, d;
    doy_to_month_day(doy, &m, &d);
    sprintf(out, "%s%04d%02d%02d", nc_path, year, m, d);
}

static void write_output_hdf5(const char *filename,
                              const double *lon, const double *lat,
                              int lon_num, int lat_num, int doy,
                              const double *Clim, const double *P90) {
    hid_t file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        fprintf(stderr, "H5Fcreate failed: %s (errno=%d %s)\n",
                filename, errno, strerror(errno));
        return;
    }
    herr_t wr_status = 0;
    { hsize_t d[1]={(hsize_t)lat_num}; hid_t s=H5Screate_simple(1,d,NULL);
      hid_t ds=H5Dcreate2(file_id,"Lat",H5T_NATIVE_DOUBLE,s,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
      wr_status = H5Dwrite(ds,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,lat);
      H5Dclose(ds); H5Sclose(s); if (wr_status < 0) goto fail; }
    { hsize_t d[1]={(hsize_t)lon_num}; hid_t s=H5Screate_simple(1,d,NULL);
      hid_t ds=H5Dcreate2(file_id,"Lon",H5T_NATIVE_DOUBLE,s,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
      wr_status = H5Dwrite(ds,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,lon);
      H5Dclose(ds); H5Sclose(s); if (wr_status < 0) goto fail; }
    { hsize_t d[1]={1}; hid_t s=H5Screate_simple(1,d,NULL);
      hid_t ds=H5Dcreate2(file_id,"dayofyear",H5T_NATIVE_DOUBLE,s,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
      double v=(double)doy;
      wr_status = H5Dwrite(ds,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,&v);
      H5Dclose(ds); H5Sclose(s); if (wr_status < 0) goto fail; }
    { hsize_t d[2]={(hsize_t)lat_num,(hsize_t)lon_num}; hid_t s=H5Screate_simple(2,d,NULL);
      hid_t ds=H5Dcreate2(file_id,"Climmean",H5T_NATIVE_DOUBLE,s,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
      wr_status = H5Dwrite(ds,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,Clim);
      H5Dclose(ds); H5Sclose(s); if (wr_status < 0) goto fail; }
    { hsize_t d[2]={(hsize_t)lat_num,(hsize_t)lon_num}; hid_t s=H5Screate_simple(2,d,NULL);
      hid_t ds=H5Dcreate2(file_id,"P90_sst",H5T_NATIVE_DOUBLE,s,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
      wr_status = H5Dwrite(ds,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,P90);
      H5Dclose(ds); H5Sclose(s); if (wr_status < 0) goto fail; }
    H5Fclose(file_id);
    return;
fail:
    fprintf(stderr, "H5Dwrite failed for %s (errno=%d %s) ??deleting partial file\n",
            filename, errno, strerror(errno));
    H5Fclose(file_id);
    unlink(filename);
}


int main(int argc, char **argv) {
    double t_total0 = omp_get_wtime();

    const char *env_rank = getenv("SLURM_PROCID");
    int mpi_rank = env_rank ? atoi(env_rank) : 0;
    const char *env_size = getenv("SLURM_NTASKS");
    int mpi_size = env_size ? atoi(env_size) : 8;
    int local_rank = 0;
    const char *lr_env = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (lr_env == NULL) lr_env = getenv("SLURM_LOCALID");
    if (lr_env != NULL) local_rank = atoi(lr_env);

    double t_gpu0 = omp_get_wtime();
    int dev_count = 0;
    HIP_CHECK(hipGetDeviceCount(&dev_count));
    if (dev_count == 0) {
        fprintf(stderr, "[rank %d/%d] No GPU devices found!\n", mpi_rank, mpi_size);
        exit(EXIT_FAILURE);
    }
    HIP_CHECK(hipSetDevice(local_rank % dev_count));
    double t_devset = omp_get_wtime();

    if (argc > 1) nc_path = argv[1];
    const char *save_path = (argc > 4) ? argv[4] : DEFAULT_SAVE_PATH;
    int start_doy = 152, end_doy = 243;
    if (argc > 2) {
        start_doy = atoi(argv[2]);
        end_doy   = (argc > 3) ? atoi(argv[3]) : 243;
    } else {
        int total_doy = end_doy - start_doy + 1;
        int chunk = total_doy / mpi_size;
        int rem   = total_doy % mpi_size;
        start_doy = start_doy + mpi_rank * chunk + (mpi_rank < rem ? mpi_rank : rem);
        end_doy   = start_doy + chunk - 1 + (mpi_rank < rem ? 1 : 0);
    }
    int ndoys = end_doy - start_doy + 1;
    printf("DOY %d-%d (%d days)\n", start_doy, end_doy, ndoys);

    double *lon = (double *)malloc(LON_NUM * sizeof(double));
    double *lat = (double *)malloc(LAT_NUM * sizeof(double));
    gen_lon(lon); gen_lat(lat);
    size_t grid_size = GRID_SIZE, raw_bytes = RAW_BYTES, flt_bytes = FLT_BYTES;
    int io_threads = MAX_IO_THREADS;
    const char *env_io_th = getenv("IO_THREADS");
    if (env_io_th) io_threads = atoi(env_io_th);
    if (io_threads < N_YEARS) io_threads = N_YEARS;

    double *Clim  = (double *)malloc(grid_size * sizeof(double));
    double *P90   = (double *)malloc(grid_size * sizeof(double));
    double *Clim2 = (double *)malloc(grid_size * sizeof(double));
    double *P902  = (double *)malloc(grid_size * sizeof(double));
    float *sst_data = (float *)malloc(SAMPLE_TOTAL * flt_bytes);
    int   *slot_valid = (int *)calloc(SAMPLE_TOTAL, sizeof(int));
    void **raw_pool = (void **)malloc(N_YEARS * sizeof(void *));
    for (int t = 0; t < N_YEARS; t++)
        raw_pool[t] = aligned_alloc(32, raw_bytes);

    if (!Clim || !P90 || !Clim2 || !P902 || !sst_data || !slot_valid || !raw_pool) {
        fprintf(stderr, "Memory allocation failed.\n"); exit(EXIT_FAILURE);
    }
    { hipError_t hr = hipHostRegister(sst_data, SAMPLE_TOTAL * flt_bytes,
                                       hipHostRegisterDefault); }
    double t_hostmalloc = omp_get_wtime();

    size_t _sz_sst  = SAMPLE_TOTAL * flt_bytes;
    size_t _sz_clim = grid_size * sizeof(double);
    size_t _sz_slot = SAMPLE_TOTAL * sizeof(int);
    char *_d_base;
    HIP_CHECK(hipMalloc(&_d_base, _sz_sst + 2 * _sz_clim + _sz_slot));
    float *d_sst_data   = (float *)(_d_base);
    double *d_Clim      = (double *)(_d_base + _sz_sst);
    double *d_P90       = (double *)(_d_base + _sz_sst + _sz_clim);
    int   *d_slot_valid = (int   *)(_d_base + _sz_sst + 2 * _sz_clim);
    double t_gpumalloc = omp_get_wtime();

    hipStream_t stream, stream_h2d;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipStreamCreate(&stream_h2d));
    hipEvent_t ev_kern_start, ev_kern_stop, ev_d2h_stop, ev_h2d_done;
    HIP_CHECK(hipEventCreate(&ev_kern_start));
    HIP_CHECK(hipEventCreate(&ev_kern_stop));
    HIP_CHECK(hipEventCreate(&ev_d2h_stop));
    HIP_CHECK(hipEventCreate(&ev_h2d_done));
    double t_setup_done = omp_get_wtime();

    /* pre-open all files with 30 parallel threads */
    int fd_cache[30][366];
    memset(fd_cache, -1, sizeof(fd_cache));
    int min_doy = start_doy - DELTA_DAY, max_doy = end_doy + DELTA_DAY + 1;
    if (min_doy < 1)   min_doy = 1;
    if (max_doy > 365) max_doy = 365;

    #pragma omp parallel for collapse(2) num_threads(N_YEARS)
    for (int yi = 0; yi < N_YEARS; yi++)
        for (int doy = min_doy; doy <= max_doy; doy++) {
            char fpath[512];
            make_file_path(START_YR + yi, doy, fpath);
            fd_cache[yi][doy] = open(fpath, O_RDONLY);
        }

    /* ===== INIT ===== */
    double t_init0 = omp_get_wtime();

    #pragma omp parallel num_threads(N_YEARS)
    {
        int tid = omp_get_thread_num();
        void *raw = raw_pool[tid];

        #pragma omp for collapse(2) schedule(dynamic)
        for (int oi = 0; oi < N_OFFSETS; oi++) {
            for (int yi = 0; yi < N_YEARS; yi++) {
                int off = oi - DELTA_DAY;
                int td  = start_doy + off;
                int slot = oi * N_YEARS + yi;
                float *dst = &sst_data[(size_t)slot * grid_size];

                if (td >= 1 && td <= 365) {
                    int fd = fd_cache[yi][td];
                    if (fd >= 0 && pread(fd, raw, raw_bytes, DATA_OFFSET) == (ssize_t)raw_bytes) {
                        double * __restrict__ src = (double *)__builtin_assume_aligned(raw, 32);
                        #pragma omp simd aligned(src, dst: 32)
                        for (size_t g = 0; g < grid_size; g++)
                            dst[g] = (float)src[g];
                        slot_valid[slot] = 1;
                        hipMemcpyAsync(d_sst_data + (size_t)slot * grid_size,
                                      dst, flt_bytes, hipMemcpyHostToDevice, stream_h2d);
                        continue;
                    }
                }
                #pragma omp simd
                for (size_t g = 0; g < grid_size; g++)
                    dst[g] = NAN;
                hipMemcpyAsync(d_sst_data + (size_t)slot * grid_size,
                              dst, flt_bytes, hipMemcpyHostToDevice, stream_h2d);
            }
        }
    }

    hipMemcpyAsync(d_slot_valid, slot_valid, SAMPLE_TOTAL * sizeof(int),
                   hipMemcpyHostToDevice, stream_h2d);
    hipStreamSynchronize(stream_h2d);
    double t_init_h2d = omp_get_wtime();

    /* ===== Main loop ===== */
    double sum_t_io = 0.0, sum_t_cmp = 0.0, sum_t_wr = 0.0, sum_t_sync = 0.0;
    double sum_t_kern = 0.0, sum_t_d2h = 0.0;
    int slide = 0, cur = 0, prev_doy = -1;

    double *doy_io  = (double *)calloc(ndoys, sizeof(double));
    double *doy_cmp = (double *)calloc(ndoys, sizeof(double));
    double *doy_wr  = (double *)calloc(ndoys, sizeof(double));
    double *doy_sync = (double *)calloc(ndoys, sizeof(double));
    double *doy_wall = (double *)calloc(ndoys, sizeof(double));
    double t_doy_prev = omp_get_wtime();

    for (int doy = start_doy; doy <= end_doy; doy++) {
        double t_doy0 = omp_get_wtime();
        double *host_Clim = (cur == 0) ? Clim : Clim2;
        double *host_P90  = (cur == 0) ? P90  : P902;

        int n_valid = 0;
        for (int k = 0; k < SAMPLE_TOTAL; k++) n_valid += slot_valid[k];

        int blocks = (int)((grid_size + GPU_THREADS - 1) / GPU_THREADS);
        hipEventRecord(ev_kern_start, stream);
        compute_stats_kernel<<<blocks, GPU_THREADS, 0, stream>>>(
            d_sst_data, d_Clim, d_P90, d_slot_valid, grid_size, n_valid);
        hipEventRecord(ev_kern_stop, stream);
        hipMemcpyAsync(host_Clim, d_Clim, grid_size * sizeof(double), hipMemcpyDeviceToHost, stream);
        hipMemcpyAsync(host_P90, d_P90, grid_size * sizeof(double), hipMemcpyDeviceToHost, stream);
        hipEventRecord(ev_d2h_stop, stream);

        if (prev_doy >= start_doy) {
            double *prev_Clim = (cur == 0) ? Clim2 : Clim;
            double *prev_P90  = (cur == 0) ? P902  : P90;
            double t_wr0 = omp_get_wtime();
            int month, day;
            doy_to_month_day(prev_doy, &month, &day);
            char output_file[512];
            sprintf(output_file, "%s%02d%02d.nc", save_path, month, day);
            write_output_hdf5(output_file, lon, lat, LON_NUM, LAT_NUM,
                            prev_doy, prev_Clim, prev_P90);
            double wr_time = omp_get_wtime() - t_wr0;
            sum_t_wr += wr_time;
            doy_wr[prev_doy - start_doy] = wr_time;
        }

        double t_io0 = 0.0;
        if (doy < end_doy) {
            int future_new_doy = doy + 1 + DELTA_DAY;
            int future_slot_oi = slide % N_OFFSETS;
            t_io0 = omp_get_wtime();

            if (future_new_doy >= 1 && future_new_doy <= 365) {
                #pragma omp parallel num_threads(N_YEARS)
                {
                    int yi = omp_get_thread_num();
                    void *raw = raw_pool[yi];
                    int slot = future_slot_oi * N_YEARS + yi;
                    float *dst = &sst_data[(size_t)slot * grid_size];
                    int fd = fd_cache[yi][future_new_doy];

                    if (fd >= 0 && pread(fd, raw, raw_bytes, DATA_OFFSET) == (ssize_t)raw_bytes) {
                        double * __restrict__ src = (double *)__builtin_assume_aligned(raw, 32);
                        #pragma omp simd aligned(src, dst: 32)
                        for (size_t g = 0; g < grid_size; g++)
                            dst[g] = (float)src[g];
                        slot_valid[slot] = 1;
                    } else {
                        #pragma omp simd
                        for (size_t g = 0; g < grid_size; g++)
                            dst[g] = NAN;
                        slot_valid[slot] = 0;
                    }
                    hipMemcpyAsync(d_sst_data + (size_t)slot * grid_size,
                                  dst, flt_bytes, hipMemcpyHostToDevice, stream_h2d);
                }
                hipMemcpyAsync(d_slot_valid + future_slot_oi * N_YEARS,
                              slot_valid + future_slot_oi * N_YEARS,
                              N_YEARS * sizeof(int), hipMemcpyHostToDevice, stream_h2d);
            }
            double io_time = omp_get_wtime() - t_io0;
            sum_t_io += io_time;
            doy_io[doy + 1 - start_doy] = io_time;
            hipEventRecord(ev_h2d_done, stream_h2d);
        }

        double t_sync0 = omp_get_wtime();
        hipStreamSynchronize(stream);
        double sync0 = omp_get_wtime() - t_sync0;
        double t_sync1 = omp_get_wtime();
        hipStreamSynchronize(stream_h2d);
        double sync1 = omp_get_wtime() - t_sync1;
        double sync_time = sync0 + sync1;
        sum_t_sync += sync_time;
        doy_sync[doy - start_doy] = sync_time;

        double t_doy_now = omp_get_wtime();
        doy_wall[doy - start_doy] = t_doy_now - t_doy_prev;
        t_doy_prev = t_doy_now;

        float kern_ms, d2h_ms;
        hipEventElapsedTime(&kern_ms, ev_kern_start, ev_kern_stop);
        hipEventElapsedTime(&d2h_ms, ev_kern_stop, ev_d2h_stop);
        double cmp_time = (double)(kern_ms + d2h_ms) / 1000.0;
        sum_t_cmp += cmp_time;
        sum_t_kern += (double)kern_ms / 1000.0;
        sum_t_d2h += (double)d2h_ms / 1000.0;
        doy_cmp[doy - start_doy] = cmp_time;

        if (doy < end_doy) slide++;
        prev_doy = doy;
        cur = 1 - cur;
    }

    if (prev_doy >= start_doy) {
        double *last_Clim = (cur == 0) ? Clim2 : Clim;
        double *last_P90  = (cur == 0) ? P902  : P90;
        double t_wr0 = omp_get_wtime();
        int month, day;
        doy_to_month_day(prev_doy, &month, &day);
        char output_file[512];
        sprintf(output_file, "%s%02d%02d.nc", save_path, month, day);
        write_output_hdf5(output_file, lon, lat, LON_NUM, LAT_NUM,
                         prev_doy, last_Clim, last_P90);
        sum_t_wr += omp_get_wtime() - t_wr0;
    }

    /* close FDs */
    for (int yi = 0; yi < N_YEARS; yi++)
        for (int d = min_doy; d <= max_doy; d++)
            if (fd_cache[yi][d] >= 0) close(fd_cache[yi][d]);

    /* cleanup */
    double t_clean0 = omp_get_wtime();
    hipEventDestroy(ev_kern_start); hipEventDestroy(ev_kern_stop);
    hipEventDestroy(ev_d2h_stop);  hipEventDestroy(ev_h2d_done);
    hipStreamDestroy(stream);      hipStreamDestroy(stream_h2d);
    hipFree(_d_base);

    free(lon); free(lat);
    free(Clim); free(P90); free(Clim2); free(P902);
    free(sst_data); free(slot_valid);
    free(doy_io); free(doy_cmp); free(doy_wr);
    free(doy_sync); free(doy_wall);
    for (int t = 0; t < N_YEARS; t++) free(raw_pool[t]);
    free(raw_pool);
    double t_cleanup = omp_get_wtime() - t_clean0;

    double t_total = omp_get_wtime() - t_total0;
    double mainloop_total = sum_t_io + sum_t_cmp + sum_t_wr + sum_t_sync;
    double t_measured = (t_init_h2d - t_init0) + mainloop_total;
    double t_unaccounted = t_total - t_measured
        - (t_devset - t_gpu0) - (t_hostmalloc - t_devset)
        - (t_gpumalloc - t_hostmalloc) - (t_setup_done - t_gpumalloc)
        - t_cleanup;

    printf("SETUP: devset=%.3fs hostmalloc=%.3fs gpumalloc=%.3fs str+ev=%.3fs\n",
           t_devset - t_gpu0, t_hostmalloc - t_devset,
           t_gpumalloc - t_hostmalloc, t_setup_done - t_gpumalloc);
    printf("INIT:  io+h2d=%.3fs\n", t_init_h2d - t_init0);
    for (int i = 0; i < ndoys; i++)
        printf("DOY %3d: io=%.3fs cmp=%.3fs wr=%.3fs sync=%.3fs wall=%.3fs\n",
               start_doy + i, doy_io[i], doy_cmp[i], doy_wr[i],
               doy_sync[i], doy_wall[i]);
    printf("MAIN:  io=%.3fs cmp=%.3fs(kern=%.3f+d2h=%.3f) wr=%.3fs sync=%.3fs total=%.3fs\n",
           sum_t_io, sum_t_cmp, sum_t_kern, sum_t_d2h, sum_t_wr, sum_t_sync, mainloop_total);
    printf("WALL:  %.3f s (measured=%.3f unacc=%.3f cleanup=%.3f)\n",
           t_total, t_measured, t_unaccounted, t_cleanup);
    fflush(stdout);

    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    fprintf(stderr, "EXIT_TS: %ld.%09ld\n", (long)ts.tv_sec, (long)ts.tv_nsec);
    fflush(stderr);

    _exit(0);
}



#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "hdf5.h"
#include <omp.h>
#include <mpi.h>

#ifdef USE_HIP
#include <hip/hip_runtime.h>
#define GPU_THREADS 256
#endif

#define START_YR 1991
#define END_YR   2020
#define DELTA_DAY 5
#define N_YEARS  30
#define N_OFFSETS 11
#define SAMPLE_TOTAL (N_YEARS * N_OFFSETS)  /* 330 */

#define DEFAULT_SAVE_PATH "/public/home/mcc20262029/zhx/mcc_3/ERA5/Climatology/"
#define DEFAULT_NC_PATH "/public/home/achwjznh4b/Newdata/"

#define VAR_SST "data"
#define VAR_LON "lon"
#define VAR_LAT "lat"

/* Raw data offset in HDF5 file */
#define DATA_OFFSET 23488

/* Block size for compute gather */
#define COMP_BLK 64

#ifdef USE_HIP
#define HIST_BINS 256

__global__ void compute_stats_kernel(
    const float * __restrict__ sst_data,
    double * __restrict__ Clim,
    double * __restrict__ P90,
    const int * __restrict__ d_slot_valid,
    size_t grid_size,
    int n_valid)
{
    unsigned long long gid =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    if (gid >= (unsigned long long)grid_size) return;

    const unsigned long long stride = (unsigned long long)grid_size;

    if (n_valid == SAMPLE_TOTAL) {
        float sum = 0.0f;
        float vmin = 3.402823466e+38f;
        float vmax = -3.402823466e+38f;

        unsigned long long idx = gid;
        for (int k = 0; k < SAMPLE_TOTAL; k++) {
            float val = sst_data[idx];
            sum += val;
            if (val < vmin) vmin = val;
            if (val > vmax) vmax = val;
            idx += stride;
        }

        Clim[gid] = (double)(sum / (float)SAMPLE_TOTAL);
        if (vmin == vmax) { P90[gid] = (double)vmin; return; }

        unsigned short hist[HIST_BINS];
        for (int b = 0; b < HIST_BINS; b++) hist[b] = 0;
        float range = vmax - vmin;
        float inv_width = (float)HIST_BINS / range;
        idx = gid;
        for (int k = 0; k < SAMPLE_TOTAL; k++) {
            float val = sst_data[idx];
            int bin = (int)((val - vmin) * inv_width);
            if (bin < 0) bin = 0;
            if (bin >= HIST_BINS) bin = HIST_BINS - 1;
            hist[bin]++;
            idx += stride;
        }

        int target = (int)ceilf(0.9f * (float)SAMPLE_TOTAL);
        int acc = 0, p90_bin = HIST_BINS - 1;
        for (int b = 0; b < HIST_BINS; b++) {
            acc += hist[b];
            if (acc >= target) { p90_bin = b; break; }
        }
        P90[gid] = (double)(vmin + ((float)p90_bin + 0.5f) * (range / (float)HIST_BINS));
    } else {
        float sum = 0.0f;
        float vmin = 3.402823466e+38f;
        float vmax = -3.402823466e+38f;
        int count = 0;

        unsigned long long idx = gid;
        for (int k = 0; k < SAMPLE_TOTAL; k++) {
            float val = sst_data[idx];
            if (val == val) {
                sum += val; count++;
                if (val < vmin) vmin = val;
                if (val > vmax) vmax = val;
            }
            idx += stride;
        }

        if (count == 0) { Clim[gid] = NAN; P90[gid] = NAN; return; }
        Clim[gid] = (double)(sum / (float)count);
        if (count == 1 || vmin == vmax) { P90[gid] = (double)vmin; return; }

        unsigned short hist[HIST_BINS];
        for (int b = 0; b < HIST_BINS; b++) hist[b] = 0;
        float range = vmax - vmin;
        float inv_width = (float)HIST_BINS / range;
        idx = gid;
        for (int k = 0; k < SAMPLE_TOTAL; k++) {
            float val = sst_data[idx];
            if (val == val) {
                int bin = (int)((val - vmin) * inv_width);
                if (bin < 0) bin = 0;
                if (bin >= HIST_BINS) bin = HIST_BINS - 1;
                hist[bin]++;
            }
            idx += stride;
        }

        int target = (int)ceilf(0.9f * (float)count);
        int acc = 0, p90_bin = HIST_BINS - 1;
        for (int b = 0; b < HIST_BINS; b++) {
            acc += hist[b];
            if (acc >= target) { p90_bin = b; break; }
        }
        P90[gid] = (double)(vmin + ((float)p90_bin + 0.5f) * (range / (float)HIST_BINS));
    }
}
#endif /* USE_HIP */

/* Data path — overridable via command line */
static const char *nc_path = DEFAULT_NC_PATH;

#define H5_CHECK(x) do { \
    if ((x) < 0) { \
        fprintf(stderr, "HDF5 error at line %d\n", __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

static size_t file_elem_sz = 8;
static int    file_is_f32  = 0;

int file_exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0;
}

void doy_to_month_day(int doy, int *month, int *day) {
    int mdays[12] = {31,28,31,30,31,30,31,31,30,31,30,31};
    for (int m = 0; m < 12; m++) {
        if (doy <= mdays[m]) {
            *month = m + 1;
            *day = doy;
            return;
        }
        doy -= mdays[m];
    }
}

void make_file_path(int year, int doy, char *out) {
    int m, d;
    doy_to_month_day(doy, &m, &d);
    sprintf(out, "%s%04d%02d%02d", nc_path, year, m, d);
}

void read_1d_dataset(hid_t file_id, const char *name, double **data, int *len) {
    hid_t dset = H5Dopen2(file_id, name, H5P_DEFAULT);
    H5_CHECK(dset);
    hid_t space = H5Dget_space(dset);
    H5_CHECK(space);
    hsize_t dims[1];
    int ndims = H5Sget_simple_extent_dims(space, dims, NULL);
    if (ndims != 1) {
        fprintf(stderr, "Dataset %s is not 1D.\n", name);
        exit(EXIT_FAILURE);
    }
    *len = (int)dims[0];
    *data = (double *)malloc((*len) * sizeof(double));
    H5_CHECK(H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, *data));
    H5_CHECK(H5Sclose(space));
    H5_CHECK(H5Dclose(dset));
}

void read_lon_lat(double **lon, double **lat, int *lon_num, int *lat_num) {
    char demo_file[512];
    sprintf(demo_file, "%s19910101", DEFAULT_NC_PATH);
    hid_t file_id = H5Fopen(demo_file, H5F_ACC_RDONLY, H5P_DEFAULT);
    H5_CHECK(file_id);
    read_1d_dataset(file_id, VAR_LON, lon, lon_num);
    read_1d_dataset(file_id, VAR_LAT, lat, lat_num);
    H5_CHECK(H5Fclose(file_id));
}

void detect_sst_fmt(void) {
    char demo[512];
    sprintf(demo, "%s19910101", DEFAULT_NC_PATH);
    hid_t fid = H5Fopen(demo, H5F_ACC_RDONLY, H5P_DEFAULT);
    H5_CHECK(fid);
    hid_t dset = H5Dopen2(fid, VAR_SST, H5P_DEFAULT);
    H5_CHECK(dset);
    hid_t dtype = H5Dget_type(dset);
    H5_CHECK(dtype);

    file_elem_sz = H5Tget_size(dtype);
    H5T_class_t cls = H5Tget_class(dtype);

    if (cls == H5T_FLOAT && file_elem_sz == 4) {
        file_is_f32 = 1;
        printf("SST raw format: float32\n");
    } else if (cls == H5T_FLOAT && file_elem_sz == 8) {
        file_is_f32 = 0;
        printf("SST raw format: float64  (converting to float32)\n");
    } else {
        fprintf(stderr, "SST type not float (class=%d, size=%zu). Abort.\n",
                cls, file_elem_sz);
        exit(EXIT_FAILURE);
    }

    H5Tclose(dtype);
    H5Dclose(dset);
    H5Fclose(fid);
}

int read_sst_raw(const char *filename, void *buf, size_t nbytes) {
    int fd = open(filename, O_RDONLY);
    if (fd < 0) return -1;
    ssize_t n = pread(fd, buf, nbytes, DATA_OFFSET);
    close(fd);
    if (n != (ssize_t)nbytes) {
        fprintf(stderr, "pread short read: %s (got %zd, expected %zu)\n",
                filename, n, nbytes);
        return -1;
    }
    return 0;
}

void write_1d_dataset(hid_t file_id, const char *name, double *data, int len) {
    hsize_t dims[1] = {(hsize_t)len};
    hid_t space = H5Screate_simple(1, dims, NULL);
    H5_CHECK(space);
    hid_t dset = H5Dcreate2(file_id, name, H5T_NATIVE_DOUBLE, space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5_CHECK(dset);
    H5_CHECK(H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data));
    H5_CHECK(H5Dclose(dset));
    H5_CHECK(H5Sclose(space));
}

void write_2d_dataset(hid_t file_id, const char *name, double *data,
                      int lat_num, int lon_num) {
    hsize_t dims[2] = {(hsize_t)lat_num, (hsize_t)lon_num};
    hid_t space = H5Screate_simple(2, dims, NULL);
    H5_CHECK(space);
    hid_t dset = H5Dcreate2(file_id, name, H5T_NATIVE_DOUBLE, space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5_CHECK(dset);
    H5_CHECK(H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data));
    H5_CHECK(H5Dclose(dset));
    H5_CHECK(H5Sclose(space));
}

void write_output_hdf5(const char *filename, double *lon, double *lat,
                       int lon_num, int lat_num, int doy,
                       double *Clim, double *P90) {
    hid_t file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    H5_CHECK(file_id);
    write_1d_dataset(file_id, "Lat", lat, lat_num);
    write_1d_dataset(file_id, "Lon", lon, lon_num);

    hsize_t dims[1] = {1};
    hid_t space = H5Screate_simple(1, dims, NULL);
    hid_t dset = H5Dcreate2(file_id, "dayofyear", H5T_NATIVE_DOUBLE, space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    double doy_val = (double)doy;
    H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &doy_val);
    H5Dclose(dset);
    H5Sclose(space);

    write_2d_dataset(file_id, "Climmean", Clim, lat_num, lon_num);
    write_2d_dataset(file_id, "P90_sst", P90, lat_num, lon_num);
    H5_CHECK(H5Fclose(file_id));
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    double t_total0 = omp_get_wtime();

    if (argc > 1) nc_path = argv[1];
    const char *save_path = (argc > 4) ? argv[4] : DEFAULT_SAVE_PATH;
    int start_doy = 152, end_doy = 243;
    if (argc > 2) {
        start_doy = atoi(argv[2]);
        end_doy   = (argc > 3) ? atoi(argv[3]) : 243;
    } else {
        /* Auto-split DOY range by MPI rank */
        int total_doy = end_doy - start_doy + 1;  /* 92 */
        int chunk = total_doy / mpi_size;
        int rem   = total_doy % mpi_size;
        start_doy = start_doy + mpi_rank * chunk + (mpi_rank < rem ? mpi_rank : rem);
        end_doy   = start_doy + chunk - 1 + (mpi_rank < rem ? 1 : 0);
    }
    printf("[rank %d/%d] Data path: %s\n", mpi_rank, mpi_size, nc_path);
    printf("[rank %d/%d] DOY range: %d - %d (%d days)\n",
           mpi_rank, mpi_size, start_doy, end_doy, end_doy - start_doy + 1);

    detect_sst_fmt();

    double *lon = NULL, *lat = NULL;
    int lon_num, lat_num;
    read_lon_lat(&lon, &lat, &lon_num, &lat_num);
    printf("lon_num = %d, lat_num = %d\n", lon_num, lat_num);

    size_t grid_size    = (size_t)lat_num * lon_num;
    size_t raw_bytes    = grid_size * file_elem_sz;
    size_t flt_bytes    = grid_size * sizeof(float);
    int    max_threads  = omp_get_max_threads();

    double *Clim = (double *)malloc(grid_size * sizeof(double));
    double *P90  = (double *)malloc(grid_size * sizeof(double));

    /* sst_data: [SAMPLE_TOTAL][grid_size] — pread directly into rows */
    float *sst_data = (float *)malloc(SAMPLE_TOTAL * grid_size * sizeof(float));

    void  **raw_pool = (void  **)malloc(max_threads * sizeof(void *));
    for (int t = 0; t < max_threads; t++) {
        raw_pool[t] = malloc(raw_bytes);
    }

    if (!Clim || !P90 || !sst_data) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }

#ifdef USE_HIP
    float *d_sst_data = NULL;
    double *d_Clim = NULL, *d_P90 = NULL;
    int *d_slot_valid = NULL;

    hipError_t hip_err;
    hip_err = hipMalloc(&d_sst_data, SAMPLE_TOTAL * flt_bytes);
    if (hip_err != hipSuccess) {
        fprintf(stderr, "hipMalloc sst_data failed: %s\n", hipGetErrorString(hip_err));
        exit(EXIT_FAILURE);
    }
    hip_err = hipMalloc(&d_Clim, grid_size * sizeof(double));
    if (hip_err != hipSuccess) {
        fprintf(stderr, "hipMalloc Clim failed: %s\n", hipGetErrorString(hip_err));
        exit(EXIT_FAILURE);
    }
    hip_err = hipMalloc(&d_P90, grid_size * sizeof(double));
    if (hip_err != hipSuccess) {
        fprintf(stderr, "hipMalloc P90 failed: %s\n", hipGetErrorString(hip_err));
        exit(EXIT_FAILURE);
    }
    hip_err = hipMalloc(&d_slot_valid, SAMPLE_TOTAL * sizeof(int));
    if (hip_err != hipSuccess) {
        fprintf(stderr, "hipMalloc slot_valid failed: %s\n", hipGetErrorString(hip_err));
        exit(EXIT_FAILURE);
    }
    printf("GPU memory allocated: sst=%.1f GB, results=%.1f MB\n",
           (double)(SAMPLE_TOTAL * flt_bytes) / (1024*1024*1024),
           (double)(grid_size * sizeof(double) * 2) / (1024*1024));

    hipStream_t stream;
    hipStreamCreate(&stream);
#endif

    printf("Initializing buffer with NaN...\n");
    #pragma omp parallel for
    for (size_t i = 0; i < SAMPLE_TOTAL * grid_size; i++) {
        sst_data[i] = NAN;
    }

    /* Pre-scan file existence */
    printf("Scanning file existence...\n");
    int *file_valid = (int *)malloc(30 * 366 * sizeof(int));
    for (int yi = 0; yi < 30; yi++) {
        int yr = START_YR + yi;
        for (int d = 1; d <= 365; d++) {
            char fpath[512];
            int m, day;
            doy_to_month_day(d, &m, &day);
            sprintf(fpath, "%s%04d%02d%02d", nc_path, yr, m, day);
            file_valid[yi * 366 + d] = file_exists(fpath);
        }
    }
    printf("File scan complete.\n");

    int *slot_valid = (int *)calloc(SAMPLE_TOTAL, sizeof(int));

    /* Phase 1: load initial window for start DOY */
    printf("Loading initial window for DOY %d...\n", start_doy);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        void *raw = raw_pool[tid];

        #pragma omp for schedule(dynamic)
        for (int k = 0; k < SAMPLE_TOTAL; k++) {
            int yi  = k / N_OFFSETS;
            int off = (k % N_OFFSETS) - DELTA_DAY;
            int td  = start_doy + off;

            if (td >= 1 && td <= 365) {
                int fid = yi * 366 + td;
                if (file_valid[fid]) {
                    char fpath[512];
                    make_file_path(START_YR + yi, td, fpath);

                    if (read_sst_raw(fpath, raw, raw_bytes) == 0) {
                        float *dst = &sst_data[k * grid_size];
                        if (file_is_f32) {
                            memcpy(dst, raw, flt_bytes);
                        } else {
                            double *src = (double *)raw;
                            #pragma omp simd
                            for (size_t g = 0; g < grid_size; g++)
                                dst[g] = (float)src[g];
                        }
                        slot_valid[k] = 1;
                    }
                }
            }
        }
    }

#ifdef USE_HIP
    /* Copy initial sst_data and slot_valid to GPU (async) */
    hipMemcpyAsync(d_sst_data, sst_data, SAMPLE_TOTAL * flt_bytes, hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(d_slot_valid, slot_valid, SAMPLE_TOTAL * sizeof(int), hipMemcpyHostToDevice, stream);
    hipStreamSynchronize(stream);
    printf("Initial data copied to GPU.\n");
#endif

    double sum_t_io  = 0.0;
    double sum_t_cmp = 0.0;
    double sum_t_wr  = 0.0;
    int slide = 0;

    for (int doy = start_doy; doy <= end_doy; doy++) {
        double t_doy0 = omp_get_wtime();

        /* ---- I/O: pread 30 files for next DOY + scatter to sst_data ---- */
        double t_io0 = omp_get_wtime();
        int next_new_doy = (doy < end_doy) ? doy + 1 + DELTA_DAY : 0;
        int next_slot_oi = (doy < end_doy) ? slide % N_OFFSETS    : 0;

        if (doy > start_doy && next_new_doy >= 1 && next_new_doy <= 365) {
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                void *raw = raw_pool[tid];

                #pragma omp for schedule(dynamic)
                for (int yi = 0; yi < N_YEARS; yi++) {
                    int slot = yi * N_OFFSETS + next_slot_oi;
                    int fid  = yi * 366 + next_new_doy;

                    if (file_valid[fid]) {
                        char fpath[512];
                        make_file_path(START_YR + yi, next_new_doy, fpath);

                        if (read_sst_raw(fpath, raw, raw_bytes) == 0) {
                            float *dst = &sst_data[slot * grid_size];
                            if (file_is_f32) {
                                memcpy(dst, raw, flt_bytes);
                            } else {
                                double *src = (double *)raw;
                                #pragma omp simd
                                for (size_t g = 0; g < grid_size; g++)
                                    dst[g] = (float)src[g];
                            }
                            slot_valid[slot] = 1;
                        } else {
                            #pragma omp simd
                            for (size_t g = 0; g < grid_size; g++)
                                sst_data[slot * grid_size + g] = NAN;
                            slot_valid[slot] = 0;
                        }
                    } else {
                        #pragma omp simd
                        for (size_t g = 0; g < grid_size; g++)
                            sst_data[slot * grid_size + g] = NAN;
                        slot_valid[slot] = 0;
                    }
                }
            }
        }
        double t_io1 = omp_get_wtime();
        sum_t_io += t_io1 - t_io0;

        /* ---- Compute ---- */
        double t_cmp0 = omp_get_wtime();

        int n_valid = 0;
        for (int k = 0; k < SAMPLE_TOTAL; k++) n_valid += slot_valid[k];

#ifdef USE_HIP
        /* Update changed slots on GPU (after first DOY) - async batched */
        if (doy > start_doy && next_new_doy >= 1 && next_new_doy <= 365) {
            for (int yi = 0; yi < N_YEARS; yi++) {
                int slot = yi * N_OFFSETS + next_slot_oi;
                hipMemcpyAsync(d_sst_data + slot * grid_size,
                              sst_data + slot * grid_size,
                              flt_bytes, hipMemcpyHostToDevice, stream);
            }
            hipMemcpyAsync(d_slot_valid, slot_valid, SAMPLE_TOTAL * sizeof(int),
                          hipMemcpyHostToDevice, stream);
        }

        compute_stats_kernel<<<grid_size, GPU_THREADS, 0, stream>>>(
            d_sst_data, d_Clim, d_P90, d_slot_valid, grid_size, n_valid);
        hipMemcpyAsync(Clim, d_Clim, grid_size * sizeof(double), hipMemcpyDeviceToHost, stream);
        hipMemcpyAsync(P90, d_P90, grid_size * sizeof(double), hipMemcpyDeviceToHost, stream);
        hipStreamSynchronize(stream);
#else
        int total_blk = (int)((grid_size + COMP_BLK - 1) / COMP_BLK);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int n_th = omp_get_num_threads();
            int chunk = (total_blk + n_th - 1) / n_th;
            int bi0 = tid * chunk;
            int bi1 = bi0 + chunk;
            if (bi1 > total_blk) bi1 = total_blk;

            for (int bi = bi0; bi < bi1; bi++) {
                size_t gs = (size_t)bi * COMP_BLK;
                size_t ge = gs + COMP_BLK;
                if (ge > grid_size) ge = grid_size;
                size_t nb = ge - gs;

                float buf[COMP_BLK][SAMPLE_TOTAL];
                double sum_buf[COMP_BLK] = {0};

                /* Gather + accumulate mean: outer over time slots */
                if (n_valid == SAMPLE_TOTAL) {
                    for (int k = 0; k < SAMPLE_TOTAL; k++) {
                        const float * __restrict src = &sst_data[k * grid_size + gs];
                        for (size_t i = 0; i < nb; i++) {
                            float x = src[i];
                            buf[i][k] = x;
                            sum_buf[i] += x;
                        }
                    }
                    for (size_t i = 0; i < nb; i++)
                        Clim[gs + i] = sum_buf[i] / SAMPLE_TOTAL;
                } else {
                    for (int k = 0; k < SAMPLE_TOTAL; k++) {
                        const float * __restrict src = &sst_data[k * grid_size + gs];
                        int sv = slot_valid[k];
                        for (size_t i = 0; i < nb; i++) {
                            float x = src[i];
                            buf[i][k] = x;
                            if (sv) sum_buf[i] += x;
                        }
                    }
                    for (size_t i = 0; i < nb; i++) {
                        size_t g = gs + i;
                        int cnt = n_valid;
                        if (cnt > 0)
                            Clim[g] = sum_buf[i] / cnt;
                        else
                            Clim[g] = NAN;
                    }
                }

                /* Compute P90 per grid point — quickselect in-place on buf[i] */
                for (size_t i = 0; i < nb; i++) {
                    size_t g = gs + i;
                    float * __restrict col = buf[i];

                    if (n_valid == SAMPLE_TOTAL) {
                        float pos = 0.9f * (float)(SAMPLE_TOTAL - 1);
                        int kk = (int)pos;
                        int lo = 0, hi = SAMPLE_TOTAL - 1;
                        while (lo < hi) {
                            int mid = (lo + hi) / 2;
                            float a = col[lo], b = col[mid], c = col[hi];
                            float piv = (a < b)
                                ? ((b < c) ? b : (a < c) ? c : a)
                                : ((a < c) ? a : (b < c) ? c : b);
                            int ii = lo - 1, jj = hi + 1;
                            while (1) {
                                do ii++; while (col[ii] < piv);
                                do jj--; while (col[jj] > piv);
                                if (ii >= jj) break;
                                float t = col[ii]; col[ii] = col[jj]; col[jj] = t;
                            }
                            if (jj < kk) lo = jj + 1;
                            else         hi = jj;
                        }
                        int lo2 = (int)floorf(pos);
                        int hi2 = (int)ceilf(pos);
                        float frac = pos - (float)lo2;
                        if (lo2 == hi2) {
                            P90[g] = (double)col[lo2];
                        } else {
                            float v_lo = col[lo2];
                            float v_hi = col[lo2 + 1];
                            for (int m = lo2 + 2; m < SAMPLE_TOTAL; m++)
                                if (col[m] < v_hi) v_hi = col[m];
                            P90[g] = (double)(v_lo + frac * (v_hi - v_lo));
                        }
                    } else {
                        /* NaN path: compact valid values to front, then quickselect */
                        int cnt = 0;
                        for (int k = 0; k < SAMPLE_TOTAL; k++) {
                            float x = col[k];
                            if (!isnan(x)) col[cnt++] = x;
                        }
                        if (cnt == 0) { P90[g] = NAN; continue; }

                        float pos = 0.9f * (float)(cnt - 1);
                        int kk = (int)pos;
                        int lo = 0, hi = cnt - 1;
                        while (lo < hi) {
                            int mid = (lo + hi) / 2;
                            float a = col[lo], b = col[mid], c = col[hi];
                            float piv = (a < b)
                                ? ((b < c) ? b : (a < c) ? c : a)
                                : ((a < c) ? a : (b < c) ? c : b);
                            int ii = lo - 1, jj = hi + 1;
                            while (1) {
                                do ii++; while (col[ii] < piv);
                                do jj--; while (col[jj] > piv);
                                if (ii >= jj) break;
                                float t = col[ii]; col[ii] = col[jj]; col[jj] = t;
                            }
                            if (jj < kk) lo = jj + 1;
                            else         hi = jj;
                        }
                        int lo2 = (int)floorf(pos);
                        int hi2 = (int)ceilf(pos);
                        float frac = pos - (float)lo2;
                        if (lo2 == hi2) {
                            P90[g] = (double)col[lo2];
                        } else {
                            float v_lo = col[lo2];
                            float v_hi = col[lo2 + 1];
                            for (int m = lo2 + 2; m < cnt; m++)
                                if (col[m] < v_hi) v_hi = col[m];
                            P90[g] = (double)(v_lo + frac * (v_hi - v_lo));
                        }
                    }
                }
            }
        }
#endif /* USE_HIP */

        double t_cmp1 = omp_get_wtime();
        sum_t_cmp += t_cmp1 - t_cmp0;

        /* ---- Write output ---- */
        int month, day;
        doy_to_month_day(doy, &month, &day);
        char output_file[512];
        sprintf(output_file, "%s%02d%02d.nc", save_path, month, day);
        write_output_hdf5(output_file, lon, lat, lon_num, lat_num, doy, Clim, P90);

        double t_wr1 = omp_get_wtime();
        sum_t_wr += t_wr1 - t_cmp1;

        if (doy < end_doy) slide++;

        printf("DOY %3d: io=%.2fs  cmp=%.4fs  wr=%.2fs  total=%.2fs\n",
               doy, t_io1 - t_io0, t_cmp1 - t_cmp0,
               t_wr1 - t_cmp1, omp_get_wtime() - t_doy0);
        fflush(stdout);
    }

#ifdef USE_HIP
    hipStreamDestroy(stream);
    hipFree(d_sst_data);
    hipFree(d_Clim);
    hipFree(d_P90);
    hipFree(d_slot_valid);
#endif

    free(lon); free(lat); free(Clim); free(P90);
    free(sst_data); free(file_valid); free(slot_valid);
    for (int t = 0; t < max_threads; t++) free(raw_pool[t]);
    free(raw_pool);

    double t_total = omp_get_wtime() - t_total0;

    int my_doy_count = end_doy - start_doy + 1;
    printf("\n============================================\n");
    printf("  TIMING SUMMARY [rank %d/%d] (%d DOYs)\n",
           mpi_rank, mpi_size, my_doy_count);
    printf("============================================\n");
    printf("  File I/O (pread+scatter):  %8.2f s  (%5.1f%%)\n",
           sum_t_io,  sum_t_io  / t_total * 100);
    printf("  Compute  (quickselect):    %8.2f s  (%5.1f%%)\n",
           sum_t_cmp, sum_t_cmp / t_total * 100);
    printf("  Output write:              %8.2f s  (%5.1f%%)\n",
           sum_t_wr,  sum_t_wr  / t_total * 100);
    printf("  ----------------------------------------\n");
    printf("  WALL TIME:                 %8.2f s\n", t_total);
    printf("============================================\n");
    fflush(stdout);

    MPI_Finalize();
    return 0;
}

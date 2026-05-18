#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/time.h>
#include "hdf5.h"
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <omp.h>

#define START_YR 1991
#define END_YR 2020
#define DELTA_DAY 5

#define DATA_OFFSET 23488

#define IO_THREADS_INIT 4
#define IO_THREADS_PREFETCH 2

#define YEAR_NUM (END_YR - START_YR + 1)
#define WINDOW_NUM (2 * DELTA_DAY + 1)
#define SAMPLE_TOTAL (YEAR_NUM * WINDOW_NUM)

#define COMP_BLK 32

#define NC_PATH "/public/home/achwjznh4b/Newdata/"
#define SAVE_PATH "/public/home/mcc20262029/lyh/code_c_v5/ERA5/Climatology/"

#define VAR_SST "data"
#define VAR_LON "lon"
#define VAR_LAT "lat"

#define H5_CHECK(x)                                               \
    do                                                            \
    {                                                             \
        if ((x) < 0)                                              \
        {                                                         \
            fprintf(stderr, "HDF5 error at line %d\n", __LINE__); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

int file_exists(const char *path){
    struct stat st;
    return stat(path, &st) == 0;
}

double get_elapsed_time(struct timeval start, struct timeval end){
    return (end.tv_sec - start.tv_sec) +
           (end.tv_usec - start.tv_usec) / 1000000.0;
}

void disable_hdf5_file_locking(void){
    setenv("HDF5_USE_FILE_LOCKING", "FALSE", 1);
}

hid_t create_no_lock_fapl(void){
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5_CHECK(fapl);

    H5_CHECK(H5Pset_file_locking(fapl, 0, 1));

    return fapl;
}

void doy_to_month_day(int doy, int *month, int *day){
    int mdays[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};

    for (int m = 0; m < 12; m++){
        if (doy <= mdays[m]){
            *month = m + 1;
            *day = doy;
            return;
        }
        doy -= mdays[m];
    }
}

void make_date_string(int year, int doy, char *out){
    int month, day;
    doy_to_month_day(doy, &month, &day);
    sprintf(out, "%04d%02d%02d", year, month, day);
}

void read_sst_full_raw(const char *filename, double *sst_full, int lat_num, int lon_num){
    int fd = open(filename, O_RDONLY);

    if (fd < 0){
        fprintf(stderr, "Failed to open file: %s, errno=%d\n", filename, errno);
        exit(EXIT_FAILURE);
    }

    size_t nbytes = (size_t)lat_num * lon_num * sizeof(double);

    ssize_t nread = pread(fd, sst_full, nbytes, DATA_OFFSET);

    if (nread < 0){
        fprintf(stderr, "pread failed: %s, errno=%d\n", filename, errno);
        close(fd);
        exit(EXIT_FAILURE);
    }

    if ((size_t)nread != nbytes){
        fprintf(stderr,
                "pread incomplete: %s, expected %zu bytes, got %zd bytes\n",
                filename, nbytes, nread);
        close(fd);
        exit(EXIT_FAILURE);
    }

    close(fd);
}

void read_1d_dataset(hid_t file_id, const char *name, double **data, int *len){
    hid_t dset = H5Dopen2(file_id, name, H5P_DEFAULT);
    H5_CHECK(dset);

    hid_t space = H5Dget_space(dset);
    H5_CHECK(space);

    hsize_t dims[1];
    int ndims = H5Sget_simple_extent_dims(space, dims, NULL);

    if (ndims != 1){
        fprintf(stderr, "Dataset %s is not 1D.\n", name);
        exit(EXIT_FAILURE);
    }

    *len = (int)dims[0];
    *data = (double *)malloc((size_t)(*len) * sizeof(double));

    if (*data == NULL){
        fprintf(stderr, "Memory allocation failed for %s.\n", name);
        exit(EXIT_FAILURE);
    }

    H5_CHECK(H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, *data));

    H5_CHECK(H5Sclose(space));
    H5_CHECK(H5Dclose(dset));
}

void read_lon_lat(double **lon, double **lat, int *lon_num, int *lat_num){
    char demo_file[512];
    sprintf(demo_file, "%s19910101", NC_PATH);

    if (!file_exists(demo_file)){
        fprintf(stderr, "Demo file does not exist: %s\n", demo_file);
        exit(EXIT_FAILURE);
    }

    hid_t fapl = create_no_lock_fapl();

    hid_t file_id = H5Fopen(demo_file, H5F_ACC_RDONLY, fapl);
    H5_CHECK(file_id);

    H5_CHECK(H5Pclose(fapl));

    read_1d_dataset(file_id, VAR_LON, lon, lon_num);
    read_1d_dataset(file_id, VAR_LAT, lat, lat_num);

    H5_CHECK(H5Fclose(file_id));
}

void write_1d_dataset(hid_t file_id, const char *name, double *data, int len){
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

void write_scalar_dataset(hid_t file_id, const char *name, double value){
    hid_t space = H5Screate(H5S_SCALAR);
    H5_CHECK(space);

    hid_t dset = H5Dcreate2(file_id, name, H5T_NATIVE_DOUBLE, space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5_CHECK(dset);

    H5_CHECK(H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value));

    H5_CHECK(H5Dclose(dset));
    H5_CHECK(H5Sclose(space));
}

void write_2d_dataset(hid_t file_id, const char *name, double *data, int lat_num, int lon_num){
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

void write_output_hdf5(const char *filename, double *lon, double *lat,int lon_num, int lat_num, int doy,double *Clim, double *P90){
    if (file_exists(filename)){
        remove(filename);
    }

    hid_t fapl = create_no_lock_fapl();

    hid_t file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5_CHECK(file_id);

    H5_CHECK(H5Pclose(fapl));

    write_1d_dataset(file_id, "Lat", lat, lat_num);
    write_1d_dataset(file_id, "Lon", lon, lon_num);
    write_scalar_dataset(file_id, "dayofyear", (double)doy);
    write_2d_dataset(file_id, "Climmean", Clim, lat_num, lon_num);
    write_2d_dataset(file_id, "P90_sst", P90, lat_num, lon_num);

    H5_CHECK(H5Fclose(file_id));
}

void copy_full_to_sample(float *sst_temp, double *sst_full,int sample_idx, int lat_num, int lon_num){
    if (sample_idx < 0 || sample_idx >= SAMPLE_TOTAL){
        fprintf(stderr, "sample_idx out of range: %d\n", sample_idx);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < lat_num; i++){
        size_t dst = (size_t)i * SAMPLE_TOTAL * lon_num + (size_t)sample_idx * lon_num;
        size_t src = (size_t)i * lon_num;

        #pragma omp simd
        for (int j = 0; j < lon_num; j++){
            sst_temp[dst + j] = (float)sst_full[src + j];
        }
    }
}

void fill_sample_nan(float *sst_temp, int sample_idx, int lat_num, int lon_num){
    if (sample_idx < 0 || sample_idx >= SAMPLE_TOTAL){
        fprintf(stderr, "sample_idx out of range: %d\n", sample_idx);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < lat_num; i++){
        size_t base = (size_t)i * SAMPLE_TOTAL * lon_num + (size_t)sample_idx * lon_num;

        #pragma omp simd
        for (int j = 0; j < lon_num; j++){
            sst_temp[base + j] = (float)NAN;
        }
    }
}

void load_initial_window(int doy, float *sst_temp, double *sst_full_pool,int lat_num, int lon_num, int *slot_valid){
    
    #pragma omp parallel for num_threads(IO_THREADS_INIT) schedule(dynamic)
    for (int sample_idx = 0; sample_idx < SAMPLE_TOTAL; sample_idx++){
        int yr_idx = sample_idx / WINDOW_NUM;
        int day_idx = sample_idx % WINDOW_NUM;

        int yr = START_YR + yr_idx;
        int offset = day_idx - DELTA_DAY;
        int target_doy = doy + offset;

        if (target_doy < 1 || target_doy > 365){
            slot_valid[sample_idx] = 0;
            fill_sample_nan(sst_temp, sample_idx, lat_num, lon_num);
            continue;
        }

        char date_str[16];
        char input_file[512];

        make_date_string(yr, target_doy, date_str);
        sprintf(input_file, "%s%s", NC_PATH, date_str);

        if (!file_exists(input_file)){
            printf("Warning: %s does not exist, skip.\n", input_file);
            fflush(stdout);

            slot_valid[sample_idx] = 0;
            fill_sample_nan(sst_temp, sample_idx, lat_num, lon_num);
            continue;
        }

        int tid = omp_get_thread_num();
        double *local_sst_full = &sst_full_pool[(size_t)tid * lat_num * lon_num];

        read_sst_full_raw(input_file, local_sst_full, lat_num, lon_num);
        copy_full_to_sample(sst_temp, local_sst_full, sample_idx, lat_num, lon_num);

        slot_valid[sample_idx] = 1;
    }
}

void load_new_to_slot(int doy, int overwrite_day_idx, float *sst_temp, double *sst_full_pool,int lat_num, int lon_num, int *slot_valid){
    int target_doy = doy + DELTA_DAY;

    #pragma omp parallel for num_threads(IO_THREADS_INIT) schedule(dynamic)
    for (int yr = START_YR; yr <= END_YR; yr++){
        int yr_idx = yr - START_YR;
        int sample_idx = yr_idx * WINDOW_NUM + overwrite_day_idx;

        slot_valid[sample_idx] = 0;

        if (target_doy < 1 || target_doy > 365){
            fill_sample_nan(sst_temp, sample_idx, lat_num, lon_num);
            continue;
        }

        char date_str[16];
        char input_file[512];

        make_date_string(yr, target_doy, date_str);
        sprintf(input_file, "%s%s", NC_PATH, date_str);

        if (!file_exists(input_file)){
            printf("Warning: %s does not exist, skip.\n", input_file);
            fflush(stdout);

            fill_sample_nan(sst_temp, sample_idx, lat_num, lon_num);
            continue;
        }

        int tid = omp_get_thread_num();
        double *local_sst_full = &sst_full_pool[(size_t)tid * lat_num * lon_num];

        read_sst_full_raw(input_file, local_sst_full, lat_num, lon_num);
        copy_full_to_sample(sst_temp, local_sst_full, sample_idx, lat_num, lon_num);

        slot_valid[sample_idx] = 1;
    }
}

void scatter_stage_to_slot(float *sst_temp, double *stage_buf, int *stage_ok,int *slot_valid, int lat_num, int lon_num,int overwrite_day_idx){
    size_t grid_size = (size_t)lat_num * lon_num;

    #pragma omp parallel for schedule(static)
    for (int yr_idx = 0; yr_idx < YEAR_NUM; yr_idx++){
        int sample_idx = yr_idx * WINDOW_NUM + overwrite_day_idx;

        if (stage_ok[yr_idx]){
            double *src = &stage_buf[(size_t)yr_idx * grid_size];
            copy_full_to_sample(sst_temp, src, sample_idx, lat_num, lon_num);
            slot_valid[sample_idx] = 1;
        }
        else{
            fill_sample_nan(sst_temp, sample_idx, lat_num, lon_num);
            slot_valid[sample_idx] = 0;
        }
    }
}

void prefetch_next_right_edge(int next_doy, double *stage_buf, int *stage_ok,int lat_num, int lon_num, int tid, int n_io){
    size_t grid_size = (size_t)lat_num * lon_num;

    int chunk = (YEAR_NUM + n_io - 1) / n_io;
    int y0 = tid * chunk;
    int y1 = y0 + chunk;

    if (y1 > YEAR_NUM){
        y1 = YEAR_NUM;
    }

    for (int yr_idx = y0; yr_idx < y1; yr_idx++){
        int yr = START_YR + yr_idx;

        stage_ok[yr_idx] = 0;

        if (next_doy < 1 || next_doy > 365){
            continue;
        }

        char date_str[16];
        char input_file[512];

        make_date_string(yr, next_doy, date_str);
        sprintf(input_file, "%s%s", NC_PATH, date_str);

        if (!file_exists(input_file)){
            continue;
        }

        double *dst = &stage_buf[(size_t)yr_idx * grid_size];

        read_sst_full_raw(input_file, dst, lat_num, lon_num);

        stage_ok[yr_idx] = 1;
    }
}

int main(){
    disable_hdf5_file_locking();

    struct timeval total_start, total_end;
    gettimeofday(&total_start, NULL);

    double *lon = NULL;
    double *lat = NULL;
    int lon_num, lat_num;

    read_lon_lat(&lon, &lat, &lon_num, &lat_num);

    printf("lon_num = %d, lat_num = %d\n", lon_num, lat_num);

    double *Clim = (double *)malloc((size_t)lat_num * lon_num * sizeof(double));
    double *P90 = (double *)malloc((size_t)lat_num * lon_num * sizeof(double));

    float *sst_temp = (float *)malloc((size_t)lat_num * SAMPLE_TOTAL * lon_num * sizeof(float));
    double *sst_full_pool = (double *)malloc((size_t)IO_THREADS_INIT * lat_num * lon_num * sizeof(double));

    int *slot_valid = (int *)calloc(SAMPLE_TOTAL, sizeof(int));

    size_t grid_size = (size_t)lat_num * lon_num;
    double *stage_buf = (double *)malloc((size_t)YEAR_NUM * grid_size * sizeof(double));
    int *stage_ok = (int *)calloc(YEAR_NUM, sizeof(int));
    int stage_ready = 0;
    int stage_slot = -1;

    if (stage_buf == NULL || stage_ok == NULL){
        fprintf(stderr, "Memory allocation failed for stage buffer.\n");
        exit(EXIT_FAILURE);
    }

    if (Clim == NULL || P90 == NULL || sst_temp == NULL || sst_full_pool == NULL || slot_valid == NULL){
        fprintf(stderr, "Memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }

    for (int doy = 152; doy <= 243; doy++){
        struct timeval doy_start, doy_end;
        gettimeofday(&doy_start, NULL);

        printf("Calculating climatology for day %d...\n", doy);

        if (doy == 152){
            load_initial_window(doy, sst_temp, sst_full_pool, lat_num, lon_num, slot_valid);
        }
        else{
            int overwrite_day_idx = (doy - 153) % WINDOW_NUM;

            if (overwrite_day_idx < 0){
                overwrite_day_idx += WINDOW_NUM;
            }

            if (stage_ready && stage_slot == overwrite_day_idx){
                scatter_stage_to_slot(sst_temp, stage_buf, stage_ok,slot_valid, lat_num, lon_num,overwrite_day_idx);
                stage_ready = 0;
                stage_slot = -1;
            }
            else{
                load_new_to_slot(doy, overwrite_day_idx,sst_temp, sst_full_pool,lat_num, lon_num, slot_valid);
                stage_ready = 0;
                stage_slot = -1;
            }
        }

        int n_valid = 0;
        for (int k = 0; k < SAMPLE_TOTAL; k++){
            n_valid += slot_valid[k];
        }

        int all_valid = (n_valid == SAMPLE_TOTAL);

        int max_threads = omp_get_max_threads();

        int n_io = 0;
        if (doy < 243){
            n_io = IO_THREADS_PREFETCH;
        }

        int n_cmp = max_threads - n_io;
        if (n_cmp <= 0){
            fprintf(stderr, "n_cmp <= 0. Please use more OpenMP threads than IO_THREADS_PREFETCH.\n");
            exit(EXIT_FAILURE);
        }

        int next_doy = doy + 1 + DELTA_DAY;
        int next_overwrite_day_idx = -1;

        if (doy < 243){
            next_overwrite_day_idx = (doy + 1 - 153) % WINDOW_NUM;
            if (next_overwrite_day_idx < 0)
            {
                next_overwrite_day_idx += WINDOW_NUM;
            }
        }

        #pragma omp parallel num_threads(max_threads)
        {
            int tid = omp_get_thread_num();

            if (tid < n_io){
                prefetch_next_right_edge(next_doy, stage_buf, stage_ok,lat_num, lon_num, tid, n_io);
            }
            else{
                int ctid = tid - n_io;

                int rows_per_thread = (lat_num + n_cmp - 1) / n_cmp;

                int i0 = ctid * rows_per_thread;
                int i1 = i0 + rows_per_thread;

                if (i1 > lat_num){
                    i1 = lat_num;
                }

                for (int i = i0; i < i1; i++){
                    size_t row_out_base = (size_t)i * lon_num;
                    size_t row_tmp_base = (size_t)i * SAMPLE_TOTAL * lon_num;

                    float buf[COMP_BLK][SAMPLE_TOTAL];

                    for (int jb = 0; jb < lon_num; jb += COMP_BLK){     //分块操作
                        int nb = COMP_BLK;

                        if (jb + nb > lon_num){
                            nb = lon_num - jb;
                        }

                        for (int k = 0; k < SAMPLE_TOTAL; k++){
                            size_t src_base = row_tmp_base + (size_t)k * lon_num + jb;

                            #pragma omp simd
                            for (int b = 0; b < nb; b++){
                                buf[b][k] = sst_temp[src_base + b];
                            }
                        }

                        for (int b = 0; b < nb; b++){
                            int j = jb + b;
                            size_t out_idx = row_out_base + (size_t)j;

                            float *src = buf[b];

                            float v[SAMPLE_TOTAL];
                            float sum = 0.0f;
                            int count = 0;

                            if (all_valid){                      //预先进行isnan()的判断 减少isnan()的开销
                                for (int k = 0; k < SAMPLE_TOTAL; k++){
                                    float val = src[k];
                                    v[k] = val;
                                    sum += val;
                                }

                                count = SAMPLE_TOTAL;
                            }
                            else{
                                for (int k = 0; k < SAMPLE_TOTAL; k++){
                                    float val = src[k];

                                    if (!isnan(val)){
                                        v[count++] = val;
                                        sum += val;
                                    }
                                }

                                if (count == 0){
                                    Clim[out_idx] = NAN;
                                    P90[out_idx] = NAN;
                                    continue;
                                }
                            }

                            Clim[out_idx] = (double)(sum / (float)count);    //计算mean

                            float pos = 0.9f * (float)(count - 1);
                            int low_k = (int)floorf(pos);
                            int high_k = (int)ceilf(pos);
                            float frac = pos - (float)low_k;

                            int left = 0;
                            int right = count - 1;
                            int target = low_k;

                            while (left < right){                         //找P90
                                float pivot = v[(left + right) / 2];

                                int ii = left - 1;
                                int jj = right + 1;

                                while (1){
                                    do{
                                        ii++;
                                    } while (v[ii] < pivot);

                                    do{
                                        jj--;
                                    } while (v[jj] > pivot);

                                    if (ii >= jj){
                                        break;
                                    }

                                    float tmp = v[ii];
                                    v[ii] = v[jj];
                                    v[jj] = tmp;
                                }

                                if (jj < target){
                                    left = jj + 1;
                                }
                                else{
                                    right = jj;
                                }
                            }

                            float v_low = v[low_k];

                            if (low_k == high_k){
                                P90[out_idx] = (double)v_low;
                            }
                            else{
                                float v_high = v[low_k + 1];

                                for (int t = low_k + 2; t < count; t++){
                                    if (v[t] < v_high){
                                        v_high = v[t];
                                    }
                                }

                                P90[out_idx] = (double)(v_low * (1.0f - frac) + v_high * frac);
                            }
                        }
                    }
                }
            }
        }

        if (doy < 243){
            stage_ready = 1;
            stage_slot = next_overwrite_day_idx;
        }

        int month, day;
        doy_to_month_day(doy, &month, &day);

        char output_file[512];
        sprintf(output_file, "%s%02d%02d.nc", SAVE_PATH, month, day);

        write_output_hdf5(output_file, lon, lat, lon_num, lat_num, doy, Clim, P90);

        gettimeofday(&doy_end, NULL);
        double doy_elapsed = get_elapsed_time(doy_start, doy_end);

        printf("Day %d elapsed wall time: %.2f seconds\n", doy, doy_elapsed);
        printf("------------------------------------\n");

        fflush(stdout);
    }

    free(lon);
    free(lat);
    free(Clim);
    free(P90);
    free(sst_temp);
    free(sst_full_pool);
    free(slot_valid);
    free(stage_buf);
    free(stage_ok);

    gettimeofday(&total_end, NULL);
    double total_elapsed = get_elapsed_time(total_start, total_end);

    printf("\n====================================\n");
    printf("Total elapsed wall time: %.2f seconds\n", total_elapsed);
    printf("====================================\n");
    fflush(stdout);

    return 0;
}

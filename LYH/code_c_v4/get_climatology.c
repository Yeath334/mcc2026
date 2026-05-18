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
#define END_YR   2020
#define DELTA_DAY 5

#define DATA_OFFSET 23488

#define YEAR_NUM   (END_YR - START_YR + 1)
#define WINDOW_NUM (2 * DELTA_DAY + 1)
#define SAMPLE_TOTAL (YEAR_NUM * WINDOW_NUM)

#define NC_PATH "/public/home/achwjznh4b/Newdata/"
#define SAVE_PATH "/public/home/mcc20262029/lyh/code_c_v4/ERA5/Climatology/"

#define VAR_SST "data"
#define VAR_LON "lon"
#define VAR_LAT "lat"

#define H5_CHECK(x) do { \
    if ((x) < 0) { \
        fprintf(stderr, "HDF5 error at line %d\n", __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

int file_exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0;
}

double get_elapsed_time(struct timeval start, struct timeval end) {
    return (end.tv_sec - start.tv_sec) +
           (end.tv_usec - start.tv_usec) / 1000000.0;
}

void disable_hdf5_file_locking(void) {
    setenv("HDF5_USE_FILE_LOCKING", "FALSE", 1);
}

hid_t create_no_lock_fapl(void) {
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5_CHECK(fapl);

    H5_CHECK(H5Pset_file_locking(fapl, 0, 1));

    return fapl;
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

void make_date_string(int year, int doy, char *out) {
    int month, day;
    doy_to_month_day(doy, &month, &day);
    sprintf(out, "%04d%02d%02d", year, month, day);
}

int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a;
    double y = *(const double *)b;
    return (x > y) - (x < y);
}

double nanmean(double *arr, int n) {    //计算平均数
    double sum = 0.0;
    int count = 0;

    for (int i = 0; i < n; i++) {
        if (!isnan(arr[i])) {
            sum += arr[i];
            count++;
        }
    }

    if (count == 0) return NAN;
    return sum / count;
}

double percentile90(double *arr, int n) {    //计算90%分位数
    double tmp[SAMPLE_TOTAL];
    int count = 0;

    for (int i = 0; i < n; i++) {
        if (!isnan(arr[i])) {
            tmp[count++] = arr[i];
        }
    }

    if (count == 0) return NAN;

    qsort(tmp, count, sizeof(double), cmp_double);

    double pos = 0.9 * (count - 1);
    int low = (int)floor(pos);
    int high = (int)ceil(pos);
    double frac = pos - low;

    if (low == high) return tmp[low];

    return tmp[low] * (1.0 - frac) + tmp[high] * frac;
}

void read_sst_full_raw(const char *filename, double *sst_full, int lat_num, int lon_num) {
    int fd = open(filename, O_RDONLY);

    if (fd < 0) {
        fprintf(stderr, "Failed to open file: %s, errno=%d\n", filename, errno);
        exit(EXIT_FAILURE);
    }

    size_t nbytes = (size_t)lat_num * lon_num * sizeof(double);

    ssize_t nread = pread(fd, sst_full, nbytes, DATA_OFFSET);

    if (nread < 0) {
        fprintf(stderr, "pread failed: %s, errno=%d\n", filename, errno);
        close(fd);
        exit(EXIT_FAILURE);
    }

    if ((size_t)nread != nbytes) {
        fprintf(stderr,
                "pread incomplete: %s, expected %zu bytes, got %zd bytes\n",
                filename, nbytes, nread);
        close(fd);
        exit(EXIT_FAILURE);
    }

    close(fd);
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
    *data = (double *)malloc((size_t)(*len) * sizeof(double));

    if (*data == NULL) {
        fprintf(stderr, "Memory allocation failed for %s.\n", name);
        exit(EXIT_FAILURE);
    }

    H5_CHECK(H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, *data));

    H5_CHECK(H5Sclose(space));
    H5_CHECK(H5Dclose(dset));
}

void read_lon_lat(double **lon, double **lat, int *lon_num, int *lat_num) {
    char demo_file[512];
    sprintf(demo_file, "%s19910101", NC_PATH);

    if (!file_exists(demo_file)) {
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

void write_1d_dataset(hid_t file_id, const char *name, double *data, int len) {
    hsize_t dims[1] = {(hsize_t)len};

    hid_t space = H5Screate_simple(1, dims, NULL);
    H5_CHECK(space);

    hid_t dset = H5Dcreate2(file_id, name, H5T_NATIVE_DOUBLE,space,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
    H5_CHECK(dset);

    H5_CHECK(H5Dwrite(dset, H5T_NATIVE_DOUBLE,H5S_ALL, H5S_ALL,H5P_DEFAULT,data));

    H5_CHECK(H5Dclose(dset));
    H5_CHECK(H5Sclose(space));
}

void write_scalar_dataset(hid_t file_id, const char *name, double value) {
    hid_t space = H5Screate(H5S_SCALAR);
    H5_CHECK(space);

    hid_t dset = H5Dcreate2(file_id, name, H5T_NATIVE_DOUBLE,space,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
    H5_CHECK(dset);

    H5_CHECK(H5Dwrite(dset, H5T_NATIVE_DOUBLE,H5S_ALL, H5S_ALL,H5P_DEFAULT,&value));

    H5_CHECK(H5Dclose(dset));
    H5_CHECK(H5Sclose(space));
}

void write_2d_dataset(hid_t file_id, const char *name,double *data, int lat_num, int lon_num) {
    hsize_t dims[2] = {(hsize_t)lat_num, (hsize_t)lon_num};

    hid_t space = H5Screate_simple(2, dims, NULL);
    H5_CHECK(space);

    hid_t dset = H5Dcreate2(file_id, name, H5T_NATIVE_DOUBLE,space,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
    H5_CHECK(dset);

    H5_CHECK(H5Dwrite(dset, H5T_NATIVE_DOUBLE,H5S_ALL, H5S_ALL,H5P_DEFAULT,data));

    H5_CHECK(H5Dclose(dset));
    H5_CHECK(H5Sclose(space));
}

void write_output_hdf5(const char *filename,double *lon,double *lat,int lon_num,int lat_num,int doy,double *Clim,double *P90) {
    if (file_exists(filename)) {
        remove(filename);
    }

    hid_t fapl = create_no_lock_fapl();

    hid_t file_id = H5Fcreate(filename,H5F_ACC_TRUNC,H5P_DEFAULT,fapl);
    H5_CHECK(file_id);

    H5_CHECK(H5Pclose(fapl));

    write_1d_dataset(file_id, "Lat", lat, lat_num);
    write_1d_dataset(file_id, "Lon", lon, lon_num);
    write_scalar_dataset(file_id, "dayofyear", (double)doy);
    write_2d_dataset(file_id, "Climmean", Clim, lat_num, lon_num);
    write_2d_dataset(file_id, "P90_sst", P90, lat_num, lon_num);

    H5_CHECK(H5Fclose(file_id));
}

void copy_full_to_sample(double *sst_temp,double *sst_full,int sample_idx,int lat_num,int lon_num) {
    if (sample_idx < 0 || sample_idx >= SAMPLE_TOTAL) {
        fprintf(stderr, "sample_idx out of range: %d\n", sample_idx);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < lat_num; i++) {
        size_t dst = (size_t)i * SAMPLE_TOTAL * lon_num
                   + (size_t)sample_idx * lon_num;

        size_t src = (size_t)i * lon_num;

        memcpy(
            &sst_temp[dst],
            &sst_full[src],
            (size_t)lon_num * sizeof(double)
        );
    }
}

void fill_sample_nan(double *sst_temp,int sample_idx,int lat_num,int lon_num) {
    if (sample_idx < 0 || sample_idx >= SAMPLE_TOTAL) {
        fprintf(stderr, "sample_idx out of range: %d\n", sample_idx);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < lat_num; i++) {
        size_t base = (size_t)i * SAMPLE_TOTAL * lon_num
                    + (size_t)sample_idx * lon_num;

        for (int j = 0; j < lon_num; j++) {
            sst_temp[base + j] = NAN;
        }
    }
}

void load_initial_window(int doy,double *sst_temp,double *sst_full,int lat_num,int lon_num) {

    for (size_t i = 0; i < (size_t)lat_num * SAMPLE_TOTAL * lon_num; i++) {
        sst_temp[i] = NAN;
    }

    for (int yr = START_YR; yr <= END_YR; yr++) {
        int yr_idx = yr - START_YR;

        for (int offset = -DELTA_DAY; offset <= DELTA_DAY; offset++) {
            int day_idx = offset + DELTA_DAY;
            int sample_idx = yr_idx * WINDOW_NUM + day_idx;
            int target_doy = doy + offset;

            if (target_doy < 1 || target_doy > 365) {
                fill_sample_nan(sst_temp, sample_idx, lat_num, lon_num);
                continue;
            }

            char date_str[16];
            char input_file[512];

            make_date_string(yr, target_doy, date_str);
            sprintf(input_file, "%s%s", NC_PATH, date_str);

            if (!file_exists(input_file)) {
                printf("Warning: %s does not exist, skip.\n", input_file);
                fflush(stdout);

                fill_sample_nan(sst_temp, sample_idx, lat_num, lon_num);
                continue;
            }

            read_sst_full_raw(input_file, sst_full, lat_num, lon_num);

            copy_full_to_sample(sst_temp,sst_full,sample_idx,lat_num,lon_num);
        }
    }
}

void slide_window_one_day(double *sst_temp,int lat_num,int lon_num) {
    for (int i = 0; i < lat_num; i++) {
        for (int yr_idx = 0; yr_idx < YEAR_NUM; yr_idx++) {
            size_t base =
                (size_t)i * SAMPLE_TOTAL * lon_num
              + (size_t)yr_idx * WINDOW_NUM * lon_num;

            memmove(
                &sst_temp[base],
                &sst_temp[base + lon_num],
                (size_t)(WINDOW_NUM - 1) * lon_num * sizeof(double)
            );
        }
    }
}

void load_new_right_edge(int doy,double *sst_temp,double *sst_full,int lat_num,int lon_num) {
    int day_idx = WINDOW_NUM - 1;
    int target_doy = doy + DELTA_DAY;

    printf("Loading new right edge for doy %d, target_doy %d...\n", doy, target_doy);
    fflush(stdout);

    for (int yr = START_YR; yr <= END_YR; yr++) {
        int yr_idx = yr - START_YR;
        int sample_idx = yr_idx * WINDOW_NUM + day_idx;

        fill_sample_nan(sst_temp, sample_idx, lat_num, lon_num);

        if (target_doy < 1 || target_doy > 365) {
            continue;
        }

        char date_str[16];
        char input_file[512];

        make_date_string(yr, target_doy, date_str);
        sprintf(input_file, "%s%s", NC_PATH, date_str);

        if (!file_exists(input_file)) {
            printf("Warning: %s does not exist, skip.\n", input_file);
            fflush(stdout);
            continue;
        }

        read_sst_full_raw(input_file, sst_full, lat_num, lon_num);

        copy_full_to_sample(sst_temp,sst_full,sample_idx,lat_num,lon_num);
    }
}

int main() {
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
    double *sst_temp = (double *)malloc((size_t)lat_num * SAMPLE_TOTAL * lon_num * sizeof(double));
    double *sst_full = (double *)malloc((size_t)lat_num * lon_num * sizeof(double));
    

    if (Clim == NULL || P90 == NULL || sst_temp == NULL || sst_full == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(EXIT_FAILURE); 
    }

    for (int doy = 152; doy <= 243; doy++) {

        struct timeval doy_start, doy_end;
        gettimeofday(&doy_start, NULL);

        printf("Calculating climatology for day %d...\n", doy);     

        for (int i = 0; i < lat_num * lon_num; i++) {
            Clim[i] = NAN;
            P90[i] = NAN;
        }

        
        if (doy == 152) {
            load_initial_window(doy,sst_temp,sst_full,lat_num,lon_num);
        } else {
            slide_window_one_day(sst_temp,lat_num,lon_num);

            load_new_right_edge(doy,sst_temp,sst_full,lat_num,lon_num);
        }

        #pragma omp parallel for schedule(static)
        for(int i = 0; i < lat_num; i++){

            double local_sample_values[SAMPLE_TOTAL];

            size_t x = i * lon_num;

            for (int j = 0; j < lon_num; j++) {

                size_t z = i * (lon_num * SAMPLE_TOTAL);

                for (int k = 0; k < SAMPLE_TOTAL; k++) {
                    local_sample_values[k] = sst_temp[z + k * lon_num + j];
                }

                Clim[x + j] = nanmean(local_sample_values, SAMPLE_TOTAL);
                P90[x + j] = percentile90(local_sample_values, SAMPLE_TOTAL);
            }

        }
        
        printf("Finished all rows for day %d\n", doy);
        fflush(stdout);
        
            
        int month, day;
        doy_to_month_day(doy, &month, &day);

        char output_file[512];
        sprintf(output_file, "%s%02d%02d.nc", SAVE_PATH, month, day);

        write_output_hdf5(output_file, lon, lat, lon_num, lat_num, doy, Clim, P90);

        printf("%s was created successfully.\n", output_file);

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
    free(sst_full);

    gettimeofday(&total_end, NULL);
    double total_elapsed = get_elapsed_time(total_start, total_end);

    printf("\n====================================\n");
    printf("Total elapsed wall time: %.2f seconds\n", total_elapsed);
    printf("====================================\n");
    fflush(stdout);

    return 0;
}

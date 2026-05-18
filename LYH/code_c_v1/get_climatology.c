#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include "hdf5.h"

#define START_YR 1991
#define END_YR   2020
#define DELTA_DAY 5
#define SAMPLE_TOTAL 330

#define NC_PATH "/public/home/achwjznh4b/Newdata/"
#define SAVE_PATH "/public/home/mcc20262029/lyh/code_c/ERA5/Climatology/"

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

double nanmean(double *arr, int n) {
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

double percentile90(double *arr, int n) {
    double tmp[SAMPLE_TOTAL];
    int count = 0;

    for (int i = 0; i < n; i++) {
        if (!isnan(arr[i])) {
            tmp[count++] = arr[i];
        }
    }

    if (count == 0) return NAN;

    qsort(tmp, count, sizeof(double), cmp_double);4c

    double pos = 0.9 * (count - 1);
    int low = (int)floor(pos);
    int high = (int)ceil(pos);
    double frac = pos - low;

    if (low == high) return tmp[low];

    return tmp[low] * (1.0 - frac) + tmp[high] * frac;
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

    hid_t file_id = H5Fopen(demo_file, H5F_ACC_RDONLY, H5P_DEFAULT);
    H5_CHECK(file_id);

    read_1d_dataset(file_id, VAR_LON, lon, lon_num);
    read_1d_dataset(file_id, VAR_LAT, lat, lat_num);

    H5_CHECK(H5Fclose(file_id));
}

void read_sst_row(const char *filename, int row, int lon_num, double *sst_row) {
    hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    H5_CHECK(file_id);

    hid_t dset = H5Dopen2(file_id, VAR_SST, H5P_DEFAULT);
    H5_CHECK(dset);

    hid_t filespace = H5Dget_space(dset);
    H5_CHECK(filespace);

    hsize_t start[2] = {(hsize_t)row, 0};
    hsize_t count[2] = {1, (hsize_t)lon_num};

    H5_CHECK(H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, NULL, count, NULL));

    hsize_t mem_dims[1] = {(hsize_t)lon_num};
    hid_t memspace = H5Screate_simple(1, mem_dims, NULL);
    H5_CHECK(memspace);

    H5_CHECK(H5Dread(dset, H5T_NATIVE_DOUBLE, memspace, filespace, H5P_DEFAULT, sst_row));

    H5_CHECK(H5Sclose(memspace));
    H5_CHECK(H5Sclose(filespace));
    H5_CHECK(H5Dclose(dset));
    H5_CHECK(H5Fclose(file_id));
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

void write_scalar_dataset(hid_t file_id, const char *name, double value) {
    hid_t space = H5Screate(H5S_SCALAR);
    H5_CHECK(space);

    hid_t dset = H5Dcreate2(file_id, name, H5T_NATIVE_DOUBLE, space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5_CHECK(dset);

    H5_CHECK(H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value));

    H5_CHECK(H5Dclose(dset));
    H5_CHECK(H5Sclose(space));
}

void write_2d_dataset(hid_t file_id, const char *name, double *data, int lat_num, int lon_num) {
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

void write_output_hdf5(const char *filename,double *lon,double *lat,int lon_num,int lat_num,int doy,double *Clim,double *P90
) {
    if (file_exists(filename)) {
        remove(filename);
    }

    hid_t file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    H5_CHECK(file_id);

    write_1d_dataset(file_id, "Lat", lat, lat_num);
    write_1d_dataset(file_id, "Lon", lon, lon_num);
    write_scalar_dataset(file_id, "dayofyear", (double)doy);
    write_2d_dataset(file_id, "Climmean", Clim, lat_num, lon_num);
    write_2d_dataset(file_id, "P90_sst", P90, lat_num, lon_num);

    H5_CHECK(H5Fclose(file_id));
}

int main() {
    double *lon = NULL;
    double *lat = NULL;
    int lon_num, lat_num;

    read_lon_lat(&lon, &lat, &lon_num, &lat_num);

    printf("lon_num = %d, lat_num = %d\n", lon_num, lat_num);

    double *Clim = (double *)malloc((size_t)lat_num * lon_num * sizeof(double));
    double *P90 = (double *)malloc((size_t)lat_num * lon_num * sizeof(double));
    double *sst_temp = (double *)malloc((size_t)SAMPLE_TOTAL * lon_num * sizeof(double));
    double *sst_row = (double *)malloc((size_t)lon_num * sizeof(double));
    double sample_values[SAMPLE_TOTAL];

    if (Clim == NULL || P90 == NULL || sst_temp == NULL || sst_row == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }

    for (int doy = 152; doy <= 243; doy++) {
        printf("Calculating climatology for day %d...\n", doy);

        for (int i = 0; i < lat_num * lon_num; i++) {
            Clim[i] = NAN;
            P90[i] = NAN;
        }

        for (int row = 0; row < lat_num; row++) {
            for (int i = 0; i < SAMPLE_TOTAL * lon_num; i++) {
                sst_temp[i] = NAN;
            }

            int temp_idx = 0;

            for (int yr = START_YR; yr <= END_YR; yr++) {
                for (int offset = -DELTA_DAY; offset <= DELTA_DAY; offset++) {
                    int target_doy = doy + offset;

                    if (target_doy < 1 || target_doy > 365) {
                        temp_idx++;
                        continue;
                    }

                    char date_str[16];
                    char input_file[512];

                    make_date_string(yr, target_doy, date_str);
                    sprintf(input_file, "%s%s", NC_PATH, date_str);

                    if (!file_exists(input_file)) {
                        printf("Warning: %s does not exist, skip.\n", input_file);
                        temp_idx++;
                        continue;
                    }

                    read_sst_row(input_file, row, lon_num, sst_row);

                    for (int j = 0; j < lon_num; j++) {
                        sst_temp[temp_idx * lon_num + j] = sst_row[j];
                    }

                    temp_idx++;
                }
            }

            for (int j = 0; j < lon_num; j++) {
                for (int k = 0; k < SAMPLE_TOTAL; k++) {
                    sample_values[k] = sst_temp[k * lon_num + j];
                }

                Clim[row * lon_num + j] = nanmean(sample_values, SAMPLE_TOTAL);
                P90[row * lon_num + j] = percentile90(sample_values, SAMPLE_TOTAL);
            }

            if ((row + 1) % 10 == 0 || row == lat_num - 1) {
                printf("Finished row %d / %d\n", row + 1, lat_num);
                fflush(stdout);
            }
        }

        int month, day;
        doy_to_month_day(doy, &month, &day);

        char output_file[512];
        sprintf(output_file, "%s%02d%02d.nc", SAVE_PATH, month, day);

        write_output_hdf5(output_file, lon, lat, lon_num, lat_num, doy, Clim, P90);

        printf("%s was created successfully.\n", output_file);
        fflush(stdout);
    }

    free(lon);
    free(lat);
    free(Clim);
    free(P90);
    free(sst_temp);
    free(sst_row);

    return 0;
}

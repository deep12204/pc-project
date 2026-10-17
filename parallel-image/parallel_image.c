
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <mpi.h>

// Read a PPM token and skip comments
int read_token(FILE *fp, char *token)
{
    int ch;
    int i = 0;

    do {
        ch = fgetc(fp);

        if (ch == '#') {
            while (ch != '\n' && ch != EOF)
                ch = fgetc(fp);
        }
    } while (isspace(ch));

    if (ch == EOF)
        return 0;

    while (ch != EOF && !isspace(ch) && ch != '#') {
        if (i < 99)
            token[i++] = (char)ch;
        ch = fgetc(fp);
    }

    if (ch == '#') {
        while (ch != '\n' && ch != EOF)
            ch = fgetc(fp);
    }

    token[i] = '\0';
    return 1;
}

// Convert one RGB pixel to grayscale
void convert_to_grayscale(int *rgb, int pixels)
{
    for (int i = 0; i < pixels; i++) {
        int r = rgb[3 * i];
        int g = rgb[3 * i + 1];
        int b = rgb[3 * i + 2];

        int gray = (int)(0.299 * r +
                         0.587 * g +
                         0.114 * b + 0.5);

        rgb[3 * i]     = gray;
        rgb[3 * i + 1] = gray;
        rgb[3 * i + 2] = gray;
    }
}

int main(int argc, char *argv[])
{
    int rank, size;
    int width = 0, height = 0, maxval = 0;
    int success = 1;

    int *image = NULL;
    int *result = NULL;
    int *local_image = NULL;

    int *counts = NULL;
    int *displs = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start = MPI_Wtime();

    if (rank == 0) {
        if (argc < 3) {
            printf("Usage: mpirun -np 4 ./parallel_image "
                   "input.ppm output.ppm\n");
            success = 0;
        } else {
            FILE *fp = fopen(argv[1], "r");

            if (fp == NULL) {
                printf("Error: Cannot open input image.\n");
                success = 0;
            } else {
                char token[100];

                if (!read_token(fp, token) ||
                    strcmp(token, "P3") != 0) {
                    printf("Error: Only P3 PPM images "
                           "are supported.\n");
                    success = 0;
                }

                if (success &&
                    (!read_token(fp, token))) {
                    success = 0;
                } else if (success) {
                    width = atoi(token);
                }

                if (success && !read_token(fp, token))
                    success = 0;
                else if (success)
                    height = atoi(token);

                if (success && !read_token(fp, token))
                    success = 0;
                else if (success)
                    maxval = atoi(token);

                if (success &&
                    (width <= 0 || height <= 0 ||
                     maxval != 255 ||
                     (long long)width * height > 10000000)) {
                    printf("Error: Invalid dimensions or "
                           "max value. Use max value 255.\n");
                    success = 0;
                }

                if (success) {
                    int total = width * height * 3;

                    image = malloc((size_t)total *
                                   sizeof(int));
                    result = malloc((size_t)total *
                                    sizeof(int));

                    if (image == NULL || result == NULL) {
                        printf("Error: Not enough memory.\n");
                        success = 0;
                    }
                }

                if (success) {
                    int total = width * height * 3;

                    for (int i = 0; i < total; i++) {
                        if (!read_token(fp, token)) {
                            success = 0;
                            break;
                        }

                        image[i] = atoi(token);

                        if (image[i] < 0 || image[i] > 255) {
                            success = 0;
                            break;
                        }
                    }

                    if (!success)
                        printf("Error: Invalid or incomplete "
                               "PPM pixel data.\n");
                }

                fclose(fp);
            }
        }
    }

    // Tell all processes whether loading succeeded
    MPI_Bcast(&success, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (!success) {
        free(image);
        free(result);
        MPI_Finalize();
        return 1;
    }

    // Share image dimensions
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int total_pixels = width * height;

    counts = malloc((size_t)size * sizeof(int));
    displs = malloc((size_t)size * sizeof(int));

    int base = total_pixels / size;
    int remainder = total_pixels % size;
    int offset = 0;

    for (int i = 0; i < size; i++) {
        int pixels = base + (i < remainder ? 1 : 0);

        counts[i] = pixels * 3;
        displs[i] = offset * 3;
        offset += pixels;
    }

    int local_count = counts[rank];

    local_image = malloc((size_t)(local_count > 0 ?
                                  local_count : 1) *
                         sizeof(int));

    // Distribute RGB pixels among processes
    MPI_Scatterv(image, counts, displs, MPI_INT,
                 local_image, local_count, MPI_INT,
                 0, MPI_COMM_WORLD);

    // Each process converts its own pixels
    convert_to_grayscale(local_image, local_count / 3);

    // Gather processed pixels at the master process
    MPI_Gatherv(local_image, local_count, MPI_INT,
                result, counts, displs, MPI_INT,
                0, MPI_COMM_WORLD);

    double local_time = MPI_Wtime() - start;
    double total_time;

    MPI_Reduce(&local_time, &total_time, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        FILE *fp = fopen(argv[2], "w");

        if (fp == NULL) {
            printf("Error: Cannot create output image.\n");
        } else {
            // Write the output as an ASCII PPM image
            fprintf(fp, "P3\n");
            fprintf(fp, "%d %d\n255\n", width, height);

            for (int i = 0; i < total_pixels * 3; i++) {
                fprintf(fp, "%d ", result[i]);

                if ((i + 1) % 12 == 0)
                    fprintf(fp, "\n");
            }

            fprintf(fp, "\n");
            fclose(fp);

            printf("\nImage processing completed!\n");
            printf("Image size: %d x %d\n", width, height);
            printf("MPI processes: %d\n", size);
            printf("Output image: %s\n", argv[2]);
            printf("Parallel elapsed time: %.6f seconds\n",
                   total_time);
        }
    }

    free(image);
    free(result);
    free(local_image);
    free(counts);
    free(displs);

    MPI_Finalize();
    return 0;
}

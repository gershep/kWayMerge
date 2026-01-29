#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda_simple_wrapper.h"

int comp(const void *a, const void *b) {
    double c = *(double *)a;
    double d = *(double *)b;
    if (c < d) {
        return -1;
    } else if (c > d) {
        return 1;
    }

    return 0;
}

int main(int argc, char **argv) {
    FILE *fS;                   // binary file for list sizes
    FILE *fE;                   // binary file for list elements

    int k;                      // number of lists
    int n = 0;                  // number of elements
    int max_n;                  // max number of elements per list

    int     *S = NULL;          // list sizes
    double  *E = NULL;          // list elements
    double  *W = NULL;          // working buffer

    int idx = 0;                // index of the current element

    if (argc != 3) {
        fprintf(stderr, "Usage: %s k max_n\n", argv[0]);
        exit(1);
    }

    k = atoi(argv[1]);
    max_n = atoi(argv[2]);

    srand(time(0));
    host_alloc(S, int, k);

    for (int i = 0; i < k; ++i) {
        S[i] = (rand() % max_n) + 1;
        n += S[i];
    }

    host_alloc(E, double, n);
    host_alloc(W, double, max_n);

    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < S[i]; ++j) {
            W[j] = (double) rand() / RAND_MAX;
        }

        qsort(W, S[i], sizeof(double), comp);

        for (int j = 0; j < S[i]; ++j) {
            E[idx++] = W[j];
        }
    }

    open_file(fS, "sizes.dat", "wb");
    open_file(fE, "elements.dat", "wb");

    write_file(S, sizeof(int), k, fS);
    write_file(E, sizeof(double), n, fE);

    free(S);
    free(E);

    close_file(fS);
    close_file(fE);

    return 0;
}

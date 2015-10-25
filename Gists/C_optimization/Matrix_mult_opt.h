void matrix_mult_opt(double** src1, double** src2,double** dest, int SIZE, int CacheLineSize){
    int itemsPerCacheLine = CacheLineSize / sizeof(double);
    int i,j,k,jj;

    for(i = 0; i < SIZE; i++) {
        for(j = 0; j < SIZE; j += itemsPerCacheLine ) {
            for(jj = 0; jj < itemsPerCacheLine && j+jj < SIZE; jj++) {
                dest[i][j+jj] = 0;
            }
            for(k=0; k < SIZE; k++) {
                for(jj = 0; jj < itemsPerCacheLine && j+jj < SIZE; jj++) {
                    dest[i][j+jj] += src1[i][k] * src2[k][j+jj];
                }
            }
        }
    }
}

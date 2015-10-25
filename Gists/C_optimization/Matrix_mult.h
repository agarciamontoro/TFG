void matrix_mult(double** src1, double** src2, double** dest, int SIZE){
    int i,j,k;

    for(i = 0; i < SIZE; i++) {
       for(j = 0; j < SIZE; j++) {
          dest[i][j] = 0;
          for(k = 0; k < SIZE; k++) {
             dest[i][j] += src1[i][k] * src2[k][j];
          }
       }
    }
}

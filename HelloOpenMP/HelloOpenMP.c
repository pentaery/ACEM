#include <omp.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
#pragma omp parallel
  {
    int nthreads = omp_get_num_threads(), thread_id = omp_get_thread_num();
    #pragma omp masked
    {
      printf("Goodbye slow serial world and Hello OpenMP!\n");
      printf("I have %d threads and my thread id is %d\n", nthreads, thread_id);
    }
  }
}
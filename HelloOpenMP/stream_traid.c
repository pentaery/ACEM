#include <stdio.h>
#include <omp.h>
#include <time.h>
#include "timer.h"

#define NTIMES 16
#define STREAM_ARRAY_SIZE 80000000
static double a[STREAM_ARRAY_SIZE], b[STREAM_ARRAY_SIZE], c[STREAM_ARRAY_SIZE];

int main(int argc, char *argv[]) {
#pragma omp parallel
  if(omp_get_thread_num() == 0)
    printf("Running with %d threads\n", omp_get_num_threads());

  struct timespec tstart;
  double scalar = 3.0, time_sum = 0.0;
#pragma omp parallel for
  for(int i = 0; i < STREAM_ARRAY_SIZE; i++) {
    a[i] = 1.0;
    b[i] = 2.0;
  }

  for(int k =0; k < NTIMES; k++) {
    cpu_timer_start(&tstart);
    #pragma omp parallel for
    for(int i = 0; i < STREAM_ARRAY_SIZE; i++) {
      c[i] = a[i] + scalar * b[i];
    }
    time_sum += cpu_timer_stop(tstart);
    c[1] = c[2];
  }

  printf("Average runtime is %lf seconds\n", time_sum / NTIMES);
}
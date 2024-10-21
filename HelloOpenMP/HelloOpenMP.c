#include <omp.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
#pragma omp parallel
  {
    int nthreads, thread_id;
    nthreads = omp_get_num_threads();
    thread_id = omp_get_thread_num();
#pragma omp single
    {
      printf("Goodbye slow serial world and Hellp OpenMP!\n");
      printf("I have %d thread(s) and my thread id is %d\n", nthreads,
             thread_id);
    }
  }
}
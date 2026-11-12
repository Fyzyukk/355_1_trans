#include <hip/hip_runtime.h>
#include <cstdio>
int main(){int n;hipGetDeviceCount(&n);for(int i=0;i<n;i++){char b[64];hipDeviceGetPCIBusId(b,64,i);printf("%d %s\n",i,b);}}

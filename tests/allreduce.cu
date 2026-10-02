// Minimal NCCL all-reduce test, no MPI or PyTorch needed.
// Usage: ./allreduce <rank> <nranks> <rank0_mgmt_ip>   (ITERS env = timed iterations, default 10)
// Rank 0 hands the NCCL unique id to the other ranks over TCP port 29555.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <cuda_runtime.h>
#include <nccl.h>
#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA %s\n",cudaGetErrorString(e));exit(1);}}while(0)
#define NK(x) do{ncclResult_t e=(x); if(e){printf("NCCL %s @%d\n",ncclGetErrorString(e),__LINE__);exit(1);}}while(0)
static void xchg(int rank,int n,const char*ip,ncclUniqueId*id){
  int port=29555; sockaddr_in a{}; a.sin_family=AF_INET; a.sin_port=htons(port);
  if(rank==0){
    int s=socket(AF_INET,SOCK_STREAM,0),o=1; setsockopt(s,SOL_SOCKET,SO_REUSEADDR,&o,sizeof o);
    a.sin_addr.s_addr=INADDR_ANY; bind(s,(sockaddr*)&a,sizeof a); listen(s,8);
    for(int i=1;i<n;i++){int c=accept(s,0,0); write(c,id,sizeof *id); close(c);} close(s);
  } else {
    inet_pton(AF_INET,ip,&a.sin_addr);
    for(int t=0;t<120;t++){int c=socket(AF_INET,SOCK_STREAM,0);
      if(!connect(c,(sockaddr*)&a,sizeof a)){ size_t g=0; while(g<sizeof *id){ssize_t r=read(c,(char*)id+g,sizeof *id-g); if(r<=0)break; g+=r;} close(c); return;}
      close(c); sleep(1);}
    printf("no rank0\n"); exit(1);
  }
}
int main(int argc,char**argv){
  int rank=atoi(argv[1]),n=atoi(argv[2]); const char*ip=argv[3];
  int v; ncclGetVersion(&v); if(!rank) printf("NCCL runtime version code %d\n",v);
  ncclUniqueId id; if(rank==0) NK(ncclGetUniqueId(&id)); xchg(rank,n,ip,&id);
  CK(cudaSetDevice(0)); ncclComm_t comm; NK(ncclCommInitRank(&comm,n,id,rank));
  cudaStream_t st; CK(cudaStreamCreate(&st));
  size_t sizes[]={1<<20,64<<20,1024ul<<20};
  for(size_t b: sizes){
    float*x; CK(cudaMalloc(&x,b)); size_t cnt=b/4;
    float one=1.f; float*h=(float*)malloc(b); for(size_t i=0;i<cnt;i++)h[i]=one; CK(cudaMemcpy(x,h,b,cudaMemcpyHostToDevice));
    for(int i=0;i<3;i++) NK(ncclAllReduce(x,x,cnt,ncclFloat,ncclSum,comm,st)); CK(cudaStreamSynchronize(st));
    // reset to ones, then check one reduce
    CK(cudaMemcpy(x,h,b,cudaMemcpyHostToDevice));
    NK(ncclAllReduce(x,x,cnt,ncclFloat,ncclSum,comm,st)); CK(cudaStreamSynchronize(st));
    float chk; CK(cudaMemcpy(&chk,x,4,cudaMemcpyDeviceToHost));
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1); int it=getenv("ITERS")?atoi(getenv("ITERS")):10;
    cudaEventRecord(e0,st); for(int i=0;i<it;i++) NK(ncclAllReduce(x,x,cnt,ncclFloat,ncclSum,comm,st)); cudaEventRecord(e1,st); CK(cudaEventSynchronize(e1));
    float ms; cudaEventElapsedTime(&ms,e0,e1); ms/=it;
    double busbw=(double)b*8*2*(n-1)/n/(ms/1e3)/1e9;
    if(!rank) printf("%5zu MiB  %8.2f ms  busbw %7.1f Gb/s  check=%.0f (expected %d)\n",b>>20,ms,busbw,chk,n);
    cudaFree(x); free(h);
  }
  ncclCommDestroy(comm); if(!rank) printf("DONE\n"); return 0;
}

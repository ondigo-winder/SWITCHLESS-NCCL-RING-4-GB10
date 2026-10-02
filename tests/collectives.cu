// NCCL collective tests for a switchless ring. Usage: ./colls <rank> <nranks> <rank0_ip> <test>
// test: allgather | reducescatter | broadcast | reduce | sendrecv1 | sendrecv2 | alltoall | all
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <cuda_runtime.h>
#include <nccl.h>
#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA %s\n",cudaGetErrorString(e));exit(1);}}while(0)
#define NK(x) do{ncclResult_t e=(x); if(e){printf("NCCL %s @%d\n",ncclGetErrorString(e),__LINE__);exit(1);}}while(0)
static int R,N; static ncclComm_t comm; static cudaStream_t st;
static void xchg(const char*ip,ncclUniqueId*id){
  sockaddr_in a{}; a.sin_family=AF_INET; a.sin_port=htons(29556);
  if(R==0){ int s=socket(AF_INET,SOCK_STREAM,0),o=1; setsockopt(s,SOL_SOCKET,SO_REUSEADDR,&o,sizeof o);
    a.sin_addr.s_addr=INADDR_ANY; bind(s,(sockaddr*)&a,sizeof a); listen(s,8);
    for(int i=1;i<N;i++){int c=accept(s,0,0); if(write(c,id,sizeof *id)<0){} close(c);} close(s);
  } else { inet_pton(AF_INET,ip,&a.sin_addr);
    for(int t=0;t<120;t++){int c=socket(AF_INET,SOCK_STREAM,0);
      if(!connect(c,(sockaddr*)&a,sizeof a)){size_t g=0; while(g<sizeof *id){ssize_t r=read(c,(char*)id+g,sizeof *id-g); if(r<=0)break; g+=r;} close(c); return;}
      close(c); sleep(1);} printf("no rank0\n"); exit(1); }
}
static void fill(float*d,size_t cnt,float v){ float*h=(float*)malloc(cnt*4); for(size_t i=0;i<cnt;i++)h[i]=v; CK(cudaMemcpy(d,h,cnt*4,cudaMemcpyHostToDevice)); free(h);}
static float at(float*d,size_t i){float v; CK(cudaMemcpy(&v,d+i,4,cudaMemcpyDeviceToHost)); return v;}
static void report(const char*name,size_t bytes,double busfac,float ms,bool ok,const char*detail){
  double bw=(double)bytes*8*busfac/(ms/1e3)/1e9;
  printf("RESULT rank%d %-14s %6zu MiB %8.2f ms busbw %7.1f Gb/s %s %s\n",R,name,bytes>>20,ms,bw,ok?"OK":"FAIL",detail);
}
template<class F> float timeit(F f){ for(int i=0;i<2;i++)f(); CK(cudaStreamSynchronize(st));
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); int it=10; cudaEventRecord(a,st);
  for(int i=0;i<it;i++)f(); cudaEventRecord(b,st); CK(cudaEventSynchronize(b)); float ms; cudaEventElapsedTime(&ms,a,b); return ms/it; }
int main(int argc,char**argv){
  setvbuf(stdout,NULL,_IONBF,0);
  R=atoi(argv[1]); N=atoi(argv[2]); std::string t=argv[4];
  ncclUniqueId id; if(R==0) NK(ncclGetUniqueId(&id)); xchg(argv[3],&id);
  CK(cudaSetDevice(0)); NK(ncclCommInitRank(&comm,N,id,R)); CK(cudaStreamCreate(&st));
  const size_t TOT=1024ul<<20, cnt=TOT/4, per=cnt/N; float *x,*y; CK(cudaMalloc(&x,TOT)); CK(cudaMalloc(&y,TOT));
  char d[160]; bool all=(t=="all");
  if(all||t=="allgather"){ fill(x,per,R+1); float ms=timeit([&]{NK(ncclAllGather(x,y,per,ncclFloat,comm,st));});
    bool ok=true; for(int k=0;k<N;k++){ ok&=at(y,k*per)==k+1 && at(y,k*per+per-1)==k+1; }
    snprintf(d,sizeof d,"(block k = k+1)"); report("all_gather",TOT,(double)(N-1)/N,ms,ok,d); }
  if(all||t=="reducescatter"){ fill(x,cnt,R+1); fill(y,per,0);
    float ms=timeit([&]{NK(ncclReduceScatter(x,y,per,ncclFloat,ncclSum,comm,st));});
    float e=N*(N+1)/2.f; bool ok=at(y,0)==e && at(y,per-1)==e; snprintf(d,sizeof d,"(%g, expected %g)",at(y,0),e);
    report("reduce_scatter",TOT,(double)(N-1)/N,ms,ok,d); }
  for(int root: {0,2}) if(all||t=="broadcast"){ fill(x,cnt,R==root?42:0);
    float ms=timeit([&]{NK(ncclBroadcast(x,x,cnt,ncclFloat,root,comm,st));});
    bool ok=at(x,0)==42 && at(x,cnt-1)==42; snprintf(d,sizeof d,"(root %d: %g, expected 42)",root,at(x,cnt-1));
    report("broadcast",TOT,1.0,ms,ok,d); }
  for(int root: {0,2}) if(all||t=="reduce"){ fill(x,cnt,R+1); fill(y,cnt,0);
    float ms=timeit([&]{NK(ncclReduce(x,y,cnt,ncclFloat,ncclSum,root,comm,st));});
    fill(y,cnt,0); NK(ncclReduce(x,y,cnt,ncclFloat,ncclSum,root,comm,st)); CK(cudaStreamSynchronize(st));
    float e=N*(N+1)/2.f; bool ok = R!=root || (at(y,0)==e && at(y,cnt-1)==e);
    snprintf(d,sizeof d,"(root %d%s)",root,R==root?"":", not the root"); report("reduce",TOT,1.0,ms,ok,d); }
  for(int k: {1,2}) if((all&&k==1)||t==("sendrecv"+std::to_string(k))){
    int to=(R+k)%N, from=(R-k+N)%N; fill(x,cnt,R+1); fill(y,cnt,0);
    float ms=timeit([&]{ NK(ncclGroupStart()); NK(ncclSend(x,cnt,ncclFloat,to,comm,st)); NK(ncclRecv(y,cnt,ncclFloat,from,comm,st)); NK(ncclGroupEnd()); });
    bool ok=at(y,0)==from+1 && at(y,cnt-1)==from+1; snprintf(d,sizeof d,"(to %d, from %d: %g, expected %d)",to,from,at(y,0),from+1);
    report(k==1?"sendrecv_next":"sendrecv_+2",TOT,1.0,ms,ok,d); }
  if(t=="alltoall"){ fill(x,cnt,R+1); fill(y,cnt,0);
    float ms=timeit([&]{ NK(ncclGroupStart()); for(int p=0;p<N;p++){ NK(ncclSend(x+p*per,per,ncclFloat,p,comm,st)); NK(ncclRecv(y+p*per,per,ncclFloat,p,comm,st)); } NK(ncclGroupEnd()); });
    bool ok=true; for(int p=0;p<N;p++) ok&=at(y,p*per)==p+1; snprintf(d,sizeof d,"(block p = p+1)"); report("all_to_all",TOT,(double)(N-1)/N,ms,ok,d); }
  ncclCommDestroy(comm); printf("DONE rank%d\n",R); return 0;
}

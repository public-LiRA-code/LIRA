#define _GNU_SOURCE
#include <dlfcn.h>
#include <mpi.h>
#include <string.h>
#include <time.h>

#define MAXR 4096
#define MAXREQ 16384
static int far_of[MAXR];
static double t_far, t_near, t_barrier, t_coll, t_test, b_far, b_near, n_data, n_wait, n_barrier, n_coll, n_test;
static struct { MPI_Request req; int peer; int used; } tab[MAXREQ];
static int tab_hi;

static inline double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + 1e-9 * ts.tv_nsec; }
static inline int isfar(int peer) { return peer >= 0 && peer < MAXR && far_of[peer]; }
static void put(MPI_Request r, int peer) {
    for (int i = 0; i < MAXREQ; i++) if (!tab[i].used) { tab[i].req = r; tab[i].peer = peer; tab[i].used = 1; if (i >= tab_hi) tab_hi = i + 1; return; }
}
static int find(MPI_Request r, int take) {
    for (int i = 0; i < tab_hi; i++) if (tab[i].used && tab[i].req == r) { if (take) tab[i].used = 0; return tab[i].peer; }
    return -1;
}
static inline void add(int peer, double dt, double bytes) { if (isfar(peer)) { t_far += dt; b_far += bytes; } else { t_near += dt; b_near += bytes; } }
static double nbytes(int count, MPI_Datatype t) { int s = 0; PMPI_Type_size(t, &s); return (double)count * s; }

static double t_swap_far, t_swap_near, n_swap_far, n_swap_near;
static int cur_peer = -1;
static void *real_sym(const char *name) {
    void *f = dlsym(RTLD_NEXT, name);
    if (!f) { void *h = dlopen("libcustatevec.so.1", RTLD_LAZY | RTLD_NOLOAD); if (h) f = dlsym(h, name); }
    return f;
}
int custatevecSVSwapWorkerSetParameters(void *handle, void *worker, const void *params, int peer) {
    static int (*real)(void *, void *, const void *, int);
    if (!real) real = (int (*)(void *, void *, const void *, int))real_sym("custatevecSVSwapWorkerSetParameters");
    cur_peer = peer;
    return real(handle, worker, params, peer);
}
int custatevecSVSwapWorkerExecute(void *handle, void *worker, long long begin, long long end) {
    static int (*real)(void *, void *, long long, long long);
    static int (*sync)(void);
    if (!real) {
        real = (int (*)(void *, void *, long long, long long))real_sym("custatevecSVSwapWorkerExecute");
        sync = (int (*)(void))dlsym(RTLD_DEFAULT, "cudaDeviceSynchronize");
    }
    double t = now();
    int r = real(handle, worker, begin, end);
    if (sync) sync();
    double dt = now() - t;
    if (isfar(cur_peer)) { t_swap_far += dt; n_swap_far++; } else { t_swap_near += dt; n_swap_near++; }
    return r;
}

void mpiprof_set_far(const int *f, int n) { memset(far_of, 0, sizeof far_of); for (int i = 0; i < n && i < MAXR; i++) far_of[i] = f[i]; }
void mpiprof_reset(void) {
    t_far = t_near = t_barrier = t_coll = t_test = b_far = b_near = n_data = n_wait = n_barrier = n_coll = n_test = 0.0;
    t_swap_far = t_swap_near = n_swap_far = n_swap_near = 0.0;
}
void mpiprof_get(double *o) {
    o[0] = t_far; o[1] = t_near; o[2] = t_barrier; o[3] = t_coll; o[4] = b_far; o[5] = b_near;
    o[6] = n_data; o[7] = n_wait; o[8] = n_barrier; o[9] = n_coll; o[10] = n_test; o[11] = t_test;
    o[12] = t_swap_far; o[13] = t_swap_near; o[14] = n_swap_far; o[15] = n_swap_near;
}

int MPI_Send(const void *buf, int count, MPI_Datatype dt, int dest, int tag, MPI_Comm comm) {
    double t = now(); int r = PMPI_Send(buf, count, dt, dest, tag, comm); add(dest, now() - t, nbytes(count, dt)); n_data++; return r;
}
int MPI_Recv(void *buf, int count, MPI_Datatype dt, int src, int tag, MPI_Comm comm, MPI_Status *st) {
    double t = now(); int r = PMPI_Recv(buf, count, dt, src, tag, comm, st); add(src, now() - t, 0.0); n_data++; return r;
}
int MPI_Sendrecv(const void *sb, int sc, MPI_Datatype st, int dest, int stag, void *rb, int rc, MPI_Datatype rt, int src, int rtag,
                 MPI_Comm comm, MPI_Status *status) {
    double t = now(); int r = PMPI_Sendrecv(sb, sc, st, dest, stag, rb, rc, rt, src, rtag, comm, status);
    add(dest, now() - t, nbytes(sc, st)); n_data++; return r;
}
int MPI_Send_c(const void *buf, MPI_Count count, MPI_Datatype dt, int dest, int tag, MPI_Comm comm) {
    int s = 0; PMPI_Type_size(dt, &s);
    double t = now(); int r = PMPI_Send_c(buf, count, dt, dest, tag, comm); add(dest, now() - t, (double)count * s); n_data++; return r;
}
int MPI_Recv_c(void *buf, MPI_Count count, MPI_Datatype dt, int src, int tag, MPI_Comm comm, MPI_Status *st) {
    double t = now(); int r = PMPI_Recv_c(buf, count, dt, src, tag, comm, st); add(src, now() - t, 0.0); n_data++; return r;
}
int MPI_Sendrecv_c(const void *sb, MPI_Count sc, MPI_Datatype st, int dest, int stag, void *rb, MPI_Count rc, MPI_Datatype rt, int src, int rtag,
                   MPI_Comm comm, MPI_Status *status) {
    int s = 0; PMPI_Type_size(st, &s);
    double t = now(); int r = PMPI_Sendrecv_c(sb, sc, st, dest, stag, rb, rc, rt, src, rtag, comm, status);
    add(dest, now() - t, (double)sc * s); n_data++; return r;
}
int MPI_Isend(const void *buf, int count, MPI_Datatype dt, int dest, int tag, MPI_Comm comm, MPI_Request *req) {
    double t = now(); int r = PMPI_Isend(buf, count, dt, dest, tag, comm, req); add(dest, now() - t, nbytes(count, dt)); put(*req, dest); n_data++; return r;
}
int MPI_Irecv(void *buf, int count, MPI_Datatype dt, int src, int tag, MPI_Comm comm, MPI_Request *req) {
    double t = now(); int r = PMPI_Irecv(buf, count, dt, src, tag, comm, req); add(src, now() - t, 0.0); put(*req, src); n_data++; return r;
}
int MPI_Wait(MPI_Request *req, MPI_Status *st) {
    int peer = find(*req, 1); double t = now(); int r = PMPI_Wait(req, st); add(peer, now() - t, 0.0); n_wait++; return r;
}
int MPI_Waitall(int count, MPI_Request reqs[], MPI_Status sts[]) {
    int anyfar = 0; for (int i = 0; i < count; i++) if (isfar(find(reqs[i], 1))) anyfar = 1;
    double t = now(); int r = PMPI_Waitall(count, reqs, sts); double dt = now() - t;
    if (anyfar) t_far += dt; else t_near += dt; n_wait++; return r;
}
int MPI_Test(MPI_Request *req, int *flag, MPI_Status *st) {
    MPI_Request r0 = *req; double t = now(); int r = PMPI_Test(req, flag, st); double dt = now() - t;
    add(find(r0, *flag), dt, 0.0); t_test += dt; n_test++; return r;
}
int MPI_Testall(int count, MPI_Request reqs[], int *flag, MPI_Status sts[]) {
    int anyfar = 0; MPI_Request keep[64]; int k = count < 64 ? count : 64;
    for (int i = 0; i < k; i++) { keep[i] = reqs[i]; if (isfar(find(reqs[i], 0))) anyfar = 1; }
    double t = now(); int r = PMPI_Testall(count, reqs, flag, sts); double dt = now() - t;
    if (*flag) for (int i = 0; i < k; i++) find(keep[i], 1);
    if (anyfar) t_far += dt; else t_near += dt; t_test += dt; n_test++; return r;
}
int MPI_Barrier(MPI_Comm comm) { double t = now(); int r = PMPI_Barrier(comm); t_barrier += now() - t; n_barrier++; return r; }
int MPI_Allreduce(const void *sb, void *rb, int count, MPI_Datatype dt, MPI_Op op, MPI_Comm comm) {
    double t = now(); int r = PMPI_Allreduce(sb, rb, count, dt, op, comm); t_coll += now() - t; n_coll++; return r;
}
int MPI_Reduce(const void *sb, void *rb, int count, MPI_Datatype dt, MPI_Op op, int root, MPI_Comm comm) {
    double t = now(); int r = PMPI_Reduce(sb, rb, count, dt, op, root, comm); t_coll += now() - t; n_coll++; return r;
}
int MPI_Bcast(void *buf, int count, MPI_Datatype dt, int root, MPI_Comm comm) {
    double t = now(); int r = PMPI_Bcast(buf, count, dt, root, comm); t_coll += now() - t; n_coll++; return r;
}
int MPI_Allgather(const void *sb, int sc, MPI_Datatype st, void *rb, int rc, MPI_Datatype rt, MPI_Comm comm) {
    double t = now(); int r = PMPI_Allgather(sb, sc, st, rb, rc, rt, comm); t_coll += now() - t; n_coll++; return r;
}
int MPI_Gather(const void *sb, int sc, MPI_Datatype st, void *rb, int rc, MPI_Datatype rt, int root, MPI_Comm comm) {
    double t = now(); int r = PMPI_Gather(sb, sc, st, rb, rc, rt, root, comm); t_coll += now() - t; n_coll++; return r;
}

typedef struct {
    int nodes[3];
    double matriz[6]; // C11 C22 C33 C12 C13 C23
} elementri;

typedef struct {
    float x;
    float y;
    int i;
    bool calc;
    int ne;
    int elements[10];
} node;

// HEADER
__global__ void kernel_pre();
__global__ void kernel_iter();
extern "C" void teste_Arrays(int, int, elementri *, node *);

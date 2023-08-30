typedef struct {
    int *data;
    int size;
} Tensor;

Tensor* create_tensor(int *data, int size);
void free_tensor(Tensor *tensor);
int get_tensor_element(Tensor *tensor, int index);

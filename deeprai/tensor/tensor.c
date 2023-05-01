#include <stdlib.h>

// Function to create a tensor with the given shape
float* create_tensor(int shape[], int num_dims) {
    int num_elements = 1;
    for (int i = 0; i < num_dims; i++) {
        num_elements *= shape[i];
    }
    float* data = (float*) malloc(num_elements * sizeof(float));
    return data;
}

// Function to get the value of the tensor at the given index
float get_value(float* data, int shape[], int num_dims, int indices[]) {
    int index = 0;
    for (int i = 0; i < num_dims; i++) {
        index = index * shape[i] + indices[i];
    }
    return data[index];
}

// Function to set the value of the tensor at the given index
void set_value(float* data, int shape[], int num_dims, int indices[], float value) {
    int index = 0;
    for (int i = 0; i < num_dims; i++) {
        index = index * shape[i] + indices[i];
    }
    data[index] = value;
}

// Function to add two tensors element-wise
void add_tensors(float* a_data, float* b_data, float* c_data, int shape[], int num_dims) {
    int num_elements = 1;
    for (int i = 0; i < num_dims; i++) {
        num_elements *= shape[i];
    }
    for (int i = 0; i < num_elements; i++) {
        c_data[i] = a_data[i] + b_data[i];
    }
}

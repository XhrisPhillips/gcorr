#pragma once

#include <cstddef>
#include <vector>

#include <cuda_runtime.h>

template <typename T>
class gpu_array {
    T* data_ = nullptr;
    std::size_t n_ = 0;

    void gpu_move(gpu_array&& other) {
	if (data_) cudaFree(data_);
	data_ = other.data_;
	n_ = other.n_;
	other.data_ = nullptr;
	other.n_ = 0;
    }

    void copy_from_host(const T* addr, std::size_t c) {
	if (data_) cudaFree(data_);
	alloc(c);
	cudaMemcpy(data_, addr, c*sizeof(T), cudaMemcpyHostToDevice);
	n_ = c;
    }

    void alloc(std::size_t c) {
	data_ = nullptr;
	if (c>0) {
	    cudaMalloc(&data_, c*sizeof(T));
	    if (cudaGetLastError()!=cudaSuccess) throw std::bad_alloc();
	}
	n_ = c;
    }

public:
    explicit gpu_array(std::size_t n) {
	alloc(n);
    }

    gpu_array(const gpu_array&) = delete;

    gpu_array(gpu_array&& other) {
	gpu_move(std::move(other));
    }

    gpu_array(const std::vector<T>& vec) {
	copy_from_host(vec.data(), vec.size());
    }

    gpu_array& operator=(const gpu_array&) = delete;
    gpu_array& operator=(gpu_array&& other) {
	gpu_move(std::move(other));
	return *this;
    }

    gpu_array& operator=(const std::vector<T>& other) {
	copy_from_host(other.data(), other.size());
	return *this;
    }

    ~gpu_array() {
	if (data_) cudaFree(data_);
    }

    T* data() const { return data_; }
    std::size_t size() const { return n_; }

    operator std::vector<T>() const& {
	std::vector<T> v(n_);
	if (n_>0) {
	    cudaMemcpy(v.data(), data_, n_*sizeof(T), cudaMemcpyDeviceToHost);
	}
	return v;
    }

    operator std::vector<T>() && {
	std::vector<T> v(n_);
	if (n_>0) {
	    cudaMemcpy(v.data(), data_, n_*sizeof(T), cudaMemcpyDeviceToHost);
	    cudaFree(data_);
	    n_ = 0;
	    data_ = nullptr;
	}
	return v;
    }
};


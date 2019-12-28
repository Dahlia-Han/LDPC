#pragma once

#include <stdint.h>

struct CheckMatrix;
struct CheckMatrix_GPU;

extern "C" _declspec(dllexport)
bool LoadCheckMatrix(const char* filename, CheckMatrix & matrix);

extern "C" _declspec(dllexport)
bool LoadCheckMatrix_GPU(const char* filename, CheckMatrix_GPU & matrix);

/*
校验矩阵
*/
struct CheckMatrix {
	uint32_t Row, Col; // 校验矩阵的行数，列数
	uint32_t NonZerosNum; // 校验矩阵的非零元素数
	uint32_t* H_IdxRow = nullptr, * H_IdxCol = nullptr; // 校验矩阵非零元素的所在行和列

	// 变量节点与校验节点之间关系
	uint32_t* IdxConnectQij = nullptr, * IdxConnectRji = nullptr; 
	uint32_t* IdxConnectQij_idx = nullptr, * IdxConnectRji_idx = nullptr;
};

struct CheckMatrix_GPU {
	uint32_t Row, Col; // 校验矩阵的行数，列数
	uint32_t NonZerosNum; // 校验矩阵的非零元素数
	uint32_t* H_IdxRow = nullptr, * H_IdxCol = nullptr; // 校验矩阵非零元素的所在行和列
	uint32_t* H_IdxCol_d = nullptr; // 校验矩阵非零元素的所在行和列

	// 变量节点与校验节点之间关系
	uint32_t* IdxConnectQij = nullptr, * IdxConnectRji = nullptr;
	uint32_t* IdxConnectQij_d = nullptr, * IdxConnectRji_d = nullptr;
	uint32_t* IdxConnectQij_idx = nullptr, * IdxConnectRji_idx = nullptr;
	uint32_t* IdxConnectQij_idx_d = nullptr, * IdxConnectRji_idx_d = nullptr;

	uint64_t GPU_Memory = 0;
};
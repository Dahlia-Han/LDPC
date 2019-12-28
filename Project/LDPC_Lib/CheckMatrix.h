#pragma once

#include <stdint.h>

struct CheckMatrix;

extern "C" _declspec(dllexport)
bool LoadCheckMatrix(const char* filename, CheckMatrix & matrix);

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
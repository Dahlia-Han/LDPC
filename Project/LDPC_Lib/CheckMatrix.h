#pragma once

#include <stdint.h>

struct CheckMatrix;
struct CheckMatrix_GPU;

extern "C" _declspec(dllexport)
bool LoadCheckMatrix(const char* filename, CheckMatrix & matrix);

extern "C" _declspec(dllexport)
bool LoadCheckMatrix_GPU(const char* filename, CheckMatrix_GPU & matrix);

/*
У�����
*/
struct CheckMatrix {
	uint32_t Row, Col; // У����������������
	uint32_t NonZerosNum; // У�����ķ���Ԫ����
	uint32_t* H_IdxRow = nullptr, * H_IdxCol = nullptr; // У��������Ԫ�ص������к���

	// �����ڵ���У��ڵ�֮���ϵ
	uint32_t* IdxConnectQij = nullptr, * IdxConnectRji = nullptr; 
	uint32_t* IdxConnectQij_idx = nullptr, * IdxConnectRji_idx = nullptr;
};

struct CheckMatrix_GPU {
	uint32_t Row, Col; // У����������������
	uint32_t NonZerosNum; // У�����ķ���Ԫ����
	uint32_t* H_IdxRow = nullptr, * H_IdxCol = nullptr; // У��������Ԫ�ص������к���
	uint32_t* H_IdxCol_d = nullptr; // У��������Ԫ�ص������к���

	// �����ڵ���У��ڵ�֮���ϵ
	uint32_t* IdxConnectQij = nullptr, * IdxConnectRji = nullptr;
	uint32_t* IdxConnectQij_d = nullptr, * IdxConnectRji_d = nullptr;
	uint32_t* IdxConnectQij_idx = nullptr, * IdxConnectRji_idx = nullptr;
	uint32_t* IdxConnectQij_idx_d = nullptr, * IdxConnectRji_idx_d = nullptr;

	uint64_t GPU_Memory = 0;
};
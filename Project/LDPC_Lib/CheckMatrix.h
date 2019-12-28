#pragma once

#include <stdint.h>

struct CheckMatrix;

extern "C" _declspec(dllexport)
bool LoadCheckMatrix(const char* filename, CheckMatrix & matrix);

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
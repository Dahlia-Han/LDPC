#include "CheckMatrix.h"
#include <io.h>
#include <fstream>

bool LoadCheckMatrix(const char* filename, CheckMatrix& matrix)
{
	// 检测文件是否存在
	if (_access(filename, 0) != 0)
		return false;

	std::ifstream in(filename);

	in >> matrix.Row >> matrix.Col >> matrix.NonZerosNum;
	matrix.H_IdxRow = new uint32_t[matrix.NonZerosNum];
	matrix.H_IdxCol = new uint32_t[matrix.NonZerosNum];
	uint32_t H_Idx;
	for (uint32_t i = 0; i < matrix.NonZerosNum; i++)
	{
		in >> H_Idx;
		matrix.H_IdxRow[i] = H_Idx % matrix.Row;
		matrix.H_IdxCol[i] = H_Idx / matrix.Row;
	}
	in.close();

	// 生成边连接关系
	uint32_t sumidxq = 0, sumidxr = 0;
	for (uint32_t i = 0; i < matrix.NonZerosNum; i++)
	{
		for (uint32_t j = 0; j < matrix.NonZerosNum; j++)
			if (matrix.H_IdxRow[j] == matrix.H_IdxRow[i] && matrix.H_IdxCol[j] != matrix.H_IdxCol[i])
				sumidxq++;
		for (uint32_t j = 0; j < matrix.NonZerosNum; j++)
			if (matrix.H_IdxCol[j] == matrix.H_IdxCol[i] && matrix.H_IdxRow[j] != matrix.H_IdxRow[i])
				sumidxr++;
	}
	matrix.IdxConnectQij = new uint32_t[sumidxq];
	matrix.IdxConnectRji = new uint32_t[sumidxr];
	matrix.IdxConnectQij_idx = new uint32_t[(__int64)matrix.NonZerosNum * 2];
	matrix.IdxConnectRji_idx = new uint32_t[(__int64)matrix.NonZerosNum * 2];
	sumidxq = 0;
	sumidxr = 0;
	for (uint32_t i = 0; i < matrix.NonZerosNum; i++)
	{
		matrix.IdxConnectQij_idx[2 * i] = sumidxq;
		for (uint32_t j = 0; j < matrix.NonZerosNum; j++)
			if (matrix.H_IdxRow[j] == matrix.H_IdxRow[i] && matrix.H_IdxCol[j] != matrix.H_IdxCol[i])
				matrix.IdxConnectQij[sumidxq++] = j;
		matrix.IdxConnectQij_idx[2 * i + 1] = sumidxq;

		matrix.IdxConnectRji_idx[2 * i] = sumidxr;
		for (uint32_t j = 0; j < matrix.NonZerosNum; j++)
			if (matrix.H_IdxCol[j] == matrix.H_IdxCol[i] && matrix.H_IdxRow[j] != matrix.H_IdxRow[i])
				matrix.IdxConnectRji[sumidxr++] = j;
		matrix.IdxConnectRji_idx[2 * i + 1] = sumidxr;
	}

	return true;
}

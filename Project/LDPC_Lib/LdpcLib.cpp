#include "LdpcLib.h"
#include <random>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define PI 3.141592654

// 生成正态分布随机数
double gaussrand()
{
	static double U, V;
	static int phase = 0;
	double Z;

	if (phase == 0)
	{
		U = (rand() + 1.0) / ((double)RAND_MAX + 1.0);
		V = (rand() + 1.0) / ((double)RAND_MAX + 1.0);
		Z = sqrt(-2.0 * log(U)) * sin(2.0 * PI * V);
	}
	else
	{
		Z = sqrt(-2.0 * log(U)) * cos(2.0 * PI * V);
	}

	phase = 1 - phase;
	return Z;
}

void AddAWGN(const uint32_t CodeBitLength, const float R, const float sigma, const uint8_t* Code, float* Signal)
{
	for (uint32_t i = 0; i < CodeBitLength; i++)
		Signal[i] = (Code[i] * 2.0 - 1) + gaussrand() * sigma;
}

void PunchBits(const uint32_t Frames, const uint32_t CodeBitLength, const uint32_t BitsPunchedPerFrame, float* Signal)
{
	for (uint32_t i = 0; i < Frames; i++)
		for (uint32_t j = 0; j < BitsPunchedPerFrame; j++)
			Signal[i * CodeBitLength + j] = 0;
}

void LdpcDecode_BP(const CheckMatrix& matrix, const float sigma, const uint32_t MaxItr, const bool isExitBeforeMaxItr,
	const float* Signal, uint8_t* Decode)
{
	// 变量声明: 先验信息ci，变量节点到校验节点信息qij，校验节点到变量节点信息rji
	float* ci0 = nullptr, * qij0 = nullptr, * rji0 = nullptr;
	float* ci1 = nullptr, * qij1 = nullptr, * rji1 = nullptr;
	ci0 = new float[matrix.Col];
	ci1 = new float[matrix.Col];
	qij0 = new float[matrix.NonZerosNum];
	qij1 = new float[matrix.NonZerosNum];
	rji0 = new float[matrix.NonZerosNum];
	rji1 = new float[matrix.NonZerosNum];
	uint8_t* decode = new uint8_t[matrix.Col];
	uint32_t* check = new uint32_t[matrix.Row];
	memset(ci0, 0, matrix.Col * sizeof(float));
	memset(ci1, 0, matrix.Col * sizeof(float));
	memset(qij0, 0, matrix.NonZerosNum * sizeof(float));
	memset(qij1, 0, matrix.NonZerosNum * sizeof(float));
	memset(rji0, 0, matrix.NonZerosNum * sizeof(float));
	memset(rji1, 0, matrix.NonZerosNum * sizeof(float));

	// 初始化 ci/qij
	for (uint32_t i = 0; i < matrix.Col; i++)
	{
		ci0[i] = 1 / (1 + exp(2 * Signal[i] / (sigma * sigma)));
		ci1[i] = 1 / (1 + exp(-2 * Signal[i] / (sigma * sigma)));
	}
	for (uint32_t i = 0; i < matrix.NonZerosNum; i++)
	{
		qij0[i] = ci0[matrix.H_IdxCol[i]];
		qij1[i] = ci1[matrix.H_IdxCol[i]];
	}

	// 迭代过程
	for (uint32_t itr = 0; itr < MaxItr; itr++)
	{
		// 更新rji
		for (uint32_t i = 0; i < matrix.NonZerosNum; i++)
		{
			float s = 1;
			for (uint32_t j = matrix.IdxConnectQij_idx[i * 2]; j < matrix.IdxConnectQij_idx[2 * i + 1]; j++)
				s = s * (1 - 2 * qij1[matrix.IdxConnectQij[j]]);
			rji0[i] = 0.5 * (1.0 + s);
			rji1[i] = 1 - rji0[i];
		}

		// 更新qij
		for (uint32_t i = 0; i < matrix.NonZerosNum; i++)
		{
			float s0 = 1;
			float s1 = 1;
			for (uint32_t j = matrix.IdxConnectRji_idx[i * 2]; j < matrix.IdxConnectRji_idx[i * 2 + 1]; j++)
			{
				s0 *= rji0[matrix.IdxConnectRji[j]];
				s1 *= rji1[matrix.IdxConnectRji[j]];
			}
			qij0[i] = ci0[matrix.H_IdxCol[i]] * s0;
			qij1[i] = ci1[matrix.H_IdxCol[i]] * s1;
			qij0[i] = qij0[i] / (qij0[i] + qij1[i]);
			qij1[i] = 1 - qij0[i];
		}

		// 提前退出
		for (uint32_t i = 0; i < matrix.Col; i++)
		{
			float s0 = 1;
			float s1 = 1;
			for (uint32_t j = 0; j < matrix.NonZerosNum; j++)
			{
				if (matrix.H_IdxCol[j] == i)
				{
					s0 *= rji0[j];
					s1 *= rji1[j];
				}
			}
			s0 *= ci0[i];
			s1 *= ci1[i];
			if (s0 < s1)
				decode[i] = 1;
			else
				decode[i] = 0;
		}
		bool isExist = true;
		memset(check, 0, matrix.Row * sizeof(uint32_t));
		for (uint32_t i = 0; i < matrix.NonZerosNum; i++)
			check[matrix.H_IdxRow[i]] += decode[matrix.H_IdxCol[i]];
		for (uint32_t i = 0; i < matrix.Row; i++)
			if (check[i] % 2 != 0)
				isExist = false;
		if (isExist)
			break;
	}

	// 判决
	for (uint32_t i = 0; i < matrix.Col; i++)
	{
		float s0 = 1;
		float s1 = 1;
		for (uint32_t j = 0; j < matrix.NonZerosNum; j++)
		{
			if (matrix.H_IdxCol[j] == i)
			{
				s0 *= rji0[j];
				s1 *= rji1[j];
			}
		}
		s0 *= ci0[i];
		s1 *= ci1[i];
		if (s0 < s1)
			Decode[i] = 1;
		else
			Decode[i] = 0;
	}

	delete[]decode;
	delete[]check;
	delete[]ci0;
	delete[]ci1;
	delete[]qij0;
	delete[]qij1;
	delete[]rji0;
	delete[]rji1;
}

void LdpcDecode_LogBP(const CheckMatrix& matrix, const float sigma, const uint32_t MaxItr, const bool isExitBeforeMaxItr, const float* Signal, uint8_t* Decode)
{
	// 变量声明
	float* ci = nullptr, * qij = nullptr, * rji = nullptr;
	ci = new float[matrix.Col];
	qij = new float[matrix.NonZerosNum];
	rji = new float[matrix.NonZerosNum];
	uint8_t* decode = new uint8_t[matrix.Col];
	uint32_t* check = new uint32_t[matrix.Row];
	memset(ci, 0, matrix.Col * sizeof(float));
	memset(qij, 0, matrix.NonZerosNum * sizeof(float));
	memset(rji, 0, matrix.NonZerosNum * sizeof(float));

	// 初始化 ci/qij
	for (uint32_t i = 0; i < matrix.Col; i++)
		ci[i] = -2.0 * Signal[i] / (sigma * sigma);
	for (uint32_t i = 0; i < matrix.NonZerosNum; i++)
		qij[i] = ci[matrix.H_IdxCol[i]];

	// 迭代过程
	for (uint32_t itr = 0; itr < MaxItr; itr++)
	{
		// 更新rji
		for (uint32_t i = 0; i < matrix.NonZerosNum; i++)
		{
			float s = 1;
			for (uint32_t j = matrix.IdxConnectQij_idx[i * 2]; j < matrix.IdxConnectQij_idx[2 * i + 1]; j++)
				s = s * tanh(qij[matrix.IdxConnectQij[j]] / 2);
			rji[i] = 2 * atanh(s);
		}

		// 更新qij
		for (uint32_t i = 0; i < matrix.NonZerosNum; i++)
		{
			float s = 0;
			for (uint32_t j = matrix.IdxConnectRji_idx[i * 2]; j < matrix.IdxConnectRji_idx[i * 2 + 1]; j++)
				s += rji[matrix.IdxConnectRji[j]];
			qij[i] = ci[matrix.H_IdxCol[i]] + s;
			if (qij[i] > 1e12)
				qij[i] = 1e12;
			if (qij[i] < -1e12)
				qij[i] = -1e12;
		}

		// 提前退出
		for (uint32_t i = 0; i < matrix.Col; i++)
		{
			float s = 0;
			for (uint32_t j = 0; j < matrix.NonZerosNum; j++)
				if (matrix.H_IdxCol[j] == i)
					s += rji[j];
			s += ci[i];
			if (s < 0)
				decode[i] = 1;
			else
				decode[i] = 0;
		}
		bool isExist = true;
		memset(check, 0, matrix.Row * sizeof(uint32_t));
		for (uint32_t i = 0; i < matrix.NonZerosNum; i++)
			check[matrix.H_IdxRow[i]] += decode[matrix.H_IdxCol[i]];
		for (uint32_t i = 0; i < matrix.Row; i++)
			if (check[i] % 2 != 0)
				isExist = false;
		if (isExist)
			break;
	}

	// 判决
	for (uint32_t i = 0; i < matrix.Col; i++)
	{
		float s = 0;
		for (uint32_t j = 0; j < matrix.NonZerosNum; j++)
			if (matrix.H_IdxCol[j] == i)
				s += rji[j];
		s += ci[i];
		if (s < 0)
			Decode[i] = 1;
		else
			Decode[i] = 0;
	}

	delete[]decode;
	delete[]check;
	delete[]ci;
	delete[]qij;
	delete[]rji;
}

void LdpcDecode_GPU_BP(const CheckMatrix_GPU& matrix, const uint32_t FrameLength, const float sigma, const uint32_t MaxItr, const float* Signal, uint8_t* Decode)
{
	// 变量声明
	uint32_t Blocks = 0;
	uint32_t ThreadsPerBlock = 256;
	float* Signal_d = nullptr, * ci0_d = nullptr, * ci1_d = nullptr, * qij0_d = nullptr, * qij1_d = nullptr, * rji0_d = nullptr, * rji1_d = nullptr;
	uint8_t* Decode_d = nullptr;

	// 变量初始化
	cudaMalloc(&Signal_d, FrameLength * (__int64)matrix.Col * sizeof(float));
	cudaMalloc(&ci0_d, FrameLength * (__int64)matrix.Col * sizeof(float));
	cudaMalloc(&ci1_d, FrameLength * (__int64)matrix.Col * sizeof(float));
	cudaMalloc(&qij0_d, FrameLength * (__int64)matrix.NonZerosNum * sizeof(float));
	cudaMalloc(&qij1_d, FrameLength * (__int64)matrix.NonZerosNum * sizeof(float));
	cudaMalloc(&rji0_d, FrameLength * (__int64)matrix.NonZerosNum * sizeof(float));
	cudaMalloc(&rji1_d, FrameLength * (__int64)matrix.NonZerosNum * sizeof(float));
	cudaMalloc(&Decode_d, FrameLength * (__int64)matrix.Col * sizeof(uint8_t));

	// 变量赋值
	cudaMemcpy(Signal_d, Signal, FrameLength * (__int64)matrix.Col * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemset(ci0_d, 0, FrameLength * (__int64)matrix.Col * sizeof(float));
	cudaMemset(ci1_d, 0, FrameLength * (__int64)matrix.Col * sizeof(float));
	cudaMemset(qij0_d, 0, FrameLength * (__int64)matrix.NonZerosNum * sizeof(float));
	cudaMemset(qij1_d, 0, FrameLength * (__int64)matrix.NonZerosNum * sizeof(float));
	cudaMemset(rji0_d, 0, FrameLength * (__int64)matrix.NonZerosNum * sizeof(float));
	cudaMemset(rji1_d, 0, FrameLength * (__int64)matrix.NonZerosNum * sizeof(float));
	cudaMemset(Decode_d, 0, FrameLength * (__int64)matrix.Col * sizeof(uint8_t));

	Blocks = ceil(FrameLength * (__int64)matrix.Col / (double)ThreadsPerBlock);
	LdpcDecodeBP_CiInit << <Blocks, ThreadsPerBlock >> > (FrameLength, matrix.Col, sigma, Signal_d, ci0_d, ci1_d);


	Blocks = ceil(FrameLength * (__int64)matrix.NonZerosNum / (double)ThreadsPerBlock);
	LdpcDecodeBP_QiInit << <Blocks, ThreadsPerBlock >> > (FrameLength, matrix.NonZerosNum, matrix.Col, matrix.H_IdxCol_d, ci0_d, ci1_d, qij0_d, qij1_d);

	// 迭代
	Blocks = ceil(FrameLength * matrix.NonZerosNum / (double)ThreadsPerBlock);
	for (uint32_t itr = 0; itr < MaxItr; itr++)
	{
		// 正常迭代		
		LdpcDecodeBP_UpdateRji << <Blocks, ThreadsPerBlock >> > (FrameLength, matrix.NonZerosNum, matrix.IdxConnectQij_d, matrix.IdxConnectQij_idx_d, qij0_d, qij1_d, rji0_d, rji1_d);
		LdpcDecodeBP_UpdateQij << <Blocks, ThreadsPerBlock >> > (FrameLength, matrix.NonZerosNum, matrix.Col, matrix.H_IdxCol_d, matrix.IdxConnectRji_d, matrix.IdxConnectRji_idx_d,
			ci0_d, ci1_d, rji0_d, rji1_d, qij0_d, qij1_d);
	}

	Blocks = ceil(FrameLength * matrix.Col / (double)ThreadsPerBlock);
	LdpcDecodeBP_Decide << <Blocks, ThreadsPerBlock >> > (FrameLength, matrix.Col, matrix.NonZerosNum, rji0_d, rji1_d, ci0_d, ci1_d, matrix.H_IdxCol_d, Decode_d);
	cudaMemcpy(Decode, Decode_d, FrameLength * (__int64)matrix.Col * sizeof(uint8_t), cudaMemcpyDeviceToHost);


	cudaFree(Signal_d);
	cudaFree(ci0_d);
	cudaFree(ci1_d);
	cudaFree(qij0_d);
	cudaFree(qij1_d);
	cudaFree(rji0_d);
	cudaFree(rji1_d);
	cudaFree(Decode_d);
}

__global__ void LdpcDecodeBP_CiInit(const uint32_t ParallelFrames, const uint32_t Col, const float sigma, const float* Signal_d, float* ci0_d, float* ci1_d)
{
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= ParallelFrames * Col)
		return;
	ci0_d[idx] = 1 / (1 + exp(2 * Signal_d[idx] / (sigma * sigma)));
	ci1_d[idx] = 1 / (1 + exp(-2 * Signal_d[idx] / (sigma * sigma)));
}

__global__ void LdpcDecodeBP_QiInit(const uint32_t ParallelFrames, const uint32_t NonZerosNum, const uint32_t Col, const uint32_t* IdxCol_d, const float* ci0_d, const float* ci1_d, float* qij0_d, float* qij1_d)
{
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= ParallelFrames * NonZerosNum)
		return;
	uint32_t frame = idx / NonZerosNum;

	qij0_d[idx] = ci0_d[frame * Col + IdxCol_d[idx % NonZerosNum]];
	qij1_d[idx] = ci1_d[frame * Col + IdxCol_d[idx % NonZerosNum]];
}

__global__ void LdpcDecodeBP_UpdateRji(const uint32_t ParallelFrames, const uint32_t NonZerosNum, const uint32_t* IdxConnectQij_d, const uint32_t* Qij_idx_d, const float* qij0_d, const float* qij1_d, float* rji0_d, float* rji1_d)
{
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= ParallelFrames * NonZerosNum)
		return;
	uint32_t frame = idx / NonZerosNum;

	float s = 1;
	for (uint32_t i = Qij_idx_d[(idx % NonZerosNum) * 2]; i < Qij_idx_d[2 * (idx % NonZerosNum) + 1]; i++)
		s = s * (1 - 2 * qij1_d[frame * NonZerosNum + IdxConnectQij_d[i]]);
	rji0_d[idx] = 0.5 * (1.0 + s);
	rji1_d[idx] = 1 - rji0_d[idx];
}

__global__ void LdpcDecodeBP_UpdateQij(const uint32_t ParallelFrames, const uint32_t NonZerosNum, const uint32_t Col, const uint32_t* IdxCol_d, const uint32_t* IdxConnectRji_d, const uint32_t* Rji_idx_d, const float* ci0_d, const float* ci1_d, const float* rji0_d, const float* rji1_d, float* qij0_d, float* qij1_d)
{
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= ParallelFrames * NonZerosNum)
		return;
	uint32_t frame = idx / NonZerosNum;

	float s0 = 1;
	float s1 = 1;
	for (uint32_t i = Rji_idx_d[(idx % NonZerosNum) * 2]; i < Rji_idx_d[(idx % NonZerosNum) * 2 + 1]; i++)
	{
		s0 *= rji0_d[frame * NonZerosNum + IdxConnectRji_d[i]];
		s1 *= rji1_d[frame * NonZerosNum + IdxConnectRji_d[i]];
	}
	qij0_d[idx] = ci0_d[frame * Col + IdxCol_d[idx % NonZerosNum]] * s0;
	qij1_d[idx] = ci1_d[frame * Col + IdxCol_d[idx % NonZerosNum]] * s1;
	qij0_d[idx] = qij0_d[idx] / (qij0_d[idx] + qij1_d[idx]);
	qij1_d[idx] = 1 - qij0_d[idx];
}

__global__ void LdpcDecodeBP_Decide(const uint32_t ParallelFrames, const uint32_t Col, const uint32_t NumOfNonzero, const float* rji0_d, const float* rji1_d, const float* ci0_d, const float* ci1_d, const uint32_t* IdxCol_d, uint8_t* Decode_d)
{
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= ParallelFrames * Col)
		return;

	uint32_t frame = idx / Col;
	float s0 = 1;
	float s1 = 1;
	for (uint32_t j = 0; j < NumOfNonzero; j++)
	{
		if (IdxCol_d[j] == idx % Col)
		{
			s0 *= rji0_d[frame * NumOfNonzero + j];
			s1 *= rji1_d[frame * NumOfNonzero + j];
		}

	}
	s0 *= ci0_d[frame * Col + idx % Col];
	s1 *= ci1_d[frame * Col + idx % Col];
	if (s0 < s1)
		Decode_d[idx] = 1;
	else
		Decode_d[idx] = 0;
}

void LdpcDecode_GPU_LogBP(const CheckMatrix_GPU& matrix, const uint32_t FrameLength, const float sigma, const uint32_t MaxItr, const float* Signal, uint8_t* Decode)
{
	uint32_t Blocks = 0;
	uint32_t ThreadsPerBlock = 128;


	// 变量声明
	float* Signal_d = nullptr, * ci_d = nullptr, * qij_d = nullptr, * rji_d = nullptr;
	uint8_t* Decode_d = nullptr;


	// 变量初始化
	cudaMalloc(&Signal_d, FrameLength * (__int64)matrix.Col * sizeof(float));
	cudaMalloc(&ci_d, FrameLength * (__int64)matrix.Col * sizeof(float));
	cudaMalloc(&qij_d, FrameLength * (__int64)matrix.NonZerosNum * sizeof(float));
	cudaMalloc(&rji_d, FrameLength * (__int64)matrix.NonZerosNum * sizeof(float));
	cudaMalloc(&Decode_d, FrameLength * (__int64)matrix.Col * sizeof(uint8_t));


	// 变量赋值
	cudaMemcpy(Signal_d, Signal, FrameLength * (__int64)matrix.Col * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemset(ci_d, 0, FrameLength * (__int64)matrix.Col * sizeof(float));
	cudaMemset(qij_d, 0, FrameLength * (__int64)matrix.NonZerosNum * sizeof(float));
	cudaMemset(rji_d, 0, FrameLength * (__int64)matrix.NonZerosNum * sizeof(float));
	cudaMemset(Decode_d, 0, FrameLength * (__int64)matrix.Col * sizeof(uint8_t));


	Blocks = ceil(FrameLength * (__int64)matrix.Col / (double)ThreadsPerBlock);
	LdpcDecodeLogBP_CiInit << <Blocks, ThreadsPerBlock >> > (FrameLength, matrix.Col, sigma, Signal_d, ci_d);


	Blocks = ceil(FrameLength * (__int64)matrix.NonZerosNum / (double)ThreadsPerBlock);
	LdpcDecodeLogBP_QiInit << <Blocks, ThreadsPerBlock >> > (FrameLength, matrix.NonZerosNum, matrix.Col, matrix.H_IdxCol_d, ci_d, qij_d);


	// 迭代
	Blocks = ceil(FrameLength * matrix.NonZerosNum / (double)ThreadsPerBlock);
	for (uint32_t itr = 0; itr < MaxItr; itr++)
	{
		// 正常迭代		
		LdpcDecodeLogBP_UpdateRji << <Blocks, ThreadsPerBlock >> > (FrameLength, matrix.NonZerosNum, matrix.IdxConnectQij_d, matrix.IdxConnectQij_idx_d, qij_d, rji_d);
		LdpcDecodeLogBP_UpdateQij << <Blocks, ThreadsPerBlock >> > (FrameLength, matrix.NonZerosNum, matrix.Col, matrix.H_IdxCol_d, matrix.IdxConnectRji_d, matrix.IdxConnectRji_idx_d,
			ci_d, rji_d, qij_d);
	}


	Blocks = ceil(FrameLength * matrix.Col / (double)ThreadsPerBlock);
	LdpcDecodeLogBP_Decide << <Blocks, ThreadsPerBlock >> > (FrameLength, matrix.Col, matrix.NonZerosNum, rji_d, ci_d, matrix.H_IdxCol_d, Decode_d);
	cudaMemcpy(Decode, Decode_d, FrameLength * (__int64)matrix.Col * sizeof(uint8_t), cudaMemcpyDeviceToHost);


	cudaFree(Signal_d);
	cudaFree(ci_d);
	cudaFree(qij_d);
	cudaFree(rji_d);
	cudaFree(Decode_d);
}

__global__ void LdpcDecodeLogBP_CiInit(const uint32_t ParallelFrames, const uint32_t Col, const float sigma, const float* Signal_d, float* ci_d)
{
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= ParallelFrames * Col)
		return;
	ci_d[idx] = -2 * Signal_d[idx] / (sigma * sigma);
}

__global__ void LdpcDecodeLogBP_QiInit(const uint32_t ParallelFrames, const uint32_t NonZerosNum, const uint32_t Col, const uint32_t* IdxCol_d, const float* ci_d, float* qij_d)
{
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= ParallelFrames * NonZerosNum)
		return;
	uint32_t frame = idx / NonZerosNum;
	qij_d[idx] = ci_d[frame * Col + IdxCol_d[idx % NonZerosNum]];
}

__global__ void LdpcDecodeLogBP_UpdateRji(const uint32_t ParallelFrames, const uint32_t NonZerosNum, const uint32_t* IdxConnectQij_d, const uint32_t* Qij_idx_d, const float* qij_d, float* rji_d)
{
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= ParallelFrames * NonZerosNum)
		return;
	uint32_t frame = idx / NonZerosNum;

	float s = 1;
	for (uint32_t i = Qij_idx_d[(idx % NonZerosNum) * 2]; i < Qij_idx_d[2 * (idx % NonZerosNum) + 1]; i++)
		s = s * tanh(qij_d[frame * NonZerosNum + IdxConnectQij_d[i]] / 2);
	rji_d[idx] = 2 * atanh(s);
}

__global__ void LdpcDecodeLogBP_UpdateQij(const uint32_t ParallelFrames, const uint32_t NonZerosNum, const uint32_t Col, const uint32_t* IdxCol_d, const uint32_t* IdxConnectRji_d, const uint32_t* Rji_idx_d, const float* ci_d, const float* rji_d, float* qij_d)
{
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= ParallelFrames * NonZerosNum)
		return;
	uint32_t frame = idx / NonZerosNum;

	float s = 0;
	for (uint32_t i = Rji_idx_d[(idx % NonZerosNum) * 2]; i < Rji_idx_d[(idx % NonZerosNum) * 2 + 1]; i++)
		s += rji_d[frame * NonZerosNum + IdxConnectRji_d[i]];
	qij_d[idx] = ci_d[frame * Col + IdxCol_d[idx % NonZerosNum]] + s;
	if (qij_d[idx] > 1e12)
		qij_d[idx] = 1e12;
	if (qij_d[idx] < -1e12)
		qij_d[idx] = -1e12;
}

__global__ void LdpcDecodeLogBP_Decide(const uint32_t ParallelFrames, const uint32_t Col, const uint32_t NumOfNonzero, const float* rji_d, const float* ci_d, const uint32_t* IdxCol_d, uint8_t* Decode_d)
{
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= ParallelFrames * Col)
		return;

	uint32_t frame = idx / Col;
	float s = 0;
	for (uint32_t j = 0; j < NumOfNonzero; j++)
		if (IdxCol_d[j] == idx % Col)
			s += rji_d[frame * NumOfNonzero + j];
	s += ci_d[frame * Col + idx % Col];
	if (s < 0)
		Decode_d[idx] = 1;
	else
		Decode_d[idx] = 0;
}

bool LdpcSimulation_BP(const CheckMatrix& matrix, const uint64_t Seed, const float EbN0indB, const float Rate,
	const uint64_t MinError, const uint64_t MaxTrans, const uint32_t MaxITR, const uint64_t NumPunchBits, 
	const bool isExitBeforeMaxItr, const uint32_t ExitMethod,
	uint64_t& ErrorBits, uint64_t& TransBits, uint64_t& ErrorFrames, uint64_t& TransFrames)
{
	srand(Seed);

	ErrorBits = 0;
	TransBits = 0;
	ErrorFrames = 0;
	TransFrames = 0;
	uint64_t error = 0;
	uint64_t Remain = MaxTrans;
	float sigma = sqrt(1 / (pow(10.0, EbN0indB / 10) * Rate) / 2.0);

	// 初始化码字、信号、译码结果存储空间
	uint8_t* Code = new uint8_t[matrix.Col];
	float_t* Signal = new float[matrix.Col];
	uint8_t* Decode = new uint8_t[matrix.Col];


	time_t Time_Start = clock();
	time_t Time_Print = clock();
	while (Remain > 0)
	{
		// 发送全零码字
		memset(Code, 0, matrix.Col * sizeof(uint8_t));

		// 加噪
		AddAWGN(matrix.Col, Rate, sigma, Code, Signal);

		// PunchBits
		if (NumPunchBits > 0)
			PunchBits(1, matrix.Col, NumPunchBits, Signal);

		// 译码
		LdpcDecode_BP(matrix, sigma, MaxITR, isExitBeforeMaxItr, Signal, Decode);

		// 错误统计
		error = 0;
		for (uint32_t i = 0; i < matrix.Col; i++)
			error += Decode[i];
		ErrorBits += error;
		if (error != 0)
			ErrorFrames++;
		TransBits += matrix.Col;
		TransFrames += 1;

		if (ExitMethod == 0)
			Remain -= 1;
		else
			Remain = (Remain > matrix.Col) ? (Remain - matrix.Col) : 0;

		double ConsumeTime = (clock() - Time_Start) / (double)CLOCKS_PER_SEC;

		if ((clock() - Time_Print) / (double)CLOCKS_PER_SEC > 0.5)
		{
			printf("Remain=%I64d, BER=%.3e, FER=%.3e, Error=%I64d, Trans=%I64d, RemainTime %.2fs/%.2fs.\r",
				Remain, ErrorBits / (double)TransBits, ErrorFrames / (double)TransFrames,
				(ExitMethod > 0) ? ErrorBits : ErrorFrames, (ExitMethod > 0) ? TransBits : TransFrames,
				ConsumeTime / ((ExitMethod > 0) ? ErrorBits : ErrorFrames)* (MinError - ((ExitMethod > 0) ? ErrorBits : ErrorFrames)),
				ConsumeTime / TransFrames * Remain);
			Time_Print = clock();
		}

		if (ExitMethod == 0)
		{
			if (ErrorFrames > MinError)
				break;
		}
		else
		{
			if (ErrorBits > MinError)
				break;
		}
	}
	delete[]Code;
	delete[]Signal;
	delete[]Decode;
	return true;
}

bool LdpcSimulation_GPU_BP(const CheckMatrix_GPU& matrix, const uint64_t Seed, const float EbN0indB, const float Rate, 
	const uint64_t MinError, const uint64_t MaxTrans, const uint32_t MaxITR, const uint64_t NumPunchBits, 
	const uint32_t ExitMethod, const float MemoryLimit,
	uint64_t& ErrorBits, uint64_t& TransBits, uint64_t& ErrorFrames, uint64_t& TransFrames)
{
	srand(Seed);

	ErrorBits = 0;
	TransBits = 0;
	ErrorFrames = 0;
	TransFrames = 0;
	uint64_t error = 0;
	uint64_t Remain = MaxTrans;
	float sigma = sqrt(1 / (pow(10.0, EbN0indB / 10) * Rate) / 2.0);

	uint64_t curFrame = 2;
	uint64_t Memory = 0;
	time_t Time_Start = clock();
	time_t Time_Print = clock();
	while (Remain > 0)
	{
		// 计算当前循环中并行帧数
		if(curFrame > ((ExitMethod > 0) ? (ceil(Remain/matrix.Col)) : Remain))
			curFrame = Remain;
		Memory = matrix.GPU_Memory;

		// 分配空间
		uint32_t ParallelFrames = curFrame;
		uint8_t* Code = new uint8_t[matrix.Col * (__int64)ParallelFrames];
		float_t* Signal = new float[matrix.Col * (__int64)ParallelFrames];
		uint8_t* Decode = new uint8_t[matrix.Col * (__int64)ParallelFrames];

		// 加噪并译码
		memset(Code, 0, matrix.Col * (__int64)ParallelFrames * sizeof(uint8_t));
		AddAWGN(matrix.Col * (__int64)ParallelFrames, Rate, sigma, Code, Signal);
		if (NumPunchBits > 0)
			PunchBits(ParallelFrames, matrix.Col, NumPunchBits, Signal);
		LdpcDecode_GPU_BP(matrix, ParallelFrames, sigma, MaxITR, Signal, Decode);

		// 内存占用统计
		Memory += ParallelFrames * (__int64)matrix.Col * sizeof(uint8_t) * 3;
		Memory += ParallelFrames * (__int64)matrix.Col * sizeof(float) * 4;
		Memory += ParallelFrames * (__int64)matrix.NonZerosNum * sizeof(float) * 4;

		// 错误统计
		for (uint32_t p = 0; p < ParallelFrames; p++)
		{
			error = 0;
			for (uint32_t i = 0; i < matrix.Col; i++)
				error += Decode[p * matrix.Col + i];
			ErrorBits += error;
			if (error != 0)
				ErrorFrames++;
		}
		TransBits += matrix.Col * ParallelFrames;
		TransFrames += ParallelFrames;

		delete[]Code;
		delete[]Signal;
		delete[]Decode;

		Remain -= (ExitMethod == 0) ? (curFrame) : (curFrame * matrix.Col);
		double ConsumeTime = (clock() - Time_Start) / (double)CLOCKS_PER_SEC;

		if ((clock() - Time_Print) / (double)CLOCKS_PER_SEC > 0.5)
		{
			printf("Remain=%I64d, BER=%.3e, FER=%.3e, Error=%I64d, Trans=%I64d, GpuMemory=%.2fMByte, Throughput=%.3fMbps, RemainTime %.2fs/%.2fs.\r",
				Remain, ErrorBits / (double)TransBits, ErrorFrames / (double)TransFrames,
				(ExitMethod > 0) ? ErrorBits : ErrorFrames, (ExitMethod > 0) ? TransBits : TransFrames,
				Memory / 1024.0 / 1024.0,
				TransBits / ConsumeTime / 1024 / 1024,
				ConsumeTime / ((ExitMethod > 0) ? ErrorBits : ErrorFrames)* (MinError - ((ExitMethod > 0) ? ErrorBits : ErrorFrames)),
				ConsumeTime / TransFrames * Remain);
			Time_Print = clock();
		}

		if (Memory < (MemoryLimit * 1024 * 1024))
			curFrame = ceil(curFrame * 1.1);
		else
			curFrame = ceil(curFrame / 1.005);
		if (ExitMethod == 0)
		{
			if (ErrorFrames > MinError)
				break;
		}
		else
		{
			if (ErrorBits > MinError)
				break;
		}
	}

	// 初始化码字、信号、译码结果存储空间
	uint8_t* Code = new uint8_t[matrix.Col];
	float_t* Signal = new float[matrix.Col];
	uint8_t* Decode = new uint8_t[matrix.Col];
	return true;
}

bool LdpcSimulation_LogBP(const CheckMatrix& matrix, const uint64_t Seed, const float EbN0indB, const float Rate, const uint64_t MinError, const uint64_t MaxTrans, const uint32_t MaxITR, const uint64_t NumPunchBits, const bool isExitBeforeMaxItr, const uint32_t ExitMethod, uint64_t& ErrorBits, uint64_t& TransBits, uint64_t& ErrorFrames, uint64_t& TransFrames)
{
	srand(Seed);

	ErrorBits = 0;
	TransBits = 0;
	ErrorFrames = 0;
	TransFrames = 0;
	uint64_t error = 0;
	uint64_t Remain = MaxTrans;
	float sigma = sqrt(1 / (pow(10.0, EbN0indB / 10) * Rate) / 2.0);

	// 初始化码字、信号、译码结果存储空间
	uint8_t* Code = new uint8_t[matrix.Col];
	float_t* Signal = new float[matrix.Col];
	uint8_t* Decode = new uint8_t[matrix.Col];


	time_t Time_Start = clock();
	time_t Time_Print = clock();
	while (Remain > 0)
	{
		// 发送全零码字
		memset(Code, 0, matrix.Col * sizeof(uint8_t));

		// 加噪
		AddAWGN(matrix.Col, Rate, sigma, Code, Signal);

		// PunchBits
		if (NumPunchBits > 0)
			PunchBits(1, matrix.Col, NumPunchBits, Signal);

		// 译码
		LdpcDecode_LogBP(matrix, sigma, MaxITR, isExitBeforeMaxItr, Signal, Decode);

		// 错误统计
		error = 0;
		for (uint32_t i = 0; i < matrix.Col; i++)
			error += Decode[i];
		ErrorBits += error;
		if (error != 0)
			ErrorFrames++;
		TransBits += matrix.Col;
		TransFrames += 1;

		if (ExitMethod == 0)
			Remain -= 1;
		else
			Remain = (Remain > matrix.Col) ? (Remain - matrix.Col) : 0;

		double ConsumeTime = (clock() - Time_Start) / (double)CLOCKS_PER_SEC;

		if ((clock() - Time_Print) / (double)CLOCKS_PER_SEC > 0.5)
		{
			printf("Remain=%I64d, BER=%.3e, FER=%.3e, Error=%I64d, Trans=%I64d, RemainTime %.2fs/%.2fs.\r",
				Remain, ErrorBits / (double)TransBits, ErrorFrames / (double)TransFrames,
				(ExitMethod > 0) ? ErrorBits : ErrorFrames, (ExitMethod > 0) ? TransBits : TransFrames,
				ConsumeTime / ((ExitMethod > 0) ? ErrorBits : ErrorFrames)* (MinError - ((ExitMethod > 0) ? ErrorBits : ErrorFrames)),
				ConsumeTime / TransFrames * Remain);
			Time_Print = clock();
		}

		if (ExitMethod == 0)
		{
			if (ErrorFrames > MinError)
				break;
		}
		else
		{
			if (ErrorBits > MinError)
				break;
		}
	}
	delete[]Code;
	delete[]Signal;
	delete[]Decode;
	return true;
}

bool LdpcSimulation_GPU_LogBP(const CheckMatrix_GPU& matrix, const uint64_t Seed, const float EbN0indB, const float Rate, const uint64_t MinError, const uint64_t MaxTrans, const uint32_t MaxITR, const uint64_t NumPunchBits, const uint32_t ExitMethod, const float MemoryLimit, uint64_t& ErrorBits, uint64_t& TransBits, uint64_t& ErrorFrames, uint64_t& TransFrames)
{
	srand(Seed);

	ErrorBits = 0;
	TransBits = 0;
	ErrorFrames = 0;
	TransFrames = 0;
	uint64_t error = 0;
	uint64_t Remain = MaxTrans;
	float sigma = sqrt(1 / (pow(10.0, EbN0indB / 10) * Rate) / 2.0);

	uint64_t curFrame = 2;
	uint64_t Memory = 0;
	time_t Time_Start = clock();
	time_t Time_Print = clock();
	while (Remain > 0)
	{
		// 计算当前循环中并行帧数
		if (curFrame > ((ExitMethod > 0) ? (ceil(Remain / matrix.Col)) : Remain))
			curFrame = Remain;
		Memory = matrix.GPU_Memory;

		// 分配空间
		uint32_t ParallelFrames = curFrame;
		uint8_t* Code = new uint8_t[matrix.Col * (__int64)ParallelFrames];
		float_t* Signal = new float[matrix.Col * (__int64)ParallelFrames];
		uint8_t* Decode = new uint8_t[matrix.Col * (__int64)ParallelFrames];

		// 加噪并译码
		memset(Code, 0, matrix.Col * (__int64)ParallelFrames * sizeof(uint8_t));
		AddAWGN(matrix.Col * (__int64)ParallelFrames, Rate, sigma, Code, Signal);
		if (NumPunchBits > 0)
			PunchBits(ParallelFrames, matrix.Col, NumPunchBits, Signal);
		LdpcDecode_GPU_LogBP(matrix, ParallelFrames, sigma, MaxITR, Signal, Decode);

		// 内存占用统计
		Memory += ParallelFrames * (__int64)matrix.Col * sizeof(uint8_t) * 3;
		Memory += ParallelFrames * (__int64)matrix.Col * sizeof(float) * 3;
		Memory += ParallelFrames * (__int64)matrix.NonZerosNum * sizeof(float) * 2;

		// 错误统计
		for (uint32_t p = 0; p < ParallelFrames; p++)
		{
			error = 0;
			for (uint32_t i = 0; i < matrix.Col; i++)
				error += Decode[p * matrix.Col + i];
			ErrorBits += error;
			if (error != 0)
				ErrorFrames++;
		}
		TransBits += matrix.Col * ParallelFrames;
		TransFrames += ParallelFrames;

		delete[]Code;
		delete[]Signal;
		delete[]Decode;

		Remain -= (ExitMethod == 0) ? (curFrame) : (curFrame * matrix.Col);
		double ConsumeTime = (clock() - Time_Start) / (double)CLOCKS_PER_SEC;

		if ((clock() - Time_Print) / (double)CLOCKS_PER_SEC > 0.5)
		{
			printf("Remain=%I64d, BER=%.3e, FER=%.3e, Error=%I64d, Trans=%I64d, GpuMemory=%.2fMByte, Throughput=%.3fMbps, RemainTime %.2fs/%.2fs.\r",
				Remain, ErrorBits / (double)TransBits, ErrorFrames / (double)TransFrames,
				(ExitMethod > 0) ? ErrorBits : ErrorFrames, (ExitMethod > 0) ? TransBits : TransFrames,
				Memory / 1024.0 / 1024.0,
				TransBits / ConsumeTime / 1024 / 1024,
				ConsumeTime / ((ExitMethod > 0) ? ErrorBits : ErrorFrames)* (MinError - ((ExitMethod > 0) ? ErrorBits : ErrorFrames)),
				ConsumeTime / TransFrames * Remain);
			Time_Print = clock();
		}

		if (Memory < (MemoryLimit * 1024 * 1024))
			curFrame = ceil(curFrame * 1.1);
		else
			curFrame = ceil(curFrame / 1.005);
		if (ExitMethod == 0)
		{
			if (ErrorFrames > MinError)
				break;
		}
		else
		{
			if (ErrorBits > MinError)
				break;
		}
	}

	// 初始化码字、信号、译码结果存储空间
	uint8_t* Code = new uint8_t[matrix.Col];
	float_t* Signal = new float[matrix.Col];
	uint8_t* Decode = new uint8_t[matrix.Col];
	return true;
}

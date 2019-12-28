#include "LdpcLib.h"
#include <random>
#include <time.h>

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

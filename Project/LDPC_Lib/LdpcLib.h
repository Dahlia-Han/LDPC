#pragma once

#include "CheckMatrix.h"
#include <cuda.h>
#include <cuda_runtime.h>

/************************************************************************************************************************/
// 添加高斯白噪声
void AddAWGN(const uint32_t CodeBitLength, const float R, const float sigma, const uint8_t *Code, float* Signal);

// Punch Bits
void PunchBits(const uint32_t Frames, const uint32_t CodeBitLength, const uint32_t BitsPunchedPerFrame, float* Signal);

/************************************************************************************************************************/
// BP译码方法
void LdpcDecode_BP(const CheckMatrix& matrix, const float sigma, const uint32_t MaxItr, const bool isExitBeforeMaxItr,
	const float* Signal, uint8_t* Decode);

// LogBP译码方法
void LdpcDecode_LogBP(const CheckMatrix& matrix, const float sigma, const uint32_t MaxItr, const bool isExitBeforeMaxItr,
	const float* Signal, uint8_t* Decode);

/************************************************************************************************************************/
// BP译码方法
void LdpcDecode_GPU_BP(const CheckMatrix_GPU& matrix, const uint32_t FrameLength, const float sigma, const uint32_t MaxItr,
	const float* Signal, uint8_t* Decode);
__global__ void LdpcDecodeBP_CiInit(const uint32_t ParallelFrames, const uint32_t Col, const float sigma, const float* Signal_d, float* ci0_d, float* ci1_d);
__global__ void LdpcDecodeBP_QiInit(const uint32_t ParallelFrames, const uint32_t NonZerosNum, const uint32_t Col,
	const uint32_t* IdxCol_d, const float* ci0_d, const float* ci1_d, float* qij0_d, float* qij1_d);
__global__ void LdpcDecodeBP_UpdateRji(const uint32_t ParallelFrames, const uint32_t NonZerosNum,
	const uint32_t* IdxConnectQij_d, const uint32_t* Qij_idx_d, const float* qij0_d, const float* qij1_d, float* rji0_d, float* rji1_d);
__global__ void LdpcDecodeBP_UpdateQij(const uint32_t ParallelFrames, const uint32_t NonZerosNum, const uint32_t Col, const uint32_t* IdxCol_d,
	const uint32_t* IdxConnectRji_d, const uint32_t* Rji_idx_d, const float* ci0_d, const float* ci1_d, const float* rji0_d, const float* rji1_d, float* qij0_d, float* qij1_d);
__global__ void LdpcDecodeBP_Decide(const uint32_t ParallelFrames, const uint32_t Col, const uint32_t NumOfNonzero,
	const float* rji0_d, const float* rji1_d, const float* ci0_d, const float* ci1_d, const uint32_t* IdxCol_d, uint8_t* Decode_d);

/************************************************************************************************************************/
// LDPC 校验矩阵 性能仿真
extern "C" _declspec(dllexport)
bool LdpcSimulation_BP(const CheckMatrix& matrix, const uint64_t Seed, const float EbN0indB, const float Rate,
	const uint64_t MinError, const uint64_t MaxTrans, const uint32_t MaxITR, const uint64_t NumPunchBits,
	const bool isExitBeforeMaxItr, const uint32_t ExitMethod,
	uint64_t& ErrorBits, uint64_t& TransBits, uint64_t& ErrorFrames, uint64_t& TransFrames);

extern "C" _declspec(dllexport)
bool LdpcSimulation_GPU_BP(const CheckMatrix_GPU & matrix, const uint64_t Seed, const float EbN0indB, const float Rate,
	const uint64_t MinError, const uint64_t MaxTrans, const uint32_t MaxITR, const uint64_t NumPunchBits,
	const uint32_t ExitMethod, const float MemoryLimit,
	uint64_t & ErrorBits, uint64_t & TransBits, uint64_t & ErrorFrames, uint64_t & TransFrames);

extern "C" _declspec(dllexport)
bool LdpcSimulation_LogBP(const CheckMatrix & matrix, const uint64_t Seed, const float EbN0indB, const float Rate,
	const uint64_t MinError, const uint64_t MaxTrans, const uint32_t MaxITR, const uint64_t NumPunchBits,
	const bool isExitBeforeMaxItr, const uint32_t ExitMethod,
	uint64_t & ErrorBits, uint64_t & TransBits, uint64_t & ErrorFrames, uint64_t & TransFrames);




